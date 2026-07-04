"""Verifiers for RLVR + eval + synthetic filtering (Phase 11 test-bench → E1b).

A verifier turns ``(response, target)`` into a checkable verdict. Three customers
share this code (`docs/eval-plan.md` principle 4): the eval battery (pass/fail),
the RLVR reward (a shaped scalar), and synthetic-data filtering (keep-if-correct).

Two layers:
- ``MathVerifier`` / ``gen_arithmetic`` — the Phase-11 arithmetic **test bench**
  (a 110M gets *some* right, so reward has variance and GRPO has a gradient).
  Unchanged; still what `grpo_trainer.py` drives.
- ``CheckResult`` + the ``check_*`` primitives — the general, per-task-type
  verifiers (numeric tolerance, SymPy symbolic equivalence, code unit-tests,
  `pint` units). Dispatched per ``Task`` by ``taskbank.verify``.

Executable checks (code) run in ``sandbox``; symbolic/units parsing runs in-process
via SymPy/pint (model text, not fully untrusted — see ``check_symbolic``).
"""

from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass
from typing import Protocol

from lithos.posttrain.sandbox import ExecutionResult, run_python

_INT = re.compile(r"-?\d+")
# int/float, optional thousands commas, optional scientific exponent.
_NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


class Verifier(Protocol):
    def reward(self, response: str, target: str) -> float: ...


def _repetition(s: str) -> float:
    t = s.split()
    if len(t) < 2:
        return 0.0
    bigrams = list(itertools.pairwise(t))
    return 1.0 - len(set(bigrams)) / len(bigrams)


class MathVerifier:
    """Arithmetic reward. ``correctness`` is the true 0/1 objective; ``reward`` is the
    *shaped* signal GRPO optimizes — correctness, plus a small bonus for emitting a
    parseable number, minus a repetition penalty. The shaping densifies the gradient
    for a weak model (which is usually *wrong* but can learn to at least answer
    cleanly), while **correctness dominates so it can't be farmed**. ALWAYS log
    correctness separately to catch reward hacking (the DPO-v1 Goodhart lesson).

    ``extract`` takes the LAST integer (models restate then answer), so "3 + 5 = 8"
    -> 8 and "The answer is 12." -> 12; commas are stripped first.
    """

    def __init__(self, *, format_bonus: float = 0.3, rep_penalty: float = 0.3) -> None:
        self.format_bonus = format_bonus
        self.rep_penalty = rep_penalty

    def extract(self, response: str) -> int | None:
        nums = _INT.findall(response.replace(",", ""))
        return int(nums[-1]) if nums else None

    def correctness(self, response: str, target: str) -> float:
        pred = self.extract(response)
        return 1.0 if pred is not None and pred == int(target) else 0.0

    def reward(self, response: str, target: str) -> float:
        r = self.correctness(response, target)
        if self.extract(response) is not None:
            r += self.format_bonus  # a clean parseable number, even if wrong
        return r - self.rep_penalty * _repetition(response)  # dock looping


_OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}


def gen_arithmetic(
    n: int, *, seed: int = 0, max_val: int = 20, ops: str = "+-"
) -> list[dict[str, str]]:
    """Procedural arithmetic tasks ``[{"prompt", "answer"}]``.

    Small operands so a 110M has a real shot (reward variance for GRPO). Subtraction
    is kept non-negative for simplicity. Deterministic given ``seed``.
    """
    rng = random.Random(seed)
    tasks: list[dict[str, str]] = []
    for _ in range(n):
        op = rng.choice(list(ops))
        a, b = rng.randint(0, max_val), rng.randint(0, max_val)
        if op == "-" and b > a:
            a, b = b, a
        tasks.append({"prompt": f"What is {a} {op} {b}?", "answer": str(_OPS[op](a, b))})
    return tasks


# --------------------------------------------------------------------------- #
# General per-task-type verifiers (E1b). Dispatched by taskbank.verify.        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CheckResult:
    """A verdict: ``correct`` is the ground truth; ``detail`` explains a failure;
    ``extracted`` is what was parsed from the response (for logging/debugging)."""

    correct: bool
    detail: str = ""
    extracted: str | None = None


def extract_final(response: str) -> str:
    """The model's answer region: everything after the last ``</think>`` (the TIR
    reasoning close, `docs/tir-format.md`), else the whole response. Stripped."""
    tail = response.rsplit("</think>", 1)[-1]
    return tail.strip()


def extract_number(text: str) -> float | None:
    """Last number in ``text`` as a float (commas stripped). Models restate the
    problem then answer, so the last number is the answer."""
    matches = _NUM.findall(text)
    for raw in reversed(matches):
        cleaned = raw.replace(",", "")
        try:
            return float(cleaned)
        except ValueError:  # pragma: no cover — regex shouldn't yield unparseable
            continue
    return None


def check_numeric(
    response: str, answer: str, *, rel_tol: float = 1e-6, abs_tol: float = 1e-9
) -> CheckResult:
    """Extract the response's final number and compare to ``answer`` within tolerance
    (relative *or* absolute — small/zero answers need the absolute floor)."""
    pred = extract_number(extract_final(response))
    if pred is None:
        return CheckResult(False, "no number in response", None)
    try:
        target = float(answer)
    except ValueError:
        return CheckResult(False, f"non-numeric answer {answer!r}", str(pred))
    ok = abs(pred - target) <= max(abs_tol, rel_tol * abs(target))
    return CheckResult(ok, "" if ok else f"{pred} != {target}", str(pred))


def check_symbolic(response: str, answer: str) -> CheckResult:
    """SymPy symbolic equivalence between the response's final expression and
    ``answer`` (e.g. ``1/2`` ≡ ``0.5``, ``x**2-1`` ≡ ``(x-1)*(x+1)``).

    Parses only the extracted final answer (not the whole response) via
    ``parse_expr`` (safer than ``sympify``/``eval``); any parse failure → not
    correct. For fully untrusted input, symbolic checking belongs in the sandbox
    (deferred); model outputs are the intended input here.
    """
    import tokenize

    try:
        from sympy import SympifyError, simplify
        from sympy.parsing.sympy_parser import parse_expr
    except ImportError:  # pragma: no cover — sympy is a core dep
        return CheckResult(False, "sympy unavailable", None)

    candidate = extract_final(response).splitlines()[-1].strip() if response.strip() else ""
    candidate = candidate.rstrip(".")
    if "=" in candidate:  # "x = 5" / "y = (x-1)*(x+1)" -> compare only the RHS expression
        candidate = candidate.split("=")[-1].strip()
    if not candidate:
        return CheckResult(False, "empty response", None)
    # parse_expr on free-form model text can raise from deep in the tokenizer; treat
    # any parse/eval failure as "not correct" rather than crashing the trainer.
    parse_errors = (SyntaxError, TypeError, ValueError, AttributeError, tokenize.TokenError, SympifyError)
    try:
        diff = simplify(parse_expr(candidate) - parse_expr(answer))
        ok = diff == 0
    except parse_errors as e:
        return CheckResult(False, f"parse error: {type(e).__name__}", candidate)
    return CheckResult(bool(ok), "" if ok else f"{candidate} != {answer}", candidate)


def check_code(
    solution: str, tests: str, *, timeout_s: float = 5.0
) -> tuple[CheckResult, ExecutionResult]:
    """Run a candidate ``solution`` against a ``tests`` harness (asserts) in the
    sandbox; correct iff the process exits 0. Returns the execution result too, so
    the caller can inject stderr as a tool result / log the failure."""
    exec_result = run_python(tests, setup=solution, timeout_s=timeout_s)
    detail = "" if exec_result.ok else (exec_result.stderr.strip().splitlines() or [""])[-1]
    return CheckResult(exec_result.ok, detail), exec_result


def check_units(
    response: str, answer: str, *, units: str, rel_tol: float = 1e-3
) -> CheckResult:
    """Value check for a units task: compare the response's number to ``answer``,
    both interpreted in ``units``, via ``pint`` when available.

    KNOWN GAP (do not overclaim): this does **not** parse the response's *own* unit,
    so it does not yet catch a wrong-*dimension* answer (Pa vs kPa) — it verifies
    magnitude only. True dimensional checking ("wrong dimension dies instantly", the
    engineering thesis) needs parsing the response's unit string, converting, and
    comparing; that is a TODO to implement + validate once ``pint`` is installed and
    the units-RLVR path is first exercised (``pint`` is not in the current env).
    Guarded: without ``pint`` it returns not-correct rather than raising.
    """
    try:
        import pint
    except ImportError:
        return CheckResult(False, "pint not installed (units checking unavailable)", None)

    ureg = pint.UnitRegistry()
    pred = extract_number(extract_final(response))
    if pred is None:
        return CheckResult(False, "no number in response", None)
    try:
        expected = float(answer) * ureg(units)
        got = pred * ureg(units)  # TODO: use the response's own parsed unit, not `units`
        ratio = (got / expected).to_base_units()
        ok = abs(float(ratio.magnitude) - 1.0) <= rel_tol and ratio.dimensionless
    except (pint.errors.PintError, ValueError) as e:
        return CheckResult(False, f"unit error: {type(e).__name__}", str(pred))
    return CheckResult(bool(ok), "" if ok else f"{got} != {expected}", str(pred))


def shaped_reward(
    response: str,
    result: CheckResult,
    *,
    format_bonus: float = 0.2,
    rep_penalty: float = 0.3,
) -> float:
    """Domain-agnostic RLVR reward from a ``CheckResult``: correctness dominates,
    plus a small bonus for producing a parseable answer, minus a looping penalty.
    Same shape as ``MathVerifier.reward`` but works for any task type. **Log the raw
    ``result.correct`` separately** — divergence from reward = shaping being farmed.
    """
    r = 1.0 if result.correct else 0.0
    if result.extracted is not None:
        r += format_bonus
    return r - rep_penalty * _repetition(response)


def heuristic_gaming_check(code: str, answer: str) -> str | None:
    """Cheap pre-screen for the most obvious reward hacks (the LLM judge, E1e, is the
    real defense — deferred). Returns a reason string if suspicious, else None.

    Flags: (1) the ground-truth answer printed as a literal with no computation;
    (2) a no-op call (no operators, no function calls). Conservative — a false
    negative just falls through to the real judge; a false positive would wrongly
    zero a reward, so keep the rules tight.
    """
    stripped = code.strip()
    if not stripped:
        return "empty tool call"
    literal = re.search(r"""print\(\s*['"]?\s*(-?[\d.,]+)\s*['"]?\s*\)""", stripped)
    if literal and literal.group(1).replace(",", "") == answer.replace(",", ""):
        has_math = any(op in stripped for op in "+-*/") or "(" in stripped.replace("print(", "")
        if not has_math:
            return "hard-coded answer (printed literal, no computation)"
    if not any(op in stripped for op in "+-*/=") and "(" not in stripped.replace("print(", ""):
        return "no-op call (no computation)"
    return None

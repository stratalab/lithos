"""Verifiers for RLVR (Phase 11) — checkable rewards for a verifiable domain.

A ``Verifier`` turns ``(response, target)`` into a scalar reward. The test-bench
verifier is **arithmetic**: a 110M gets *some* right, so the reward has variance and
GRPO has a gradient (real GSM8K / code would be ~all-zero on a tiny model). The same
interface scales to GSM8K (answer-match) and code (unit tests) on the flagship, and
is shared with the eval battery.
"""

from __future__ import annotations

import random
import re
from typing import Protocol

_INT = re.compile(r"-?\d+")


class Verifier(Protocol):
    def reward(self, response: str, target: str) -> float: ...


def _repetition(s: str) -> float:
    t = s.split()
    if len(t) < 2:
        return 0.0
    bigrams = list(zip(t, t[1:]))
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

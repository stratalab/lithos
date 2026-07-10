"""C-CTX — the fork that picks the architecture (`docs/composite-plan.md` §3).

MassiveDS explains its factual wins with "the LM only needs to extract the answer," and
reports *marginal benefits* on reasoning-heavy benchmarks **for weaker models**. Lithos is
a weak model that reasons. Two causes fit that data, they predict **identical benchmark
numbers**, and they imply **opposite architectures**:

  (a) **capability** — a small model cannot integrate a retrieved fact into a multi-step
      derivation. It had the fact and the room, and still could not use it.
      → retrieval never serves the reasoner. Build in-context RAG for mutability and
        citation only. No decode-loop retrieval, ever.

  (b) **displacement** — the passages ate the context the reasoning needed. The model
      could have used the fact; it no longer had room to think.
      → a *context-free* fact channel is the only route, and kNN-LM's weak per-token gain
        may win net of displacement — justified on Gate 5 (scarce resource), never on
        capability, and still owing Gates 1–4.

  (c) **untrained** — the model was never taught what a ``Reference material:`` block *is*.
      Nothing in SFT or RLVR ever rendered one; only the server did. This predicts the same
      numbers as (a), and it is Gate 1 pointing back at us: the composite must have had a
      fair shot too.
      → retrieval-aware SFT before any capability claim. The ``inline`` arm detects it.

Nobody has separated them, because nobody runs a 500M with a short context window.

## The arms

The mechanic is one subtraction. Under a total budget *L* (a real deployment's sequence
length), ``completion_budget = L - prompt_tokens``. Retrieved passages are part of the
prompt, so they displace reasoning tokens — unless we decline to charge them.

===========  ==================  ==========================  =========================
arm          fact in prompt?     charged to the budget?      what it measures
===========  ==================  ==========================  =========================
``none``     no                  n/a                         the naked baseline
``prepend``  yes, retrieved      **yes** (honest)            what you can actually ship
``oracle``   yes, retrieved      **no** (free)               the upper bound a
                                                             context-free channel buys
``distractor`` yes, irrelevant   yes                         the price of the tokens
                                                             alone, content removed
``inline``   yes, as bare prose  yes                         whether the failure is
                                                             FORMAT, not capability
===========  ==================  ==========================  =========================

**No mechanism can deliver ``oracle`` by prepending.** That is the point. It is an upper
bound, and the gap ``oracle - prepend`` *is* the displacement.

Three comparisons, three questions:

  ``oracle - none``       does a free, relevant fact help at all?     (capability)
  ``oracle - prepend``    does charging its cost destroy the gain?    (displacement)
  ``prepend - distractor`` was it the content, or just the tokens?    (control)
  ``inline - oracle``     could it use the fact in a FAMILIAR shape?  (untrained)

## Pre-registration

``diagnose`` implements the rule from `docs/c0-spec.md` §7 **before any number exists**, so
the verdict cannot be argued into shape after the fact. It is called on a summary; it does
not get to look at the episodes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from statistics import mean
from typing import Any

from lithos.posttrain.reference import REFERENCE_FORMAT_VERSION, ContextPlacement
from lithos.posttrain.taskbank import Task, verify
from lithos.retrieval.types import Retriever
from lithos.serve.composite import CompositeModel

DEFAULT_BUDGETS: tuple[int, ...] = (64, 128, 256, 512)


class Arm(StrEnum):
    NONE = "none"
    PREPEND = "prepend"
    ORACLE = "oracle"
    DISTRACTOR = "distractor"
    #: Same fact, same charging as `prepend`, but rendered as bare prose before the question —
    #: a shape every LM has seen a billion times. Isolates format from capability.
    INLINE = "inline"


@dataclass(frozen=True)
class EpisodeRecord:
    """One (arm, budget, task) run. The unit of measurement for tool use and retrieval is
    the *answer*, not the token — hence episodes, not a per-token table."""

    served_model_digest: str
    datastore_version: str | None
    arm: str
    total_token_budget: int
    task_id: str
    kind: str
    family_id: str | None
    correct: bool
    detail: str
    prompt_tokens: int
    context_tokens: int
    completion_budget: int
    completion_tokens: int
    reasoning_tokens: int
    n_tool_calls: int
    truncated: bool
    cited_source_ids: tuple[str, ...]
    context_placement: str
    reference_format_version: str


#: arm -> (placement, is the context charged to the completion budget?)
_ARM_SPEC: dict[Arm, tuple[ContextPlacement, bool]] = {
    Arm.NONE: (ContextPlacement.BLOCK, True),  # no context; placement is moot
    Arm.PREPEND: (ContextPlacement.BLOCK, True),
    Arm.ORACLE: (ContextPlacement.BLOCK, False),  # the fact is free
    Arm.DISTRACTOR: (ContextPlacement.BLOCK, True),
    Arm.INLINE: (ContextPlacement.INLINE, True),  # same charging as prepend; familiar shape
}


def _retriever_for(arm: Arm, retriever: Retriever | None, distractor: Retriever | None):
    if arm is Arm.NONE:
        return None
    if arm is Arm.DISTRACTOR:
        if distractor is None:
            raise ValueError("the distractor arm needs a distractor_retriever")
        return distractor
    if retriever is None:
        raise ValueError(f"arm {arm.value!r} needs a retriever")
    return retriever


def run_cctx(
    model: Any,
    tok: Any,
    tasks: Sequence[Task],
    *,
    weights_sha256: str,
    retriever: Retriever | None = None,
    distractor_retriever: Retriever | None = None,
    arms: Sequence[Arm] = (Arm.NONE, Arm.PREPEND, Arm.ORACLE),
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    context_token_budget: int = 256,
    max_tool_calls: int = 2,
    device: str = "cpu",
    use_cache: bool = True,
    **gen_kwargs: Any,
) -> list[EpisodeRecord]:
    """Run every (arm × budget × task) and return one record each. No training, eval only."""
    records: list[EpisodeRecord] = []
    for arm in arms:
        r = _retriever_for(arm, retriever, distractor_retriever)
        placement, charge = _ARM_SPEC[arm]
        cm = CompositeModel(model, tok, weights_sha256=weights_sha256, retriever=r, device=device)
        for budget in budgets:
            for task in tasks:
                res = cm.generate(
                    task.prompt,
                    context_token_budget=context_token_budget if r is not None else 0,
                    total_token_budget=budget,
                    # The whole experiment, in two flags.
                    charge_context=charge,
                    placement=placement,
                    max_tool_calls=max_tool_calls,
                    use_cache=use_cache,
                    **gen_kwargs,
                )
                check = verify(res.text, task)
                records.append(
                    EpisodeRecord(
                        served_model_digest=cm.id.digest(),
                        datastore_version=cm.id.datastore_version,
                        arm=arm.value,
                        total_token_budget=budget,
                        task_id=task.id,
                        kind=task.kind,
                        family_id=getattr(task, "family_id", None),
                        correct=bool(check.correct),
                        detail=str(check.detail or ""),
                        prompt_tokens=res.prompt_tokens,
                        context_tokens=res.context_tokens,
                        completion_budget=res.completion_budget,
                        completion_tokens=res.completion_tokens,
                        reasoning_tokens=res.reasoning_tokens,
                        n_tool_calls=len(res.tool_calls),
                        truncated=res.truncated,
                        cited_source_ids=tuple(c.source_id for c in res.citations),
                        context_placement=res.context_placement,
                        reference_format_version=REFERENCE_FORMAT_VERSION,
                    )
                )
    return records


def summarize(records: Iterable[EpisodeRecord]) -> dict[str, dict[str, float]]:
    """``{"arm@budget": {accuracy, n, mean_context_tokens, mean_reasoning_tokens, ...}}``."""
    groups: dict[str, list[EpisodeRecord]] = {}
    for rec in records:
        groups.setdefault(f"{rec.arm}@{rec.total_token_budget}", []).append(rec)
    out: dict[str, dict[str, float]] = {}
    for key, rs in sorted(groups.items()):
        out[key] = {
            "accuracy": mean(1.0 if r.correct else 0.0 for r in rs),
            "n": float(len(rs)),
            "mean_context_tokens": mean(r.context_tokens for r in rs),
            "mean_completion_budget": mean(r.completion_budget for r in rs),
            "mean_reasoning_tokens": mean(r.reasoning_tokens for r in rs),
            "starved_frac": mean(1.0 if r.completion_budget == 0 else 0.0 for r in rs),
        }
    return out


@dataclass(frozen=True)
class Diagnosis:
    verdict: str  # "capability" | "displacement" | "untrained" | "inconclusive"
    oracle_gain: float  # oracle - none, at the largest budget
    displacement: float  # oracle - prepend, at the smallest budget
    converges: float  # oracle - prepend, at the largest budget (want ~0)
    content_effect: float | None  # prepend - distractor, largest budget (None if not run)
    format_effect: float | None  # inline - oracle, largest budget (None if not run)
    rationale: str


def diagnose(
    summary: dict[str, dict[str, float]],
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    eps: float = 0.05,
) -> Diagnosis:
    """The pre-registered decision rule (`docs/c0-spec.md` §7). Written before the numbers.

    * **untrained** — a free fact in our ``Reference material:`` block does not help, but the
      *same fact as bare prose* does. The model can use a fact; it cannot read our format.
      **Checked before `capability`, because the two are indistinguishable without it.**
      Nothing in SFT or RLVR ever rendered a reference block — only the server did.
    * **capability** — a free, relevant fact does not help even at the largest budget, *and*
      not inline either. ``oracle - none <= eps``. Retrieval does not serve the reasoner.
    * **displacement** — the free fact *does* help, and charging its cost destroys the gain
      at a tight budget while the two converge at a generous one. That is the signature of
      the context window binding, and the only thing that resurrects decode-loop retrieval.
    * **inconclusive** — anything else. Say so; do not squint.

    Note the asymmetry: without the ``inline`` arm, a ``capability`` verdict is **not
    earned** — it is merely the absence of an alternative we failed to test. `diagnose` says
    so rather than pretending otherwise.
    """
    lo, hi = min(budgets), max(budgets)

    def acc(arm: Arm, b: int) -> float | None:
        cell = summary.get(f"{arm.value}@{b}")
        return None if cell is None else cell["accuracy"]

    needed = [acc(a, b) for a in (Arm.NONE, Arm.PREPEND, Arm.ORACLE) for b in (lo, hi)]
    if any(v is None for v in needed):
        return Diagnosis(
            "inconclusive", 0.0, 0.0, 0.0, None, None, "missing arm×budget cells; cannot decide"
        )

    oracle_gain = acc(Arm.ORACLE, hi) - acc(Arm.NONE, hi)  # type: ignore[operator]
    displacement = acc(Arm.ORACLE, lo) - acc(Arm.PREPEND, lo)  # type: ignore[operator]
    converges = acc(Arm.ORACLE, hi) - acc(Arm.PREPEND, hi)  # type: ignore[operator]

    d_hi = acc(Arm.DISTRACTOR, hi)
    content = None if d_hi is None else acc(Arm.PREPEND, hi) - d_hi  # type: ignore[operator]

    i_hi = acc(Arm.INLINE, hi)
    fmt = None if i_hi is None else i_hi - acc(Arm.ORACLE, hi)  # type: ignore[operator]
    inline_gain = None if i_hi is None else i_hi - acc(Arm.NONE, hi)  # type: ignore[operator]

    if oracle_gain <= eps and inline_gain is not None and inline_gain > eps:
        verdict, why = (
            "untrained",
            f"a free fact in a reference block bought {oracle_gain:+.3f}, but the same fact as "
            f"bare prose bought {inline_gain:+.3f} at budget {hi}. The model can use a fact; it "
            f"cannot read our format. Nothing in SFT or RLVR ever rendered a "
            f"`{REFERENCE_FORMAT_VERSION}` block. Retrieval-aware SFT is required before any "
            f"capability claim — this is Gate 1 pointing back at us.",
        )
    elif oracle_gain <= eps:
        earned = (
            "" if inline_gain is not None else " NOTE: the `inline` arm was not run, so "
            "`untrained` was never ruled out and this verdict is not earned."
        )
        verdict, why = (
            "capability",
            f"a free, relevant fact bought {oracle_gain:+.3f} at budget {hi} (<= eps={eps}). "
            "The model cannot use the fact even when it costs nothing. Retrieval does not "
            f"serve the reasoner; build it for mutability and citation only.{earned}",
        )
    elif displacement > eps and abs(converges) <= eps:
        verdict, why = (
            "displacement",
            f"a free fact bought {oracle_gain:+.3f}, charging its cost removed "
            f"{displacement:+.3f} of it at budget {lo}, and the arms converge at {hi} "
            f"({converges:+.3f}). The context window is binding. A context-free fact channel "
            "is the only route — justified on Gate 5, never on capability.",
        )
    else:
        verdict, why = (
            "inconclusive",
            f"oracle_gain={oracle_gain:+.3f}, displacement={displacement:+.3f}, "
            f"converges={converges:+.3f} fit neither pre-registered pattern.",
        )
    return Diagnosis(verdict, oracle_gain, displacement, converges, content, fmt, why)


def write_episodes(records: Iterable[EpisodeRecord], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), separators=(",", ":")) + "\n")
    return p

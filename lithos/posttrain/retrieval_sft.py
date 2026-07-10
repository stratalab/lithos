"""Retrieval-aware SFT: teach the model to *use* a reference block — and to ignore noise.

The `inline` arm of C-CTX only *detects* the `untrained` cause (`docs/composite-plan.md` §3):
the model has never seen a ``Reference material:`` block, so a `capability` verdict is not
believable. This module *closes* that gap. It renders (query, contexts, answer) examples into
messages-JSONL that the existing `build_sft_corpus` consumes — and it renders them through the
**same** `lithos.posttrain.reference.build_messages` the server uses, so the model trains on
byte-for-byte the format it will be served. A format the trainer invents is one the model
never saw twice.

## The curriculum — three kinds, and why each exists

``GROUNDED``
    The answer *is* in the passages. Teaches extraction and reasoning over the block.
``DISTRACTOR``
    The passages are irrelevant. The target is an **abstention** (`ABSTAIN_ANSWER`). Teaches
    the single most valuable retrieval behaviour: *say the sources don't contain it* rather
    than confabulate from them. A model that cannot ignore noise is worse than one with no
    retrieval — and abstention is what keeps a composite attributable.
``MIXED``
    Relevant *and* irrelevant passages together; the answer uses only the relevant one.
    Teaches filtering within the block, which is what real top-k retrieval actually returns.

## Where this meets the tier gate

A retrieval-aware example is the "read the book, write your own explanation" path made
concrete (`docs/chisel-tier-gate.md`): the reference block may be a **restricted** textbook
passage (it lives in the loss-masked *prompt*), while the **answer** is the gradient-bearing
target and must be `open` / `lawful` / `synthetic-verified`. So a source file of these
examples is declared `prompt_tier="restricted"`, `tier="synthetic-verified"`, `grounded_on=…`
— exactly the seam `SFTSourceSpec` already carries. `grounded_source_ids` computes the union
for that field.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from lithos.posttrain.reference import ContextPlacement, build_messages

#: The canonical abstention target. Fixed string, so the model learns *one* refusal shape
#: rather than a scatter — and so a served model's abstention is recognisable to the harness.
ABSTAIN_ANSWER = "The reference material provided does not contain the information needed to answer this question."


class ExampleKind(StrEnum):
    GROUNDED = "grounded"
    DISTRACTOR = "distractor"
    MIXED = "mixed"


@dataclass(frozen=True)
class Context:
    """One reference passage. ``source_id`` feeds `grounded_on`; ``relevant`` marks whether
    the answer should draw on it (false for the noise in a MIXED/DISTRACTOR example)."""

    text: str
    source_id: str = ""
    relevant: bool = True


@dataclass(frozen=True)
class RetrievalExample:
    query: str
    answer: str
    contexts: tuple[Context, ...] = ()
    kind: ExampleKind = ExampleKind.GROUNDED
    system: str | None = None

    def relevant_source_ids(self) -> tuple[str, ...]:
        return tuple(c.source_id for c in self.contexts if c.relevant and c.source_id)


def to_messages_record(
    example: RetrievalExample, *, placement: ContextPlacement = ContextPlacement.BLOCK
) -> dict[str, Any]:
    """Render one example into an SFT messages record.

    The user turn carries the reference block via the shared renderer; the assistant turn is
    the answer (the only loss target). ``kind`` rides along for provenance — `build_sft_corpus`
    reads only ``messages`` and ignores the rest.
    """
    msgs = build_messages(
        example.query,
        [c.text for c in example.contexts],
        system=example.system,
        placement=placement,
    )
    msgs.append({"role": "assistant", "content": example.answer})
    return {"messages": msgs, "kind": example.kind.value}


def make_distractor(
    query: str, distractors: Sequence[Context], *, system: str | None = None
) -> RetrievalExample:
    """An example whose passages are all irrelevant; the target is abstention."""
    noise = tuple(Context(c.text, c.source_id, relevant=False) for c in distractors)
    return RetrievalExample(
        query=query,
        answer=ABSTAIN_ANSWER,
        contexts=noise,
        kind=ExampleKind.DISTRACTOR,
        system=system,
    )


def make_mixed(grounded: RetrievalExample, distractors: Sequence[Context]) -> RetrievalExample:
    """Interleave irrelevant passages into a grounded example; the answer is unchanged.

    The distractors are inserted **after** the relevant ones so the answer's supporting passage
    is not always first — the model must actually read, not learn "use passage [1]".
    """
    if grounded.kind is not ExampleKind.GROUNDED:
        raise ValueError("make_mixed expects a GROUNDED example to add noise to")
    noise = tuple(Context(c.text, c.source_id, relevant=False) for c in distractors)
    return RetrievalExample(
        query=grounded.query,
        answer=grounded.answer,
        contexts=(*grounded.contexts, *noise),
        kind=ExampleKind.MIXED,
        system=grounded.system,
    )


def grounded_source_ids(examples: Iterable[RetrievalExample]) -> list[str]:
    """The union of `source_id`s the answers actually draw on — for `SFTSourceSpec.grounded_on`.

    Distractor sources are excluded: nothing was learned *from* them, so grounding an answer on
    them would be a false provenance claim (the exact dishonesty the tier gate exists to prevent).
    """
    seen: dict[str, None] = {}
    for ex in examples:
        for sid in ex.relevant_source_ids():
            seen.setdefault(sid, None)
    return list(seen)


@dataclass
class RetrievalSFTStats:
    n: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    grounded_on: list[str] = field(default_factory=list)


def write_retrieval_sft(
    examples: Sequence[RetrievalExample],
    path: str | Path,
    *,
    placement: ContextPlacement = ContextPlacement.BLOCK,
) -> RetrievalSFTStats:
    """Write examples as messages-JSONL consumable by `build_sft_corpus`.

    A balanced mix matters: all-GROUNDED teaches the model that the block is always right, and
    it will then trust an irrelevant block at serve time. The caller composes the mix; this
    only records what it was, so the manifest can show it.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    by_kind: dict[str, int] = {}
    with open(p, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(to_messages_record(ex, placement=placement)) + "\n")
            by_kind[ex.kind.value] = by_kind.get(ex.kind.value, 0) + 1
    return RetrievalSFTStats(
        n=len(examples), by_kind=by_kind, grounded_on=grounded_source_ids(examples)
    )

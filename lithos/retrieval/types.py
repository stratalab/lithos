"""The retrieval seam: what a retriever hands the composite.

These live here, not in ``lithos/serve``, so the dependency runs one way:
``serve`` → ``retrieval``. A retriever never needs to know it is being served.

Retrieval lives **above the token stream** (`docs/composite-plan.md` §1): passages are
prepended to the prompt and *cited*, never interpolated into the decode loop. The cost
is **context** — the scarcest resource a 500M has — which is why ``retrieve`` takes a
``token_budget`` rather than just a ``k``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from lithos.data.tiers import DATASTORE_ALLOWED_TIERS


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk, carrying the provenance keys that join to Chisel and Petra.

    ``text_sha256`` is the **parent document's** hash — the join key fixed in the
    reconciliation. ``chunk_sha256`` identifies this span within it, which is what makes
    a citation exact rather than document-level.
    """

    text: str
    source_id: str
    record_id: str
    text_sha256: str
    tier: str
    score: float = 0.0
    chunk_sha256: str = ""


@dataclass(frozen=True)
class RetrievedContext:
    passages: tuple[Passage, ...] = ()
    #: What the retriever *believes* the passages cost. The composite re-measures against
    #: its own tokenizer (BPE can merge across the passage/query seam) and that number wins.
    tokens_used: int = 0


@runtime_checkable
class Retriever(Protocol):
    """Anything that turns a query into passages, under a **token budget**."""

    version: str

    def retrieve(self, query: str, *, token_budget: int) -> RetrievedContext: ...


class StubRetriever:
    """Fixed passages, ranked as given. Enough to exercise every seam without an index.

    Enforces the datastore half of the tier gate: `restricted` passages are welcome —
    the model *cites* what it consults, which is the whole point of moving books out of
    the weights (`docs/chisel-tier-gate.md`). `unknown` is not: an undeclared provenance
    cannot be cited.
    """

    version = "stub-v0"

    def __init__(self, passages: Sequence[Passage]) -> None:
        for p in passages:
            assert_datastore_tier(p.tier, where=f"passage {p.source_id!r}")
        self._passages = tuple(passages)

    def retrieve(self, query: str, *, token_budget: int) -> RetrievedContext:
        if token_budget <= 0:
            return RetrievedContext()
        # A real retriever ranks by similarity to `query`; the stub preserves order and
        # lets the composite do the budget accounting, since only it owns the tokenizer.
        return RetrievedContext(passages=self._passages)


def assert_datastore_tier(tier: str, *, where: str = "") -> None:
    """The datastore accepts every declared tier, ``restricted`` included — it is cited,
    never trained on. It does not accept ``unknown``."""
    if tier not in DATASTORE_ALLOWED_TIERS:
        raise ValueError(
            f"{where or 'passage'} has tier={tier!r}; the datastore accepts "
            f"{sorted(DATASTORE_ALLOWED_TIERS)} (restricted is allowed here — it is cited, "
            f"never trained on; unknown is not — an undeclared provenance cannot be cited)"
        )

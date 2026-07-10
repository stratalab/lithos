"""The retriever: query → ranked passages, under a context-token budget.

Implements the ``Retriever`` protocol against a ``Datastore``. It returns *candidates*,
ranked; the composite trims them to fit, because only the composite owns the LM tokenizer
and therefore only it can measure what a passage truly costs once rendered into a prompt.

The budget still reaches here, though: it bounds how many candidates are worth returning.
Handing back forty passages for a 128-token budget is wasted work and a misleading
``tokens_used``.
"""

from __future__ import annotations

from lithos.retrieval.embed import Embedder
from lithos.retrieval.index import Datastore
from lithos.retrieval.types import Passage, RetrievedContext


class DocumentRetriever:
    """Exact top-k retrieval over document chunks."""

    def __init__(
        self,
        store: Datastore,
        embedder: Embedder,
        *,
        top_k: int = 4,
        min_score: float = 0.0,
    ) -> None:
        if store.embedder_version != embedder.version:
            raise ValueError(
                f"embedder mismatch: datastore was built with {store.embedder_version!r} but "
                f"queries would be embedded with {embedder.version!r}. Vectors from different "
                f"embedders are not comparable, and the failure looks like bad retrieval."
            )
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score
        self.version = f"exact-{embedder.version}-k{top_k}"

    @property
    def datastore_version(self) -> str:
        return self.store.version

    def _budgeted_k(self, token_budget: int) -> int:
        """How many candidates could plausibly fit, capped at ``top_k``."""
        per_chunk = max(1, int(self.store.chunk_params.get("max_tokens", 128)))
        by_budget = max(1, token_budget // per_chunk)
        return min(self.top_k, by_budget)

    def retrieve(self, query: str, *, token_budget: int) -> RetrievedContext:
        if token_budget <= 0 or len(self.store.chunks) == 0:
            return RetrievedContext()

        qvec = self.embedder.encode([query])[0]
        hits = self.store.index.search(qvec, self._budgeted_k(token_budget))

        passages: list[Passage] = []
        tokens = 0
        for i, score in hits:
            if score <= self.min_score:
                continue  # a zero/negative match is noise; prepending it only costs context
            c = self.store.chunks[i]
            passages.append(
                Passage(
                    text=c.text,
                    source_id=c.source_id,
                    record_id=c.record_id,
                    text_sha256=c.text_sha256,
                    tier=c.tier,
                    score=score,
                    chunk_sha256=c.chunk_sha256,
                )
            )
            tokens += c.n_tokens
        return RetrievedContext(passages=tuple(passages), tokens_used=tokens)

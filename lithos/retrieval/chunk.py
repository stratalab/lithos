"""Document → chunks, with provenance carried through.

Chunking is **token-based, on the LM's own tokenizer**, because the thing we are
budgeting is the LM's context window. A chunk whose size is measured in characters gives
you no control over what it costs to prepend, and context cost is the whole question
C-CTX asks (`docs/composite-plan.md` §3).

Consequence worth stating: the chunk boundaries — and therefore the datastore's identity —
depend on the tokenizer. That is why ``Datastore.version`` folds in the tokenizer name.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from lithos.data.tiers import TIER_UNKNOWN


@dataclass(frozen=True)
class Chunk:
    """A span of one document, plus everything needed to cite it."""

    text: str
    source_id: str
    record_id: str
    text_sha256: str  # the PARENT document's hash: the join key to Chisel/Petra
    tier: str
    chunk_index: int
    chunk_sha256: str  # this span's hash: what makes the citation exact
    n_tokens: int


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_document(
    doc: dict[str, Any],
    tok: Any,
    *,
    max_tokens: int = 128,
    overlap_tokens: int = 16,
) -> list[Chunk]:
    """Split one canonical record into overlapping token windows.

    Reads provenance from ``metadata`` (``source_id``, ``text_sha256``) per the R2
    contract, falling back to the record's own ``id``. ``tier`` is a top-level key
    (`lithos.data.tiers`).

    Overlap exists so a fact straddling a boundary is retrievable from at least one
    window. It costs storage, not context — the composite still prepends whole chunks.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens >= max_tokens:
        raise ValueError(f"overlap_tokens ({overlap_tokens}) must be < max_tokens ({max_tokens})")

    text = doc.get("text") or ""
    if not text:
        return []

    meta = doc.get("metadata") or {}
    source_id = meta.get("source_id") or doc.get("source") or ""
    record_id = meta.get("record_id") or str(doc.get("id", ""))
    doc_sha = meta.get("text_sha256") or _sha256(text)
    tier = doc.get("tier") or TIER_UNKNOWN

    ids = tok.encode(text).ids
    stride = max_tokens - overlap_tokens
    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(ids), stride)):
        window = ids[start : start + max_tokens]
        if not window:
            break
        span = tok.decode(window, skip_special_tokens=True)
        if not span.strip():
            continue
        chunks.append(
            Chunk(
                text=span,
                source_id=source_id,
                record_id=record_id,
                text_sha256=doc_sha,
                tier=tier,
                chunk_index=i,
                chunk_sha256=_sha256(span),
                n_tokens=len(window),
            )
        )
        if start + max_tokens >= len(ids):
            break  # the last window covered the tail; don't emit a pure-overlap tail
    return chunks

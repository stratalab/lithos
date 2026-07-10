"""The datastore: chunks + vectors + an identity you can pin.

Three properties earn their keep here, and each has a test:

1. **The tier gate, datastore half.** ``restricted`` content is *welcome* — it is cited,
   never trained on. ``unknown`` is refused: an undeclared provenance cannot be cited.
   (`docs/chisel-tier-gate.md`.)

2. **``version`` is derived from content, never declared.** It folds in the embedder, the
   chunk parameters, the tokenizer, and the sha256 of every chunk. Change one document and
   the datastore version changes, therefore ``served_model_id`` changes, therefore two
   results from different corpora can never be silently pooled. This is what makes C5 —
   bisecting a corpus-caused regression — possible at all.

3. **``assert_disjoint_from``.** If the eval set is inside the datastore, retrieval returns
   the answer verbatim and every number you measure is worthless. `docs/c0-spec.md` §5.1
   calls this "the single highest-value line of code in this spec". Exact-hash disjointness
   is necessary and *not sufficient* — it does not remove n-gram overlap — so C-CTX also
   reports gain bucketed by overlap.

``NumpyExactIndex`` is exact inner-product search. At document-chunk scale (~1e6 chunks ×
512 dims ≈ 2 GB fp32, or far less quantised) exact search is fast enough and has no recall
knob to get wrong. An ANN index drops in behind the same call.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lithos.retrieval.chunk import Chunk, chunk_document
from lithos.retrieval.embed import Embedder
from lithos.retrieval.types import assert_datastore_tier
from lithos.utils.io import ensure_dir, write_json


class NumpyExactIndex:
    """Exact inner-product search over L2-normalised rows (so IP == cosine)."""

    def __init__(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if len(self) == 0 or k <= 0:
            return []
        scores = self.vectors @ np.asarray(query, dtype=np.float32).ravel()
        k = min(k, len(self))
        # argpartition finds the top-k without a full sort, then we sort just those.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]


class Datastore:
    """Chunks, their vectors, and a content-derived version."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        *,
        embedder_version: str,
        chunk_params: dict[str, Any],
    ) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"{len(chunks)} chunks but {vectors.shape[0]} vectors")
        for c in chunks:
            assert_datastore_tier(c.tier, where=f"chunk {c.chunk_sha256[:12]} of {c.source_id!r}")
            # Recompute, never trust. A chunk carrying a hash of text it does not contain
            # would make `version` a hash of a lie, and every downstream attestation with
            # it. A Chunk whose sha disagrees with its text simply cannot exist.
            actual = hashlib.sha256(c.text.encode("utf-8")).hexdigest()
            if actual != c.chunk_sha256:
                raise ValueError(
                    f"chunk {c.chunk_index} of {c.source_id!r}: the content and its identity "
                    f"disagree (sha256 of text is {actual[:16]}…, chunk_sha256 says "
                    f"{c.chunk_sha256[:16]}…)"
                )
        self.chunks = tuple(chunks)
        self.index = NumpyExactIndex(vectors)
        self.embedder_version = embedder_version
        self.chunk_params = dict(chunk_params)
        self.version = self._compute_version()

    # ── identity ──────────────────────────────────────────────────────────────

    def _compute_version(self) -> str:
        payload = {
            "embedder": self.embedder_version,
            "chunk_params": self.chunk_params,
            # Sorted, so the version is invariant to ingest order but not to content.
            "chunks": sorted(c.chunk_sha256 for c in self.chunks),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "ds:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # ── gates ─────────────────────────────────────────────────────────────────

    def assert_disjoint_from(self, eval_text_sha256s: Iterable[str]) -> None:
        """Fail loudly if any eval document is in the datastore.

        "Corpus-internal" means the *training* corpus, never the held-out set. If the eval
        text is retrievable, retrieval returns the answer verbatim and the measurement is
        worthless. Necessary, not sufficient — hash disjointness leaves n-gram overlap.
        """
        evalset = set(eval_text_sha256s)
        hit = {c.text_sha256 for c in self.chunks} & evalset
        if hit:
            raise ValueError(
                f"{len(hit)} eval document(s) are present in the datastore "
                f"(e.g. {sorted(hit)[0][:16]}…). Retrieval would return the answer verbatim. "
                f"'Corpus-internal' means the training corpus, never the held-out set."
            )

    # ── build / persist ───────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        docs: Iterable[dict[str, Any]],
        tok: Any,
        embedder: Embedder,
        *,
        tokenizer_name: str = "unknown",
        max_tokens: int = 128,
        overlap_tokens: int = 16,
    ) -> Datastore:
        chunks: list[Chunk] = []
        for doc in docs:
            chunks.extend(
                chunk_document(doc, tok, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
            )
        vectors = embedder.encode([c.text for c in chunks])
        return cls(
            chunks,
            vectors,
            embedder_version=embedder.version,
            chunk_params={
                "max_tokens": max_tokens,
                "overlap_tokens": overlap_tokens,
                "tokenizer": tokenizer_name,
            },
        )

    def manifest(self) -> dict[str, Any]:
        """The datastore's attestation, mirroring the corpus manifest's ``tiers`` block.

        It says plainly how much restricted content is indexed. That is not an admission —
        it is the point. Restricted content in the *datastore* is cited on every use; the
        same content in the *weights* would be unattributable.
        """
        tiers: dict[str, int] = {}
        for c in self.chunks:
            tiers[c.tier] = tiers.get(c.tier, 0) + 1
        return {
            "datastore_version": self.version,
            "embedder": self.embedder_version,
            "chunk_params": self.chunk_params,
            "num_chunks": len(self.chunks),
            "num_documents": len({c.text_sha256 for c in self.chunks}),
            "tiers": tiers,
        }

    def save(self, output_dir: str | Path) -> dict[str, Any]:
        out = ensure_dir(output_dir)
        np.save(out / "vectors.npy", self.index.vectors)
        with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.__dict__, separators=(",", ":")) + "\n")
        man = self.manifest()
        write_json(out / "datastore_manifest.json", man)
        return man

    @classmethod
    def load(cls, input_dir: str | Path) -> Datastore:
        d = Path(input_dir)
        vectors = np.load(d / "vectors.npy")
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        man = json.loads((d / "datastore_manifest.json").read_text(encoding="utf-8"))
        store = cls(
            chunks,
            vectors,
            embedder_version=man["embedder"],
            chunk_params=man["chunk_params"],
        )
        if store.version != man["datastore_version"]:
            raise ValueError(
                f"datastore version mismatch on load: recomputed {store.version} != "
                f"{man['datastore_version']} on disk. The content and its identity disagree."
            )
        return store

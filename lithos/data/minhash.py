"""MinHash + LSH near-deduplication (PRD §8.8.3).

Exact-hash dedup misses *near*-duplicates — boilerplate, re-hosted pages, tiny
edits — which waste model capacity and inflate eval contamination. MinHash
estimates Jaccard similarity over document shingles; LSH banding finds candidate
near-dupes in ~O(1); a Jaccard check on the signatures confirms them.

`MinHashDeduper` exposes the same ``is_duplicate(text) -> bool`` / ``stats()``
seam as ``ExactDocumentDeduper``, so the corpus pipeline gains near-dedup without
changing any callers (the seam promised in Phase 3 / dedup.py).

Memory note: kept-document signatures are retained (``num_perm`` uint64 each), so
the full corpus needs a few GB of RAM — fine as an offline data-build step.
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

# Mersenne prime M31. With 31-bit shingle hashes and 31-bit coefficients, every
# (a*h + b) stays < 2^62 < 2^64, so uint64 modular arithmetic never overflows.
_PRIME = (1 << 31) - 1


def _shingle_hashes(text: str, k: int) -> np.ndarray:
    """Deterministic hashes of the *set* of word k-shingles (reduced mod _PRIME)."""
    words = text.split()
    if len(words) < k:
        grams = [" ".join(words)] if words else []
    else:
        grams = [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]
    if not grams:
        return np.empty(0, dtype=np.uint64)
    hashes = np.fromiter(
        (zlib.crc32(g.encode("utf-8")) for g in grams), dtype=np.uint64, count=len(grams)
    )
    return np.unique(hashes % _PRIME)  # Jaccard is over the shingle *set*


class MinHashConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_perm: int = 128
    bands: int = 16  # rows/band = num_perm // bands; together they set the LSH threshold
    shingle_size: int = 5
    threshold: float = 0.8  # estimated-Jaccard cutoff for "duplicate"
    seed: int = 1


class MinHasher:
    """MinHash signatures under a fixed, seeded permutation family (reproducible)."""

    def __init__(self, num_perm: int = 128, shingle_size: int = 5, seed: int = 1) -> None:
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, _PRIME, size=num_perm).astype(np.uint64)
        self.b = rng.randint(0, _PRIME, size=num_perm).astype(np.uint64)

    def signature(self, text: str) -> np.ndarray:
        sh = _shingle_hashes(text, self.shingle_size)
        if sh.size == 0:
            return np.full(self.num_perm, _PRIME, dtype=np.uint64)
        # (num_perm, n_shingles): min over shingles of (a*h + b) mod prime.
        hashed = (np.outer(self.a, sh) + self.b[:, None]) % _PRIME
        return hashed.min(axis=1)


def estimate_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """MinHash Jaccard estimate = fraction of equal signature positions."""
    return float(np.mean(a == b))


class MinHashDeduper:
    """Streaming near-dedup: LSH-band candidate retrieval + Jaccard confirmation."""

    def __init__(self, cfg: MinHashConfig | None = None) -> None:
        self.cfg = cfg or MinHashConfig()
        if self.cfg.num_perm % self.cfg.bands != 0:
            raise ValueError(
                f"num_perm ({self.cfg.num_perm}) must be divisible by bands ({self.cfg.bands})"
            )
        self.rows = self.cfg.num_perm // self.cfg.bands
        self.hasher = MinHasher(self.cfg.num_perm, self.cfg.shingle_size, self.cfg.seed)
        self._buckets: list[dict[bytes, list[int]]] = [{} for _ in range(self.cfg.bands)]
        self._sigs: list[np.ndarray] = []
        self.duplicates = 0

    def _band_keys(self, sig: np.ndarray) -> list[bytes]:
        return [
            sig[i * self.rows : (i + 1) * self.rows].tobytes() for i in range(self.cfg.bands)
        ]

    def is_duplicate(self, text: str) -> bool:
        """True if ``text`` is a near-duplicate of a previously-kept document."""
        sig = self.hasher.signature(text)
        keys = self._band_keys(sig)
        candidates: set[int] = set()
        for band, key in enumerate(keys):
            candidates.update(self._buckets[band].get(key, ()))
        for cid in candidates:
            if estimate_jaccard(sig, self._sigs[cid]) >= self.cfg.threshold:
                self.duplicates += 1
                return True
        cid = len(self._sigs)
        self._sigs.append(sig)
        for band, key in enumerate(keys):
            self._buckets[band].setdefault(key, []).append(cid)
        return False

    def stats(self) -> dict[str, Any]:
        return {"unique": len(self._sigs), "duplicates": self.duplicates}

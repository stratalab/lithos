"""Embedders: text → a unit vector.

``HashingEmbedder`` is a dependency-free, deterministic lexical embedder (signed feature
hashing over word tokens, log-tf weighted, L2-normalised). It is a real baseline, not a
mock: for retrieving a passage that shares vocabulary with the query — which is most of
what a STEM datastore is asked to do — it works, and it costs no model, no download, and
no GPU.

A trained encoder drops in behind the ``Embedder`` protocol when we want semantic match.
Nothing above this line changes; only ``Datastore.version`` does, which is exactly the
point of folding the embedder's version into it.

**It hashes with blake2b, never Python's ``hash()``.** ``hash()`` on a ``str`` is salted
per process (PYTHONHASHSEED), so a datastore built in one process would not match a query
embedded in another. That bug is silent and it would look like bad retrieval.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    version: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """(n_texts, dim), L2-normalised rows."""
        ...


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _bucket_and_sign(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    h = int.from_bytes(digest, "big")
    return h % dim, (1.0 if (h >> 63) & 1 else -1.0)


class HashingEmbedder:
    """Signed feature hashing. Deterministic across processes and machines."""

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.version = f"hashing-b2b-{dim}-v1"

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts = Counter(_tokens(text))
        for token, tf in counts.items():
            idx, sign = _bucket_and_sign(token, self.dim)
            # log-tf damps a term repeated 50 times without discarding the repetition.
            vec[idx] += sign * (1.0 + math.log(tf))
        norm = float(np.linalg.norm(vec))
        # An empty/stopword-only text has no direction. Return zeros rather than NaN; it
        # will score 0 against everything, which is the honest answer.
        return vec / norm if norm > 0.0 else vec

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._one(t) for t in texts])

"""Per-domain quality classifier v0 (docs/quality-classifiers.md §4).

FastText-style architecture, owned: hashed word n-gram features (unigrams +
bigrams, hashing trick) into a linear regression head trained with SGD —
pure numpy, no new dependencies. Predicts the 0-5 rubric score; corpus-scale
scoring is a sparse dot product (fast on CPU, no GPU in the data pipeline).

v0 is the plumbing-and-smoke-test tier: trained on the ~1k pilot labels it
gives directional metrics only. The real model trains on the 5k/domain run;
if this architecture ceilings there, §4 says upgrade that domain to an
embedder head — measured, not assumed.
"""

from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")  # words + symbol chars (math/code signal)


def tokenize(text: str, *, max_tokens: int = 4000) -> list[str]:
    return _TOKEN_RE.findall(text.lower())[:max_tokens]


def featurize(text: str, *, dim_bits: int = 18) -> np.ndarray:
    """L2-normalized hashed counts of word 1-2 grams. Dense output (2^dim_bits)."""
    dim = 1 << dim_bits
    toks = tokenize(text)
    vec = np.zeros(dim, dtype=np.float32)
    prev = None
    for t in toks:
        vec[zlib.crc32(t.encode()) & (dim - 1)] += 1.0
        if prev is not None:
            vec[zlib.crc32(f"{prev}_{t}".encode()) & (dim - 1)] += 1.0
        prev = t
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


@dataclass
class QualityModel:
    """Linear head over hashed n-grams: score(text) = w·phi(text) + b."""

    weights: np.ndarray  # (dim,)
    bias: float
    dim_bits: int
    domain: str
    rubric_version: int
    metrics: dict

    def score(self, text: str) -> float:
        return float(self.weights @ featurize(text, dim_bits=self.dim_bits) + self.bias)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, weights=self.weights, bias=np.float64(self.bias),
            dim_bits=np.int64(self.dim_bits),
            meta=np.frombuffer(json.dumps({
                "domain": self.domain, "rubric_version": self.rubric_version,
                "metrics": self.metrics,
            }).encode(), dtype=np.uint8),
        )

    @classmethod
    def load(cls, path: str | Path) -> QualityModel:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(bytes(z["meta"]).decode())
        return cls(weights=z["weights"], bias=float(z["bias"]), dim_bits=int(z["dim_bits"]),
                   domain=meta["domain"], rubric_version=meta["rubric_version"],
                   metrics=meta["metrics"])


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman via pearson on ranks (average-rank ties handling omitted — fine here)."""
    ra, rb = np.argsort(np.argsort(a)).astype(np.float64), np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / denom) if denom else 0.0


def evaluate(pred: np.ndarray, y: np.ndarray) -> dict:
    clipped = np.clip(np.round(pred), 0, 5)
    return {
        "n": len(y),
        "mae": float(np.abs(pred - y).mean()),
        "within_1": float((np.abs(clipped - y) <= 1).mean()),
        "spearman": round(_rank_correlation(pred, y), 3),
        "baseline_mae": float(np.abs(y.mean() - y).mean()),  # predict-the-mean strawman
    }


def train(
    texts: list[str],
    scores: list[int],
    *,
    domain: str,
    rubric_version: int,
    dim_bits: int = 18,
    epochs: int = 30,
    lr: float = 0.5,
    l2: float = 1e-5,
    holdout_frac: float = 0.2,
    seed: int = 13,
) -> QualityModel:
    """SGD ridge regression on hashed features; metrics from a held-out split."""
    if len(texts) != len(scores) or len(texts) < 20:
        raise ValueError("need >=20 aligned (text, score) pairs")
    rng = np.random.default_rng(seed)
    X = np.stack([featurize(t, dim_bits=dim_bits) for t in texts])
    y = np.asarray(scores, dtype=np.float64)
    idx = rng.permutation(len(y))
    n_hold = max(1, int(len(y) * holdout_frac))
    hold, tr = idx[:n_hold], idx[n_hold:]

    w = np.zeros(X.shape[1], dtype=np.float64)
    b = float(y[tr].mean())  # start at the mean; SGD learns the deviations
    for _ in range(epochs):
        for i in rng.permutation(tr):
            err = (w @ X[i] + b) - y[i]
            w -= lr * (err * X[i] + l2 * w)
            b -= lr * 0.01 * err
    pred_hold = X[hold] @ w + b
    metrics = {"holdout": evaluate(pred_hold, y[hold]),
               "train": evaluate(X[tr] @ w + b, y[tr])}
    return QualityModel(weights=w.astype(np.float32), bias=b, dim_bits=dim_bits,
                        domain=domain, rubric_version=rubric_version, metrics=metrics)

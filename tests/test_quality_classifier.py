"""Tests for the v0 quality classifier (lithos/data/quality_classifier.py)."""

from __future__ import annotations

import random

import numpy as np
import pytest
from lithos.data.quality_classifier import QualityModel, evaluate, featurize, tokenize, train


def _doc(quality: str, i: int) -> str:
    rng = random.Random(f"{quality}-{i}")
    if quality == "high":
        words = ["derivation", "equation", "theorem", "proof", "units", "integral",
                 "boundary", "velocity", "newtons", "solve"]
    else:
        words = ["click", "subscribe", "amazing", "offer", "celebrity", "gossip",
                 "buy", "sale", "wow", "trending"]
    return " ".join(rng.choice(words) for _ in range(120))


@pytest.fixture(scope="module")
def synthetic_model() -> QualityModel:
    texts = [_doc("high", i) for i in range(60)] + [_doc("low", i) for i in range(60)]
    scores = [4] * 60 + [1] * 60
    return train(texts, scores, domain="test", rubric_version=1, epochs=15)


def test_tokenize_keeps_symbols():
    assert tokenize("f(x) = x^2") == ["f", "(", "x", ")", "=", "x", "^", "2"]


def test_featurize_deterministic_and_normalized():
    a, b = featurize("derivation of the equation"), featurize("derivation of the equation")
    assert np.array_equal(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0)
    assert featurize("").sum() == 0.0  # empty doc -> zero vector, no NaN


def test_train_separates_synthetic_quality(synthetic_model):
    m = synthetic_model
    assert m.score(_doc("high", 999)) > m.score(_doc("low", 999)) + 1.0
    assert m.metrics["holdout"]["mae"] < m.metrics["holdout"]["baseline_mae"]
    assert m.metrics["holdout"]["within_1"] == 1.0


def test_save_load_roundtrip(tmp_path, synthetic_model):
    path = tmp_path / "m.npz"
    synthetic_model.save(path)
    loaded = QualityModel.load(path)
    doc = _doc("high", 5)
    assert loaded.score(doc) == pytest.approx(synthetic_model.score(doc), abs=1e-5)
    assert loaded.domain == "test"
    assert loaded.metrics["holdout"]["n"] > 0


def test_train_rejects_tiny_or_mismatched_input():
    with pytest.raises(ValueError):
        train(["a"] * 5, [1] * 5, domain="x", rubric_version=1)
    with pytest.raises(ValueError):
        train(["a"] * 30, [1] * 29, domain="x", rubric_version=1)


def test_evaluate_metrics_sane():
    y = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
    m = evaluate(y.copy(), y)  # perfect predictions
    assert m["mae"] == 0.0
    assert m["within_1"] == 1.0
    assert m["spearman"] == pytest.approx(1.0)

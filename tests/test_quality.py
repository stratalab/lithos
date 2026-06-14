"""Tests for quality-score filtering (Phase 10)."""

from lithos.data.documents import normalize
from lithos.data.quality import QualityConfig, QualityFilter


def test_normalize_carries_quality_score_only_when_requested():
    rec = {"text": "hello world", "score": 4.2}
    doc = normalize(rec, source="s", subset=None, language="en", license="x", quality_field="score")
    assert doc is not None and doc["quality_score"] == 4.2
    # without quality_field, the score is not carried
    plain = normalize(rec, source="s", subset=None, language="en", license="x")
    assert plain is not None and "quality_score" not in plain


def test_threshold_keeps_above_drops_below():
    f = QualityFilter(QualityConfig(enabled=True, threshold=3.0))
    assert f.keep({"quality_score": 3.5}) is True
    assert f.keep({"quality_score": 3.0}) is True  # >= is inclusive
    assert f.keep({"quality_score": 2.9}) is False
    s = f.stats()
    assert s["kept"] == 2 and s["dropped"] == 1


def test_missing_score_drops_by_default():
    f = QualityFilter(QualityConfig(enabled=True, threshold=3.0))
    assert f.keep({"text": "no score field"}) is False
    assert f.stats()["missing_score"] == 1 and f.stats()["dropped"] == 1


def test_missing_score_can_be_assumed():
    f = QualityFilter(QualityConfig(enabled=True, threshold=3.0, missing_score=3.0))
    assert f.keep({"text": "no score field"}) is True  # assumed 3.0 >= 3.0
    assert f.stats()["missing_score"] == 1 and f.stats()["kept"] == 1

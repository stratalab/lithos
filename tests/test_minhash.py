"""Tests for MinHash near-deduplication (PRD §8.8.3)."""

import numpy as np
import pytest
from lithos.data.dedup import ExactDocumentDeduper
from lithos.data.minhash import (
    MinHashConfig,
    MinHashDeduper,
    MinHasher,
    estimate_jaccard,
)


def test_signature_is_deterministic_across_instances():
    text = "the quick brown fox jumps over the lazy dog several times this morning"
    a = MinHasher(seed=1).signature(text)
    b = MinHasher(seed=1).signature(text)
    assert np.array_equal(a, b)


def test_jaccard_estimate_identical_is_one_and_approximates_truth():
    h = MinHasher(num_perm=256, seed=3)
    t = " ".join(f"w{i}" for i in range(200))
    assert estimate_jaccard(h.signature(t), h.signature(t)) == 1.0
    # ~50% shingle overlap -> estimate should be in the right ballpark
    a = h.signature(" ".join(f"w{i}" for i in range(200)))
    b = h.signature(" ".join(f"w{i}" for i in range(100, 300)))
    assert 0.2 < estimate_jaccard(a, b) < 0.5  # true Jaccard ~0.33


def test_identical_document_is_flagged_on_second_sight():
    d = MinHashDeduper()
    doc = " ".join(f"token{i}" for i in range(80))
    assert d.is_duplicate(doc) is False  # first time -> kept
    assert d.is_duplicate(doc) is True  # exact repeat -> duplicate
    assert d.stats() == {"unique": 1, "duplicates": 1}


def test_near_duplicate_small_edit_is_flagged():
    d = MinHashDeduper()  # default threshold 0.8
    base = " ".join(f"token{i}" for i in range(120))
    assert d.is_duplicate(base) is False
    near = base + " a few extra words appended to the very end"  # Jaccard ~0.96
    assert d.is_duplicate(near) is True


def test_distinct_documents_are_not_flagged():
    d = MinHashDeduper()
    assert d.is_duplicate(" ".join(f"alpha{i}" for i in range(120))) is False
    assert d.is_duplicate(" ".join(f"beta{i}" for i in range(120))) is False
    assert d.stats() == {"unique": 2, "duplicates": 0}


def test_drops_into_exact_dedup_seam():
    # Same interface as ExactDocumentDeduper, so it swaps into the pipeline unchanged.
    for deduper in (ExactDocumentDeduper(), MinHashDeduper()):
        assert callable(deduper.is_duplicate) and callable(deduper.stats)


def test_num_perm_must_divide_bands():
    with pytest.raises(ValueError, match="divisible"):
        MinHashDeduper(MinHashConfig(num_perm=100, bands=16))

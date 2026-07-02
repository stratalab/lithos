"""Tests for cross-corpus overlap estimation (lithos/data/overlap.py)."""

from __future__ import annotations

import random

import numpy as np
import pytest
from lithos.data.overlap import (
    build_sample,
    format_report,
    normalize_text_hash,
    normalize_url,
    pair_overlap,
)


def _doc(i: int, corpus: str, words: int = 80) -> dict:
    rng = random.Random(f"{corpus}-{i}")
    text = " ".join(f"w{rng.randrange(10_000)}" for _ in range(words))
    return {"text": text, "url": f"https://example.com/{corpus}/{i}"}


def _shared_doc(i: int, words: int = 80) -> dict:
    rng = random.Random(f"shared-{i}")
    text = " ".join(f"s{rng.randrange(10_000)}" for _ in range(words))
    return {"text": text, "url": f"https://shared.org/doc/{i}"}


# -- normalization -----------------------------------------------------------


def test_normalize_url_variants_collide():
    forms = [
        "https://www.example.com/a/b/",
        "http://example.com/a/b",
        "HTTPS://EXAMPLE.COM/a/b#frag",
    ]
    hashes = {normalize_url(u) for u in forms}
    assert len(hashes) == 1
    assert normalize_url("") is None


def test_normalize_text_hash_ws_and_case_insensitive():
    assert normalize_text_hash("Hello  World\n") == normalize_text_hash("hello world")
    assert normalize_text_hash("hello world") != normalize_text_hash("hello mars")


# -- build_sample ------------------------------------------------------------


def test_build_sample_respects_size_and_min_chars():
    docs = [{"text": "short", "url": "u"}] + [_doc(i, "a") for i in range(50)]
    s = build_sample("a", docs, total_docs=1000, sample_size=20,
                     text_field="text", url_field="url")
    assert s.n == 20  # short doc skipped, capped at sample_size
    assert s.sigs.shape[1] == 128
    assert s.url_hashes is not None


def test_build_sample_nested_url_field():
    docs = [{"text": "x " * 200, "metadata": {"url": f"https://n.com/{i}"}} for i in range(3)]
    s = build_sample("a", docs, total_docs=3, sample_size=3,
                     text_field="text", url_field="metadata.url")
    assert s.url_hashes is not None
    assert len(set(s.url_hashes.tolist())) == 3  # distinct urls hashed


def test_build_sample_no_url_field():
    s = build_sample("a", [_doc(i, "a") for i in range(5)], total_docs=10,
                     sample_size=5, text_field="text", url_field=None)
    assert s.url_hashes is None


def test_build_sample_empty_raises():
    with pytest.raises(ValueError):
        build_sample("a", [], total_docs=10, sample_size=5,
                     text_field="text", url_field=None)


# -- pair matching + the estimator -------------------------------------------


def _two_corpora(shared: int, only_a: int, only_b: int):
    """Two fully-sampled corpora with `shared` common docs."""
    a_docs = [_shared_doc(i) for i in range(shared)] + [_doc(i, "a") for i in range(only_a)]
    b_docs = [_shared_doc(i) for i in range(shared)] + [_doc(i, "b") for i in range(only_b)]
    n_a, n_b = shared + only_a, shared + only_b
    a = build_sample("A", a_docs, total_docs=n_a, sample_size=n_a,
                     text_field="text", url_field="url")
    b = build_sample("B", b_docs, total_docs=n_b, sample_size=n_b,
                     text_field="text", url_field="url")
    return a, b


def test_full_sample_overlap_recovered_exactly():
    # Samples == full corpora → estimator inversion factor is 1; estimates exact.
    a, b = _two_corpora(shared=30, only_a=70, only_b=170)
    r = pair_overlap(a, b)
    assert r.url_matched_a == r.url_matched_b == 30
    assert r.text_matched_a == 30
    assert r.near_matched_a == 30
    ab, ba = r.estimate(r.near_matched_a, r.near_matched_b)
    assert ab == pytest.approx(0.30, abs=0.001)  # 30 of A's 100
    assert ba == pytest.approx(0.15, abs=0.001)  # 30 of B's 200


def test_subsample_estimator_inverts_inclusion_probability():
    # A fully sampled; B sampled at 50% → raw matches halve, estimate must not.
    a, b_full = _two_corpora(shared=40, only_a=60, only_b=160)
    idx = np.arange(0, b_full.n, 2)  # every other doc: 100 of 200, 20 shared
    b_half = type(b_full)(
        name="B", total_docs=200, sigs=b_full.sigs[idx],
        text_hashes=b_full.text_hashes[idx], url_hashes=b_full.url_hashes[idx],
    )
    r = pair_overlap(a, b_half)
    assert r.near_matched_a == 20  # raw matches halved...
    ab, _ = r.estimate(r.near_matched_a, r.near_matched_b)
    assert ab == pytest.approx(0.40, abs=0.001)  # ...but estimate recovers 40%


def test_near_dup_catches_perturbed_text_that_exact_misses():
    base = _shared_doc(1, words=200)
    words = base["text"].split()
    words[10] = "CHANGED"  # ~0.5% perturbation → Jaccard still ≫ 0.8
    perturbed = {"text": " ".join(words), "url": "https://elsewhere.net/x"}
    a = build_sample("A", [base], total_docs=1, sample_size=1,
                     text_field="text", url_field="url")
    b = build_sample("B", [perturbed] + [_doc(i, "b") for i in range(20)],
                     total_docs=21, sample_size=21, text_field="text", url_field="url")
    r = pair_overlap(a, b)
    assert r.url_matched_a == 0
    assert r.text_matched_a == 0
    assert r.near_matched_a == 1  # only the near-dup matcher sees it


def test_no_url_field_notes_and_skips():
    a, b = _two_corpora(shared=5, only_a=5, only_b=5)
    b.url_hashes = None
    r = pair_overlap(a, b)
    assert r.url_matched_a is None
    assert any("url" in n for n in r.notes)


def test_low_match_count_flagged_noisy():
    a, b = _two_corpora(shared=3, only_a=200, only_b=200)
    r = pair_overlap(a, b)
    assert any("noisy" in n for n in r.notes)


def test_estimate_caps_at_one():
    a, b = _two_corpora(shared=10, only_a=0, only_b=0)
    r = pair_overlap(a, b)
    ab, ba = r.estimate(r.near_matched_a, r.near_matched_b)
    assert ab == 1.0
    assert ba == 1.0


# -- report ------------------------------------------------------------------


def test_report_contains_both_directions_and_percentages():
    a, b = _two_corpora(shared=30, only_a=70, only_b=170)
    report = format_report([pair_overlap(a, b)])
    assert "A → B" in report
    assert "B → A" in report
    assert "30.0%" in report
    assert "15.0%" in report

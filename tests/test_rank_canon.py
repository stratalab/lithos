"""Tests for the canon-candidate ranker (scripts/rank_canon.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "rank_canon", Path(__file__).parent.parent / "scripts" / "rank_canon.py"
)
assert spec and spec.loader
rank_canon = importlib.util.module_from_spec(spec)
sys.modules["rank_canon"] = rank_canon
spec.loader.exec_module(rank_canon)


def _meta(key, citations=10, title="T", isbn="1"):
    return {"key": key, "citations": str(citations), "title": title,
            "author": "A", "year": "2000", "isbn": isbn, "kind": "cite book"}


def test_specificity_filters_cross_domain_generics():
    weighted = {
        "canon": {"physics": 10.0, "math": 1.0},        # spec 0.91 -> kept
        "generic": {"physics": 10.0, "math": 10.0, "cs": 10.0, "eng": 10.0},  # 0.25 -> cut
    }
    meta = {k: _meta(k) for k in weighted}
    rows = rank_canon.rank_domain("physics", weighted, meta)
    assert [r["key"] for r in rows] == ["canon"]


def test_score_orders_by_weight_times_specificity():
    weighted = {
        "big_specific": {"physics": 10.0},               # score 10
        "bigger_shared": {"physics": 12.0, "math": 9.0},  # spec .57 -> score ~6.9
    }
    meta = {k: _meta(k) for k in weighted}
    rows = rank_canon.rank_domain("physics", weighted, meta)
    assert [r["key"] for r in rows] == ["big_specific", "bigger_shared"]


def test_min_citations_filter():
    weighted = {"rare": {"physics": 5.0}}
    meta = {"rare": _meta("rare", citations=2)}
    assert rank_canon.rank_domain("physics", weighted, meta) == []


def test_clean_title_strips_wiki_markup():
    assert rank_canon.clean_title("[[Artificial Intelligence: A Modern Approach]]") == (
        "Artificial Intelligence: A Modern Approach"
    )
    assert rank_canon.clean_title("[[Foo|Bar]] and ''Baz''") == "Bar and Baz"

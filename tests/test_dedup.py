"""Tests for lithos.data.dedup — exact document and line dedup (PRD §8.8)."""

from lithos.data.dedup import ExactDocumentDeduper, ExactLineDeduper


def test_exact_document_dedup():
    d = ExactDocumentDeduper()
    assert d.is_duplicate("hello") is False
    assert d.is_duplicate("world") is False
    assert d.is_duplicate("hello") is True  # repeat
    stats = d.stats()
    assert stats["unique"] == 2
    assert stats["duplicates"] == 1


def test_exact_line_dedup():
    d = ExactLineDeduper()
    first = d.filter_lines("a\nb\nc")
    assert first == "a\nb\nc"
    # b and c already seen; only new line d survives (blank lines preserved)
    second = d.filter_lines("b\nd\nc\ne")
    assert second.splitlines() == ["d", "e"]

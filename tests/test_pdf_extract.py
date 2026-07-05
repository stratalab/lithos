"""Tests for EX-2 record assembly (lithos/data/pdf_extract.py).

The Docling conversion itself is heavy (models + a real PDF) and is validated by an
end-to-end run, not pytest; make_record is a pure function and IS tested here.
"""

from __future__ import annotations

from lithos.data.pdf_extract import make_record


def test_make_record_anchors_to_canon() -> None:
    r = make_record(
        "# Chapter 1\n\n$q_*(a) \\doteq \\mathbb{E}[R_t \\mid A_t = a]$",
        source_id="sutton-barto-rl", title="Reinforcement Learning: An Introduction",
        domain="code", license="grey", tier="grey", n_pages=548, ocr=False,
        docling_version="2.x",
    )
    assert r["id"] == "pdf:sutton-barto-rl"
    assert r["subset"] == "sutton-barto-rl"
    assert r["source"] == "canon"
    # CH-12: the record is anchored to its Lithos Canon (seed_index) entry.
    assert r["metadata"]["source_id"] == "sutton-barto-rl"
    assert r["metadata"]["pages"] == 548
    assert r["metadata"]["extractor"] == "docling"
    assert r["metadata"]["tier"] == "grey"
    assert r["metadata"]["formula_enrichment"] is True
    assert "\\mathbb{E}" in r["text"]  # LaTeX math preserved verbatim

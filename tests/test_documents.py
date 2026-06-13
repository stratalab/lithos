"""Tests for lithos.data.documents — schema normalization and JSONL reading."""

import json

from lithos.data.documents import DocumentSource, iter_documents, normalize


def test_normalize_fills_defaults():
    doc = normalize({"text": "hello"}, source="src", subset=None, language="en", license="unk")
    assert doc is not None
    assert doc["text"] == "hello"
    assert doc["source"] == "src"
    assert doc["language"] == "en"
    assert doc["metadata"] == {}


def test_normalize_drops_textless_records():
    assert normalize({"text": ""}, source="s", subset=None, language="en", license="u") is None
    assert normalize({}, source="s", subset=None, language="en", license="u") is None


def test_normalize_keeps_record_provenance():
    doc = normalize(
        {"text": "x", "source": "real", "language": "fr"},
        source="default",
        subset=None,
        language="en",
        license="u",
    )
    assert doc is not None
    assert doc["source"] == "real"  # record value wins over default
    assert doc["language"] == "fr"


def test_iter_jsonl_with_limit(tmp_path):
    p = tmp_path / "docs.jsonl"
    p.write_text("\n".join(json.dumps({"text": f"doc {i}"}) for i in range(5)) + "\n")
    src = DocumentSource(kind="jsonl", paths=[str(p)], source_name="t", limit=3)
    docs = list(iter_documents(src))
    assert len(docs) == 3
    assert docs[0]["text"] == "doc 0"
    assert docs[0]["source"] == "t"

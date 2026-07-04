"""Tests for lithos.data.documents — schema normalization and JSONL reading."""

import json

import pytest
from lithos.data.documents import DocumentSource, _expand_paths, iter_documents, normalize


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


def test_expand_paths_glob_and_literal(tmp_path):
    (tmp_path / "a.jsonl").write_text("")
    (tmp_path / "b.jsonl").write_text("")
    (tmp_path / "c.txt").write_text("")
    # Glob matches sorted; literal path passes through unchanged.
    got = _expand_paths([str(tmp_path / "*.jsonl"), "/literal/path.jsonl"])
    assert got == [str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl"), "/literal/path.jsonl"]


def test_expand_paths_empty_glob_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no files match"):
        _expand_paths([str(tmp_path / "*.parquet")])


def test_iter_jsonl_via_glob(tmp_path):
    for i in range(2):
        (tmp_path / f"shard{i}.jsonl").write_text(json.dumps({"text": f"d{i}"}) + "\n")
    src = DocumentSource(kind="jsonl", paths=[str(tmp_path / "*.jsonl")], source_name="t")
    docs = list(iter_documents(src))
    assert {d["text"] for d in docs} == {"d0", "d1"}

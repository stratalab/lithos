"""Mirror contract test — the drift alarm after the producer tier moved to Chisel.

Chisel writes canonical records as ``.jsonl.zst`` (docs/chisel-lithos-r2-contract.md
§3.1); Lithos must read them back through ``documents.read_jsonl`` + ``normalize``
without loss. If the record schema drifts on either side of the repo boundary, this
fails loud. Chisel runs the mirror of this against a Lithos-shaped record in its own CI.
"""

from __future__ import annotations

import json

import zstandard
from lithos.data.documents import normalize, read_jsonl


def _write_shard(path, records: list[dict]) -> None:
    with open(path, "wb") as fh:
        w = zstandard.ZstdCompressor(level=10).stream_writer(fh)
        for r in records:
            w.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
        w.close()


def test_chisel_record_roundtrips(tmp_path) -> None:
    rec = {
        "id": "stack-python:abc",
        "text": "import numpy as np\n",
        "source": "the-stack-stem",
        "subset": "python/python",
        "language": "en",
        "license": "mit",
        "metadata": {"source_id": "the-stack-stem", "domain": "code"},
    }
    shard = tmp_path / "part-0.jsonl.zst"
    _write_shard(shard, [rec])

    got = list(read_jsonl([str(shard)]))
    assert got == [rec]  # byte-for-byte round trip through the reader

    doc = normalize(got[0], source="fallback", subset=None, language="en", license="unknown")
    assert doc is not None
    assert doc["id"] == rec["id"]
    assert doc["text"] == rec["text"]
    assert doc["source"] == "the-stack-stem"                 # the record's own value wins
    assert doc["metadata"]["source_id"] == "the-stack-stem"  # CH-12 canon anchor preserved


def test_textless_record_dropped(tmp_path) -> None:
    """normalize drops a record with no usable text — the reader's one filter."""
    shard = tmp_path / "bad.jsonl.zst"
    _write_shard(shard, [{"id": "x", "text": ""}])
    rec = next(iter(read_jsonl([str(shard)])))
    assert normalize(rec, source="s", subset=None, language="en", license="unknown") is None

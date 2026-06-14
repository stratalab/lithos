"""Tests for lithos.utils.storage — backend-agnostic artifact store."""

from lithos.utils.storage import Storage, StorageConfig


def test_local_json_and_bytes_roundtrip(tmp_path):
    s = Storage(StorageConfig(base_uri=str(tmp_path / "store")))
    s.write_json("manifests/corpus.json", {"tokens": 123})
    s.write_bytes("blob.bin", b"\x00\x01\x02")
    assert s.exists("manifests/corpus.json")
    assert s.read_json("manifests/corpus.json") == {"tokens": 123}
    assert s.read_bytes("blob.bin") == b"\x00\x01\x02"


def test_put_get_single_file(tmp_path):
    src = tmp_path / "manifest.json"
    src.write_bytes(b'{"tokens": 1}')
    s = Storage(StorageConfig(base_uri=str(tmp_path / "store")))
    s.put(src, "corpus/v0.1/manifest.json")
    assert s.exists("corpus/v0.1/manifest.json")  # exact key, no basename appended
    out = tmp_path / "pulled.json"
    s.get("corpus/v0.1/manifest.json", out)
    assert out.read_bytes() == b'{"tokens": 1}'


def test_put_get_directory(tmp_path):
    src = tmp_path / "shards"
    src.mkdir()
    (src / "shard_000001.bin").write_bytes(b"aaaa")
    (src / "shard_000002.bin").write_bytes(b"bbbb")

    s = Storage(StorageConfig(base_uri=str(tmp_path / "store")))
    s.put(src, "corpus/v0.1")
    assert s.exists("corpus/v0.1/shard_000001.bin")

    out = tmp_path / "pulled"
    s.get("corpus/v0.1", out)
    assert (out / "shard_000001.bin").read_bytes() == b"aaaa"
    assert (out / "shard_000002.bin").read_bytes() == b"bbbb"


def test_memory_backend_uri():
    # A non-local URI exercises the fsspec backend path (no creds needed).
    s = Storage(StorageConfig(base_uri="memory://lithos-test-bucket"))
    s.write_bytes("a/b.bin", b"hello")
    assert s.exists("a/b.bin")
    assert s.read_bytes("a/b.bin") == b"hello"


def test_uri_join():
    s = Storage(StorageConfig(base_uri="memory://bucket/prefix"))
    assert s.uri("corpus", "v0.1").endswith("/corpus/v0.1")
    assert s.uri("x") == s.uri("/x/")  # leading/trailing slashes are normalized

"""Tests for lithos.data.shard — dtype selection and shard write/read roundtrip."""

from lithos.data.shard import ShardWriter, dtype_for_vocab, load_shard


def test_dtype_selection():
    assert dtype_for_vocab(32000) == "uint16"
    assert dtype_for_vocab(65536) == "uint16"
    assert dtype_for_vocab(70000) == "uint32"


def test_shard_writer_roundtrip(tmp_path):
    writer = ShardWriter(tmp_path, tokens_per_shard=10, dtype="uint16", tokenizer_name="t")
    writer.add(list(range(10)))
    writer.add(list(range(10, 25)))  # crosses shard boundary; remainder of 5
    shards = writer.close()

    assert len(shards) == 3
    assert [s["num_tokens"] for s in shards] == [10, 10, 5]
    assert writer.total_tokens == 25

    reassembled: list[int] = []
    for s in shards:
        assert len(s["sha256"]) == 64
        assert s["dtype"] == "uint16"
        reassembled.extend(load_shard(s["path"], s["dtype"]).tolist())
    assert reassembled == list(range(25))


def test_shard_writer_no_remainder(tmp_path):
    writer = ShardWriter(tmp_path, tokens_per_shard=10, dtype="uint16", tokenizer_name="t")
    writer.add(list(range(20)))
    shards = writer.close()
    assert [s["num_tokens"] for s in shards] == [10, 10]

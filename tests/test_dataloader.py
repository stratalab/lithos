"""Tests for lithos.data.dataloader — packed dataset, determinism, and resume."""

import torch
from lithos.data.dataloader import PackedDataLoader, PackedDataset
from lithos.data.shard import ShardWriter


def _make_shards(tmp_path, n_tokens, per_shard):
    writer = ShardWriter(tmp_path, tokens_per_shard=per_shard, dtype="uint16", tokenizer_name="t")
    writer.add(list(range(n_tokens)))
    shards = writer.close()
    return [(s["path"], s["num_tokens"], s["dtype"]) for s in shards]


def test_packed_dataset_length_and_shift(tmp_path):
    shards = _make_shards(tmp_path, 200, 80)  # shards of 80, 80, 40 tokens
    seq_len = 8
    ds = PackedDataset(shards, seq_len)
    # per-shard sequences: (80-1)//8=9, 9, (40-1)//8=4
    assert len(ds) == 9 + 9 + 4
    x, y = ds[0]
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)
    assert torch.equal(y[:-1], x[1:])  # shifted-window property


def test_dataloader_determinism(tmp_path):
    ds = PackedDataset(_make_shards(tmp_path, 400, 150), seq_len=8)
    l1 = PackedDataLoader(ds, batch_size=2, seed=0)
    l2 = PackedDataLoader(ds, batch_size=2, seed=0)
    for _ in range(4):
        a_x, a_y = next(l1)
        b_x, b_y = next(l2)
        assert torch.equal(a_x, b_x)
        assert torch.equal(a_y, b_y)


def test_dataloader_resume_continues_at_position(tmp_path):
    ds = PackedDataset(_make_shards(tmp_path, 400, 150), seq_len=8)
    loader = PackedDataLoader(ds, batch_size=2, seed=0)

    for _ in range(3):  # advance, then snapshot
        next(loader)
    state = loader.state_dict()
    expected = [next(loader) for _ in range(3)]

    resumed = PackedDataLoader(ds, batch_size=2, seed=0)
    resumed.load_state_dict(state)
    got = [next(resumed) for _ in range(3)]

    for (ex_x, ex_y), (got_x, got_y) in zip(expected, got, strict=True):
        assert torch.equal(ex_x, got_x)
        assert torch.equal(ex_y, got_y)


def test_dataloader_reshuffles_each_epoch(tmp_path):
    ds = PackedDataset(_make_shards(tmp_path, 200, 200), seq_len=8)  # 24 sequences
    loader = PackedDataLoader(ds, batch_size=4, seed=0)
    first_epoch_perm = loader._make_perm(0)
    second_epoch_perm = loader._make_perm(1)
    assert not (first_epoch_perm == second_epoch_perm).all()  # different ordering

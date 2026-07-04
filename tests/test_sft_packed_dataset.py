"""Tests for the packed SFT loader (PackedSFTDataset, E2): windowing + masking.

Verifies the core invariant — y carries a real token exactly where the (shifted)
loss mask is 1 and IGNORE_INDEX everywhere else — and that packing raises the
loss-token fraction over the dense one-conversation-per-row path.
"""

import json
from pathlib import Path

import pytest
import torch
from lithos.posttrain.sft_corpus import (
    SFTCorpusBuildConfig,
    SFTShardWriter,
    SFTSourceSpec,
    build_sft_corpus,
)
from lithos.posttrain.sft_dataset import (
    IGNORE_INDEX,
    PackedSFTDataset,
    SFTDataset,
    load_sft_shard_specs,
)

TOKENIZER = "artifacts/tokenizer/fineweb-edu-32k/tokenizer.json"
pytestmark = pytest.mark.skipif(not Path(TOKENIZER).exists(), reason="tokenizer artifact absent")


def _shards_from(tmp_path, tokens, mask, *, dtype="uint16"):
    w = SFTShardWriter(tmp_path, tokens_per_shard=10_000, dtype=dtype, tokenizer_name="t")
    w.add(list(tokens), [bool(m) for m in mask])
    shards = w.close()
    # emulate a manifest so load_sft_shard_specs resolves the paths
    manifest = {"shards": shards}
    (tmp_path / "sft_manifest.json").write_text(json.dumps(manifest))
    return load_sft_shard_specs(tmp_path / "sft_manifest.json")


def test_windowing_and_mask_to_ignore(tmp_path):
    # tokens 0..9, loss only on the last 4 positions
    tokens = list(range(10))
    mask = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    specs = _shards_from(tmp_path, tokens, mask)
    ds = PackedSFTDataset(specs, seq_len=4)
    # num_sequences(10, 4) = (10-1)//4 = 2 windows
    assert len(ds) == 2

    x0, y0 = ds[0]  # window tokens[0:5] -> x=[0,1,2,3], y=shift [1,2,3,4] masked by mask[1:5]=0
    assert x0.tolist() == [0, 1, 2, 3]
    assert y0.tolist() == [IGNORE_INDEX] * 4  # all masked

    x1, y1 = ds[1]  # window tokens[4:9] -> x=[4,5,6,7], y=[5,6,7,8], mask[5:9]=[0,1,1,1]
    assert x1.tolist() == [4, 5, 6, 7]
    assert y1.tolist() == [IGNORE_INDEX, 6, 7, 8]


def test_dtype_and_shapes(tmp_path):
    specs = _shards_from(tmp_path, list(range(20)), [1] * 20)
    ds = PackedSFTDataset(specs, seq_len=8)
    x, y = ds[0]
    assert x.shape == (8,) and y.shape == (8,)
    assert x.dtype == torch.int64 and y.dtype == torch.int64


def test_index_bounds(tmp_path):
    specs = _shards_from(tmp_path, list(range(20)), [1] * 20)
    ds = PackedSFTDataset(specs, seq_len=8)
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_windows_span_shards(tmp_path):
    # two shards, each 8 tokens; num_sequences(8,4)=1 per shard -> 2 total
    w = SFTShardWriter(tmp_path, tokens_per_shard=8, dtype="uint16", tokenizer_name="t")
    w.add(list(range(16)), [1] * 16)
    shards = w.close()
    (tmp_path / "sft_manifest.json").write_text(json.dumps({"shards": shards}))
    specs = load_sft_shard_specs(tmp_path / "sft_manifest.json")
    assert len(shards) == 2
    ds = PackedSFTDataset(specs, seq_len=4)
    assert len(ds) == 2  # one window per shard; windows never cross shard boundaries


def test_packing_beats_dense_loss_fraction(tmp_path):
    # Build the SAME conversations two ways and compare loss-token density.
    rows = [{"messages": [{"role": "user", "content": f"question number {i}"},
                          {"role": "assistant", "content": f"the answer is {i}"}]} for i in range(40)]
    src = tmp_path / "a.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    seq_len = 64
    cfg = SFTCorpusBuildConfig(
        tokenizer_path=TOKENIZER, output_dir=str(tmp_path / "out"),
        sources=[SFTSourceSpec(path=str(src), name="a")], seq_len=seq_len, tokens_per_shard=100_000,
    )
    packed_manifest = build_sft_corpus(cfg)

    from tokenizers import Tokenizer

    dense = SFTDataset(src, Tokenizer.from_file(TOKENIZER), seq_len)
    dense_frac = dense.stats()["loss_token_fraction"]
    packed_frac = packed_manifest["loss_token_fraction"]
    # Packing removes per-conversation padding, so a far larger share of positions
    # carry loss (the FLOPs-per-loss-token win).
    assert packed_frac > dense_frac
    assert packed_frac > 2 * dense_frac


def test_masked_positions_have_valid_input_tokens(tmp_path):
    # x must always be a real token id (>=0), even where y is IGNORE_INDEX.
    tokens = list(range(1, 13))
    mask = [0, 1] * 6
    specs = _shards_from(tmp_path, tokens, mask)
    ds = PackedSFTDataset(specs, seq_len=4)
    for i in range(len(ds)):
        x, y = ds[i]
        assert (x >= 0).all()
        assert ((y == IGNORE_INDEX) | (y >= 0)).all()

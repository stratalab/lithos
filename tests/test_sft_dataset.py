"""Tests for the SFT dataset: (x, y) construction, masking, padding, drops (Phase 11)."""

import json

import pytest
import torch
from lithos.posttrain.sft_dataset import IGNORE_INDEX, SFTDataset

_NAMES = ["<pad>", "<bos>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


class FakeTok:
    def __init__(self):
        self._ids = {n: i for i, n in enumerate(_NAMES)}

    def token_to_id(self, token):
        return self._ids.get(token)

    def encode(self, text):
        class _Enc:
            ids = [100 + (ord(c) % 50) for c in text]

        return _Enc()


def _write(path, conversations):
    with open(path, "w") as f:
        for msgs in conversations:
            f.write(json.dumps({"messages": msgs}) + "\n")


def _convo(u, a):
    return [{"role": "user", "content": u}, {"role": "assistant", "content": a}]


def test_shapes_padding_and_masking(tmp_path):
    p = tmp_path / "sft.jsonl"
    _write(p, [_convo("hi", "yo"), _convo("a", "bb")])
    seq_len = 16
    ds = SFTDataset(p, FakeTok(), seq_len)
    assert len(ds) == 2

    x, y = ds[0]
    assert x.shape == (seq_len,) and y.shape == (seq_len,)
    assert x.dtype == torch.int64 and y.dtype == torch.int64

    # exactly the assistant content + its <|end|> carry loss; everything else is IGNORE
    learned_positions = (y != IGNORE_INDEX).nonzero().flatten().tolist()
    yo_ids = FakeTok().encode("yo").ids
    end_id = _NAMES.index("<|end|>")
    learned_targets = [int(y[i]) for i in learned_positions]
    assert learned_targets == [*yo_ids, end_id]
    # trailing padding is all IGNORE in labels and <pad> in inputs
    assert int(x[-1]) == _NAMES.index("<pad>")
    assert int(y[-1]) == IGNORE_INDEX


def test_drop_overlong(tmp_path):
    p = tmp_path / "sft.jsonl"
    # "hi"/"yo" -> x of length 8 (fits seq_len=8); the 50-char turn overflows and is dropped.
    _write(p, [_convo("hi", "yo"), _convo("x" * 50, "y" * 50)])
    ds = SFTDataset(p, FakeTok(), seq_len=8)
    assert len(ds) == 1  # the long one is dropped, not truncated
    assert ds.stats()["dropped"] == 1


def test_stats_and_loss_fraction(tmp_path):
    p = tmp_path / "sft.jsonl"
    _write(p, [_convo("hi", "yo")])
    ds = SFTDataset(p, FakeTok(), seq_len=16)
    s = ds.stats()
    assert s["examples"] == 1 and s["read"] == 1
    assert 0.0 < s["loss_token_fraction"] < 1.0  # only a few tokens of 16 carry loss


def test_empty_dataset_errors(tmp_path):
    p = tmp_path / "sft.jsonl"
    _write(p, [])
    with pytest.raises(ValueError, match="no usable SFT examples"):
        SFTDataset(p, FakeTok(), seq_len=16)

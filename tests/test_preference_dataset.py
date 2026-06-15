"""Tests for the DPO PreferenceDataset (Phase 11)."""

import json

import pytest
import torch
from lithos.posttrain.preference_dataset import PreferenceDataset
from lithos.posttrain.sft_dataset import IGNORE_INDEX

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


def _write(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def _pref(user, chosen, rejected):
    return {"prompt": [{"role": "user", "content": user}], "chosen": chosen, "rejected": rejected}


def test_yields_four_aligned_tensors(tmp_path):
    p = tmp_path / "prefs.jsonl"
    _write(p, [_pref("hi", "good", "bad")])
    ds = PreferenceDataset(p, FakeTok(), seq_len=16)
    assert len(ds) == 1
    cx, cy, rx, ry = ds[0]
    for t in (cx, cy, rx, ry):
        assert t.shape == (16,) and t.dtype == torch.int64
    # the shared prompt prefix is identical in chosen_x and rejected_x
    assert torch.equal(cx[:4], rx[:4])  # <bos> <|user|> h i
    # the learned (response) tokens differ between chosen and rejected
    chosen_resp = [int(cx[i]) for i in range(16) if int(cy[i]) != IGNORE_INDEX]
    rejected_resp = [int(rx[i]) for i in range(16) if int(ry[i]) != IGNORE_INDEX]
    assert chosen_resp != rejected_resp


def test_drops_pair_if_either_side_overlong(tmp_path):
    p = tmp_path / "prefs.jsonl"
    _write(p, [_pref("hi", "ok", "x" * 80)])  # rejected too long for seq_len=8
    with pytest.raises(ValueError, match="no usable preference pairs"):
        PreferenceDataset(p, FakeTok(), seq_len=8)


def test_stats(tmp_path):
    p = tmp_path / "prefs.jsonl"
    _write(p, [_pref("a", "bb", "cc"), _pref("d", "ee", "ff")])
    ds = PreferenceDataset(p, FakeTok(), seq_len=16)
    s = ds.stats()
    assert s["pairs"] == 2 and s["read"] == 2 and s["dropped"] == 0

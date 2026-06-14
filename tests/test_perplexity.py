"""Tests for lithos.evals.perplexity."""

import math

import pytest
from lithos.data import PackedDataLoader, PackedDataset
from lithos.evals.perplexity import compute_perplexity
from lithos.utils.io import read_json

from tests.helpers import make_model, make_tiny_corpus


def test_perplexity_is_finite_and_consistent(tmp_path):
    manifest = make_tiny_corpus(tmp_path / "c")
    shards = [(s["path"], s["num_tokens"], s["dtype"]) for s in read_json(manifest)["shards"]]
    loader = PackedDataLoader(PackedDataset(shards, 32), batch_size=4, seed=0)
    model = make_model(vocab_size=32, seq_len=32)

    res = compute_perplexity(model, loader, n_batches=5)
    assert math.isfinite(res["perplexity"])
    assert res["perplexity"] > 1.0
    assert res["perplexity"] == pytest.approx(math.exp(res["loss"]))
    assert res["tokens"] == 5 * 4 * 32

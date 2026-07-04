"""Tests for lithos.model.generation — greedy, sampling, KV-cache parity, eos."""

import torch
from lithos.model import generate

from tests.helpers import make_model


def test_generate_restores_training_mode():
    # F1 landmine: generate() must not leave a training model in eval mode, or the
    # GRPO loss forward after a rollout runs with dropout off (a silent bug).
    model = make_model(seed=0)
    model.train()
    generate(model, torch.tensor([[1, 2, 3]]), max_new_tokens=4, greedy=True)
    assert model.training is True
    model.eval()
    generate(model, torch.tensor([[1, 2, 3]]), max_new_tokens=4, greedy=True)
    assert model.training is False


def test_greedy_is_deterministic():
    model = make_model(seed=0)
    ids = torch.tensor([[1, 2, 3]])
    out1 = generate(model, ids, max_new_tokens=10, greedy=True)
    out2 = generate(model, ids, max_new_tokens=10, greedy=True)
    assert out1.shape == (1, 13)
    assert torch.equal(out1, out2)


def test_kv_cache_matches_full_recompute():
    # Eager backend makes the cached and full-recompute paths numerically
    # identical, so greedy decoding must yield identical tokens.
    model = make_model(seed=1, attn_backend="eager")
    ids = torch.tensor([[5, 6, 7, 8]])
    cached = generate(model, ids, max_new_tokens=12, greedy=True, use_cache=True)
    recompute = generate(model, ids, max_new_tokens=12, greedy=True, use_cache=False)
    assert torch.equal(cached, recompute)


def test_sampling_is_seeded_and_runs():
    model = make_model(seed=2)
    ids = torch.tensor([[1, 2]])
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    kw = dict(max_new_tokens=8, temperature=0.8, top_k=20, top_p=0.95)
    out1 = generate(model, ids, generator=g1, **kw)
    out2 = generate(model, ids, generator=g2, **kw)
    assert out1.shape == (1, 10)
    assert torch.equal(out1, out2)  # same seed -> same samples


def test_eos_stops_generation():
    model = make_model(seed=3)
    ids = torch.tensor([[1, 2, 3]])
    # The greedy first token, used as eos, should halt generation immediately.
    first = int(generate(model, ids, max_new_tokens=1, greedy=True)[0, -1])
    out = generate(model, ids, max_new_tokens=15, greedy=True, eos_token_id=first)
    assert out.shape == (1, 4)  # prompt(3) + the eos token
    assert int(out[0, -1]) == first

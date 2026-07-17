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


# ---- sampler logprobs (T2: the rollout record carries q) ----


def test_return_logprobs_greedy_is_zero():
    # Greedy decoding is a delta distribution: log q = 0 at every position.
    model = make_model(seed=0)
    out, lps = generate(
        model, torch.tensor([[1, 2, 3]]), max_new_tokens=5, greedy=True, return_logprobs=True
    )
    assert out.shape == (1, 8)
    assert lps.shape == (1, 5)  # one logprob per GENERATED token, prompt excluded
    assert torch.all(lps == 0.0)


def test_return_logprobs_match_sampling_distribution():
    # One sampled step must report log softmax(logits/T) at the sampled token —
    # the q an importance-sampling correction divides by.
    model = make_model(seed=2)
    ids = torch.tensor([[1, 2]])
    temperature = 0.7
    out, lps = generate(
        model, ids, max_new_tokens=1, temperature=temperature,
        generator=torch.Generator().manual_seed(7), return_logprobs=True,
    )
    with torch.no_grad():
        logits, _ = model(ids)
    expected = torch.log_softmax(logits[:, -1, :] / temperature, dim=-1)[0, out[0, -1]]
    assert torch.allclose(lps[0, 0], expected, atol=1e-5)


def test_return_logprobs_zero_on_forced_eos_padding():
    # A row that finishes early is padded with FORCED eos tokens while the rest of
    # the batch continues — those positions were never sampled, so their recorded
    # logprob must be exactly 0.0. Probe an unconstrained run to learn row 0's
    # first sampled token, then rerun with that token as eos.
    model = make_model(seed=4)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    kw = dict(max_new_tokens=6, temperature=1.0, return_logprobs=True)
    probe, _ = generate(model, ids, generator=torch.Generator().manual_seed(11), **kw)
    eos = int(probe[0, ids.shape[1]])  # row 0's first sampled token
    out, lps = generate(
        model, ids, generator=torch.Generator().manual_seed(11), eos_token_id=eos, **kw
    )
    gen = out[:, ids.shape[1] :]
    assert int(gen[0, 0]) == eos  # same seed -> row 0 finishes at step 0
    assert gen.shape[1] > 1  # row 1 kept the batch alive past row 0's finish
    assert torch.all(gen[0, 1:] == eos)  # row 0's tail is forced padding...
    assert torch.all(lps[0, 1:] == 0.0)  # ...and carries no sampler logprob
    assert lps[0, 0] < 0.0  # the genuinely sampled token does
    # row 1: sampled positions carry real logprobs up to (and including) its own
    # first eos, 0.0 only after — the same invariant, row-relative.
    hits = (gen[1] == eos).nonzero()
    first = int(hits[0]) if len(hits) else gen.shape[1] - 1
    assert torch.all(lps[1, : first + 1] < 0.0)
    assert torch.all(lps[1, first + 1 :] == 0.0)

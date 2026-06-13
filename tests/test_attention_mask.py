"""Tests for causal masking, GQA, and SDPA/eager backend parity."""

import torch
from lithos.model.attention import repeat_kv

from tests.helpers import make_model


def _change_last_token(ids: torch.Tensor, vocab: int) -> torch.Tensor:
    out = ids.clone()
    out[0, -1] = (ids[0, -1] + 1) % vocab
    return out


def test_no_future_token_leakage():
    model = make_model(seed=0)
    model.eval()
    torch.manual_seed(1)
    ids = torch.randint(0, model.cfg.vocab_size, (1, 12))
    logits1, _ = model(ids)
    logits2, _ = model(_change_last_token(ids, model.cfg.vocab_size))
    # Changing the last token must not affect any earlier position's logits.
    torch.testing.assert_close(logits1[:, :-1], logits2[:, :-1], atol=1e-5, rtol=1e-5)


def test_repeat_kv_groups_correctly():
    x = torch.randn(1, 2, 3, 4)  # (B, n_kv=2, T, D)
    r = repeat_kv(x, 3)
    assert r.shape == (1, 6, 3, 4)
    for i in range(3):
        torch.testing.assert_close(r[:, i], x[:, 0])
        torch.testing.assert_close(r[:, 3 + i], x[:, 1])


def test_mha_forward_shape():
    model = make_model(seed=2, n_heads=4, n_kv_heads=4)  # MHA
    model.eval()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 8))
    logits, _ = model(ids)
    assert logits.shape == (1, 8, model.cfg.padded_vocab_size)


def test_gqa_runs_and_stays_causal():
    model = make_model(seed=3, n_heads=4, n_kv_heads=2)  # GQA
    model.eval()
    torch.manual_seed(4)
    ids = torch.randint(0, model.cfg.vocab_size, (1, 10))
    logits1, _ = model(ids)
    logits2, _ = model(_change_last_token(ids, model.cfg.vocab_size))
    torch.testing.assert_close(logits1[:, :-1], logits2[:, :-1], atol=1e-5, rtol=1e-5)


def test_sdpa_and_eager_backends_match():
    m_sdpa = make_model(seed=5, attn_backend="sdpa")
    m_eager = make_model(seed=5, attn_backend="eager")
    m_eager.load_state_dict(m_sdpa.state_dict())
    m_sdpa.eval()
    m_eager.eval()
    torch.manual_seed(6)
    ids = torch.randint(0, m_sdpa.cfg.vocab_size, (2, 9))
    l_sdpa, _ = m_sdpa(ids)
    l_eager, _ = m_eager(ids)
    torch.testing.assert_close(l_sdpa, l_eager, atol=1e-4, rtol=1e-4)

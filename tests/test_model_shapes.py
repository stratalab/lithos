"""Tests for forward/loss shapes, backprop, parameter count, and vocab padding."""

from pathlib import Path

import torch
from lithos.model import LithosForCausalLM, ModelConfig
from lithos.utils.config import load_and_validate

from tests.helpers import make_model

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_forward_logits_shape():
    model = make_model()
    ids = torch.randint(0, model.cfg.vocab_size, (3, 7))
    logits, loss = model(ids)
    assert logits.shape == (3, 7, model.cfg.padded_vocab_size)
    assert loss is None


def test_loss_is_scalar_and_backprops():
    model = make_model()
    ids = torch.randint(0, model.cfg.vocab_size, (2, 16))
    _, loss = model(ids, labels=ids)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_toy_config_param_count_in_range():
    cfg = load_and_validate(REPO_ROOT / "configs/model/lithos-toy.yaml", ModelConfig)
    model = LithosForCausalLM(cfg)
    assert 4_000_000 < model.num_parameters() < 20_000_000


def test_loss_ignores_vocab_padding():
    # vocab_size not a multiple of pad_vocab_to -> padding columns exist.
    model = make_model(vocab_size=100, pad_vocab_to=128)
    assert model.cfg.padded_vocab_size == 128
    ids = torch.randint(0, 100, (2, 8))
    logits, loss = model(ids, labels=ids)
    pad_min = torch.finfo(logits.dtype).min
    assert torch.all(logits[..., 100:] == pad_min)  # padding never predicted
    assert torch.isfinite(loss)

"""Tests for lithos.train.optim — AdamW weight-decay split (PRD §9.3)."""

from lithos.train.config import OptimConfig
from lithos.train.optim import build_optimizer

from tests.helpers import make_model


def test_weight_decay_split_by_dimensionality():
    model = make_model()
    opt = build_optimizer(model, OptimConfig(weight_decay=0.1))
    decay_group, no_decay_group = opt.param_groups

    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    # Matmul / embedding weights (>=2D) decay; norm weights (1D) do not.
    assert all(p.dim() >= 2 for p in decay_group["params"])
    assert all(p.dim() == 1 for p in no_decay_group["params"])
    assert decay_group["params"]  # non-empty
    assert no_decay_group["params"]


def test_optimizer_hyperparams():
    model = make_model()
    opt = build_optimizer(model, OptimConfig(lr=1e-3, betas=(0.9, 0.95), eps=1e-8))
    assert opt.defaults["lr"] == 1e-3
    assert opt.defaults["betas"] == (0.9, 0.95)
    assert opt.defaults["eps"] == 1e-8

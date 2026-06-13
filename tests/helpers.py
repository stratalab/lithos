"""Shared test helpers: tiny model/config factories for fast CPU tests."""

from __future__ import annotations

from typing import Any

import torch
from lithos.model import LithosForCausalLM, ModelConfig

_TINY: dict[str, Any] = {
    "vocab_size": 64,
    "n_layers": 2,
    "hidden": 32,
    "n_heads": 4,
    "n_kv_heads": 4,
    "intermediate_size": 64,
    "seq_len": 32,
    "rope_theta": 10000.0,
    "qk_norm": False,
    "tie_embeddings": True,
    "dropout": 0.0,
    "attn_backend": "sdpa",
}


def make_config(**overrides: Any) -> ModelConfig:
    return ModelConfig(**{**_TINY, **overrides})


def make_model(seed: int = 0, **overrides: Any) -> LithosForCausalLM:
    torch.manual_seed(seed)
    return LithosForCausalLM(make_config(**overrides))

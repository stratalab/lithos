"""Shared test helpers: tiny model/config factories for fast CPU tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from lithos.data.shard import ShardWriter
from lithos.model import LithosForCausalLM, ModelConfig
from lithos.utils.io import write_json

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


def make_tiny_corpus(
    directory: Path, *, n_tokens: int = 4000, period: int = 16, tokens_per_shard: int = 2000
) -> str:
    """Write a tiny, perfectly-learnable periodic corpus + manifest; return its path.

    Each token's successor is a deterministic function of the token value, so a
    small model can drive the loss to ~0 (used by the overfit test).
    """
    pattern = [(i % period) + 1 for i in range(n_tokens)]
    writer = ShardWriter(
        Path(directory) / "tokenized",
        tokens_per_shard=tokens_per_shard,
        dtype="uint16",
        tokenizer_name="tiny",
    )
    writer.add(pattern)
    shards = writer.close()
    manifest_path = Path(directory) / "corpus_manifest.json"
    write_json(
        manifest_path, {"tokenizer": "tiny", "num_tokens": writer.total_tokens, "shards": shards}
    )
    return str(manifest_path)

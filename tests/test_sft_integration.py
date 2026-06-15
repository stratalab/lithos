"""Integration tests for SFT wiring: config validation + weight-only init (Phase 11)."""

import pytest
import torch
from lithos.model import LithosForCausalLM
from lithos.model.config import ModelConfig
from lithos.train.checkpoint import (
    load_model_from_checkpoint,
    load_model_weights,
    save_checkpoint,
)
from lithos.train.config import DataConfig
from safetensors.torch import save_model


def _tiny_model_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        n_layers=2,
        hidden=32,
        n_heads=2,
        n_kv_heads=2,
        intermediate_size=64,
        seq_len=16,
        qk_norm=True,
        tie_embeddings=True,
    )


def test_sft_data_requires_tokenizer_path():
    with pytest.raises(ValueError, match="tokenizer_path"):
        DataConfig(kind="sft", corpus_manifest="data/sft/train.jsonl", seq_len=64)


def test_packed_data_defaults_need_no_tokenizer():
    cfg = DataConfig(corpus_manifest="corpus.json", seq_len=64)
    assert cfg.kind == "packed" and cfg.tokenizer_path is None


def test_load_model_weights_roundtrip(tmp_path):
    torch.manual_seed(0)
    src = LithosForCausalLM(_tiny_model_cfg())
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    save_model(src, str(ckpt / "model.safetensors"))

    torch.manual_seed(1)  # a differently-initialised destination
    dst = LithosForCausalLM(_tiny_model_cfg())
    assert not all(torch.equal(a, b) for a, b in zip(src.parameters(), dst.parameters()))

    load_model_weights(ckpt, dst)  # loads weights only — no optimizer/RNG/data state
    assert all(torch.equal(a, b) for a, b in zip(src.parameters(), dst.parameters()))


def test_checkpoint_embeds_arch_and_loads_size_agnostic(tmp_path):
    import json

    src = LithosForCausalLM(_tiny_model_cfg())
    opt = torch.optim.AdamW(src.parameters(), lr=1e-3)
    ckpt = tmp_path / "step_000001"
    save_checkpoint(ckpt, model=src, optimizer=opt, step=1, tokens_seen=0, dataloader_state={}, meta={})

    # the architecture is embedded in meta.json (self-describing checkpoint)
    meta = json.loads((ckpt / "meta.json").read_text())
    assert meta["model"]["n_layers"] == 2 and meta["model"]["hidden"] == 32

    # the loader rebuilds the right model + weights WITHOUT being told the shape
    loaded = load_model_from_checkpoint(ckpt)
    assert all(torch.equal(a, b) for a, b in zip(src.parameters(), loaded.parameters()))

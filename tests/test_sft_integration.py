"""Integration tests for SFT wiring: config validation + weight-only init (Phase 11)."""

import pytest
import torch
from lithos.model import LithosForCausalLM
from lithos.model.config import ModelConfig
from lithos.train.checkpoint import load_model_weights
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

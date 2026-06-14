"""Tests for lithos.serve.export — HF/Qwen3 export + transformers round-trip."""

import pytest
import torch
from lithos.serve.export import export_hf, hf_config

from tests.helpers import make_model


def test_hf_config_picks_architecture_by_qk_norm():
    assert hf_config(make_model(qk_norm=False))["model_type"] == "llama"
    qwen = hf_config(make_model(qk_norm=True))
    assert qwen["model_type"] == "qwen3"
    assert qwen["architectures"] == ["Qwen3ForCausalLM"]


def test_export_writes_standard_files(tmp_path):
    out = export_hf(make_model(vocab_size=100), tmp_path / "hf", dtype="float32")
    for fname in ("config.json", "model.safetensors", "generation_config.json"):
        assert (out / fname).is_file()


def _assert_roundtrip(model, out_dir, vocab):
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    model.eval()
    out = export_hf(model, out_dir, dtype="float32")
    hf = AutoModelForCausalLM.from_pretrained(out)
    hf.eval()
    ids = torch.randint(0, vocab, (1, 12))
    with torch.no_grad():
        ours, _ = model(ids)
        theirs = hf(ids).logits
    # Our logits over the real vocab must match the HF model's.
    torch.testing.assert_close(ours[:, :, :vocab], theirs, atol=2e-3, rtol=2e-3)


def test_llama_export_matches_transformers(tmp_path):
    _assert_roundtrip(make_model(seed=3, vocab_size=96, qk_norm=False), tmp_path / "hf", 96)


def test_qwen3_export_matches_transformers(tmp_path):
    _assert_roundtrip(make_model(seed=4, vocab_size=96, qk_norm=True), tmp_path / "hf", 96)

"""Tests for importing Qwen3 weights into Lithos (lithos/serve/hf_import.py, E7).

The parity tests are the inverse of test_export.py: build a tiny random Qwen3 in
transformers with a DECOUPLED head_dim + GQA (tied and untied), import it into
LithosForCausalLM, and assert the logits match — the affirmative answer to "does
the Qwen-lineage hero share one tooling path with the from-scratch models?"
"""

import pytest
import torch
from lithos.model.config import ModelConfig
from lithos.serve.hf_import import lithos_config_from_hf, load_qwen3

transformers = pytest.importorskip("transformers")


def _tiny_qwen3(*, tie: bool, **over):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    # hidden//n_heads = 32//4 = 8, but head_dim = 16 -> DECOUPLED (the E7 crux);
    # n_kv_heads=2 -> GQA. rms_eps/rope_theta set to Qwen3 values so parity is exact.
    kwargs = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        tie_word_embeddings=tie,
    )
    kwargs.update(over)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(**kwargs))
    model.eval()
    return model


def _assert_import_parity(hf_model, vocab):
    lithos = load_qwen3(hf_model)
    ids = torch.randint(0, vocab, (1, 12))
    with torch.no_grad():
        theirs = hf_model(ids).logits
        ours, _ = lithos(ids)
    # Our logits over the real (unpadded) vocab must match the source Qwen3's.
    torch.testing.assert_close(ours[:, :, :vocab], theirs, atol=2e-3, rtol=2e-3)


def test_config_from_hf_maps_decoupled_head_dim():
    cfg = lithos_config_from_hf(_tiny_qwen3(tie=True).config)
    assert isinstance(cfg, ModelConfig)
    assert cfg.hidden == 32 and cfg.n_heads == 4 and cfg.n_kv_heads == 2
    assert cfg.head_dim == 16  # decoupled, not 32//4=8
    assert cfg.qk_norm is True and cfg.tie_embeddings is True
    assert cfg.rope_theta == 1_000_000.0 and cfg.rms_eps == 1e-6


def test_config_from_hf_rejects_non_qwen3():
    class _Cfg:
        model_type = "llama"

    with pytest.raises(ValueError, match="Qwen3"):
        lithos_config_from_hf(_Cfg())


def test_import_untied_qwen3_matches_logits():
    _assert_import_parity(_tiny_qwen3(tie=False), 64)


def test_import_tied_qwen3_matches_logits():
    # Qwen3-0.6B ships tied embeddings (no lm_head.weight) — the importer shares it.
    _assert_import_parity(_tiny_qwen3(tie=True), 64)


@pytest.mark.parametrize(
    "over, match",
    [
        ({"attention_bias": True}, "attention_bias"),  # dropped biases -> silent-wrong
        ({"hidden_act": "gelu"}, "hidden_act"),  # different activation
        ({"use_sliding_window": True, "sliding_window": 4}, "sliding-window"),  # different attention
    ],
)
def test_unsupported_qwen3_features_refused(over, match):
    # These import WITHOUT error today but produce wrong logits — must refuse loudly.
    with pytest.raises(ValueError, match=match):
        load_qwen3(_tiny_qwen3(tie=True, **over))


def test_imported_model_generates():
    from lithos.model.generation import generate

    lithos = load_qwen3(_tiny_qwen3(tie=True))
    out = generate(lithos, torch.tensor([[1, 2, 3]]), max_new_tokens=5, greedy=True)
    assert out.shape == (1, 8)  # drives the Lithos generation path unchanged

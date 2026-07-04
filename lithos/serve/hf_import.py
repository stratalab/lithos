"""Import Qwen3 weights INTO the Lithos architecture (epic E7, review §3.4).

The inverse of ``lithos/serve/export.py``: Lithos was built to the Qwen3 export
envelope (qk_norm, GQA, SwiGLU, rotate_half RoPE, no biases, matching leaf names),
so a Qwen3 checkpoint loads into ``LithosForCausalLM`` with **logit parity** — the
prerequisite for the family's "one deployment recipe" (the same Lithos loop /
generation / GRPO-TIR driving both the from-scratch 500M and the Qwen-lineage hero).

The single structural difference Qwen3 needs is a **decoupled head_dim** (e.g.
Qwen3-0.6B: 128 ≠ hidden//n_heads), now expressible via ``ModelConfig.head_dim``.
Weight names map back one-to-one (strip ``model.``, ``self_attn``→``attn``,
``input_layernorm``→``attn_norm``, ``post_attention_layernorm``→``mlp_norm``); no
q/k permutation (both use rotate_half). Vocab is padded up to Lithos's
``padded_vocab_size``; tied checkpoints share ``embed_tokens`` into ``lm_head``.
"""

from __future__ import annotations

from typing import Any

import torch

from lithos.model import LithosForCausalLM
from lithos.model.config import ModelConfig


def _rope_theta(config: Any) -> float:
    """Read RoPE theta defensively: transformers 5.12 may store it under a
    ``rope_parameters`` dict rather than a flat ``rope_theta`` attribute."""
    theta = getattr(config, "rope_theta", None)
    if theta is not None:
        return float(theta)
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and rope_parameters.get("rope_theta") is not None:
        return float(rope_parameters["rope_theta"])
    return 10000.0


def _check_importable(config: Any) -> None:
    """Raise if the Qwen3 config uses features Lithos cannot represent — otherwise
    the import would **silently** produce wrong logits (dropped biases, a different
    activation, sliding-window attention, or scaled RoPE)."""
    bad: list[str] = []
    if getattr(config, "attention_bias", False):
        bad.append("attention_bias=True (Lithos attention is bias-free)")
    act = getattr(config, "hidden_act", "silu")
    if act != "silu":
        bad.append(f"hidden_act={act!r} (Lithos MLP is SwiGLU/SiLU)")
    if getattr(config, "use_sliding_window", False) and getattr(config, "sliding_window", None):
        bad.append("sliding-window attention (Lithos is full-causal)")
    layer_types = getattr(config, "layer_types", None)
    if layer_types and any(t != "full_attention" for t in layer_types):
        bad.append("mixed/sliding layer_types (Lithos is full-causal)")
    # RoPE: reject only genuine scaling (yarn/linear/…); the "default" dict that
    # transformers populates for plain RoPE is fine.
    rope_cfg = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
    if isinstance(rope_cfg, dict):
        rope_type = rope_cfg.get("rope_type") or rope_cfg.get("type")
        if rope_type not in (None, "default"):
            bad.append(f"rope_type={rope_type!r} (Lithos uses plain RoPE)")
    if bad:
        raise ValueError("cannot import Qwen3 checkpoint — unsupported: " + "; ".join(bad))


def lithos_config_from_hf(config: Any, *, vocab_size: int | None = None) -> ModelConfig:
    """Build a ``ModelConfig`` matching a transformers Qwen3 config."""
    model_type = getattr(config, "model_type", None)
    if model_type != "qwen3":
        raise ValueError(
            f"expected a Qwen3 config (model_type='qwen3'), got {model_type!r}; "
            "the importer targets the qk-norm Qwen3 envelope"
        )
    _check_importable(config)
    return ModelConfig(
        vocab_size=vocab_size or config.vocab_size,
        n_layers=config.num_hidden_layers,
        hidden=config.hidden_size,
        n_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,  # Qwen3 always sets this (may be decoupled)
        intermediate_size=config.intermediate_size,
        seq_len=config.max_position_embeddings,
        rope_theta=_rope_theta(config),
        rms_eps=config.rms_norm_eps,
        qk_norm=True,  # Qwen3 has q_norm/k_norm
        tie_embeddings=bool(config.tie_word_embeddings),
    )


def _pad_rows(weight: torch.Tensor, target_rows: int) -> torch.Tensor:
    """Pad a ``[vocab, hidden]`` tensor up to ``target_rows`` (Lithos pads vocab for
    tensor-core efficiency; the extra rows/cols are masked out of the logits)."""
    n, d = weight.shape
    if n == target_rows:
        return weight.clone()
    if n > target_rows:
        raise ValueError(f"vocab {n} exceeds padded target {target_rows}")
    padded = torch.zeros(target_rows, d, dtype=weight.dtype)
    padded[:n] = weight
    return padded


def _from_hf_state_dict(hf: dict[str, torch.Tensor], cfg: ModelConfig) -> dict[str, torch.Tensor]:
    """Rename Qwen3 state-dict keys to Lithos names; pad vocab; wire tying."""
    padded = cfg.padded_vocab_size
    out: dict[str, torch.Tensor] = {
        "embed_tokens.weight": _pad_rows(hf["model.embed_tokens.weight"], padded),
        "norm.weight": hf["model.norm.weight"].clone(),
    }
    # Tied checkpoints (e.g. Qwen3-0.6B) omit lm_head.weight — share the embedding.
    out["lm_head.weight"] = (
        out["embed_tokens.weight"] if cfg.tie_embeddings else _pad_rows(hf["lm_head.weight"], padded)
    )
    for i in range(cfg.n_layers):
        h, p = f"model.layers.{i}.", f"layers.{i}."
        out[p + "attn_norm.weight"] = hf[h + "input_layernorm.weight"].clone()
        out[p + "mlp_norm.weight"] = hf[h + "post_attention_layernorm.weight"].clone()
        out[p + "attn.q_proj.weight"] = hf[h + "self_attn.q_proj.weight"].clone()
        out[p + "attn.k_proj.weight"] = hf[h + "self_attn.k_proj.weight"].clone()
        out[p + "attn.v_proj.weight"] = hf[h + "self_attn.v_proj.weight"].clone()
        out[p + "attn.o_proj.weight"] = hf[h + "self_attn.o_proj.weight"].clone()
        out[p + "attn.q_norm.weight"] = hf[h + "self_attn.q_norm.weight"].clone()
        out[p + "attn.k_norm.weight"] = hf[h + "self_attn.k_norm.weight"].clone()
        out[p + "mlp.gate_proj.weight"] = hf[h + "mlp.gate_proj.weight"].clone()
        out[p + "mlp.up_proj.weight"] = hf[h + "mlp.up_proj.weight"].clone()
        out[p + "mlp.down_proj.weight"] = hf[h + "mlp.down_proj.weight"].clone()
    return out


def _expected_hf_keys(cfg: ModelConfig) -> set[str]:
    """Every Qwen3 state-dict key the importer consumes — used to detect any weight
    that would be *silently dropped* (e.g. attention biases from a trained checkpoint)."""
    keys = {"model.embed_tokens.weight", "model.norm.weight"}
    if not cfg.tie_embeddings:
        keys.add("lm_head.weight")
    per_layer = (
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    )
    for i in range(cfg.n_layers):
        keys |= {f"model.layers.{i}.{s}" for s in per_layer}
    return keys


def load_qwen3(hf_model: Any, *, vocab_size: int | None = None) -> LithosForCausalLM:
    """Load a transformers ``Qwen3ForCausalLM`` into a fresh ``LithosForCausalLM``.

    Returns the Lithos model in eval mode; its logits match the source Qwen3's on
    the real vocab (verified in ``tests/test_hf_import.py`` and the real-0.6B spike).
    Refuses to import if any source weight would be dropped (``load_state_dict`` alone
    only checks the Lithos side, so a dropped bias would import silently-wrong).
    """
    cfg = lithos_config_from_hf(hf_model.config, vocab_size=vocab_size)
    hf_sd = hf_model.state_dict()
    # `lm_head.weight` is allowed extra: a tied checkpoint may still carry it.
    dropped = set(hf_sd) - _expected_hf_keys(cfg) - {"lm_head.weight"}
    if dropped:
        raise ValueError(
            f"Qwen3 checkpoint has {len(dropped)} weight(s) Lithos cannot represent, "
            f"e.g. {sorted(dropped)[:4]} — refusing to import silently"
        )
    model = LithosForCausalLM(cfg)
    model.load_state_dict(_from_hf_state_dict(hf_sd, cfg))
    model.eval()
    return model

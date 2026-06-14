"""Export a Lithos checkpoint to a HuggingFace-loadable directory (PRD §13.3, §26.8).

Models without QK-norm export as **Llama** (LlamaForCausalLM); models with QK-norm
export as **Qwen3** (Qwen3ForCausalLM) -- the envelope that preserves our refinements
while loading drop-in in transformers / vLLM / llama.cpp.

The exported artifact has the standard layout: config.json, model.safetensors,
generation_config.json, optional tokenizer.json + model_card.md.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from lithos.model import LithosForCausalLM
from lithos.utils.io import ensure_dir, write_json

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _to_hf_state_dict(model: LithosForCausalLM) -> dict[str, torch.Tensor]:
    """Rename Lithos params to HF Llama/Qwen3 names; trim vocab padding rows."""
    cfg = model.cfg
    v = cfg.vocab_size
    sd = model.state_dict()
    out: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": sd["embed_tokens.weight"][:v].clone(),
        "model.norm.weight": sd["norm.weight"].clone(),
    }
    if not cfg.tie_embeddings:
        out["lm_head.weight"] = sd["lm_head.weight"][:v].clone()
    for i in range(cfg.n_layers):
        p, h = f"layers.{i}.", f"model.layers.{i}."
        out[h + "input_layernorm.weight"] = sd[p + "attn_norm.weight"].clone()
        out[h + "post_attention_layernorm.weight"] = sd[p + "mlp_norm.weight"].clone()
        out[h + "self_attn.q_proj.weight"] = sd[p + "attn.q_proj.weight"].clone()
        out[h + "self_attn.k_proj.weight"] = sd[p + "attn.k_proj.weight"].clone()
        out[h + "self_attn.v_proj.weight"] = sd[p + "attn.v_proj.weight"].clone()
        out[h + "self_attn.o_proj.weight"] = sd[p + "attn.o_proj.weight"].clone()
        if cfg.qk_norm:
            out[h + "self_attn.q_norm.weight"] = sd[p + "attn.q_norm.weight"].clone()
            out[h + "self_attn.k_norm.weight"] = sd[p + "attn.k_norm.weight"].clone()
        out[h + "mlp.gate_proj.weight"] = sd[p + "mlp.gate_proj.weight"].clone()
        out[h + "mlp.up_proj.weight"] = sd[p + "mlp.up_proj.weight"].clone()
        out[h + "mlp.down_proj.weight"] = sd[p + "mlp.down_proj.weight"].clone()
    return out


def hf_config(
    model: LithosForCausalLM,
    *,
    dtype: str = "float32",
    special_ids: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    """Build a Llama/Qwen3-compatible config.json dict."""
    cfg = model.cfg
    is_qwen3 = cfg.qk_norm
    ids = special_ids or {}
    return {
        "architectures": ["Qwen3ForCausalLM" if is_qwen3 else "LlamaForCausalLM"],
        "model_type": "qwen3" if is_qwen3 else "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "max_position_embeddings": cfg.seq_len,
        "rms_norm_eps": cfg.rms_eps,
        "rope_theta": cfg.rope_theta,
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": cfg.tie_embeddings,
        "torch_dtype": dtype,
        "bos_token_id": ids.get("<bos>"),
        "eos_token_id": ids.get("<eos>"),
        "pad_token_id": ids.get("<pad>"),
    }


def export_hf(
    model: LithosForCausalLM,
    output_dir: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
    dtype: str = "float32",
    special_ids: dict[str, int | None] | None = None,
    model_card: str | None = None,
) -> Path:
    """Write a HuggingFace-loadable model directory; returns its path."""
    out = ensure_dir(output_dir)
    torch_dtype = _DTYPES[dtype]
    state = {k: v.to(torch_dtype).contiguous() for k, v in _to_hf_state_dict(model).items()}
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    write_json(out / "config.json", hf_config(model, dtype=dtype, special_ids=special_ids))
    ids = special_ids or {}
    write_json(
        out / "generation_config.json",
        {
            "bos_token_id": ids.get("<bos>"),
            "eos_token_id": ids.get("<eos>"),
            "pad_token_id": ids.get("<pad>"),
        },
    )
    if tokenizer_path is not None:
        shutil.copyfile(tokenizer_path, out / "tokenizer.json")
        write_json(
            out / "tokenizer_config.json",
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "clean_up_tokenization_spaces": False,
            },
        )
    if model_card is not None:
        (out / "model_card.md").write_text(model_card, encoding="utf-8")
    return out

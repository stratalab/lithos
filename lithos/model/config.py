"""Model configuration (PRD §6.1, §6.2).

A single validated dataclass describing a modernized-Llama decoder-only model:
GQA-native, RoPE (configurable theta), RMSNorm, SwiGLU, optional QK-norm,
configurable vocab padding, and an SDPA/eager attention backend.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def default_intermediate_size(hidden: int) -> int:
    """SwiGLU intermediate size: ~8/3 * hidden, rounded up to a multiple of 256."""
    return _round_up((8 * hidden) // 3, 256)


class ModelConfig(BaseModel):
    """Validated model configuration.

    ``n_kv_heads`` and ``intermediate_size`` default to MHA / the standard SwiGLU
    sizing when left unset. Unknown keys are rejected so typos fail loudly.
    """

    model_config = ConfigDict(extra="forbid")

    vocab_size: int
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int = 0  # 0 -> auto (defaults to n_heads, i.e. MHA)
    head_dim: int = 0  # 0 -> auto (hidden // n_heads); set to decouple (e.g. Qwen3-0.6B: 128)
    intermediate_size: int = 0  # 0 -> auto (~8/3 * hidden, rounded to 256)
    seq_len: int = 512
    rope_theta: float = 10000.0
    qk_norm: bool = False
    tie_embeddings: bool = True
    dropout: float = 0.0
    rms_eps: float = 1e-5
    init_std: float = 0.02
    pad_vocab_to: int = 128
    attn_backend: Literal["sdpa", "eager"] = "sdpa"

    @model_validator(mode="after")
    def _resolve_and_validate(self) -> ModelConfig:
        if self.n_kv_heads == 0:
            self.n_kv_heads = self.n_heads
        if self.intermediate_size == 0:
            self.intermediate_size = default_intermediate_size(self.hidden)
        if self.head_dim == 0:  # auto: derive from hidden/n_heads (needs clean division)
            if self.hidden % self.n_heads != 0:
                raise ValueError(
                    f"hidden={self.hidden} must be divisible by n_heads={self.n_heads} "
                    "when head_dim is auto (set head_dim explicitly to decouple)."
                )
            self.head_dim = self.hidden // self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} must be divisible by n_kv_heads={self.n_kv_heads}."
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even for RoPE.")
        return self

    @property
    def n_kv_groups(self) -> int:
        """How many query heads share each KV head (GQA repeat factor)."""
        return self.n_heads // self.n_kv_heads

    @property
    def padded_vocab_size(self) -> int:
        """Vocab size rounded up to ``pad_vocab_to`` for tensor-core efficiency."""
        return _round_up(self.vocab_size, self.pad_vocab_to)

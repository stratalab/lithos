"""Pre-norm transformer block (PRD §6.1.8)."""

from __future__ import annotations

import torch
from torch import nn

from lithos.model.attention import CausalSelfAttention, KVCache
from lithos.model.config import ModelConfig
from lithos.model.mlp import SwiGLU
from lithos.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden, cfg.rms_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden, cfg.rms_eps)
        self.mlp = SwiGLU(cfg.hidden, cfg.intermediate_size)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x), cos, sin, attn_mask, kv_cache))
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x

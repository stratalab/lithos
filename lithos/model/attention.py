"""GQA-native causal self-attention (PRD §6.1.2-3, §6.1.5, §6.1.9-10).

Features:
- Grouped-query attention via ``n_kv_heads`` (set == ``n_heads`` for plain MHA).
- SDPA backend (FlashAttention-class kernels) with an eager, explicit-mask
  fallback; both must produce identical outputs (tested).
- Optional QK-normalization (RMSNorm on per-head q/k) for stability.
- Incremental ``KVCache`` for decoding.

Masking: when there is no cached context (``attn_mask is None``) the query and
key lengths match, so we use the fast causal path. With a cache, the model
passes an explicit boolean mask built from absolute positions.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from lithos.model.config import ModelConfig
from lithos.model.norm import RMSNorm
from lithos.model.rope import apply_rotary


class KVCache:
    """Per-layer key/value cache holding pre-GQA-repeat tensors (B, n_kv, T, D)."""

    def __init__(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is not None and self.v is not None:
            k = torch.cat((self.k, k), dim=2)
            v = torch.cat((self.v, v), dim=2)
        self.k, self.v = k, v
        return k, v


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads for GQA: (B, n_kv, T, D) -> (B, n_kv*n_rep, T, D)."""
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_kv_groups
        self.head_dim = cfg.head_dim
        self.backend = cfg.attn_backend
        self.qk_norm = cfg.qk_norm

        self.q_proj = nn.Linear(cfg.hidden, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden, bias=False)

        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.rms_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.rms_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)  # (B, n_heads, T, D)
        k = k.transpose(1, 2)  # (B, n_kv, T, D)
        v = v.transpose(1, 2)

        q, k = apply_rotary(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        if self.backend == "sdpa":
            if attn_mask is None:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = self._eager_attention(q, k, v, attn_mask)

        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        return self.o_proj(out)

    def _eager_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is None:
            t_q, t_k = q.shape[2], k.shape[2]
            attn_mask = torch.ones(t_q, t_k, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~attn_mask, float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        return attn @ v

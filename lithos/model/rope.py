"""Rotary positional embeddings (PRD §6.1.4).

Uses the HF Llama/Qwen "rotate_half" convention (first/second-half split, not
interleaved) so checkpoints stay within the Qwen3 export envelope (PRD §26.8).
``cos``/``sin`` are indexed by absolute position, so incremental decoding with a
KV cache passes the correct positions for the new tokens.
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # registered buffer; annotated for type checkers

    def __init__(self, head_dim: int, theta: float = 1000000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: (T,) -> (cos, sin), each (T, head_dim), float32."""
        freqs = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q, k of shape (B, H, T, D). cos/sin are (T, D)."""
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out

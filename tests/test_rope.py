"""Tests for lithos.model.rope — shapes, norm preservation, relative-position."""

import torch
from lithos.model.rope import RotaryEmbedding, apply_rotary


def test_rope_cos_sin_shapes():
    rope = RotaryEmbedding(head_dim=8, theta=10000.0)
    cos, sin = rope(torch.arange(5))
    assert cos.shape == (5, 8)
    assert sin.shape == (5, 8)


def test_apply_rotary_preserves_shape_and_norm():
    rope = RotaryEmbedding(head_dim=8)
    cos, sin = rope(torch.arange(6))
    torch.manual_seed(0)
    q = torch.randn(2, 3, 6, 8)  # (B, H, T, D)
    k = torch.randn(2, 3, 6, 8)
    q2, k2 = apply_rotary(q, k, cos, sin)
    assert q2.shape == q.shape
    assert k2.shape == k.shape
    # RoPE is a rotation, so per-vector norms are preserved.
    torch.testing.assert_close(q2.norm(dim=-1), q.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_rope_dot_product_depends_only_on_relative_offset():
    rope = RotaryEmbedding(head_dim=16)
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def rotated_dot(pos_q: int, pos_k: int) -> float:
        qr, _ = apply_rotary(q, q, *rope(torch.tensor([pos_q])))
        kr, _ = apply_rotary(k, k, *rope(torch.tensor([pos_k])))
        return float((qr * kr).sum())

    # Same relative offset (3) -> same rotated dot product.
    assert abs(rotated_dot(5, 2) - rotated_dot(9, 6)) < 1e-4

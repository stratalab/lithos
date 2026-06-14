"""Optimizer construction (PRD §9.3).

AdamW with the standard weight-decay split: decay 2D+ weights (matmuls,
embeddings), no decay on norms and biases. Parameters are kept in fp32 so the
optimizer keeps fp32 master state under bf16 autocast (PRD §9.5, §27).
"""

from __future__ import annotations

import torch
from torch import nn

from lithos.train.config import OptimConfig


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.AdamW:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)

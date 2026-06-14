"""Learning-rate schedule: linear warmup -> cosine decay -> min lr (PRD §9.4)."""

from __future__ import annotations

import math

import torch

from lithos.train.config import ScheduleConfig


def cosine_lr(step: int, cfg: ScheduleConfig, base_lr: float) -> float:
    """LR for 0-based optimization ``step`` under warmup + cosine decay."""
    min_lr = base_lr * cfg.min_lr_ratio
    if step < cfg.warmup_steps:
        return base_lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr

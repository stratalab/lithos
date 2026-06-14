"""Validation perplexity (PRD §11.1).

Token-weighted average cross-entropy over a fixed held-out loader, reported as
loss and perplexity. The val set should be a fixed, decontaminated hold-out
(PRD §27); at smoke scale any packed loader works.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from lithos.data.dataloader import PackedDataLoader


@torch.no_grad()
def compute_perplexity(
    model: torch.nn.Module, loader: PackedDataLoader, n_batches: int, device: str = "cpu"
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for _ in range(n_batches):
        x, y = next(loader)
        _, loss = model(x.to(device), targets=y.to(device))
        n = y.numel()
        total_loss += float(loss) * n
        total_tokens += n
    if was_training:
        model.train()
    avg = total_loss / max(1, total_tokens)
    return {"loss": avg, "perplexity": math.exp(avg), "tokens": total_tokens, "batches": n_batches}

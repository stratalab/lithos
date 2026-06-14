"""Checkpoint save/load (PRD §9.8-9.9).

A checkpoint is a directory::

    checkpoints/step_000500/
      model.safetensors   # weights (safetensors; handles tied embeddings)
      train_state.pt      # optimizer, step, tokens, dataloader position, RNG
      meta.json           # step/tokens + tokenizer & corpus references

Resume restores model + optimizer + RNG + dataloader position so training
continues at the exact same state and data position (PRD §9.9, §27).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_model, save_model
from torch import nn

from lithos.utils.io import ensure_new_dir, write_json


def save_checkpoint(
    ckpt_dir: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    dataloader_state: dict[str, Any],
    meta: dict[str, Any],
) -> Path:
    out = ensure_new_dir(ckpt_dir)
    save_model(model, str(out / "model.safetensors"))
    train_state = {
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "dataloader": dataloader_state,
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(train_state, out / "train_state.pt")
    write_json(out / "meta.json", {**meta, "step": step, "tokens_seen": tokens_seen})
    return out


def load_checkpoint(
    ckpt_dir: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    restore_rng: bool = True,
) -> dict[str, Any]:
    d = Path(ckpt_dir)
    load_model(model, str(d / "model.safetensors"))
    state: dict[str, Any] = torch.load(d / "train_state.pt", weights_only=False)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if restore_rng:
        torch.set_rng_state(state["rng"]["torch"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
    return state


def find_latest_checkpoint(run_root: str | Path) -> Path | None:
    ckpts = sorted(Path(run_root, "checkpoints").glob("step_*"))
    return ckpts[-1] if ckpts else None

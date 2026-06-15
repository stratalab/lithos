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

from lithos.utils.io import ensure_new_dir, read_json, write_json


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
    full_meta = {**meta, "step": step, "tokens_seen": tokens_seen}
    mcfg = getattr(model, "cfg", None)  # self-describing: embed the arch for size-agnostic reload
    if mcfg is not None and hasattr(mcfg, "model_dump"):
        full_meta["model"] = mcfg.model_dump(mode="json")
    write_json(out / "meta.json", full_meta)
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


def load_model_weights(ckpt_dir: str | Path, model: nn.Module) -> None:
    """Load ONLY model weights from a checkpoint (for fine-tune/SFT init).

    Unlike ``load_checkpoint``, this ignores optimizer/RNG/dataloader state — the
    fine-tune run starts a fresh optimizer and schedule from step 0.
    """
    load_model(model, str(Path(ckpt_dir) / "model.safetensors"))


def _model_config_for(ckpt_dir: str | Path, model_config_cls):
    """Resolve the model architecture for a checkpoint (size-agnostic)."""
    d = Path(ckpt_dir)
    meta = read_json(d / "meta.json") if (d / "meta.json").exists() else {}
    if "model" in meta:  # new checkpoints embed the arch
        return model_config_cls(**meta["model"])
    # Backward-compat: older checkpoints predate self-describing meta — fall back to
    # the run's resolved_config.yaml (checkpoints/step_X -> run root).
    resolved = d.parent.parent / "resolved_config.yaml"
    if resolved.exists():
        import yaml

        return model_config_cls(**yaml.safe_load(resolved.read_text())["model"])
    raise ValueError(
        f"{d}: no model config in meta.json and no resolved_config.yaml to fall back to "
        "(new checkpoints embed the arch automatically)."
    )


def load_model_from_checkpoint(ckpt_dir: str | Path, device: str = "cpu"):
    """Build the right model from a checkpoint's embedded architecture and load weights.

    Size-agnostic: the arch comes from the checkpoint (meta.json, or an older run's
    resolved_config.yaml), so this works on a 100M or a 3B checkpoint without being
    told the shape. Returns a ``LithosForCausalLM`` in eval mode on ``device``.
    """
    from lithos.model import LithosForCausalLM
    from lithos.model.config import ModelConfig

    model = LithosForCausalLM(_model_config_for(ckpt_dir, ModelConfig)).to(device).eval()
    load_model_weights(ckpt_dir, model)
    return model


def find_latest_checkpoint(run_root: str | Path) -> Path | None:
    ckpts = sorted(Path(run_root, "checkpoints").glob("step_*"))
    return ckpts[-1] if ckpts else None

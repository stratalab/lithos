#!/usr/bin/env python
"""Train a Lithos model from a config (PRD §16, §23).

# single GPU
python scripts/train_model.py --config configs/train/single-gpu-smoke.yaml

# multi-GPU (DDP) via torchrun
torchrun --nproc_per_node=2 scripts/train_model.py --config configs/train/100m.yaml
"""

from __future__ import annotations

import argparse

from lithos.train.config import TrainConfig
from lithos.train.loop import train
from lithos.utils.config import load_and_validate


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a Lithos model.")
    ap.add_argument("--config", required=True, help="Path to a training YAML config.")
    ap.add_argument("--resume", default=None, help="Checkpoint directory to resume from.")
    ap.add_argument("--override", nargs="*", default=[], help="Dotted-key overrides.")
    args = ap.parse_args()

    cfg = load_and_validate(args.config, TrainConfig, args.override)
    run = train(cfg, resume_from=args.resume)
    if run is not None:  # rank 0 (or single process)
        print(f"Run complete -> {run.root}")


if __name__ == "__main__":
    main()

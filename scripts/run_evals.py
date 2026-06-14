#!/usr/bin/env python
"""Evaluate a Lithos checkpoint (PRD §16, §23).

    python scripts/run_evals.py --config configs/eval/base.yaml \
        --checkpoint runs/<run>/checkpoints/step_001000
"""

from __future__ import annotations

import argparse

from lithos.evals.config import EvalConfig
from lithos.evals.run import evaluate_checkpoint
from lithos.utils.config import load_and_validate


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a Lithos checkpoint.")
    ap.add_argument("--config", required=True, help="Path to an eval YAML config.")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint directory to evaluate.")
    ap.add_argument("--override", nargs="*", default=[], help="Dotted-key overrides.")
    args = ap.parse_args()

    cfg = load_and_validate(args.config, EvalConfig, args.override)
    out = evaluate_checkpoint(cfg, args.checkpoint)
    print(f"Eval report -> {out}")


if __name__ == "__main__":
    main()

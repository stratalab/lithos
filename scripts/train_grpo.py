#!/usr/bin/env python
"""GRPO / RLVR entrypoint (Phase 11).

    python scripts/train_grpo.py --config configs/grpo/lithos-100m-arith.yaml

Starts from the SFT checkpoint (config ``init_from``), ``data.kind: grpo``. Samples
rollouts from the policy, scores them with a verifier, and updates via group-relative
policy gradient + a KL leash to the frozen reference.
"""

from __future__ import annotations

import argparse

from lithos.posttrain.grpo_trainer import train_grpo
from lithos.train.config import TrainConfig
from lithos.utils.config import load_and_validate


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO / RL with verifiable rewards.")
    ap.add_argument("--config", required=True, help="Path to a GRPO YAML config.")
    ap.add_argument("--override", nargs="*", default=[], help="Dotted-key overrides.")
    args = ap.parse_args()

    cfg = load_and_validate(args.config, TrainConfig, args.override)
    run = train_grpo(cfg)
    print(f"GRPO run -> {run.root if run else '(non-main rank)'}")


if __name__ == "__main__":
    main()

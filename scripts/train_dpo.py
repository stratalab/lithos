#!/usr/bin/env python
"""Direct Preference Optimization entrypoint (Phase 11).

    python scripts/train_dpo.py --config configs/dpo/lithos-100m-dolly.yaml

Starts from the SFT checkpoint (config ``init_from``), with ``data.kind: dpo``
pointing at a preferences-JSONL. Custom step (policy vs frozen reference); reuses
the shared scaffolding.
"""

from __future__ import annotations

import argparse

from lithos.posttrain.dpo_trainer import train_dpo
from lithos.train.config import TrainConfig
from lithos.utils.config import load_and_validate


def main() -> None:
    ap = argparse.ArgumentParser(description="Direct Preference Optimization (DPO).")
    ap.add_argument("--config", required=True, help="Path to a DPO YAML config.")
    ap.add_argument("--override", nargs="*", default=[], help="Dotted-key overrides.")
    args = ap.parse_args()

    cfg = load_and_validate(args.config, TrainConfig, args.override)
    run = train_dpo(cfg)
    print(f"DPO run -> {run.root if run else '(non-main rank)'}")


if __name__ == "__main__":
    main()

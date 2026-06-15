#!/usr/bin/env python
"""Supervised fine-tuning entrypoint (Phase 11).

    python scripts/train_sft.py --config configs/sft/lithos-100m-dolly.yaml

SFT reuses the pretrain loop: the config sets ``init_from`` (weight-only base init)
and ``data.kind: sft`` (messages-JSONL with loss-masked labels). Nothing else differs.
"""

from __future__ import annotations

import argparse

from lithos.train.entry import train_from_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Supervised fine-tuning (SFT).")
    ap.add_argument("--config", required=True, help="Path to an SFT YAML config.")
    ap.add_argument("--override", nargs="*", default=[], help="Dotted-key overrides.")
    ap.add_argument("--resume-from", default=None, help="Resume a paused SFT run.")
    args = ap.parse_args()

    run = train_from_config(args.config, args.override, resume_from=args.resume_from)
    print(f"SFT run -> {run.root if run else '(non-main rank)'}")


if __name__ == "__main__":
    main()

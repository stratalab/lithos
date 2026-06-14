#!/usr/bin/env python
"""Run a data-recipe ablation (Phase 10): variants -> proxies -> eval -> diff.

    python scripts/run_ablation.py --config configs/ablation/quality-threshold.yaml

Each variant rebuilds the corpus with its overrides, trains the SAME proxy, and scores it
on the SAME frozen battery; the summary diffs every variant against the baseline. Needs a
GPU (the proxy trains) and `--extra eval` (the benchmark battery).
"""

from __future__ import annotations

import argparse
import json

from lithos.evals.ablation import AblationConfig, run_ablation
from lithos.utils.config import load_and_validate


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a data-recipe ablation.")
    ap.add_argument("--config", required=True, help="Path to an ablation YAML config.")
    ap.add_argument("--override", nargs="*", default=[], help="Dotted-key overrides.")
    args = ap.parse_args()

    cfg = load_and_validate(args.config, AblationConfig, args.override)
    summary = run_ablation(cfg)
    print(json.dumps(summary, indent=2))
    if summary["winners"]:
        print(f"\nWinners (beat baseline {summary['baseline']!r}, best first): {summary['winners']}")
    else:
        print(f"\nNo variant beat the baseline {summary['baseline']!r}.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build a packed, multi-source SFT corpus into dual-stream shards (epic E2).

Renders messages-JSONL sources with the chat template, screens eval-battery leaks
(F2), blends sources at controlled ratios, and packs into memmapped
token + loss-mask shards + an SFT manifest (mirrors scripts/tokenize_corpus.py).

    python scripts/build_sft_corpus.py --config configs/sft/mix-smoke.yaml
"""

from __future__ import annotations

import argparse

from lithos.posttrain.sft_corpus import SFTCorpusBuildConfig, build_sft_corpus
from lithos.utils.config import load_and_validate


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a packed multi-source SFT corpus.")
    ap.add_argument("--config", required=True, help="Path to an SFT-corpus build YAML.")
    ap.add_argument("--override", nargs="*", default=[], help="Dotted-key overrides.")
    args = ap.parse_args()

    cfg = load_and_validate(args.config, SFTCorpusBuildConfig, args.override)
    manifest = build_sft_corpus(cfg)
    mix = ", ".join(f"{name}:{m['examples']}" for name, m in manifest["mixture"].items())
    print(
        f"Built SFT corpus {cfg.name}-{cfg.version}: "
        f"{manifest['num_examples']:,} examples, {manifest['num_tokens']:,} tokens, "
        f"loss_token_fraction={manifest['loss_token_fraction']}, "
        f"{len(manifest['shards'])} shard(s) [{mix}] -> {cfg.output_dir}"
    )


if __name__ == "__main__":
    main()

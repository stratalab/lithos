#!/usr/bin/env python
"""Build tokenized shards + a corpus manifest from a config (PRD §16, §23).

python scripts/tokenize_corpus.py --config configs/data/smoke.yaml
"""

from __future__ import annotations

import argparse

from lithos.data.pipeline import CorpusBuildConfig, build_corpus
from lithos.utils.config import load_and_validate


def main() -> None:
    ap = argparse.ArgumentParser(description="Tokenize a corpus into training shards.")
    ap.add_argument("--config", required=True, help="Path to a corpus-build YAML config.")
    ap.add_argument("--override", nargs="*", default=[], help="Dotted-key overrides.")
    args = ap.parse_args()

    cfg = load_and_validate(args.config, CorpusBuildConfig, args.override)
    manifest = build_corpus(cfg)
    print(
        f"Built corpus {cfg.name}-{cfg.version}: "
        f"{manifest['num_documents']:,} docs, {manifest['num_tokens']:,} tokens, "
        f"{len(manifest['shards'])} shard(s) -> {cfg.output_dir}"
    )


if __name__ == "__main__":
    main()

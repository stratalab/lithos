#!/usr/bin/env python
"""Train the Lithos byte-level BPE tokenizer from a config (PRD §16).

python scripts/train_tokenizer.py --config configs/tokenizer/bpe-32k.yaml
"""

from __future__ import annotations

import argparse

from lithos.tokenizer.data_source import resolve_texts
from lithos.tokenizer.tokenizer_config import TokenizerTrainConfig
from lithos.tokenizer.train_tokenizer import (
    build_manifest,
    sample_report,
    save_tokenizer,
    train_tokenizer,
)
from lithos.utils.config import load_and_validate


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Lithos byte-level BPE tokenizer.")
    ap.add_argument("--config", required=True, help="Path to a tokenizer training YAML config.")
    ap.add_argument("--out", default=None, help="Override the output directory.")
    ap.add_argument("--max-documents", type=int, default=None, help="Override data.max_documents.")
    ap.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Dotted-key overrides, e.g. tokenizer.vocab_size=8000",
    )
    args = ap.parse_args()

    cfg = load_and_validate(args.config, TokenizerTrainConfig, args.override)
    if args.out:
        cfg.output_dir = args.out
    if args.max_documents is not None:
        cfg.data.max_documents = args.max_documents

    sources, texts = resolve_texts(cfg.data)
    print(
        f"Training {cfg.tokenizer.full_name} (vocab={cfg.tokenizer.vocab_size}) from {sources}..."
    )
    tok, stats = train_tokenizer(cfg.tokenizer, texts)
    manifest = build_manifest(cfg.tokenizer, stats, sources)
    report = sample_report(tok, cfg.report_samples)
    out = save_tokenizer(tok, cfg.tokenizer, cfg.output_dir, manifest, report)

    print(
        f"Saved {tok.get_vocab_size()}-token tokenizer to {out} "
        f"(docs={stats['num_documents']:,}, chars={stats['approx_chars']:,})"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate a small synthetic smoke corpus as JSONL (PRD §16, §23).

    python scripts/prepare_smoke_data.py --out data/smoke

The content is deterministic (seeded) so the smoke pipeline is reproducible.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from lithos.utils.io import ensure_dir

WORDS = [
    "the",
    "of",
    "and",
    "to",
    "a",
    "in",
    "is",
    "it",
    "for",
    "on",
    "with",
    "as",
    "by",
    "from",
    "at",
    "this",
    "that",
    "model",
    "token",
    "train",
    "data",
    "corpus",
    "lithos",
    "language",
    "transformer",
    "attention",
    "layer",
    "weight",
    "loss",
    "gradient",
    "batch",
    "sequence",
    "vocabulary",
    "tokenizer",
    "shard",
    "pipeline",
    "evaluation",
    "sample",
]


def _make_doc(rng: random.Random, idx: int) -> dict[str, str]:
    sentences = []
    for _ in range(rng.randint(3, 8)):
        words = rng.choices(WORDS, k=rng.randint(6, 16))
        words[0] = words[0].capitalize()
        sentences.append(" ".join(words) + ".")
    return {
        "id": f"smoke-{idx:06d}",
        "text": " ".join(sentences),
        "source": "smoke",
        "language": "en",
        "license": "synthetic",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic smoke corpus.")
    ap.add_argument("--out", default="data/smoke", help="Output directory.")
    ap.add_argument("--num-docs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    path = Path(ensure_dir(args.out)) / "corpus.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(args.num_docs):
            f.write(json.dumps(_make_doc(rng, i)) + "\n")
    print(f"Wrote {args.num_docs} smoke documents to {path}")


if __name__ == "__main__":
    main()

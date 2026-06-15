#!/usr/bin/env python
"""Download databricks-dolly-15k and write it as messages-JSONL for SFT (Phase 11).

    python scripts/prepare_dolly_sft.py --out data/sft/dolly
    python scripts/prepare_dolly_sft.py --out data/sft/dolly --categories open_qa closed_qa summarization

Dolly is **human-written** (CC-BY-SA) — clean provenance, no GPT taint, which fits
the sovereignty rule even on the test bench. Each record (instruction [+ context],
response, category) becomes a one-turn user/assistant conversation:

    {"messages": [{"role": "user", "content": "<instruction>\\n\\n<context>"},
                  {"role": "assistant", "content": "<response>"}]}
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def to_messages(rec: dict) -> dict:
    instr = rec["instruction"].strip()
    ctx = (rec.get("context") or "").strip()
    user = f"{instr}\n\n{ctx}" if ctx else instr
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": rec["response"].strip()},
        ]
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Dolly-15k -> messages-JSONL for SFT.")
    ap.add_argument("--out", default="data/sft/dolly")
    ap.add_argument("--categories", nargs="*", default=None, help="filter to these dolly categories")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap total examples (0 = all)")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = [
        r
        for r in ds
        if r["instruction"].strip()
        and r["response"].strip()
        and (not args.categories or r["category"] in args.categories)
    ]
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    n_val = int(len(rows) * args.val_frac)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, recs in {"val": rows[:n_val], "train": rows[n_val:]}.items():
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(to_messages(r), ensure_ascii=False) + "\n")
        print(f"wrote {len(recs):6d} -> {path}")
    print("categories:", dict(Counter(r["category"] for r in rows)))


if __name__ == "__main__":
    main()

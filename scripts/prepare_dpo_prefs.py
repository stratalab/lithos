#!/usr/bin/env python
"""Generate on-policy DPO preferences from the SFT model (Phase 11).

    chosen   = Dolly's human answer
    rejected = the SFT model's OWN sampled answer to the same prompt

Fully self-contained and sovereign: no external preference labels, no GPT taint.
The signal is "human answer > the model's own typical output", which DPO then
sharpens (and it's the same generate-then-label pattern the flagship will use).

    uv run python scripts/prepare_dpo_prefs.py \
        --sft runs/<run>/checkpoints/step_001200 --out data/dpo/dolly_onpolicy --limit 2000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

from lithos.model import LithosForCausalLM
from lithos.model.config import ModelConfig
from lithos.model.generation import generate
from lithos.posttrain.chat_template import render_prompt, special_ids
from lithos.train.checkpoint import load_model_weights

MODEL_100M = ModelConfig(
    vocab_size=32000, n_layers=12, hidden=768, n_heads=12, n_kv_heads=12,
    intermediate_size=2048, seq_len=2048, rope_theta=10000.0, qk_norm=True, tie_embeddings=True,
)
TOKENIZER = "artifacts/tokenizer/fineweb-edu-32k/tokenizer.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="On-policy DPO preference generation.")
    ap.add_argument("--sft", required=True, help="SFT checkpoint dir.")
    ap.add_argument("--dolly", default="data/sft/dolly/train.jsonl")
    ap.add_argument("--out", default="data/dpo/dolly_onpolicy")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(TOKENIZER)
    end_id = special_ids(tok)["<|end|>"]
    model = LithosForCausalLM(MODEL_100M).to(device).eval()
    load_model_weights(args.sft, model)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    rows = [json.loads(line) for line in open(args.dolly, encoding="utf-8") if line.strip()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]

    prefs: list[dict] = []
    for i, rec in enumerate(rows):
        msgs = rec["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        human = next(m["content"] for m in msgs if m["role"] == "assistant").strip()
        pids = render_prompt([{"role": "user", "content": user}], tok)
        out = generate(
            model, torch.tensor([pids], device=device), args.max_new,
            temperature=0.8, top_p=0.95, eos_token_id=end_id, generator=gen,
        )
        resp = out[0, len(pids):].tolist()
        if end_id in resp:
            resp = resp[: resp.index(end_id)]
        rejected = tok.decode(resp, skip_special_tokens=True).strip()
        if rejected and rejected != human:  # need a genuine (and non-empty) contrast
            prefs.append(
                {"prompt": [{"role": "user", "content": user}], "chosen": human, "rejected": rejected}
            )
        if (i + 1) % 200 == 0:
            print(f"  generated {i + 1}/{len(rows)} ({len(prefs)} kept)")

    random.Random(args.seed + 1).shuffle(prefs)
    n_val = int(len(prefs) * args.val_frac)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, recs in {"val": prefs[:n_val], "train": prefs[n_val:]}.items():
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(recs):6d} -> {path}")


if __name__ == "__main__":
    main()

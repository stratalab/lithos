#!/usr/bin/env python
"""Generate distillation data from an open teacher (Phase 11).

Synthetic-data distillation: the teacher answers a set of prompts; we write its
responses as a messages-JSONL, then SFT our student on it — tokenizing with OUR
vocab (tokenizer-agnostic, sidesteps the teacher's). Default teacher:
DeepSeek-R1-Distill-Qwen-1.5B (MIT, open) — it produces reasoning traces, so this
rehearses the flagship's reasoning-distillation path.

    uv run --extra eval python scripts/distill_generate.py \
        --prompts data/sft/dolly/train.jsonl --out data/distill/r1_dolly --limit 800
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_TEACHER = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate distillation data from an open teacher.")
    ap.add_argument("--teacher", default=DEFAULT_TEACHER)
    ap.add_argument("--prompts", default="data/sft/dolly/train.jsonl", help="messages-JSONL of prompts")
    ap.add_argument("--out", default="data/distill/r1_dolly")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--max-prompt-tokens", type=int, default=384, help="skip longer prompts")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--strip-think", action="store_true", help="keep only the final answer (drop <think>..)")
    ap.add_argument(
        "--decontam-probes",
        default=None,
        help="Probe JSONL (decontam.write_probes) — screen out eval-battery leaks (F2, recommended).",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.teacher)
    tok.padding_side = "left"  # left-pad for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16).to(device).eval()
    )

    all_prompts: list[str] = []
    for line in open(args.prompts, encoding="utf-8"):
        if line.strip():
            user = next((m["content"] for m in json.loads(line)["messages"] if m["role"] == "user"), None)
            if user:
                all_prompts.append(user)
    # Skip long-context prompts: they OOM batched prefill AND would be dropped at SFT
    # (prompt + response > seq_len), so they're wasted generation on a 12GB card.
    prompts = [p for p in all_prompts if len(tok(p).input_ids) <= args.max_prompt_tokens][: args.limit]
    print(f"{len(prompts)} prompts (<= {args.max_prompt_tokens} tok) of {len(all_prompts)} total")

    recs: list[dict] = []
    lengths: list[int] = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        texts = [
            tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new, do_sample=True, temperature=0.7, top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )
        for j, p in enumerate(batch):
            gen = out[j][enc["input_ids"].shape[1] :]
            lengths.append(int((gen != tok.pad_token_id).sum()))
            resp = tok.decode(gen, skip_special_tokens=True).strip()
            if args.strip_think and "</think>" in resp:
                resp = resp.split("</think>")[-1].strip()
            if resp:
                recs.append({"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": resp}]})
        print(f"  {min(i + args.batch_size, len(prompts))}/{len(prompts)} generated")

    if lengths:
        lengths.sort()
        print(f"response tokens: median {lengths[len(lengths)//2]}, p90 {lengths[int(len(lengths)*0.9)]}, max {lengths[-1]}")

    if args.decontam_probes:
        from lithos.posttrain.decontam_gate import PostTrainDecontaminator, messages_text

        gate = PostTrainDecontaminator.from_probe_file(args.decontam_probes)
        recs = gate.screen(recs, messages_text)
        print(f"decontam: {gate.report()}")

    random.Random(0).shuffle(recs)
    n_val = int(len(recs) * args.val_frac)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rs in {"val": recs[:n_val], "train": recs[n_val:]}.items():
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rs):6d} -> {path}")


if __name__ == "__main__":
    main()

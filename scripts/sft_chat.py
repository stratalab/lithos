#!/usr/bin/env python
"""Chat with an SFT'd Lithos checkpoint (native path + chat template), or compare
the base vs the SFT'd model side by side (Phase 11).

    # base (completes text) vs SFT (answers & stops) on demo prompts:
    uv run python scripts/sft_chat.py --sft runs/<run>/checkpoints/step_001200 \
        --base models/lithos-100m-v0.1/checkpoint --compare

    # interactive chat with the SFT model:
    uv run python scripts/sft_chat.py --sft runs/<run>/checkpoints/step_001200
"""

from __future__ import annotations

import argparse

import torch
from lithos.model.generation import generate
from lithos.posttrain.chat_template import render_prompt, special_ids
from lithos.train.checkpoint import load_model_from_checkpoint
from tokenizers import Tokenizer

TOKENIZER = "artifacts/tokenizer/fineweb-edu-32k/tokenizer.json"
PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "List three primary colors.",
    "Why is the sky blue?",
]


def _load(ckpt: str, device: str):
    return load_model_from_checkpoint(ckpt, device)  # arch read from the checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with / compare an SFT'd Lithos model.")
    ap.add_argument("--sft", required=True, help="SFT checkpoint dir (model.safetensors).")
    ap.add_argument("--base", default=None, help="Base checkpoint dir (for --compare).")
    ap.add_argument("--compare", action="store_true", help="Base-vs-SFT on demo prompts.")
    ap.add_argument("--max-new", type=int, default=48)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(TOKENIZER)
    end_id = special_ids(tok)["<|end|>"]
    eos_id = special_ids(tok)["<eos>"]
    sft = _load(args.sft, device)

    def _gen(model, prompt_ids, eos):
        out = generate(
            model, torch.tensor([prompt_ids], device=device), args.max_new,
            temperature=0.7, top_p=0.9, eos_token_id=eos,
        )
        return out[0, len(prompt_ids):].tolist()

    def chat(model, text):  # chat-formatted: answer the user, stop at <|end|>
        resp = _gen(model, render_prompt([{"role": "user", "content": text}], tok), end_id)
        if end_id in resp:
            resp = resp[: resp.index(end_id)]
        return tok.decode(resp, skip_special_tokens=True).strip()

    def complete(model, text):  # raw base behaviour: just continue the text
        return tok.decode(_gen(model, tok.encode(text).ids, eos_id), skip_special_tokens=True).strip()

    if args.compare:
        base = _load(args.base, device)
        for p in PROMPTS:
            print(f"\n\033[1m### {p}\033[0m")
            print(f"  base (completes): {complete(base, p)[:280]!r}")
            print(f"  SFT  (answers)  : {chat(sft, p)[:280]!r}")
        return

    print("Chat with the SFT'd 100M. Empty line to quit.\n")
    while True:
        try:
            text = input("\033[1m>>> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text.strip():
            break
        print(chat(sft, text) + "\n")


if __name__ == "__main__":
    main()

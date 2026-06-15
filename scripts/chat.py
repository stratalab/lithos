#!/usr/bin/env python
"""Interactive text completion against a Lithos HF/Qwen3 export.

    # interactive REPL:
    uv run python scripts/chat.py models/lithos-100m-v0.1/hf_export
    # one-shot (one or more prompts, single model load):
    uv run python scripts/chat.py models/lithos-100m-v0.1/hf_export "The capital of France is" "Photosynthesis is"

NOTE: a BASE model COMPLETES text — it does not answer questions or chat. Ask it a
question and it will likely continue with *more* text in that style, not an answer.
That gap is exactly what SFT (post-training) fills.

Knobs via env: LITHOS_TEMP (default 0.8), LITHOS_MAXTOK (64), LITHOS_REPPEN (1.3).
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/chat.py <export_dir> [prompt ...]")
        sys.exit(1)
    export_dir = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(export_dir)
    model = (
        AutoModelForCausalLM.from_pretrained(export_dir, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    temp = float(os.environ.get("LITHOS_TEMP", "0.8"))
    max_new = int(os.environ.get("LITHOS_MAXTOK", "64"))
    rep_pen = float(os.environ.get("LITHOS_REPPEN", "1.3"))

    @torch.no_grad()
    def complete(prompt: str) -> str:
        enc = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=temp > 0,
            temperature=temp,
            top_p=0.95,
            repetition_penalty=rep_pen,  # base models loop hard without this
            pad_token_id=tok.eos_token_id,
        )
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    if len(sys.argv) >= 3:  # one-shot: complete each arg prompt
        for prompt in sys.argv[2:]:
            print(f"\n\033[1m>>> {prompt}\033[0m")
            print(prompt + complete(prompt))
        return

    print(f"Loaded base model from {export_dir} on {device}.")
    print("It COMPLETES text (not a chatbot). Empty line or Ctrl-D to quit.\n")
    while True:
        try:
            prompt = input("\033[1m>>> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            break
        print(prompt + complete(prompt) + "\n")


if __name__ == "__main__":
    main()

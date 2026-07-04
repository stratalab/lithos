#!/usr/bin/env python
"""Confirm Qwen3 import parity on the REAL Qwen3-0.6B weights (epic E7).

The offline test (tests/test_hf_import.py) already proves the arch/mapping/head_dim
path on tiny random weights. This downloads the real Qwen3-0.6B (public, no auth,
~1.2 GB) and checks that LithosForCausalLM reproduces its logits on real prompts —
the on-real-weights confirmation for the "one deployment recipe" claim.

    uv run --extra eval python scripts/spike_qwen3_import.py
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen3-0.6B -> Lithos import parity spike.")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tol", type=float, default=1e-2, help="max |Δlogit| to call parity")
    args = ap.parse_args()

    from lithos.serve.hf_import import load_qwen3
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.model} (transformers)…")
    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    print(f"  config: hidden={hf.config.hidden_size} heads={hf.config.num_attention_heads} "
          f"head_dim={hf.config.head_dim} kv_heads={hf.config.num_key_value_heads} "
          f"layers={hf.config.num_hidden_layers} tied={hf.config.tie_word_embeddings}")
    print("importing into LithosForCausalLM…")
    lithos = load_qwen3(hf).eval()
    tok = AutoTokenizer.from_pretrained(args.model)

    prompts = ["The capital of France is", "2 + 2 =", "def add(a, b):\n    return"]
    vocab = hf.config.vocab_size
    max_delta = 0.0
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        with torch.no_grad():
            theirs = hf(ids).logits
            ours, _ = lithos(ids)
        delta = (ours[:, :, :vocab] - theirs).abs().max().item()
        max_delta = max(max_delta, delta)
        # sanity: argmax next-token agreement
        agree = (ours[:, -1, :vocab].argmax() == theirs[:, -1].argmax()).item()
        print(f"  {p!r:40}  max|Δlogit|={delta:.2e}  next-token match={agree}")

    print(f"\nmax |Δlogit| over all prompts: {max_delta:.3e}  (tol {args.tol:.0e})")
    print("PARITY CONFIRMED — the family shares one tooling path." if max_delta < args.tol
          else "DIVERGENCE — investigate before promising one recipe.")


if __name__ == "__main__":
    main()

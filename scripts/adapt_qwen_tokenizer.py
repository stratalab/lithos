#!/usr/bin/env python
"""Cut an augmented Qwen tokenizer artifact for v1-on-Qwen post-training.

Loads a base ``tokenizer.json`` (Qwen's), adds the Lithos chat + TIR specials it lacks,
and writes ``tokenizer.json`` + ``adapt.json`` to an output dir. The SFT / RLVR builds then
consume it exactly like any other tokenizer — point their ``tokenizer_path`` at the output.

    # obtain Qwen's tokenizer.json (needs HF access), then:
    python scripts/adapt_qwen_tokenizer.py \
        --base ~/.cache/.../Qwen3-1.7B/tokenizer.json \
        --out artifacts/tokenizer/qwen3-1.7b-lithos \
        --base-model Qwen/Qwen3-1.7B-Base

The model MUST be imported to match: `import_vocab_size(hf_config, result)` — the printed
`vocab_size` is what `load_qwen3(hf, vocab_size=...)` needs so every added special is a valid
(emittable) token, not masked padding. See docs/v1-on-qwen.md §6.
"""

from __future__ import annotations

import argparse

from lithos.serve.tokenizer_adapt import save_augmented_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="Augment a Qwen tokenizer with Lithos specials.")
    ap.add_argument("--base", required=True, help="Path to the base tokenizer.json (Qwen's).")
    ap.add_argument("--out", required=True, help="Output dir for the augmented artifact.")
    ap.add_argument("--base-model", default=None, help="e.g. Qwen/Qwen3-1.7B-Base (recorded).")
    args = ap.parse_args()

    res = save_augmented_tokenizer(args.base, args.out, base_model=args.base_model)
    print(
        f"Augmented tokenizer -> {args.out}\n"
        f"  base vocab:  {res.base_vocab_size:,}\n"
        f"  new vocab:   {res.vocab_size:,}  (+{len(res.added)} added, {len(res.reused)} reused)\n"
        f"  added:       {list(res.added)}\n"
        f"  reused:      {list(res.reused)}\n"
        f"  -> import the model with vocab_size >= {res.vocab_size}"
    )


if __name__ == "__main__":
    main()

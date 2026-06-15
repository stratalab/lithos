#!/usr/bin/env python
"""Generate on-policy DPO preferences from the SFT model (Phase 11).

Two modes (both self-contained / sovereign — no external labels, no GPT taint):

  --mode two-sample  (default, the stable recipe): sample TWO responses from the
      model and keep the better one as `chosen`, the worse as `rejected`, ranked by
      a quality judge = token-F1 to the human reference minus a repetition penalty.
      Both responses are in-distribution, so DPO sharpens within reach (avoids the
      coherence collapse seen when `chosen` is the far-OOD human text).

  --mode human       (the simple baseline): chosen = Dolly human answer, rejected =
      the model's own sample. Larger distribution gap; tends to destabilise a tiny
      model.

    uv run python scripts/prepare_dpo_prefs.py --sft runs/<run>/checkpoints/step_001200 \
        --mode two-sample --out data/dpo/dolly_twosample --limit 2000
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
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
_REP_WEIGHT = 0.3   # how much looping is penalised
_MIN_MARGIN = 0.05  # keep a pair only if the judge clearly separates the two samples


def _toks(s: str) -> list[str]:
    return s.lower().split()


def _f1(pred: str, ref: str) -> float:
    p, r = _toks(pred), _toks(ref)
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def _repetition(s: str) -> float:
    t = _toks(s)
    if len(t) < 2:
        return 0.0
    bigrams = list(zip(t, t[1:]))
    return 1.0 - len(set(bigrams)) / len(bigrams)  # fraction of duplicate bigrams


def _quality(sample: str, human: str) -> float:
    """Sovereign judge: content overlap with the reference, minus looping."""
    return _f1(sample, human) - _REP_WEIGHT * _repetition(sample)


def main() -> None:
    ap = argparse.ArgumentParser(description="On-policy DPO preference generation.")
    ap.add_argument("--sft", required=True, help="SFT checkpoint dir.")
    ap.add_argument("--mode", choices=["two-sample", "human"], default="two-sample")
    ap.add_argument("--dolly", default="data/sft/dolly/train.jsonl")
    ap.add_argument("--out", default="data/dpo/dolly_twosample")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(TOKENIZER)
    end_id = special_ids(tok)["<|end|>"]
    model = LithosForCausalLM(MODEL_100M).to(device).eval()
    load_model_weights(args.sft, model)
    g = torch.Generator(device=device).manual_seed(args.seed)

    def sample(prompt: str) -> str:
        pids = render_prompt([{"role": "user", "content": prompt}], tok)
        out = generate(
            model, torch.tensor([pids], device=device), args.max_new,
            temperature=args.temperature, top_p=0.95, eos_token_id=end_id, generator=g,
        )
        resp = out[0, len(pids):].tolist()
        if end_id in resp:
            resp = resp[: resp.index(end_id)]
        return tok.decode(resp, skip_special_tokens=True).strip()

    rows = [json.loads(line) for line in open(args.dolly, encoding="utf-8") if line.strip()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]

    prefs: list[dict] = []
    for i, rec in enumerate(rows):
        msgs = rec["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        human = next(m["content"] for m in msgs if m["role"] == "assistant").strip()

        if args.mode == "human":
            rej = sample(user)
            if rej and rej != human:
                chosen, rejected = human, rej
            else:
                chosen = rejected = None
        else:  # two-sample: rank two model samples by the judge
            a, b = sample(user), sample(user)
            if not (a or b) or a == b:
                chosen = rejected = None
            else:
                sa, sb = _quality(a, human), _quality(b, human)
                if abs(sa - sb) < _MIN_MARGIN:
                    chosen = rejected = None  # judge can't separate them -> no signal
                else:
                    chosen, rejected = (a, b) if sa > sb else (b, a)
                    if not chosen.strip():
                        chosen = rejected = None

        if chosen is not None:
            prefs.append(
                {"prompt": [{"role": "user", "content": user}], "chosen": chosen, "rejected": rejected}
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

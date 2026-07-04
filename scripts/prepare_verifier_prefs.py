#!/usr/bin/env python
"""Generate verifier-labeled DPO preferences from a verifiable task bank (epic E8).

For each task: sample K completions from the (SFT'd) model, verify each with the
E1 verifier, and pair a correct one (chosen) against an incorrect one (rejected).
Both are the model's own samples — on-policy and in-distribution (the sovereign
pref path). Output is the standard pref JSONL that DPO already consumes.

    uv run python scripts/prepare_verifier_prefs.py \
        --sft runs/<sft-run>/checkpoints/step_XXXXXX \
        --tasks corpus/problems/math.jsonl --out data/dpo/verifier --samples 6
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from lithos.model.generation import generate
from lithos.posttrain.chat_template import render_prompt, special_ids
from lithos.posttrain.taskbank import load_tasks, verify
from lithos.posttrain.verifier_prefs import build_verifier_prefs
from lithos.train.checkpoint import load_model_from_checkpoint
from tokenizers import Tokenizer

DEFAULT_TOKENIZER = "artifacts/tokenizer/fineweb-edu-32k/tokenizer.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Verifier-labeled DPO preference generation.")
    ap.add_argument("--sft", required=True, help="SFT checkpoint dir.")
    ap.add_argument("--tasks", required=True, help="Verifiable task bank JSONL (kind=problems).")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--out", default="data/dpo/verifier")
    ap.add_argument("--samples", type=int, default=6, help="completions sampled per task")
    ap.add_argument("--max-pairs", type=int, default=1, help="pairs kept per task")
    ap.add_argument("--limit", type=int, default=None, help="cap number of tasks")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling diversity")
    ap.add_argument("--timeout-s", type=float, default=5.0, help="per-verify sandbox budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--decontam-probes", default=None,
        help="Probe JSONL (decontam.write_probes) — screen out eval-battery leaks (F2, recommended).",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(args.tokenizer)
    end_id = special_ids(tok)["<|end|>"]
    model = load_model_from_checkpoint(args.sft, device)  # arch read from the checkpoint
    g = torch.Generator(device=device).manual_seed(args.seed)

    def sample(task) -> list[str]:
        pids = render_prompt([{"role": "user", "content": task.prompt}], tok)
        prompt = torch.tensor([pids], device=device).repeat(args.samples, 1)
        out = generate(
            model, prompt, args.max_new, temperature=args.temperature, top_p=0.95,
            eos_token_id=end_id, generator=g,
        )
        texts = []
        for row in out.tolist():
            resp = row[len(pids):]
            if end_id in resp:
                resp = resp[: resp.index(end_id)]
            texts.append(tok.decode(resp, skip_special_tokens=True).strip())
        return texts

    def is_correct(text: str, task) -> bool:
        return verify(text, task, timeout_s=args.timeout_s).correct

    tasks = load_tasks(args.tasks)
    random.Random(args.seed).shuffle(tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    prefs, stats = build_verifier_prefs(
        tasks, sample, is_correct, samples_per_task=args.samples, max_pairs_per_task=args.max_pairs
    )
    print(f"generated: {stats}")

    if args.decontam_probes:
        from lithos.posttrain.decontam_gate import PostTrainDecontaminator, prefs_text

        gate = PostTrainDecontaminator.from_probe_file(args.decontam_probes)
        prefs = gate.screen(prefs, prefs_text)
        print(f"decontam: {gate.report()}")

    if not prefs:
        raise SystemExit("no preference pairs — tasks may be all-solved/all-failed at this capability")

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

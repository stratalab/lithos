# Post-training pipeline pressure test (E1–E8)

**Goal:** shake out the *real* (non-toy) post-training code paths end-to-end on the
real 100M base (`models/lithos-100m-v0.1`, 57k steps / 30B tokens, fineweb-edu-32k).
Model quality is irrelevant here — the question is only whether the machinery runs.
Run on branch `posttrain-e1e8-pressure-test` (2026-07-05), 4070 SUPER.

## What ran

| Stage | Real code path | Result |
|---|---|---|
| **E2** packed SFT | `build_sft_corpus` → dual-stream shards → `sft_packed` memmap loader → SFT | ✅ loss 2.84→2.45, ckpt |
| **E8** verifier-DPO | on-policy sampling → E1 verifier labeling → pair-build → DPO | ✅ 3 pairs → 20 steps, ckpt |
| **E4** TIR-GRPO | TIR render (E3) → multi-segment rollout → reward/KL → GRPO update | ✅ 2 steps, ckpt |
| E1 sandbox exec | Python-in-sandbox tool execution | ✅ unit tests only (see below) |

To reach E4 at all, the base's fineweb-edu-32k tokenizer (vocab 32000) was extended
into a `stem-32k` tokenizer by adding the 6 TIR tokens (`<think>`, `</think>`,
`<|python|>`, `<|octave|>`, `<|/tool|>`, `<|tool_result|>` → ids 32000–32005), and
GRPO was run on a *fresh random model* at that vocab (`init_from` is optional). This
exercises the rollout/GRPO code without a TIR-trained base.

## Findings

1. **Task bank required `id`.** `taskbank.task_from_record` did `rec["id"]` with no
   default → bare `KeyError` on a hand-written bank. **Fixed:** `id` is optional,
   derived from the prompt when absent; a missing `prompt` now raises a clear error.
2. **DPO/GRPO wrote no `run_manifest.json`.** The pretrain/SFT loop writes one; the
   DPO and GRPO trainers have their own loops and silently omitted it. **Fixed:** a
   shared `logging.write_run_manifest` helper, called by both.
3. **⭐ TIR-GRPO cannot cold-start tool use.** A random / non-TIR-SFT'd model emits
   **zero** well-formed tool calls (`tool_calls_per_rollout = 0`), so the sandbox
   branch never fires and RL reward stays flat at 0. This is not a bug — it is a
   **bring-up ordering requirement**: the model must first be SFT'd on TIR-format
   traces so it *can* emit `<|python|>…<|/tool|>`; only then does TIR-GRPO have tool
   calls to reward and the sandbox to run. The sandbox execution path itself is
   covered by `tests/test_sandbox.py` + `tests/test_tir_rollout.py`.

## Conclusion

The real E1–E8 machinery is sound — every stage runs end-to-end on the real base
with no crashes. The findings are polish (a data-contract default, an observability
gap) plus one structural insight (#3). It did **not** test realistic conditions
(a competent model, real STEM data, meaningful rewards); that is the separate "signal
run", gated on: **stem-32k base → TIR-format SFT → TIR-GRPO.**

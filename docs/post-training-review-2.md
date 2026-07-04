# Post-Training Pipeline Review 2 — The Seams (2026-07-04)

Second end-to-end review, after the review-1 buildout landed (D1, F1/F2, E1–E4,
E7, E8 — all adversarially reviewed; suite 243 → 349). Review 1
(`docs/post-training-review.md`) asked *is the machinery flagship-grade?* — it
wasn't, and the gaps became the implementation plan. This review asks the next
question: *does the chain hold end to end, and what can it still not do?* The
gaps found here are **integration seams and missing feedstock**, not machinery —
and they are actioned as **Wave 4** in `docs/post-training-implementation-plan.md`.

**Verdict: the machinery is done; the pipeline is starving.** Every stage of
SFT → RLVR-TIR → DPO exists, is reviewed, and is proven end-to-end on toy
models. The code is now ahead of the data on every axis: the pipeline is a
working factory with no feedstock. Critically, **most of the feedstock work is
CPU-local, not GPU-blocked** — the bottleneck has moved from engineering to
data, and the true critical path (mix decision → tokenizer freeze → 500M) runs
through corpus acquisition/extraction, which is VM/CPU work too.

## A. The end-to-end walk (what's green)

harvested JSONL → **F2** decontam → **E2** mixer + packed dual-stream shards →
SFT via unchanged `train()` (`kind: sft_packed`) → **E4** GRPO-TIR (the **E1**
sandbox executes tool calls, the verifier rewards, the action mask excludes
injected results from PG + KL) → **E8** verifier-labeled DPO → export — plus
**E7**'s bit-exact Qwen3 import making the 4B hero a first-class citizen of the
same stack. Every arrow has a passing integration test. The Goodhart defenses
(reward/accuracy split-logging, gaming pre-screen, tight-KL DPO recipe) are
wired in, not aspirational.

## B. Gaps — almost all are "the pipeline has no food"

### Tier 1 — missing links in the chain (all CPU-local, all unblocked NOW)

1. **No converters for the harvested P0 datasets** (→ epic **E11**). Only
   `prepare_dolly_sft.py` exists. OpenMathReasoning, AceReason, Tülu-3,
   StarCoder2-Instruct: no acquisition, no converter into the messages/segments
   JSONL that E2 eats. The SFT stage would train on Dolly today. Converters need
   no tokenizer and no GPU — including `<think>`-text → segments parsing (D1
   chose the format to make this near-identity).
2. **No real task banks** (→ **E12**). E4/E8 run off one 6-problem toy JSONL.
   The `kind=problems` acquisitions (GSM8K, MBPP-style, FE-style units; `level`/
   `year` columns for curriculum + split) don't exist. Pure data engineering;
   also feeds the eval battery.
3. **No rollout→segments converter — the self-generation engine's missing link**
   (→ **E13**). `tir_rollout` returns tokens/text; the §2.5 engine (generate →
   verify → keep → SFT) needs verified rollouts converted into E3's segments
   format to become training data. Its absence also means there is **no test
   that SFT's loss mask and RL's action mask agree on the same episode** — if
   they diverge, SFT and RLVR push on different token sets (a subtle
   performance-bug class). The converter makes that consistency test possible.
4. **The verifier's third customer was never wired** (→ **E14**). E1's design is
   one verifier serving RLVR + prefs + **eval**; `lithos/evals/` only runs
   lm-eval. No executable-battery runner scores a checkpoint on a task bank via
   `verify_batch` — which is exactly the GSM8K/MBPP+ pass@1 instrument
   `docs/eval-plan.md` battery v2 calls for, and is a thin loop over existing
   machinery.
5. **Engineering verification is still latent** (→ **E15**). pint/CoolProp/
   python-control not installed; `check_units` is magnitude-only (documented
   TODO). "Units as a verifier dimension" — engineering's differentiation — has
   never executed once.

### Tier 2 — smaller seams

6. **Teacher-side TIR generation has no path.** `distill_generate.py` is plain
   text; a teacher can't drive tools in our sandbox. **E7 partially unlocks
   this**: a Qwen3-family teacher can be `load_qwen3`'d into Lithos and driven
   by `tir_rollout` directly — unplanned synthesis, makes eng-TIR trace
   generation with small/mid Qwen teachers locally possible. (Tracked inside
   E13's scope note.)
7. **No compatibility guards at the manifest/model seam** (→ **F3**).
   `kind: sft_packed` doesn't check the manifest's tokenizer/seq_len against the
   model config (the yaml comment says "must match"; nothing enforces it).
8. **Regression-gate automation** (base battery + bpb after SFT; SFT battery
   after RL) is still manual — a review-1 item that never became an epic (folded
   into E14's scope).
9. **`tir_rollout` re-prefills the whole history on every resume** — the KV
   cache isn't carried across tool segments. E5 territory; named here as its
   first specific optimization.

## C. GPU-blocked ledger (honestly small)

E5 (measure/batch rollout throughput), E6 (memory re-cost), E9 (loss-choice
ablations), E10 (long-context phase), all Part-B experiments (P1–P6). Plus the
two structural gates: the **mix decision** (100M sweep bursts) → **tokenizer
freeze** → retokenize → 500M. Note: the freeze is gated on the *mix*, not just
D1 — so the tokenizer isn't purely waiting on hardware either; it's waiting on
corpus acquisition/extraction (VM/CPU work).

## D. Unknowns — unchanged by design

The Part-B table (trace length, SFT token budget, RL hyperparams at scale,
base-vs-SFT-first, DPO-after-RLVR, curriculum pacing) stands as written; nothing
built since changes those questions, and they were always experiment-gated.

## E. Recommended pre-GPU order

(1) **E11** — acquire + convert the P0 SFT sets; build the first *real*
multi-source packed corpus (exercises E2's mixer/decontam on real data for the
first time). (2) **E12** — first real task banks. (3) **E14** — the executable
eval runner (closes the three-customer loop; the battery-v2 instrument).
(4) **E13** — rollout→segments + the SFT/RL mask-consistency test.
(5) **E15** — make engineering verification dimensionally real.

One line: **end to end, the pipeline is structurally complete and internally
consistent — but it has never touched real data, and feeding it doesn't need
the GPU.**

## Pointers

- Review 1: `docs/post-training-review.md` (machinery gaps → Waves 0–3, all
  landed). This review's items → **Wave 4** in
  `docs/post-training-implementation-plan.md`.
- Doctrine: `docs/data-construction.md` Part 2 (the P0 inventory E11 acquires;
  §2.5 self-generation engine E13 completes). Gates: `docs/eval-plan.md`
  (battery v2 = E14's instrument).

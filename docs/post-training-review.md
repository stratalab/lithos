# Post-Training Pipeline Review — Gaps & Unknowns (2026-07-03)

Code-level review of the full post-training stack — `lithos/posttrain/` (~800
lines), the driver scripts (`train_sft/dpo/grpo.py`, `prepare_dpo_prefs.py`,
`distill_generate.py`), `lithos/model/generation.py`, and the SFT/DPO/GRPO
configs — against the flagship recipe (SFT → RLVR-TIR → DPO, per
`docs/data-construction.md` Part 2). Companion to `docs/eval-plan.md` (gates)
and `docs/tokenizer.md` (the freeze this review discovered a dependency on).

**Verdict: a validated test bench, not yet a flagship recipe.** The math is
correct, the masking is exact, and the Goodhart instincts (reward vs. accuracy
split-logging, on-policy DPO, tight KL) are baked in — the 100M shakedown did
its job. But it was built to prove *mechanics* at 110M on arithmetic and Dolly.
Most gaps below are known-deferred; the Tier-1 items are **not on any deferral
list** — and one (§2.5) is upstream of *pretraining*, not just post-training.

## 1. What's solid (no action)

- Chat template inserts special tokens **by ID**, never by string-parsing
  rendered text, so loss masks are exact regardless of how content tokenizes
  (`posttrain/chat_template.py`).
- Overlong SFT examples are **dropped, never right-truncated** — truncation
  would train half-replies (`posttrain/sft_dataset.py:45`).
- SFT and DPO share `build_xy`, so chosen/rejected masking cannot diverge from
  SFT masking (`posttrain/preference_dataset.py`).
- Fused cross-entropy for log-probs — never materializes the (B,T,V)
  log-softmax; fp32 under autocast (`posttrain/dpo.py:23`).
- GRPO KL uses the k3 estimator (non-negative, low-variance)
  (`posttrain/grpo_trainer.py:145`).
- Shaped `reward` and true `accuracy` logged separately, with the explicit
  rule that divergence = the shaping being farmed (`posttrain/verifier.py`).
- Scaffolding reuse: run dirs, resolved configs, JSONL metrics, W&B mirror,
  self-describing checkpoints — post-training runs are as reproducible as
  pretraining runs.

## 2. Tier 1 — capability gaps (block the thesis, not just the schedule)

### 2.1 TIR is unrepresentable — the chat template has no tool turn

`ROLE_TOKEN` = system/user/assistant, full stop (`posttrain/chat_template.py:26`).
The defining capability needs: a tool-call delimiter, a tool-result role, and —
the detail that decides quality — **tool outputs masked out of the loss** (train
the model to *make* the call, never to *predict the sandbox's answer*; training
on tool outputs teaches hallucinating results instead of calling). Cascades:

- GRPO rollouts become multi-segment: generate → detect tool call → pause →
  execute in sandbox → append result → resume. Today a rollout is one
  `generate()` call (`posttrain/grpo_trainer.py:99`).
- PG loss **and** KL must exclude tool-result tokens.
- SFT dataset format needs multi-step episodes (call → result → continue).

Largest single work item in the stack.

### 2.2 The reasoning-format decision is unmade, and it's load-bearing

Do Lithos models think in `<think>…</think>` spans? Special-token delimited or
plain text? Included in SFT targets at what length? `distill_generate.py
--strip-think` is a toy-scale dodge, not a decision. Gates: bulk conversion of
the harvested trace datasets (millions of rows rendered *into* the chosen
format), eval answer-extraction, and the edge latency budget (thinking tokens
are user-visible cost on-device).

### 2.3 Sequence length will silently discard the best SFT data

Pretraining context is 2048 and the SFT loader **drops** what doesn't fit.
OpenMathReasoning/AceReason traces routinely run 4k–16k tokens — at seq_len
2048 the `dropped` counter quietly eats the majority of the highest-value data
(the `stats()` dict is the waste meter; watch it). Positions available:

1. **Long-context extension phase in pretraining** (RoPE-theta scaling +
   long-doc anneal) — standard, but must be planned into the 500M run.
2. **Filter to short traces** — loses the best data.
3. **Generate own short-trace data** — on the (plausible, testable) thesis
   that a 500M can't absorb 16k-token reasoning anyway (§5).

Probably some of all three — but it's a decision with a **pretraining
dependency**, not an SFT config knob.

### 2.4 RLVR has no task-bank interface and no real verifier

`train_grpo` hardcodes `gen_arithmetic` + `MathVerifier`
(`posttrain/grpo_trainer.py:68,89`). Known-deferred (sandbox = Phase 12); the
full missing list, so nothing gets rediscovered later:

- problem-bank loading (`kind=problems` acquisitions → task pool);
- per-task-type verifier dispatch behind the existing `Verifier` protocol;
- sandboxed execution: isolation, timeouts, resource caps;
- **async/parallel verification** — unit tests take seconds; run serially per
  rollout, the GPU sits dead;
- pass-rate curriculum machinery for the exam difficulty ladder (doctrine
  only — GRPO's signal lives at pass-rate ∈ (0,1));
- the anti-gaming judge on tool calls (doctrine only, lifted from GLM-5.2);
- year-split enforcement between RLVR pool and eval sets (doctrine only).

The verifier remains the three-customer artifact (eval + RLVR reward +
preference labeling) and the highest-leverage pre-hardware build.

### 2.5 SEQUENCING CONSTRAINT: tool/think token design precedes the tokenizer freeze

`docs/tokenizer.md` §3.3 reserves special-token IDs 7–15, but sizing that block
correctly requires deciding the TIR format **first**: how many tool tokens,
whether think delimiters are special tokens, FIM tokens for code. Tokenizer
v1.0 → corpus retokenization → 500M pretraining all sit downstream. **The TIR
format design is therefore upstream of pretraining.** Most important finding in
this review.

## 3. Tier 2 — throughput/scale gaps (cap experiment count, which *is* performance)

### 3.1 GRPO rollouts are ~10–100× off flagship speed

Three compounding issues (`posttrain/grpo_trainer.py:96-102`):

1. Prompts processed sequentially — only G-way batching within one prompt.
2. The shared prompt is re-prefilled G times (`.repeat(G, 1)`) — no prefix
   sharing.
3. `generate()` runs **outside autocast on fp32 weights** — the model is
   `.to(device)` with no dtype cast; only the loss forwards are autocast.

Fix tiers: **(a)** batch P×G together + bf16 weights + prefix-share — order of
10×, doable in-repo; **(b)** vLLM/SGLang rollout sidecar with weight sync for
500M+ — already on the deferral list; budget it as weeks, not days. At
`grpo_max_new` 512–2048 (real math, vs 16 today) rollout time dominates
everything.

### 3.2 The SFT data path won't hold flagship data

`SFTDataset` materializes dense padded int64 arrays in RAM: 1M examples × 4096
seq × 8B × 2 ≈ **65GB before training starts** (`posttrain/sft_dataset.py:85`).
And one-conversation-per-row padding means most FLOPs go to pad tokens at scale
(`loss_token_fraction` is the meter). Needs streaming/memmapped SFT shards +
**sequence packing** (or length-bucketing) — standard practice, a 2–5× swing on
the SFT compute bill.

### 3.3 "Fits one GPU through ~3B" needs arithmetic before it's believed

DPO/GRPO hold policy + frozen reference + Adam moments: at 3B fp32 that is
~12 + 12 + 24 GB **before activations** — not a 12GB card, marginal on 80GB.
Known fixes, none wired: bf16 weights (+ master-weight strategy), 8-bit
optimizer, gradient accumulation in GRPO (currently absent — the whole P×G
batch goes through one backward, `posttrain/grpo_trainer.py:148-151`), grad
checkpointing (off in the SFT configs), eventually LoRA (deferred). Re-cost per
rung before the flagship post-training schedule is committed. The 4B hero
(untied, 151k vocab) is bigger still.

### 3.4 The 4B hero path has unproven tooling compatibility

The whole stack drives `LithosForCausalLM` with the 7-token Lithos template and
32k vocab; the hero is Qwen3-lineage with Qwen chat tokens and a 151k vocab.
Either Qwen3 weights import into the Lithos architecture (the export envelope
was designed for the *export* direction — import is untested: untied
embeddings, 151k vocab, template remap in `special_ids()`), or the hero runs on
different tooling and "one deployment recipe" quietly becomes two. **Cheap
spike:** load Qwen3-0.6B into `LithosForCausalLM`, verify logit parity against
transformers.

## 4. Tier 3 — landmines & hygiene (cheap now, expensive later)

1. **`generate()` flips the policy to eval mode permanently mid-step.**
   `generation.py:64` calls `model.eval()`; `train_grpo` sets `policy.train()`
   once before the loop (`grpo_trainer.py:85`), so every loss forward after
   step 0 runs in eval mode. Benign today (dropout=0, no batchnorm); silent
   behavior change the day dropout is nonzero. One-line fix.
2. **GRPO loss choices are toy-validated, not settled.** Sequence log-prob is
   *summed*, not length-normalized (`grpo_trainer.py:143`) — longer responses
   get proportionally larger gradients, an unchosen length pressure (the
   Dr. GRPO critique axis; DeepSeekMath normalizes per-token). Std-normalized
   advantages carry a known easy/hard-question bias. No importance-ratio
   clipping means rollouts can't be reused for multiple updates — fine on-policy,
   but once rollouts are expensive (§3.1) PPO-style clipping amortizes them.
   Decide each deliberately at flagship; write the choice down.
3. **No decontamination in the post-training data path.** The corpus pipeline
   has `DecontaminationFilter`; the SFT/pref/distill converters write JSONL
   with no screen. SFT data is the *most* contamination-dense input we handle
   (OpenMathReasoning ← AoPS ← MATH/AIME). Wire the 13-gram probe into every
   converter as a standard stage — an afternoon; protects every parity claim.
4. **No SFT mixture control.** One `train.jsonl` per run. Flagship SFT is a
   deliberate blend (general backbone + math traces + code + physics) with
   per-source caps — the LIMA lesson in reverse: don't let 2.7M AceReason rows
   drown 1k excellent general examples. Needs a weighted mixer with a recorded
   manifest + per-source loss logging (which slice is starving?).
5. **Safety posture is entirely absent.** Possibly correct for an edge STEM
   tool — but make it a written decision, not an omission. Minimal version: a
   few hundred refusal examples in the SFT blend + one red-team pass on the
   shipped GGUF.
6. **On-policy DPO judge is toy-bound** (token-F1 to reference,
   `prepare_dpo_prefs.py:63` — right for Dolly). Flagship pairs should be
   verifier-labeled correct/incorrect rollouts from the same prompt — the
   machinery is ~identical to GRPO's rollout+verify loop; build once.

## 5. Unknowns (genuinely empirical — plan experiments, don't debate)

| Unknown | Why it matters | Cheap probe |
|---|---|---|
| Optimal trace length for 500M–3B | long-CoT SFT may exceed small-model capacity; literature favors shorter traces for smaller students | 100M/500M A/B: full traces vs truncated-reasoning versions of the same data |
| SFT token budget before regression | when does bulk STEM SFT collapse the base battery? | regression-gate curve vs SFT tokens on the rig |
| RL hyperparams at scale | KL 0.05 / lr 1e-6 / G=8 validated at 110M with 16-token rollouts; RL transfers poorly across scale | re-sweep at 500M on the easiest exam-ladder rung before long runs |
| Base-vs-SFT-first RLVR | R1-Zero worked at 671B; almost certainly needs SFT-first at 500M — but the ladder's lowest rungs might not | one GRPO-from-base run at 500M |
| Does DPO add anything after RLVR | already demoted in doctrine; confirm before spending pair-generation compute | battery delta on one flagship candidate |
| Curriculum pacing | when to advance the difficulty ladder | pass-rate windows on the 100M/500M |

## 6. Order of attack (pre-hardware, dependency-sorted)

> Operationalized into sequenced epics + an experiment plan in
> `docs/post-training-implementation-plan.md`. The list below is the summary;
> that doc carries the deps, done-criteria, and the Part-B unknowns schedule.


1. **Decide the TIR + reasoning format** (tool tokens, think delimiters,
   masking rules) — gates the tokenizer freeze (§2.5), which gates everything.
2. **Build the verifier + sandbox** (task-bank interface, async execution,
   isolation) — three customers, longest pole.
3. **Wire decontam into all post-training converters** — an afternoon.
4. Fix the eval-mode landmine; add GRPO grad accumulation — small.
5. **Rewrite the SFT data path** (streamed shards + packing + weighted mixer)
   against the harvested P0 datasets.
6. Extend template/datasets/GRPO for tool turns; dry-run the full
   SFT → GRPO-with-sandbox loop on the 100M.
7. Spike the Qwen3-import question for the hero (§3.4).
8. Re-cost post-training memory/throughput per rung; decide where vLLM enters.

## Pointers

- Code reviewed: `lithos/posttrain/{chat_template,sft_dataset,preference_dataset,dpo,dpo_trainer,grpo_trainer,verifier}.py`,
  `lithos/model/generation.py`, `scripts/{train_sft,train_dpo,train_grpo,prepare_dpo_prefs,distill_generate}.py`,
  `configs/{sft,dpo,grpo}/*.yaml`.
- Doctrine: `docs/data-construction.md` Part 2 (stages, inventory, sovereignty
  filter, self-generation engine); banked Phase-11 lessons in
  `lithos-implementation-plan.md`.
- Gates: `docs/eval-plan.md` (regression gates, parity frontier, reward-hacking
  audit). Tokenizer dependency: `docs/tokenizer.md` §3.3 (reserved block).

# Post-Training Implementation Plan — Closing the Review Gaps (2026-07-03)

Turns `docs/post-training-review.md` into sequenced, dependency-aware work: the
buildout that takes Phase 12's post-training from *validated test bench* to
*flagship recipe*. Part A fixes the gaps; Part B tests the unknowns. Companion
to `lithos-implementation-plan.md` (Phase 12), `docs/eval-plan.md` (gates),
`docs/tokenizer.md` (the freeze the spine runs through).

Each item tags its **review §**, **deps**, **size** (S ≤1 day · M ≤1 week · L
>1 week), and **HW** (`local` = 4070/CPU · `rented` = needs a GPU burst). Epic
IDs are stable handles (D#/F#/E#/X#) — reference them directly.

## 0. The shape of the work

**One spine, everything else parallel.** The only hard ordering constraint is
§2.5: the TIR/reasoning **format decision (D1)** sizes the tokenizer's reserved
special-token block, and the frozen tokenizer gates the 500M pretrain, which
gates the *keeper* post-training run. So:

```
D1 (format) ─► tokenizer v1.0 freeze ─► retokenize ─► 500M pretrain + E10 (long-ctx) ─► KEEPER post-train
   │                (docs/tokenizer.md)                                                      ▲
   ├─► E3 (TIR template) ─► E4 (TIR rollout) ─► E8 (verifier-DPO) ───────────────────────────┤
   └─► E1 (verifier/sandbox) ──────────────────┘                                            │
        E2 (SFT data path) ───────────────────────────────────────────────────────────────┘
```

Everything except D1 and the keeper run can be built and unit-tested **now, on
the 4070, against the current 100M + fineweb-edu tokenizer** — the pipeline is
scale- and tokenizer-invariant by design. So the build is not blocked on
hardware; only the *experiments* (Part B) and the keeper runs are. That is the
whole point of doing this during the wait.

**Three kinds of item, different "done":** **decisions** (D#, X1 — a written
choice, some needing the user), **fixes/builds** (F#, E# — code + tests), and
**experiments** (Part B — a measured answer). Don't let them blur; a build
whose decision isn't made yet is a build waiting to be redone.

## Part A — Implementation

### Wave 0 — Decisions & cheap landmines (start immediately, no build deps)

**D1 · TIR + reasoning format** · §2.1/2.2/2.5 · deps none · L · local · ✅ **DONE (2026-07-03)**
The spine's head. `docs/tir-format.md` decides the wire format concretely enough
to render data and freeze the tokenizer against: `<think>…</think>` reasoning
(loss target), `<|python|>`/`<|octave|>`/`<|/tool|>` tool calls (raw-source
payload), `<|tool_result|>…<|end|>` results (masked), token IDs 7–12 assigned in
the reserved block. The two user decisions are ratified: **(A)** no-think toggle
**deferred** (always-on thinking for the MVP); **(B)** the 4k–16k trace-length /
2048-context collision resolved by a **long-context pretrain phase** (epic E10),
keeping full traces rather than capping. Feeds `docs/tokenizer.md` §3.3.

**F1 · Eval-mode landmine** · §4.1 · deps none · S · local · ✅ **DONE (2026-07-03)**
`generate()` now saves/restores the model's train/eval mode via try/finally
(`lithos/model/generation.py`), so a mid-training rollout can't leave the policy
in eval mode for the loss forward — fixes all callers, not just GRPO. Regression
test in `tests/test_generation.py::test_generate_restores_training_mode`.

**F2 · Decontam in the post-training data path** · §4.3 · deps none · S · local · ✅ **DONE (2026-07-03)**
`lithos/posttrain/decontam_gate.py` (`PostTrainDecontaminator`) wraps the corpus
13-gram `DecontaminationFilter`; `screen()` drops records overlapping the eval
battery with a drop-rate report. Wired into `prepare_dpo_prefs.py` and
`distill_generate.py` via `--decontam-probes` (built from `decontam.write_probes`,
no network). Tests in `tests/test_decontam_gate.py`. The E2 SFT ingest will reuse
it when built.

**X1 · Safety posture** · §4.5 · deps none · S · local (decision)
A written decision, not an omission. Recommend the minimal viable posture for an
edge STEM tool: a few hundred refusal examples folded into the E2 SFT blend +
one red-team pass on the shipped GGUF as a keeper-gate line in `eval-plan.md`.
User ratifies scope.

**E7 · Qwen3-import spike** · §3.4 · deps none · S · local
Cheap, independent, de-risks the whole hero track. Load Qwen3-0.6B weights into
`LithosForCausalLM`; verify logit parity vs transformers on a fixed prompt set
(untied embeddings, 151k vocab, RoPE-theta, norm placement). **Done:** parity
within fp tolerance → the family shares one tooling path; or a written list of
what diverges → "one deployment recipe" is re-scoped before it's promised.

### Wave 1 — The two long poles (parallel; E1 informed by D1's task schema)

**E1 · Verifier + sandbox** · §2.4 · deps D1 (task/tool schema) · L · local · ◑ **core landed (2026-07-03)**
The three-customer artifact (eval reward · RLVR reward · preference labels) and
the longest pole. Sub-epics:
- **E1a — sandboxed executor.** ✅ `lithos/posttrain/sandbox.py`: `run_python`/
  `run_octave`/`run_tool`, subprocess isolation, wall-clock timeout + process-group
  kill, POSIX CPU/address-space rlimits, output truncation. **Deferred:** network/FS
  isolation (needs container/nsjail — threat model documented in the module);
  CoolProp/python-control/pint are install-time deps (not yet in the env).
- **E1b — verifier dispatch.** ✅ `posttrain/verifier.py`: `CheckResult` +
  `check_numeric` (tolerance), `check_symbolic` (SymPy equivalence), `check_code`
  (sandbox unit-tests), `check_units` (`pint`, import-guarded), `shaped_reward`,
  `heuristic_gaming_check`. `MathVerifier`/`gen_arithmetic` kept for the arithmetic
  test bench.
- **E1c — task-bank interface.** ✅ `posttrain/taskbank.py`: `Task` (+ validation),
  `load_tasks`, `verify` dispatch by kind, `filter_by_level`.
- **E1d — async/parallel verification.** ✅ `verify_batch` (ThreadPoolExecutor;
  order-preserving; overlaps subprocess waits). Full weight-synced sidecar is E5.
- **E1e — anti-gaming judge.** ◑ heuristic pre-screen `heuristic_gaming_check`
  landed (hard-coded-answer / no-op detection); **the LLM judge is deferred** (needs
  a judge model + the dummy-reward hook in E4).
- **E1f — year-split enforcement.** ✅ `split_by_year` + `assert_disjoint`.
- **Remaining for "done":** real problem-bank acquisitions (`kind=problems` JSONL),
  install the engineering packages, and the E4 rollout wiring so eval and reward
  literally call this. Tests: `test_sandbox.py`, `test_verifier_checks.py`,
  `test_taskbank.py` (38 new tests, suite green at 281).

**E2 · SFT data path rewrite** · §3.2/4.4 · deps F2 · L · local · ✅ **DONE (2026-07-03)**
Offline SFT-corpus build (`lithos/posttrain/sft_corpus.py`: `SFTShardWriter`
dual-stream tokens+mask, `build_sft_corpus`, `SFTCorpusBuildConfig`) mirrors
`build_corpus`; loader `PackedSFTDataset` (`sft_dataset.py`) memmaps both streams
and duck-types `PackedDataset`, so `train()` is unchanged via new `kind: sft_packed`.
- **Streamed/memmapped shards** ✅ — dual `.bin` streams (uint16/32 tokens + uint8
  loss mask), 3 B/tok, no RAM materialization.
- **Sequence packing** ✅ — **decision recorded: cross-doc bleed** (matches
  pretraining `packing.py`; block-diagonal masking is the deferred PRD §27 shared
  upgrade — needs top-level forward + dataloader-contract changes). Zero model
  changes.
- **Weighted mixer** ✅ — per-source `max_examples` cap + `repeats` upsample
  (LIMA-in-reverse), realized mixture recorded in the manifest.
- **Per-source logging** ◑ — build-time per-source accounting in the manifest
  (`mixture[src] = {examples, kept_unique, tokens, loss_tokens, decontam_dropped,
  dropped_overlong, ...}`). Per-source *training*-loss logging (source-tagged
  windows) remains the fast-follow.
- **Verified:** the smoke build on Dolly gives **loss_token_fraction 0.47 vs the
  dense path's 0.078 (~6×)** — the core FLOPs-per-loss-token win; `train()` overfits
  a packed stream via `kind: sft_packed`. F2 decontam wired into the build.
  `scripts/build_sft_corpus.py` + `configs/sft/{mix-smoke,lithos-100m-packed}.yaml`.
  17 new tests; suite green at 298.

### Wave 2 — TIR plumbing + throughput (after Wave 1)

**E3 · TIR template + dataset episodes** · §2.1 · deps D1, E2 · M · local · ✅ **DONE (2026-07-03)**
`chat_template.py` renders TIR episodes: an assistant turn is an ordered list of
typed `segments` (`think`/`text`/`tool`/`tool_result`) — the inline single-turn
form D1's grammar mandates. `_encode_segments` applies the D1 §4 mask by ID
(think/tool/text learned; the `tool_result` span incl. its closing `<|end|>`
masked), shared by `render_conversation` + `render_prompt`. TIR tokens
(`tir_token_ids`) resolve **lazily** — only when a segments turn is rendered — so
non-TIR SFT still works on today's tokenizer; flat `content` is unchanged
(byte-identical, existing tests pass). `build_xy`/`SFTDataset`/`PackedSFTDataset`/
`build_sft_corpus` inherit TIR for free (no change). `decontam_gate.messages_text`
now pulls segment text so F2 screens TIR traces. **Scope:** rendering only (the
generate→execute→inject loop is E4); building *real* TIR corpora needs the STEM
tokenizer (TIR tokens must live in-vocab). Adversarial review hardened
malformed-input handling (clear errors for missing/non-string segment fields,
non-dict/non-list segments, content⊕segments; decontam made crash-proof on messy
records). 20 new tests; suite green at 321.

**E4 · GRPO multi-segment rollout** · §2.1 · deps E1, E3 · L · local+rented · ✅ **DONE (2026-07-04)**
`lithos/posttrain/tir_rollout.py`: `tir_rollout` loops generate (stop at
`<|/tool|>`/`<|end|>` — new `stop_token_ids` in `generation.py`) → `parse_tool_call`
by ID → `run_tool` (E1 sandbox) → inject `<|tool_result|>…<|end|>` → resume, with
a per-token **action mask** (False on prompt + injected results). `grpo_trainer.py`
gains a TIR mode (`cfg.grpo_tir`): G sequential rollouts/prompt, reward =
`shaped_reward(verify(...))` with the E1 verifier + task bank, zeroed by
`heuristic_gaming_check` (E1e hook). Labels built from the action mask via the
shared `_labels_from_action_mask` + extracted `_grpo_loss`, so **tool-result tokens
drop out of PG and KL for free** (the existing `IGNORE_INDEX` machinery). Arithmetic
path preserved (locked by a step test). **Verified:** a scripted-model rollout
executes a real `print(2+2)` in the sandbox and masks the injected "4"; the
exclusion invariant proven numerically (0 PG + 0 KL contribution at tool-result
positions); arith + TIR GRPO steps run end-to-end on toy models. `configs/grpo/
lithos-tir-toy.yaml` + task bank. **Scope:** sequential rollouts (E5 batches);
real 100M TIR run waits on the STEM tokenizer (TIR tokens in-vocab). 11 new tests;
suite green at 332. Review hardening: empty-task-bank guard (was ZeroDivisionError);
`generate` docstring warns that batched `stop_token_ids` without eos needs
per-row trimming (E5); TIR reward uses `completion_text` (right for answer-checked
tasks — code-kind TIR verifying the executed tool code is a documented TODO).

**E5 · Rollout throughput** · §3.1 · deps E4 · M · rented (to measure)
Tier (a), in-repo: batch P×G together, cast rollout weights to bf16 (today
`generate` runs fp32 outside autocast), share the prompt prefill across the group.
Target ~10× at realistic `grpo_max_new` (512–2048). **Then decide** tier (b): a
vLLM/SGLang rollout sidecar with weight-sync — budget weeks, only if (a) leaves
rollouts dominating wall-clock at 500M+. Record the decision either way.

### Wave 3 — Scale readiness (gate before any 500M+ run)

**E6 · Memory re-cost + wire the fixes** · §3.3 · deps E4 · M · local+rented
Actually cost policy+frozen-reference+Adam per rung (3B fp32 ≈ 48GB before
activations). Wire: bf16 weights + fp32 master, 8-bit optimizer, **GRPO gradient
accumulation** (today the whole P×G batch is one backward, `grpo_trainer.py:148`),
grad checkpointing on in the SFT/RL configs. **Done:** a per-rung memory table in
this doc with headroom on the intended card; LoRA flagged as the fallback if a
rung still doesn't fit.

**E8 · Verifier-labeled DPO** · §4.6 · deps E4 · M · local
Replace the token-F1 judge (`prepare_dpo_prefs.py:63`, right for Dolly) with
E1-verifier-labeled correct/incorrect rollouts from the same prompt — machinery
is ~identical to E4's rollout+verify loop. **Done:** an on-policy verifier-labeled
pref set generated + a DPO run consuming it.

**E9 · GRPO loss decisions** · §4.2 · deps E4, Part B RL sweep · S (code) · rented (validate)
Decide and document, each validated by the Part B RL experiment: per-token vs
summed sequence log-prob (length pressure — Dr. GRPO axis; `grpo_trainer.py:143`
sums today), advantage normalization (std-normalized easy/hard bias), importance-
ratio clipping (lets expensive rollouts feed multiple updates once E5 makes them
costly). **Done:** each choice recorded with its ablation delta.

**E10 · Long-context pretrain extension** · §2.3 · deps D1 · M · rented · *(pretraining-side; from D1(B))*
The 500M gains a context-extension phase (RoPE-theta scaling + long-doc anneal)
so it natively handles the 4k–16k harvested reasoning traces instead of dropping
them — the D1(B) resolution. Sits on the **spine**: it's part of the 500M
pretrain, before keeper post-training. First step is measurement: profile the
trace-length distribution across the harvested P0 datasets to set the target
context (8k vs 16k — headroom vs cost), which also sets E2's SFT `seq_len` and
feeds Part-B P1. **Done:** a 500M (or 100M proxy) trains stably at the extended
context with intact short-context bpb; target length recorded. Belongs to the
pretraining plan (`lithos-implementation-plan.md` Phase 12 Track S) — tracked
here because it fell out of this review chain.

### Coverage check (every review item lands somewhere)

§2.1→E3+E4 · §2.2→D1 · §2.3→E10(long-context extension)+E2(length control)+Part B(trace-length) ·
§2.4→E1 · §2.5→D1 · §3.1→E5 · §3.2→E2 · §3.3→E6 · §3.4→E7 · §4.1→F1 · §4.2→E9 ·
§4.3→F2 · §4.4→E2 · §4.5→X1 · §4.6→E8.

## Part B — Experiment plan (testing the unknowns)

Principles inherited from `eval-plan.md`: **bpb decides below 500M** (benchmarks
are ~chance); **regression-gate every stage** (don't collapse the prior stage);
**watch real rollouts, not the reward curve**. Each experiment names the
**decision it gates**, its **rig**, the **prerequisite build**, and a **cost
tier** (100M run ≈ $10–15 rented; 500M runs materially more).

| # | Unknown → decision | Rig · IV → metric | Prereq | When |
|---|---|---|---|---|
| **P1** | Optimal trace length for a small student → how to render the harvested traces (full vs truncated-reasoning) | 100M then 500M · trace-length variants of one dataset → executable-battery acc + base-battery regression | E2 (length control), E3 | after Wave 2 |
| **P2** | SFT token budget before base collapse → when to stop SFT | 100M · SFT tokens → base battery + bpb curve | E2 | early (cheap, no TIR needed) |
| **P3** | RL hyperparams at scale → KL/lr/G for the 500M run | 500M · KL·lr·G sweep on the easiest exam-ladder rung → pass-rate + reward/accuracy divergence | E1, E4, E5 | before any long 500M RL |
| **P4** | Base-vs-SFT-first RLVR → does the 500M need SFT before RL | 500M · GRPO-from-base vs from-SFT → capability + stability | E1, E4 | with P3 |
| **P5** | Does DPO add anything after RLVR → spend pair-gen compute or not | 500M · +E8-DPO on an RLVR'd candidate → battery delta | E8 | late (post-RLVR) |
| **P6** | Curriculum pacing → when to advance the difficulty ladder | 100M/500M · pass-rate windows → advancement rule | E1c, E4 | with P3 |

**Sequencing against the build.** P2 is the cheapest and needs no TIR — run it on
the 100M as soon as E2 lands, as early de-risking (it also calibrates the
regression gate itself). P1 follows Wave 2. P3/P4/P6 are the RL-tuning cluster:
they need the real verifier + TIR rollout (E1/E4) and the 500M base, so they gate
the *keeper* RL run — budget a dedicated rented burst for them, and re-sweep
rather than trusting the 110M/16-token GRPO settings (RL transfers poorly across
scale). P5 is last, and may return "skip DPO" — doctrine already demoted it, so a
null result just confirms the saved compute.

**Standing rule:** every experiment records its config + real-transcript
spot-check into the run dir, same as pretraining ablations — the answer is the
deliverable, and an un-inspected reward curve is not an answer.

## Sequencing summary

1. **Now (pre-compute, local):** D1 (get the two user decisions), F1, F2, X1, E7
   — plus start E1 and E2, the long poles.
2. **Freeze gate:** D1 ratified → tokenizer v1.0 (`docs/tokenizer.md` process) →
   retokenize. Runs in parallel with E1/E2 finishing.
3. **Wave 2:** E3 → E4 → E5, dry-run the full SFT → GRPO-with-sandbox loop on the
   100M. Run P2 (and P1 once ready) here.
4. **Scale gate:** E6 memory re-cost + E9 decisions before committing 500M RL;
   run the P3/P4/P6 cluster on a rented 500M burst.
5. **Keeper:** 500M pretrain **+ E10 long-context phase** (frozen tokenizer) →
   SFT (E2 mixer) → RLVR-TIR (E4) → optional DPO (E8, gated by P5) →
   `eval-plan.md` gates + parity matrix.

**Note — E10 is a new pretraining workstream** (from D1(B)): the 500M pretrain
now carries a context-extension phase. It doesn't block the post-training builds
(E1–E9 proceed against the current 100M), but it does gate the keeper run and
adds a measurement step (trace-length profiling) that should happen early since
it also sets E2's `seq_len`.

## Pointers

- Source review: `docs/post-training-review.md` (the gaps this closes).
- Phase context: `lithos-implementation-plan.md` Phase 12 (the family + TIR
  recipe this operationalizes), Phase 11 (the test-bench lessons).
- Gates & experiments frame: `docs/eval-plan.md`. Format→freeze dependency:
  `docs/tokenizer.md` §3.3.

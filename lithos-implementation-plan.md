# Lithos Implementation Plan

Companion to `lithos-prd.md`. This turns the PRD (esp. §19 milestones, §21 build order, §26 resolved decisions) into an ordered, buildable sequence. Each phase is a vertical slice that runs on the local dev GPU (RTX 4070 Super, 12GB) — CPU-capable for the fast unit tests — before anything scales to cloud.

> **Reconciled 2026-06-14** to reflect actual execution (FineWeb-Edu corpus, DDP-not-FSDP, W&B, R2, remote-provisioning scripts) and the **data-centric pivot**: owning a high-quality dataset is a co-equal foundation to the model, and we build a *scale-invariant* pipeline at small scale so the 3B is a config flip. **Part I** (walking skeleton + first real run) is built; the **100M run is live**. **Part II** (the data-centric era) is the forward plan.

## Guiding strategy

1. **Walking skeleton first.** Get the entire §23 command sequence working end-to-end at *toy/smoke* scale (Phases 0–6) before the first real training run. Correctness and reproducibility before scale (PRD §3.8, §20).
2. **Single-GPU/CPU before distributed.** No multi-GPU complexity until the single-process path is green (§20.13–14). Distributed (DDP) is Phase 8, after the 100M model already trains.
3. **Every artifact gets a manifest; every module gets a test.** No silent overwrites, no unprovenanced data (§20.5–9).
4. **Each phase has an acceptance gate.** Do not start phase N+1 until phase N's gate is green.
5. **Scale-invariant pipeline.** Build the whole pipeline (data → pretrain → post-train → eval) at small/cheap scale; the 3B run is a *config flip plus scaling-law math*, not a rewrite. Tune hyperparameters on small proxies (**μP/μTransfer**) and read token budgets off **scaling-law fits**, so 100M → 1B → 3B changes config only.
6. **Data as a co-equal foundation.** Owning a high-quality dataset compounds across every model trained on it; the gains are highest at ≤3B. Ablate data recipes on cheap proxies (the **100M rig**) measured against a **frozen eval harness**. This is the same bet as StrataDB — owning foundational technology pays off in multiples as the stack evolves.

## Locked decisions (reconciled from PRD §26 + this session)

- **Compute:** local **RTX 4070 Super (12GB)** for dev/smoke + cloud for scale — rented **2×H100 (~$8/hr)** for the 100M; **8×B200** for the big runs. A 5090 (32GB) is the candidate local-iteration upgrade (full-FT 1B, LoRA 3B/7B locally); Pro 6000 (96GB) deferred until justified.
- **Corpus:** **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`, sample-10BT, ODC-By, **non-gated**) is the ACTIVE corpus. `nvidia/Nemotron-CC-v2` (6.5T tokens) is **deferred — gated, awaiting NVIDIA approval**. Over-train (~100–200 tok/param). Byte-level BPE 32k.
- **Distributed: DDP, not FSDP.** 80GB H100 / 192GB B200 fit the whole model + optimizer ≤~7B per GPU, so plain data-parallel suffices and is far simpler. FSDP deferred until a single model exceeds one GPU's memory.
- **Storage:** durable artifacts in **Cloudflare R2** (`lithos-data-fineweb-edu`) via a config-driven `Storage` abstraction + `LITHOS_STORAGE_BASE_URI`; HF Hub for published models. `uv` for envs.
- **Tracking:** optional **W&B** (rank-0, lazy-imported, disabled by default) mirroring the canonical local `metrics.jsonl`.
- **Determinism scoped:** bitwise CPU, best-effort GPU.
- **Architecture:** modernized Llama — GQA-native, optional QK-norm, configurable RoPE theta, SDPA backend, KV cache, depth-scaled init; MoE/MLA/sliding-window deferred behind seams (§6.1). Export targets the **Qwen3 envelope**.
- **Model ladder: 100M → 1B (two-track) → 3B.** (The old 300M milestone is folded into the data-ablation proxy role.)

**Inference (§26.8):** build & own the in-repo PyTorch generator; export checkpoints **HF/Qwen3-compatible** as the single hub feeding eval, vLLM (documented cloud serving, not built), and llama.cpp/GGUF (deferred). Local FastAPI `/generate` only — no hosted product (§3.7).

---

# Part I — Walking skeleton & first real run (Phases 0–8)

*Status: built. The 100M run (Phase 6) is live on 2×H100.*

## Phase 0 — Repo & tooling skeleton  ·  ✅ done

Installable package; `pytest` + `ruff check .` green; the config system everything depends on (YAML → pydantic → CLI overrides → resolved-config dump, fail-loud on missing keys). `utils/{config,seed,device,io,checks}`, `train/logging` skeleton, CI (ruff + CPU unit tests on free runners). Acceptance met (PRD §19 M0).

## Phase 1 — Toy transformer (modernized Llama)  ·  ✅ done

Correct decoder-only model on CPU with generation + the full shape/correctness suite: `model/{config,norm,rope,attention,mlp,layers,transformer,generation}`. GQA-native attention (SDPA + eager fallback), optional QK-norm, KV cache, vocab-padding masked from loss, depth-scaled init. Tests: GQA==MHA, SDPA==eager, KV-cache==full-recompute, loss-ignores-padding. Acceptance met (PRD §19 M1).

## Phase 2 — Tokenizer  ·  ✅ done

Versioned byte-level BPE 32k with stable special tokens. `tokenizer/{tokenizer_config,train_tokenizer,inspect_tokenizer}`, `scripts/train_tokenizer.py`, training manifest + sample report. **Reconciled:** trained on a **FineWeb-Edu** sample (300k docs) → `artifacts/tokenizer/fineweb-edu-32k`; decontaminate against eval sets before training (full decontam tooling → Phase 9/10). Roundtrip tests pass; special-token IDs pinned.

## Phase 3 — Data pipeline v0  ·  ✅ done

Raw text → tokenized, packed, resumable shards with provenance. `data/{documents,filters,dedup,tokenize,shard,packing,dataloader,manifest,pipeline}`, `scripts/{tokenize_corpus,prepare_smoke_data}.py`. **Reconciled:** source is **FineWeb-Edu** (`kind: hf` streaming); exact-doc dedup live, **MinHash near-dedup still stubbed → promoted to Phase 10**. Resumable, rank-shardable dataloader (position + RNG checkpointed). Packing/resume/determinism tests pass (PRD §19 M2).

## Phase 4 — Training loop v0 (single process)  ·  ✅ done

Explicit, readable loop (no hidden trainer): forward → CE loss → backward → clip → AdamW (fp32 states) → warmup+cosine → log/eval/checkpoint → exact resume → graceful interrupt. `train/{optim,scheduler,loop,checkpoint,logging,entry}`. Run dir with `resolved_config.yaml` / `metrics.jsonl` / checkpoints; grad-accum, grad-checkpointing flag, torch.compile flag. **Reconciled:** optional **W&B** reporter (rank-0, runtime device/GPU logged); idempotent checkpoint saves (interrupt-safe). Tiny-overfits + resume-reproduces tests pass (PRD §19 M3).

## Phase 5 — Evaluation v0  ·  ✅ done (v0)

Internal **perplexity** + fixed-prompt **samples** + the core **HF/Qwen3 exporter** (brought forward so lm-eval runs via `--model hf`) + versioned report. `evals/{perplexity,generate_samples,report,run,config}`, `scripts/run_evals.py`, `configs/eval/{base,lm-eval}.yaml`. **Reconciled — note the v0/v1 split:** lm-evaluation-harness is currently a *documented manual recipe* (export, then run `lm_eval` by hand); the **wired, frozen, decontaminated, scorecard harness is Phase 9 (Eval harness v1)**.

> **GATE — Walking skeleton green. ✅** The full §23 command sequence runs end-to-end at smoke scale on the local 4070; CI green. Cloud scale-up unlocked.

## Phase 6 — Lithos 100M end-to-end  ·  ⏳ RUNNING

First real run — the integration test of everything above. **Reconciled:** corpus is **FineWeb-Edu** (9.84B tokens, 99 shards, in R2); validated with a 2-GPU DDP smoke, then the full run on a rented **2×H100** at **~332k tok/s**, `grad_checkpointing: false`, **W&B-tracked**, `max_steps=57000` (~15B tokens, ~1.5 epochs). Checkpoints sync to R2 every 10 min (resumable from a fresh box). **Acceptance:** trains stably without loss explosion (loss 9.5→3.0 by step 7.7k); eval report + model-card draft on completion (PRD §19 M4).

## Phase 7 — Inference, export & serving  ·  ◑ partial

Core **HF/Qwen3 export** ships (Phase 5, enables `--model hf` eval and round-trips in `transformers`). Remaining: full `serve/generate.py` CLI, local FastAPI `/generate`, vLLM documented path, GGUF/quantize (deferred). Acceptance (export loads drop-in) met for the core; serving packaging is follow-on.

## Phase 8 — Distributed (DDP)  ·  ✅ done  *(was "FSDP")*

Multi-GPU via **torchrun + DDP** (not FSDP — see locked decisions). `train/distributed.py` (process-group init, rank-aware logging, nccl/gloo), rank-sharded dataloader (lockstep position), `no_sync` grad-accum, all-reduced loss, **rank-0-only writes**, barrier'd checkpoints. 2-process gloo test (CPU) + a real 2×H100 smoke pass. FSDP deferred until a model exceeds single-GPU memory (PRD §19 M6, adapted).

## Cross-cutting (Part I) — Remote provisioning & ops  ·  ✅ done

Not in the original plan; now a real deliverable. `scripts/`: **`build_corpus.sh`** (CPU box → tokenizer + corpus → R2; survives native-lib shutdown crashes), **`setup_server.sh`** (one-shot GPU-box provision: preflight → uv → deps → CUDA check → corpus pull → 2-GPU smoke), **`launch_train.sh`** + **`sync_checkpoints.sh`** (tmux run, auto-resume, checkpoint→R2), **`sync.py`**, shared `lib.sh`. Secrets via a git-ignored `.env`; `docs/remote-training.md` runbook. (Note: bare-image GPU boxes need an NVIDIA-driver install — see runbook.)

---

# Part II — Data-centric era (post-100M)

*The strategic pivot: own the data layer as a co-equal foundation, build a rock-solid scale-invariant pipeline at small scale, and let the **gap** between from-scratch and distillation be the deliverable. All of Part II builds locally + on the 100M rig; only Phases 12–13 need secured cloud compute.*

## Phase 9 — Eval harness v1 (the measuring stick)  ·  ◑ harness built

**Goal:** a frozen, reproducible, scale-invariant benchmark harness — the yardstick for *all* data and post-training work. Nothing data-centric is measurable without it.

**Deliverables**
- **Wire lm-eval-harness in code:** checkpoint → HF export → `lm_eval.simple_evaluate(model="hf", tasks=…, num_fewshot, limit, dtype)` → per-task scores folded into the report. One command, no hand-copying. Add `lm-eval` to the `eval` extra.
- **Version-frozen battery** in config (hellaswag, arc-easy/challenge, piqa, winogrande, lambada_openai, sciq, openbookqa…), identical at 100M → 3B; battery version recorded with every result.
- **Held-out, decontaminated eval:** a held-out FineWeb-Edu slice (never trained on) for clean perplexity + an **n-gram decontamination check** (benchmark test sets vs the corpus).
- **Comparable scorecard:** append-only results table keyed by (model, size, data-recipe, battery-version), pushed to R2, so the ablation loop can diff recipe A vs B.

**Tests:** lm-eval adapter returns scores on a tiny export; decontam flags a planted contaminant; scorecard diffs two runs.

**Acceptance:** `run_evals.py --config … --checkpoint …` yields perplexity + the full frozen scorecard + samples for any checkpoint, identically at any scale — ready to score the 100M the moment it lands.

**Status (2026-06-14):** harness built + unit-tested — lm-eval wiring (`evals/benchmarks.py`, lazy import, mocked in tests), frozen **v1 battery**, n-gram **decontamination** (`evals/decontam.py`), and a comparable **scorecard** (`evals/scorecard.py`); `configs/eval/lithos-100m.yaml`; `lm-eval` added to the `eval` extra. **Pending:** carve the held-out decontaminated FineWeb-Edu slice for clean perplexity, and the first real `--extra eval` run against the 100M on completion.

## Phase 10 — Data quality v1 ("every trick")  ·  ◻

**Goal:** turn FineWeb-Edu into a best-in-class corpus, and stand up the data-recipe ablation loop. This is the data-layer foundation.

**Deliverables**
- **MinHash/LSH near-dedup** (promote the stubbed interface from Phase 3).
- **Model-based quality classifier** (strong-LLM-labeled sample → cheap classifier → score & select the whole corpus, FineWeb-Edu-style).
- **Synthetic generation / rewrite** (WRAP-style web-rephrase, textbook/Q&A generation) — **open teachers only** (licensing + sovereignty).
- **Decontamination tooling** (shared with Phase 9).
- **Curriculum / mixture weights + an annealing-set** for the LR-cooldown; provenance recorded in the corpus manifest.
- **Ablation harness:** a data intervention → train a small proxy (100M / lean 1B) → score on Phase 9 → keep only winners.

**Acceptance:** a *measured* improvement on the frozen battery from at least one data intervention, ablated on the proxy and recorded in the scorecard.

## Phase 11 — Post-training stack (SFT → DPO → distillation)  ·  ◻  *(absorbs old "SFT v0")*

**Goal:** turn a base into a usable assistant — the scale-invariant fine-tuning pipeline.

**Deliverables**
- **SFT:** versioned **chat template**, messages-format datasets, **loss-masking on non-assistant tokens**, conversation packing, SFT trainer (reuses the pretrain loop). `posttrain/{datasets,collator,sft}.py`.
- **Preference / DPO:** preference-pair datasets, DPO trainer (frozen reference model, or adapter-off reference under LoRA), KTO/SimPO seams. Prefer DPO over PPO for a solo team.
- **Distillation:** **synthetic-data distillation** first (open teacher generates/scores a corpus; tokenize with the Lithos 32k tokenizer to sidestep the teacher's vocab); logit/on-policy distillation as a follow-on seam.
- LoRA/QLoRA path so 3B/7B fine-tunes run on a single card; full-FT of 1B local, 3B/7B full-FT as short single-GPU cloud jobs.

**Acceptance:** SFT model follows simple instructions better than base; DPO improves a held-out preference eval; distillation yields a measurably-better small model — all on the frozen battery.

## Phase 12 — Two-track 1B experiment  ·  ◻  *(reframes old "1B")*

**Goal:** the controlled comparison — *what does a strong teacher buy at 1B?*

- **Track A:** pretrain a 1B from scratch → SFT → DPO (full owned recipe).
- **Track B:** distill **Qwen-72B** (open) into an **identical** 1B.
- Hold architecture, tokenizer, param budget, and the frozen eval constant; vary only the capability source. **The gap between A and B is the deliverable.**

**Acceptance:** both 1Bs scored on the identical frozen battery; the gap quantified and written up; full run manifests + model cards. Track A also proves the owned end-to-end recipe (pretrain + post-train).

## Phase 13 — Lithos 3B (the keeper)  ·  ◻

**Goal:** the scale-invariant **config flip** on secured compute — the payoff of building everything small first.

- 3B model config; reuse the *entire* pipeline unchanged (data, loop, DDP, post-training, eval, export). Token budget from the **scaling-law fits**; hyperparameters **μP-transferred** from the small runs.
- Pretrain → post-train → eval → HF export → R2 → model card. Cost ~$10–25k / 1–4 weeks (Part II §"cost" estimates), driven by token budget; data quality buys it down.

**Acceptance:** the 3B runs as a **config change only** (no code changes beyond the model config + scaled batch/LR/steps); best-in-class-for-its-size on the frozen battery (or its chosen niche). Confirmation run, not the experiment loop.

## Cross-cutting (Part II)

- **Scale-invariance:** μP/μTransfer (tune LR/init on the 100M, transfer to 3B with no re-tuning) + scaling-law fits (read token budget/batch off the small runs). Goal: 100M → 3B is config-only.
- **Docs / model cards / reproducibility / quality gates** — as in Part I; synthetic-data + teacher-provenance disclosures mandatory.

---

## Critical path & parallelization (reconciled)

```
Part I (built):  P0–P5 ─▶ [SKELETON ✅] ─▶ P6 100M (running) ─▶ P7 export ✅ / P8 DDP ✅
                                                   │
Part II:   P9 eval-harness-v1 ─┬─▶ P10 data-quality ──┐
                               └─▶ P11 post-training ──┴─▶ P12 1B two-track ─▶ P13 3B
```
- **P9 (eval harness) gates all of Part II** — the measuring stick; build first.
- **P10 (data quality)** and **P11 (post-training)** run in parallel once P9 is green.
- **P12 (1B two-track)** needs P9 + P10 + P11. **P13 (3B)** is the scale-up of the locked recipe.
- Part II builds at small/cheap scale (local 4070/5090 + the 100M rig); only **P12/P13 need secured cloud compute**.

## Top risks & mitigations

| Risk | Mitigation |
|---|---|
| Loss spikes / instability at scale | QK-norm default-on ≥100M; grad clipping; bf16 w/ fp32 optimizer states; documented rollback. |
| Resume silently re-sees data | Dataloader position checkpointed + tested (P3/P4); same-run resume is bitwise-exact. |
| **Benchmark contamination inflates evals** | **Decontamination tooling (P9/P10)** against the frozen battery before any keeper run. |
| **Synthetic-data collapse / licensing** | Mix synthetic with curated; **open teachers only**; diversity checks; mixture disclosed in the manifest + model card. |
| **Cross-tokenizer distillation misaligns** | Default to **synthetic-data distillation** (teacher generates text; tokenize with Lithos's vocab); logit-distill only with tokenizer alignment. |
| **Scale-up surprises (100M → 3B)** | **Scale-invariant pipeline + μP + scaling-law fits**; the 3B is a config flip, validated on proxies first. |
| Distributed complexity too early | DDP (not FSDP), introduced only after the 100M trains single-process. |
| Modern refinements break engine compat | Export targets the **Qwen3 envelope**; MLA/sliding-window are a conscious, documented cost. |

## Definition of done

- **v0 — Part I (✅ essentially complete):** clone+install · toy trains locally · tokenizer trains from FineWeb-Edu · corpus tokenizes to shards in R2 · 100M trains on 2×H100 (DDP) · metrics logged (JSONL + W&B) · checkpoints resume from R2 · perplexity/export/generation work · provisioning scripts bring up a box one-shot · tests pass · docs explain the workflow.
- **v1 — Part II (the data-centric era):** a frozen, decontaminated eval harness · a *measurably better* proprietary corpus · a scale-invariant post-training pipeline (SFT + DPO + distillation) · the two-track 1B comparison quantified · a **3B keeper** that is best-for-its-size on the frozen battery, produced as a **config flip** — every layer (data, pretrain, post-train, eval, export) owned end to end.

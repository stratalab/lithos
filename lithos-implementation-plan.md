# Lithos Implementation Plan

Companion to `lithos-prd.md`. This turns the PRD (esp. §19 milestones, §21 build order, §26 resolved decisions) into an ordered, buildable sequence. Each phase is a vertical slice that runs on the local dev GPU (RTX 4070 Super, 12GB) — CPU-capable for the fast unit tests — before anything scales to cloud.

> **Reconciled 2026-06-14** to reflect actual execution (FineWeb-Edu corpus, DDP-not-FSDP, W&B, R2, remote-provisioning scripts) and the **data-centric pivot**: owning a high-quality dataset is a co-equal foundation to the model, and we build a *scale-invariant* pipeline at small scale so the 3B is a config flip. **Part I** (walking skeleton + first real run) is built; the **100M run is live**. **Part II** (the data-centric era) is the forward plan.

> **Domain pivot 2026-06-14** — the flagship's target is now a **compact cross-domain STEM reasoner** (code + math + physics + engineering) for **edge deployment**, the niche StrataDB already serves. At ≤1B params the budget is spent on *technical reasoning that transfers across domains* (the Musk-archetype STEM generalist), not general-web breadth — an underserved role precisely because it is nobody's benchmark target. The corpus is **painstakingly constructed** from open technical sources, and the data mix is chosen *empirically* by sweeping slice-splits on the cheap 100M rig (measured by per-domain bits-per-byte, since the STEM benchmarks are at chance below ~500M), then scaling the winning recipe 100M → 500M → 1B → 3B.

## Guiding strategy

1. **Walking skeleton first.** Get the entire §23 command sequence working end-to-end at *toy/smoke* scale (Phases 0–6) before the first real training run. Correctness and reproducibility before scale (PRD §3.8, §20).
2. **Single-GPU/CPU before distributed.** No multi-GPU complexity until the single-process path is green (§20.13–14). Distributed (DDP) is Phase 8, after the 100M model already trains.
3. **Every artifact gets a manifest; every module gets a test.** No silent overwrites, no unprovenanced data (§20.5–9).
4. **Each phase has an acceptance gate.** Do not start phase N+1 until phase N's gate is green.
5. **Scale-invariant pipeline.** Build the whole pipeline (data → pretrain → post-train → eval) at small/cheap scale; the 3B run is a *config flip plus scaling-law math*, not a rewrite. Tune hyperparameters on small proxies (**μP/μTransfer**) and read token budgets off **scaling-law fits**, so 100M → 1B → 3B changes config only.
6. **Data as a co-equal foundation.** Owning a high-quality dataset compounds across every model trained on it; the gains are highest at ≤3B. Ablate data recipes on cheap proxies (the **100M rig**) measured against a **frozen eval harness**. This is the same bet as StrataDB — owning foundational technology pays off in multiples as the stack evolves.
7. **Domain focus over general breadth.** The flagship is a **compact cross-domain STEM reasoner** (code + math + physics + engineering) for the **edge** — StrataDB's niche. A small model wins by spending its whole capacity on *transferable technical reasoning* in a high-signal, **verifiable** domain (code runs, math checks, units balance), not on general-web breadth. Verifiability is the unlock: it makes quality filtering executable, synthetic data safe-and-checkable, and evals trustworthy. The target is the *role shape* (a strong STEM generalist), measured by **cross-domain transfer**, not per-domain leaderboard wins.
8. **The data mix is an empirical result, not a guess.** Find the best slice-split by **sweeping mixes on the 100M rig**, decided on *per-domain bits-per-byte* over frozen held-out sets (the STEM benchmarks flat-line at chance below ~500M and can't pick the mix). Use a **smart directional sweep** (~5–6 runs, not a 2ⁿ grid). **Re-validate the top-2 recipes at each scale-up** — the 100M-optimal mix is *not assumed* to be the 3B-optimal mix (capability emergence + capacity effects are scale-dependent). The ladder (100M → 500M → 1B → 3B) doubles as **scaling-law data**, so the 3B's loss is a prediction, not a gamble.

## Locked decisions (reconciled from PRD §26 + this session)

- **Compute:** local **RTX 4070 Super (12GB)** for dev/smoke + cloud for scale — rented **2×H100 (~$8/hr)** for the 100M; **8×B200** for the big runs. A 5090 (32GB) is the candidate local-iteration upgrade (full-FT 1B, LoRA 3B/7B locally); Pro 6000 (96GB) deferred until justified.
- **Corpus → constructed STEM corpus.** The current 100M runs on **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`, sample-10BT, ODC-By, non-gated) — the pipeline-shakedown corpus. The *flagship* corpus is **purpose-built for code + math + physics + engineering**, assembled from open, mostly-permissive sources as **separate, mixable per-domain manifests**: *code* (The Stack v2, GitHub issues/PRs, notebooks), *math* (FineMath, OpenWebMath, Proof-Pile-2 / AlgebraicStack, arXiv math), *physics + engineering* (arXiv physics/cond-mat/eng, **Stack Exchange** Q&A, OpenStax / LibreTexts, USPTO patents), *verified synthetic* (generated-and-checked solutions / reasoning traces), and a ~15% *general-English glue* slice (FineWeb-Edu) so the model can explain, not just emit. The **intersections** (physics-via-code, math-as-proof-and-program — Jupyter notebooks, papers-with-code, scientific-computing repos) are over-weighted on purpose: that's where transfer is taught. `nvidia/Nemotron-CC-v2` deferred (gated). Over-train (~1,000–1,500 tok/param at 500M). Byte-level BPE — **retrained on the STEM corpus** (indentation, LaTeX, symbols).
- **Distributed: DDP, not FSDP.** 80GB H100 / 192GB B200 fit the whole model + optimizer ≤~7B per GPU, so plain data-parallel suffices and is far simpler. FSDP deferred until a single model exceeds one GPU's memory.
- **Storage:** durable artifacts in **Cloudflare R2** (`lithos-data-fineweb-edu`) via a config-driven `Storage` abstraction + `LITHOS_STORAGE_BASE_URI`; HF Hub for published models. `uv` for envs.
- **Tracking:** optional **W&B** (rank-0, lazy-imported, disabled by default) mirroring the canonical local `metrics.jsonl`.
- **Determinism scoped:** bitwise CPU, best-effort GPU.
- **Architecture:** modernized Llama — GQA-native, optional QK-norm, configurable RoPE theta, SDPA backend, KV cache, depth-scaled init; MoE/MLA/sliding-window deferred behind seams (§6.1). Export targets the **Qwen3 envelope**.
- **Model ladder: 100M (mix-sweep rig) → 500M (STEM flagship, first keeper) → 1B → 3B.** Successive 100M runs sweep the data mix; the winning recipe scales up. (The 300M milestone is folded into the proxy role; the **two-track distillation comparison** — owned-from-scratch vs distill Qwen-72B — becomes an experiment *within* the 1B step, not the headline.)

**Inference (§26.8):** build & own the in-repo PyTorch generator; export checkpoints **HF/Qwen3-compatible** as the single hub feeding eval, vLLM (documented cloud serving, not built), and **llama.cpp/GGUF — now a priority** export target, since on-device/edge is the flagship's deployment niche (the model runs *next to* StrataDB on the device). Local FastAPI `/generate` only — no hosted product (§3.7).

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

*The strategic pivot: own the data layer as a co-equal foundation, build a rock-solid scale-invariant pipeline at small scale, and **construct a best-in-class STEM corpus** whose mix is chosen empirically on the cheap 100M rig — the deliverable is a **compact cross-domain STEM reasoner for the edge** (the two-track from-scratch-vs-distillation gap becomes a side-experiment within the 1B step). All of Part II builds locally + on the 100M rig; only the 500M/1B/3B scale-ups need secured cloud compute.*

## Phase 9 — Eval harness v1 (the measuring stick)  ·  ◑ harness built

**Goal:** a frozen, reproducible, scale-invariant benchmark harness — the yardstick for *all* data and post-training work. Nothing data-centric is measurable without it.

**Deliverables**
- **Wire lm-eval-harness in code:** checkpoint → HF export → `lm_eval.simple_evaluate(model="hf", tasks=…, num_fewshot, limit, dtype)` → per-task scores folded into the report. One command, no hand-copying. Add `lm-eval` to the `eval` extra.
- **Version-frozen battery** in config (hellaswag, arc-easy/challenge, piqa, winogrande, lambada_openai, sciq, openbookqa…), identical at 100M → 3B; battery version recorded with every result.
- **Held-out, decontaminated eval:** a held-out FineWeb-Edu slice (never trained on) for clean perplexity + an **n-gram decontamination check** (benchmark test sets vs the corpus).
- **Comparable scorecard:** append-only results table keyed by (model, size, data-recipe, battery-version), pushed to R2, so the ablation loop can diff recipe A vs B.
- **STEM / executable battery (domain pivot):** extend the frozen battery with the capabilities the flagship is *for* — **HumanEval / MBPP** (code, executed in a sandbox), **GSM8K / MATH** (math, answer-checked), and a physics/engineering QA set — plus a **cross-domain transfer** probe (*derive-then-implement*, *code-to-solve-a-physics-setup*) that measures the actual product: reasoning that **crosses** silos, not per-domain scores. Executable scoring (run the code, check the answer) is the moat — only possible because the domain is verifiable.
- **Per-domain bits-per-byte (the mix-selection metric):** frozen, decontaminated per-slice held-out sets (code / math / physics / eng / general) scored as **bpb**. This is the *primary* signal for the 100M mix-sweep (Phase 12), because the STEM benchmarks are at chance below ~500M and can't rank recipes.

**Tests:** lm-eval adapter returns scores on a tiny export; decontam flags a planted contaminant; scorecard diffs two runs.

**Acceptance:** `run_evals.py --config … --checkpoint …` yields perplexity + the full frozen scorecard + samples for any checkpoint, identically at any scale — ready to score the 100M the moment it lands.

**Status (2026-06-14):** harness built, unit-tested, **and validated end-to-end** — lm-eval wiring (`evals/benchmarks.py`), frozen **v1 battery**, n-gram **decontamination** (`evals/decontam.py`), comparable **scorecard** (`evals/scorecard.py`), `configs/eval/lithos-100m.yaml`. Real-path smoke passed: tiny checkpoint → Qwen3 export → **real `lm_eval` (arc_easy) → scorecard** on transformers 5.12; benchmark scores render in `results.md`. The smoke caught a missing transitive dep — **`accelerate`** added to the `eval` extra. **Calibrated against ground truth:** Qwen2.5-0.5B/1.5B run through *our* harness reproduce their published 0-shot battery within ~1–2%, score clearly above chance, and are **monotonic** (1.5B > 0.5B on every task) — both committed as reference anchors in `configs/eval/reference_scorecard.jsonl` so every future Lithos model diffs against a known baseline. **Pending:** the first `--extra eval` run against the 100M on completion. (The held-out perplexity *mechanism* now exists — Phase 10's `holdout_docs` + decontam carve a disjoint, decontaminated val set at build time; an actual held-out set is produced on the next corpus build. The current 100M trained on all of sample-10BT, so it has none of its own.) **Domain-pivot additions (pending):** the executable STEM battery (HumanEval/MBPP/GSM8K/MATH), the cross-domain transfer probe, and the per-domain **bpb** held-out sets — the mix-sweep's decision metric.

## Phase 10 — Data quality + STEM corpus construction  ·  ◑ in progress

**Goal:** stand up the data-recipe ablation loop (✅ done) **and painstakingly construct the STEM corpus** — code + math + physics + engineering as separate, mixable slices. This is the data-layer foundation and the project's real moat.

**Deliverables**
- **MinHash/LSH near-dedup** (promote the stubbed interface from Phase 3).
- **Model-based quality classifier** (strong-LLM-labeled sample → cheap classifier → score & select the whole corpus, FineWeb-Edu-style).
- **Synthetic generation / rewrite** (WRAP-style web-rephrase, textbook/Q&A generation) — **open teachers only** (licensing + sovereignty).
- **Decontamination tooling** (shared with Phase 9).
- **Curriculum / mixture weights + an annealing-set** for the LR-cooldown; provenance recorded in the corpus manifest.
- **Ablation harness:** a data intervention → train a small proxy (100M / lean 1B) → score on Phase 9 → keep only winners.

**Domain corpus & data-mix machinery (domain pivot)**
- **Per-domain sub-corpora as separate, mixable manifests** — code / math / physics / engineering / general, each independently filtered, MinHash-deduped, **cross-source decontaminated**, and tokenized **once**, then reused across every mix experiment. Built once; mixing is thereafter a config diff. This is the heavy lifting — the source ingestion (Stack v2, FineMath, OpenWebMath, Proof-Pile-2, arXiv, Stack Exchange, OpenStax, patents).
- **Weighted-mix spec in the corpus builder** — sample across domain manifests by ratio (e.g. `mix: {code: 0.40, math: 0.18, physics_eng: 0.20, general: 0.15, synthetic: 0.07}`); exact weights + per-source provenance recorded in the corpus manifest.
- **Per-domain bpb held-out sets** (shared with Phase 9) — disjoint, decontaminated, one per slice; the mix-sweep's decision metric.
- **Verified synthetic (un-deferred, now with a safe purpose)** — in a *verifiable* domain, synthetic stops being a collapse risk: generate solutions / **reasoning traces** with an **open teacher**, then **keep only what passes a checker** (code executes, math answer matches, units balance). Traces (the *why*, step by step) over fact recall — this is where a compact reasoner is actually made. Teacher + provenance disclosed in the manifest.
- **Executable quality filters** — beyond the carried edu-score: does the code parse/run? does the solution check out? Domain signal the general edu-classifier can't give.
- **STEM tokenizer retrain** — byte-level BPE refit on the STEM corpus (whitespace/indentation, brackets, LaTeX, symbols), since the current 32k is general-web.

**Data-construction toolkit — the ingestion engine (cross-cutting; *emergent*, not upfront)**
*Feed it a source → get back a verified dataset in canonical format. This is the engine that builds the per-domain sub-corpora above — and an owned foundation in its own right (it compounds across every future dataset / model / domain).*
- **Pluggable source adapters** (one per input: arXiv-LaTeX, Stack Exchange, The Stack v2, …) → emit raw docs + metadata. The only per-source code.
- **Canonical document schema** — the "required format" everything normalizes to (seed exists: `DocumentSource` / `normalize` in `data/documents.py`).
- **Extraction layer** — LaTeX/PDF (Nougat) / HTML (trafilatura) / AST-aware code → clean text.
- **Shared processing backend** — filter → quality → dedup → decontam → tag → mix (**~60% already built**: MinHash, decontam, quality, manifests, packing).
- **Verification layer (the novel, high-value part)** — executable checks per record: run the code, check the math answer, validate units, schema-validate; each emitted record carries verification provenance. What a *verifiable* domain uniquely allows.
- **Catalog / state store** — sources, per-doc provenance, dedup signatures, verification results, dataset lineage/versions. **Dogfood StrataDB here** (Lithos's data layer cataloged in Strata's memory layer — the stack eating its own cooking); behind an interface, SQLite/parquet first, swap in StrataDB when hardening it on a real workload is the goal.
- **Agent boundary (load-bearing for cost):** AI agents operate at the **meta level only** — *building adapters* for messy sources, *calibrating filters*, the *generate* half of generate-then-verify, triaging edge cases. **Never an LLM call per document on the hot path** (astronomically expensive + non-deterministic at trillion-token scale). The FineWeb pattern: strong model labels a *sample* → distill a cheap classifier → run the classifier at scale. Agent designs/calibrates; deterministic code executes.
- **Don't reinvent the plumbing** — build a *thin* layer on the patterns of HF **`datatrove`** (what FineWeb used) / AI2 **Dolma toolkit**; spend novel effort only on what's ours (verification layer, canonical schema, StrataDB catalog, agent-assist).
- **Build emergently:** write two concrete adapters first (**arXiv-LaTeX + Stack Exchange**) with their verifiers as plain code → extract the framework from the shared shape on the third. Risk = building infra *instead of* the model; antidote = pulled-by-need, never abstract in a vacuum.

**Acceptance:** a *measured* improvement on the frozen battery from at least one data intervention, ablated on the proxy and recorded in the scorecard; **and** the five per-domain STEM sub-corpora built **through the ingestion engine** (≥2 source adapters + the verification layer landed, provenance cataloged), decontaminated, and mixable by weight — ready for the Phase 12 sweep.

**Status (2026-06-14):** ✅ **MinHash/LSH near-dedup** (`data/minhash.py`, drops into the exact-dedup seam, wired behind `near_dedup`). ✅ **Decontamination wiring** — `data/decontam.py` moved from `evals/` (data, not evals, owns it) + `DecontaminationFilter` (per-doc `is_contaminated`) + `load_benchmark_probes` (best-effort battery test-text extraction; 7/8 tasks — piqa's script-based dataset is skipped pending a parquet mirror); wired into `build_corpus` behind `decontam`, verified end-to-end dropping a doc that leaks a benchmark probe. ✅ **Held-out holdout** — `CorpusBuildConfig.holdout_docs` diverts the first N kept (filtered/deduped/decontaminated) docs into a disjoint `held_out/` manifest that loads as any val set, so any corpus gets a clean perplexity set by construction (closes Phase 9's held-out item). ✅ **Quality filtering (existing scorer)** — `data/quality.py` thresholds the edu-classifier `score` FineWeb-Edu already carries per doc (*zero inference*; `DocumentSource.quality_field` threads it through normalization), wired into `build_corpus` behind `quality`. Running a classifier *ourselves* over unscored/synthetic data is the deferred next step. ✅ **Ablation harness** — `evals/ablation.py` runs variant → build corpus → train proxy → score on the frozen battery → diff scorecard → rank winners (`scripts/run_ablation.py`; `configs/ablation/quality-threshold.yaml` + `configs/train/proxy.yaml`); orchestration unit-tested with the three heavy steps mocked. **Remaining:** synthetic generation/rewrite (now un-deferred for the verifiable STEM domain — generate-then-check), and a *real* ablation run (gated on a free GPU) to hit the acceptance criterion. **Domain-pivot forward work (the real build):** construct the five per-domain STEM sub-corpora + the weighted-mix spec + per-domain bpb sets + STEM tokenizer retrain. The ablation harness already does variant → build → train → eval → diff; it now ranges over **domain mixes**, scored by **per-domain bpb**, feeding the Phase 12 sweep.

## Phase 11 — Post-training stack (SFT → RLVR → DPO → distillation)  ·  ◻  *(absorbs old "SFT v0")*

**Goal:** turn the STEM base into a usable, *reasoning* assistant — the scale-invariant fine-tuning pipeline. In a **verifiable** domain, alignment is mostly *generate-then-check*, not human-labeled; the dataset inventory + sovereignty posture live in `docs/data-construction.md` §Post-training.

**Deliverables**
- **SFT:** versioned **chat template**, messages-format datasets, **loss-masking on non-assistant tokens**, conversation packing, SFT trainer (reuses the pretrain loop). `posttrain/{datasets,collator,sft}.py`.
- **RLVR (RL with verifiable rewards) — the domain-native stage.** Math (answer-checked) + code (unit-tested) problems → on-policy rollouts scored by a **verifier shared with Phase 9's executable battery** → GRPO-style update. No reward model, no human labels. For a STEM reasoner this is *ahead of* DPO in value. `posttrain/{verifier,rlvr}.py` — the verifier is the same module that scores HumanEval/MBPP/GSM8K/MATH.
- **Preference / DPO:** preference-pair datasets, DPO trainer (frozen reference, or adapter-off reference under LoRA), KTO/SimPO seams. Prefer DPO over PPO for a solo team. STEM preference pairs generated **on-policy** (verifier labels correct=chosen / incorrect=rejected) rather than bought.
- **Distillation:** **synthetic-data distillation** first (open teacher generates/scores a corpus; tokenize with the Lithos 32k tokenizer to sidestep the teacher's vocab); logit/on-policy distillation as a follow-on seam.
- **Datasets (seed + eval; the *engine* is our own verified generation):** **open-teacher / human / self-aligned only** — Tülu 3 (general backbone), OpenMathInstruct-2 + NuminaMath (math), StarCoder2-Instruct + OSS-Instruct (code), Camel-AI/SciInstruct (physics/eng), OpenR1/OpenThoughts (reasoning traces); RLVR problems from GSM8K/MATH/NuminaMath + MBPP/APPS/CodeContests. **Avoid GPT-tainted sets** (OpenHermes/UltraFeedback/Alpaca) — provenance taint vs the sovereignty thesis; per-dataset license check before any keeper run.
- LoRA/QLoRA path so 3B/7B fine-tunes run on a single card; full-FT of 1B local, 3B/7B full-FT as short single-GPU cloud jobs.

**Acceptance:** SFT model follows instructions better than base; **RLVR lifts pass@k on held-out math/code** (verifier-scored); DPO improves a held-out preference eval; distillation yields a measurably-better small model — all on the frozen **STEM + transfer** battery.

**SFT — concrete design (2026-06-15, in progress).** Code recon confirmed the model (`F.cross_entropy(ignore_index=-100)`) and the training loop (consumes pre-shifted `(x,y)`; `PackedDataLoader` is dataset-agnostic) need **no changes** — SFT is a *data source* + *weight-only init*. The tokenizer **already carries chat special tokens** at fixed IDs 0–6 (`<pad><bos><eos><|system|><|user|><|assistant|><|end|>`), so **no tokenizer retrain is needed for SFT** (the 100M just never trained those embeddings; SFT does). Build: `posttrain/chat_template.py` (render messages → `(ids, loss_mask)`, template **`lithos-chat-v1`**: `<bos><|user|>…<|end|><|assistant|>…<|end|>`, special tokens inserted by ID not string-parsing); `posttrain/sft_dataset.py` (`SFTDataset` → `(x,y)`, plugs into `PackedDataLoader`); a guarded `init_from` weight-only load + `data.kind: sft` branch in `train()` (pretrain path untouched); `configs/sft/*.yaml` + `scripts/train_sft.py` reusing `train()`. **v1 simplifications:** one conversation per sequence padded to `seq_len` (packing later); loss on assistant *content* + its closing `<|end|>` (learns to stop), masking role headers + user/system. **Validated on the 100M test bench** (proves the pipeline; the keeper SFT lands on the 500M).

## Phase 12 — Lithos STEM flagship: 100M mix-sweep → 500M → 1B  ·  ◻  *(reframes old "two-track 1B")*

**Goal:** find the best STEM data recipe on the cheap rig, then scale it into the first **keeper** — a compact cross-domain STEM reasoner for the edge.

**Stage 1 — 100M mix-sweep (cheap, many).** Successive 100M runs over a **smart directional sweep** of slice-splits (anchor mix + more-code / more-math / more-physics / more-general perturbations — ~5–6 runs, *not* a 2ⁿ grid), each a *shorter* fixed token budget (enough for the per-domain loss curves to separate, not a full over-train). **Decide on per-domain bpb**; the benchmark battery is a secondary read (mostly floor at this scale). Hold proxy architecture + token budget fixed across runs; vary *only* the mix; identical decontam everywhere. Output: the winning recipe + the **bpb tradeoff surface** ("+10% code costs X bpb on prose, buys Y on code"). ~hours and low-hundreds of $ per run.

**Stage 2 — 500M flagship (first keeper).** Scale the winning recipe to **500M on ~500–750B tokens** (~1,000–1,500 tok/param) → SFT → DPO (Phase 11). Carry the **top-2** recipes up — *don't* blind-inherit the 100M winner; let scale break the tie. STEM benchmarks now register; the **executable + transfer** evals become the real yardstick. **~$3–6k clean run; ~$8–15k all-in** incl. reruns + ablation R&D. This is the model the whole stack is for: edge-deployable (GGUF), runs next to StrataDB.

**Stage 3 — 1B + the two-track comparison.** Scale to 1B; *within* this step run the controlled experiment — **Track A** (owned from-scratch + post-train) vs **Track B** (distill open **Qwen-72B** into an identical 1B). Hold architecture / tokenizer / budget / eval constant; **the gap is a deliverable** — but now a side-experiment, not the headline.

**Acceptance:** the 500M flagship is best-in-its-class on the **STEM** battery (code/math executed, **transfer** measured), edge-deployable; the winning mix + bpb surface + scaling-law points recorded; the 1B two-track gap quantified. Each scale-up is a **re-validated config flip**, not a rewrite.

## Phase 13 — Lithos 3B (the keeper)  ·  ◻

**Goal:** the scale-invariant **config flip** on secured compute — the payoff of building everything small first.

- 3B model config; reuse the *entire* pipeline unchanged (data, loop, DDP, post-training, eval, export). Token budget from the **scaling-law fits**; hyperparameters **μP-transferred** from the small runs.
- Pretrain → post-train → eval → HF export → R2 → model card. Cost ~$10–25k / 1–4 weeks (Part II §"cost" estimates), driven by token budget; data quality buys it down.

**Acceptance:** the 3B runs as a **config change only** (no code changes beyond the model config + scaled batch/LR/steps), inheriting the **locked STEM recipe** + the ladder's scaling-law-predicted token budget; best-in-class-for-its-size on the **STEM** battery (its chosen niche), edge-deployable (GGUF). Confirmation run, not the experiment loop.

## Cross-cutting (Part II)

- **Scale-invariance:** μP/μTransfer (tune LR/init on the 100M, transfer to 3B with no re-tuning) + scaling-law fits (read token budget/batch off the small runs). Goal: 100M → 3B is config-only.
- **Data-construction toolkit (Phase 10):** the reusable ingestion engine — any source → a verified, canonical dataset. Agents at the meta level only (never the per-doc hot path); **StrataDB is the catalog dogfood target**. An owned foundation in its own right; built emergently from two real adapters, not designed upfront.
- **Docs / model cards / reproducibility / quality gates** — as in Part I; synthetic-data + teacher-provenance disclosures mandatory.

---

## Critical path & parallelization (reconciled)

```
Part I (built):  P0–P5 ─▶ [SKELETON ✅] ─▶ P6 100M (running) ─▶ P7 export ✅ / P8 DDP ✅
                                                   │
Part II:   P9 eval-harness-v1 ─┬─▶ P10 STEM corpus + mix machinery ──┐
                               └─▶ P11 post-training ────────────────┴─▶ P12 100M mix-sweep ─▶ 500M flagship ─▶ 1B ─▶ P13 3B
```
- **P9 (eval harness) gates all of Part II** — the measuring stick; build first. Domain pivot adds the executable STEM battery + transfer probe + per-domain bpb.
- **P10 (STEM corpus + data quality)** and **P11 (post-training)** run in parallel once P9 is green. P10's real build is the **per-domain sub-corpora + mix machinery**.
- **P12** = the **mix-sweep on the 100M rig → 500M flagship (first keeper) → 1B** (two-track distillation folded in here). **P13 (3B)** is the scale-up of the locked recipe.
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
| **100M-optimal mix ≠ 3B-optimal mix** (capability emergence + capacity effects are scale-dependent) | Decide on **per-domain bpb**, not flat-lined benchmarks; carry the **top-2 recipes** up each scale step and let scale break the tie; fit a **data-mixing / scaling law** across the ladder to extrapolate rather than assume. |
| **STEM domain over-reach at 500M** (four domains, one small budget) | Target the **transfer** role, not per-domain SOTA; over-weight the intersections; keep a ~15% general-English glue slice so it can explain; sequence (code+math first) if the bpb surface says capacity is binding. |
| Distributed complexity too early | DDP (not FSDP), introduced only after the 100M trains single-process. |
| Modern refinements break engine compat | Export targets the **Qwen3 envelope**; MLA/sliding-window are a conscious, documented cost. |

## Definition of done

- **v0 — Part I (✅ essentially complete):** clone+install · toy trains locally · tokenizer trains from FineWeb-Edu · corpus tokenizes to shards in R2 · 100M trains on 2×H100 (DDP) · metrics logged (JSONL + W&B) · checkpoints resume from R2 · perplexity/export/generation work · provisioning scripts bring up a box one-shot · tests pass · docs explain the workflow.
- **v1 — Part II (the data-centric, STEM-domain era):** a frozen, decontaminated eval harness *with an executable STEM + transfer battery* · a painstakingly **constructed STEM corpus** (code/math/physics/eng as mixable per-domain slices) with an **empirically-swept** mix · a scale-invariant post-training pipeline (SFT + DPO + distillation) · a **500M STEM flagship** that is best-for-its-size on the executable STEM + transfer battery and **edge-deployable** (runs next to StrataDB) · the two-track 1B comparison quantified · a **3B keeper** produced as a **config flip** — every layer (data, pretrain, post-train, eval, export) owned end to end.

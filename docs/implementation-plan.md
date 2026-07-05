# Lithos Implementation Plan

Companion to `prd.md`. This turns the PRD (esp. §19 milestones, §21 build order, §26 resolved decisions) into an ordered, buildable sequence. Each phase is a vertical slice that runs on the local dev GPU (RTX 4070 Super, 12GB) — CPU-capable for the fast unit tests — before anything scales to cloud.

> **Reconciled 2026-06-14** to reflect actual execution (FineWeb-Edu corpus, DDP-not-FSDP, W&B, R2, remote-provisioning scripts) and the **data-centric pivot**: owning a high-quality dataset is a co-equal foundation to the model, and we build a *scale-invariant* pipeline at small scale so the 3B is a config flip. **Part I** (walking skeleton + first real run) is built; the **100M run is live**. **Part II** (the data-centric era) is the forward plan.

> **Domain pivot 2026-06-14** — the flagship's target is now a **compact cross-domain STEM reasoner** (code + math + physics + engineering) for **edge deployment**, the niche StrataDB already serves. At ≤1B params the budget is spent on *technical reasoning that transfers across domains* (the Musk-archetype STEM generalist), not general-web breadth — an underserved role precisely because it is nobody's benchmark target. The corpus is **painstakingly constructed** from open technical sources, and the data mix is chosen *empirically* by sweeping slice-splits on the cheap 100M rig (measured by per-domain bits-per-byte, since the STEM benchmarks are at chance below ~500M), then scaling the winning recipe 100M → 500M → 1B → 3B.

> **Strategy pivot 2026-06-15 — open-base family + tool-integrated reasoning.** The *capable* tier is now built by **continued-pretraining an open Apache base** (Qwen3-4B — trained on ~**36T tokens**, i.e. ~$633k of pretraining we get for *free*) rather than pretraining 3B+ from scratch. Decisive insight: **a 36T base already *has* the STEM knowledge — the differentiation is in *deployment* (reasoning + tools), not knowledge injection.** So continued-pretrain stays **light** (re-weight to STEM + inject *verified-synthetic / reasoning* data + a high-quality anneal; ~100–300B tokens, **not** 1T of re-shown public data), and the budget goes to the differentiator: **RLVR reasoning + tool-integrated reasoning (TIR)**. (Proof: DeepSeek-R1-Zero — *pure RL on a base*, no continued-pretrain — became a world-class reasoner.) The product is a **family**: *from-scratch* small models (500M/1B — fully owned, 32k STEM tokenizer) + *continued-pretrained* capable models (4B/8B from Qwen3 — Apache-derived, 151k vocab), unified by **one owned deployment recipe**. **MVP = one of each** (a from-scratch 500M + the **4B hero**); 8B + the full family deferred. **Defining capability:** a compact STEM *reasoner* that drives **two verifiable tools — Python (SymPy/NumPy/SciPy) and MATLAB-syntax/Octave** — reasoning for judgment, tools for exact computation, running next to StrataDB on the edge. Cost model in §Economics.

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

- **Compute:** local **RTX 4070 Super (12GB)** for dev/smoke + cloud for scale — rented **2×H100 (~$8/hr)** for the 100M; **8×B200** for the big runs, **pre-emptible/spot (~$19/hr quoted)** — *acceptable*, because the pipeline is already preemption-ready (durable R2 checkpoints + bitwise-exact resume + single-node). The one add for spot: a **SIGTERM-checkpoint handler + auto-resume supervisor** (small; do before the first real cloud run). A 5090 (32GB) is the candidate local-iteration upgrade (full-FT 1B, LoRA 4B/7B locally); Pro 6000 (96GB) deferred. **Post-training stays local through ~3B** (≈0.1% of pretraining compute). **Cost anchor:** ~**$733 per 10²¹ FLOPs** ($19/hr, 40% MFU, 8×B200) — training FLOPs = 6·params·tokens.
- **Corpus → constructed STEM corpus.** The current 100M runs on **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`, sample-10BT, ODC-By, non-gated) — the pipeline-shakedown corpus. The *flagship* corpus is **purpose-built for code + math + physics + engineering**, assembled from open, mostly-permissive sources as **separate, mixable per-domain manifests**: *code* (The Stack v2, GitHub issues/PRs, notebooks), *math* (FineMath, OpenWebMath, Proof-Pile-2 / AlgebraicStack, arXiv math), *physics + engineering* (arXiv physics/cond-mat/eng, **Stack Exchange** Q&A, OpenStax / LibreTexts, USPTO patents), *verified synthetic* (generated-and-checked solutions / reasoning traces), and a ~15% *general-English glue* slice (FineWeb-Edu) so the model can explain, not just emit. The **intersections** (physics-via-code, math-as-proof-and-program — Jupyter notebooks, papers-with-code, scientific-computing repos) are over-weighted on purpose: that's where transfer is taught. `nvidia/Nemotron-CC-v2` deferred (gated). Over-train (~1,000–1,500 tok/param at 500M). Byte-level BPE — **retrained on the STEM corpus** (indentation, LaTeX, symbols).
- **Distributed: DDP, not FSDP.** 80GB H100 / 192GB B200 fit the whole model + optimizer ≤~7B per GPU, so plain data-parallel suffices and is far simpler. FSDP deferred until a single model exceeds one GPU's memory.
- **Storage:** durable artifacts in **Cloudflare R2** (`lithos-data-fineweb-edu`) via a config-driven `Storage` abstraction + `LITHOS_STORAGE_BASE_URI`; HF Hub for published models. `uv` for envs.
- **Tracking:** optional **W&B** (rank-0, lazy-imported, disabled by default) mirroring the canonical local `metrics.jsonl`.
- **Determinism scoped:** bitwise CPU, best-effort GPU.
- **Architecture:** modernized Llama — GQA-native, optional QK-norm, configurable RoPE theta, SDPA backend, KV cache, depth-scaled init; MoE/MLA/sliding-window deferred behind seams (§6.1). Export targets the **Qwen3 envelope**.
- **Model family (mixed lineage), not a from-scratch ladder.** *From-scratch* tier — **500M / 1B**, fully owned, **32k STEM tokenizer** (small models want small vocabs — a 151k vocab wastes ~30% of a 500M's params on embeddings), the sovereignty/craft statement. *Continued-pretrained* tier — **4B / 8B from Qwen3-4B/8B** (Apache; stuck with Qwen's **151k** tokenizer — you can't swap a pretrained model's tokenizer). Unified by **one owned deployment recipe** (SFT → RLVR-TIR → DPO), not by architecture/tokenizer. **MVP = one from-scratch (500M) + the 4B hero**; 8B + full family deferred. The 100M is the mix-sweep rig. Purity gradient stated in the model cards (500M = ours; 4B = "continued-pretrained from Qwen3-4B").
- **Open base + light continued-pretrain (the capital-efficient core).** Start the capable tier from an **open Apache base** (Qwen3-4B/8B). The base's 36T-token pretraining is a foundation we can't afford to replicate (~$633k-equivalent for the 4B) and don't need to — it already holds the knowledge. Continued-pretrain is **light** (~100–300B of *verified-synthetic + reasoning data + STEM re-weight + a high-quality anneal*), **not** a heavy re-pretrain on public data the base already saw (that's re-weighting, not knowledge — poor ROI). Heavy continued-pretrain (≥500B) demoted to a *documented fallback*. Catastrophic forgetting is a non-issue at this scale (≤1% of the base's tokens) with a light general-replay mix.
- **Tool-integrated reasoning (TIR) — the defining capability.** Two **verifiable** tools: **Python (SymPy symbolic + NumPy/SciPy numerical)** and **MATLAB-syntax → GNU Octave** (open/free runtime; MATLAB syntax = market reality, Octave = sovereign + shippable; toolboxes/Simulink out of scope). *No plotting* (matplotlib dropped — image output has no clean verifiable reward). Both tools return *gradeable values* → the execution **sandbox doubles as the RLVR verifier** (run the call, check value: numeric tolerance / symbolic equivalence). Tools turn a small model's computational weakness into a non-issue — the model supplies *judgment*, tools supply *exactness*. Edge stack: **Lithos + Python/Octave + StrataDB** = an open, on-device technical agent.

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

## Phase 9 — Eval harness v1 (the measuring stick)  ·  ◑ harness built  ·  *full eval plan: `docs/eval-plan.md`*

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

## Phase 11 — Post-training stack (SFT → RLVR → DPO → distillation)  ·  ✅ pipeline (test-bench)  *(absorbs old "SFT v0")*

**Goal:** turn the STEM base into a usable, *reasoning* assistant — the scale-invariant fine-tuning pipeline. In a **verifiable** domain, alignment is mostly *generate-then-check*, not human-labeled; the dataset inventory + sovereignty posture live in `docs/data-construction.md` §Post-training.

**Deliverables**
- **SFT:** versioned **chat template**, messages-format datasets, **loss-masking on non-assistant tokens**, conversation packing, SFT trainer (reuses the pretrain loop). `posttrain/{datasets,collator,sft}.py`.
- **RLVR (RL with verifiable rewards) — the domain-native stage.** Math (answer-checked) + code (unit-tested) problems → on-policy rollouts scored by a **verifier shared with Phase 9's executable battery** → GRPO-style update. No reward model, no human labels. For a STEM reasoner this is *ahead of* DPO in value. `posttrain/{verifier,rlvr}.py` — the verifier is the same module that scores HumanEval/MBPP/GSM8K/MATH.
- **Preference / DPO:** preference-pair datasets, DPO trainer (frozen reference, or adapter-off reference under LoRA), KTO/SimPO seams. Prefer DPO over PPO for a solo team. STEM preference pairs generated **on-policy** (verifier labels correct=chosen / incorrect=rejected) rather than bought.
- **Distillation:** **synthetic-data distillation** first (open teacher generates/scores a corpus; tokenize with the Lithos 32k tokenizer to sidestep the teacher's vocab); logit/on-policy distillation as a follow-on seam.
- **Datasets (seed + eval; the *engine* is our own verified generation):** **open-teacher / human / self-aligned only** — Tülu 3 (general backbone), OpenMathInstruct-2 + NuminaMath (math), StarCoder2-Instruct + OSS-Instruct (code), Camel-AI/SciInstruct (physics/eng), OpenR1/OpenThoughts (reasoning traces); RLVR problems from GSM8K/MATH/NuminaMath + MBPP/APPS/CodeContests. **Avoid GPT-tainted sets** (OpenHermes/UltraFeedback/Alpaca) — provenance taint vs the sovereignty thesis; per-dataset license check before any keeper run.
- LoRA/QLoRA path so 3B/7B fine-tunes run on a single card; full-FT of 1B local, 3B/7B full-FT as short single-GPU cloud jobs.

**Acceptance:** SFT model follows instructions better than base; **RLVR lifts pass@k on held-out math/code** (verifier-scored); DPO improves a held-out preference eval; distillation yields a measurably-better small model — all on the frozen **STEM + transfer** battery.

**SFT — concrete design (2026-06-15, in progress).** Code recon confirmed the model (`F.cross_entropy(ignore_index=-100)`) and the training loop (consumes pre-shifted `(x,y)`; `PackedDataLoader` is dataset-agnostic) need **no changes** — SFT is a *data source* + *weight-only init*. The tokenizer **already carries chat special tokens** at fixed IDs 0–6 (`<pad><bos><eos><|system|><|user|><|assistant|><|end|>`), so **no tokenizer retrain is needed for SFT** (the 100M just never trained those embeddings; SFT does). Build: `posttrain/chat_template.py` (render messages → `(ids, loss_mask)`, template **`lithos-chat-v1`**: `<bos><|user|>…<|end|><|assistant|>…<|end|>`, special tokens inserted by ID not string-parsing); `posttrain/sft_dataset.py` (`SFTDataset` → `(x,y)`, plugs into `PackedDataLoader`); a guarded `init_from` weight-only load + `data.kind: sft` branch in `train()` (pretrain path untouched); `configs/sft/*.yaml` + `scripts/train_sft.py` reusing `train()`. **v1 simplifications:** one conversation per sequence padded to `seq_len` (packing later); loss on assistant *content* + its closing `<|end|>` (learns to stop), masking role headers + user/system. **Validated on the 100M test bench** (proves the pipeline; the keeper SFT lands on the 500M). ✅ **Done + committed** (`15c17c0`): trained on a single 4070 in ~17 min, base→answers-and-stops ("Paris.").

**DPO — concrete design (2026-06-15, in progress).** Unlike SFT, DPO needs a *custom loss* (not next-token CE), so it reuses the scaffolding (optimizer/schedule/checkpoint/DDP/logging) but brings its own step. `posttrain/dpo.py`: `sequence_logprobs(logits, labels)` (sum of response-token log-probs, `-100`-masked) + `dpo_loss(policy_chosen/rejected_logps, ref_chosen/rejected_logps, beta)` → logistic loss on the reference-relative preference margin, plus reward-accuracy/margin metrics. A **frozen reference** = the SFT model (or adapter-off under LoRA). Preference data = `(prompt, chosen, rejected)` from **open/human/on-policy** sources only. The DPO trainer starts from the SFT checkpoint (`init_from`) and computes log-probs under policy + frozen ref for both responses.

**Phase 11 COMPLETE (test-bench, 2026-06-15)** — all four stages built, unit-tested, and validated on the 100M, **entirely on a single RTX 4070** (post-training is ~0.1% of pretraining compute — it stays local through ~3B). Banked lessons (the real value, each caught at ~$0):
- **SFT** ✅ base → answers-and-stops ("Paris.") (`15c17c0`).
- **DPO** ✅ needs *in-distribution* preferences + a tight KL leash. v1 (OOD human-vs-model, β=0.1, 300 steps) Goodharted — reward-acc rose while generation *regressed* (Paris→Toulouse); v2 (on-policy two-sample, β=0.5, 120 steps) held the line (`88d3ed5`, `54377b1`). Lesson: watch real outputs, not the curve.
- **RLVR/GRPO** ✅ the machinery reshapes behaviour toward a *verifiable* objective (sampled acc ~5%→16%); the reward curve oversells (greedy stays ~flat) — and **verifiable rewards beat preferences for a verifiable task** (RLVR *improved* on arithmetic where DPO *regressed* on prefs — the STEM thesis in miniature) (`a222618`).
- **Distillation** ✅ (open teacher, synthetic-data) transfers *style* not *substance* on a 110M — a flagship move (needs a capable student), not a test-bench win; more data beat it at equal scale (`a2eb9b4`).
- Checkpoints made **self-describing** for size-agnostic reload, closing the one real scale-invariance gap (`7a3c4c0`).

The **keeper** post-training lands on the flagship (the **4B hero** + the from-scratch **500M**); the cheap ladder de-risked it by catching every failure mode here for free. **Still flagship-only (deferred → built in P12):** RLVR rollout throughput (batched/vLLM), LoRA/QLoRA (memory / smaller cards), multi-GPU DPO/RLVR, and the **executable tool-sandbox verifier** (the arithmetic `MathVerifier` is the interface; SymPy/Octave execution is the real one).

## Phase 12 — Lithos family + the deployment recipe (TIR)  ·  ◻  *(reshapes "STEM flagship")*

> **Post-training buildout:** the gaps between the Phase-11 test bench and this
> flagship recipe are reviewed in `docs/post-training-review.md` and sequenced
> into epics + an experiment plan in `docs/post-training-implementation-plan.md`
> (the spine: TIR format decision → tokenizer freeze → retokenize → 500M → keeper).

**Goal:** the **MVP family** — one *from-scratch* sovereign small model (**500M**) + one *continued-pretrained* capable **hero (4B ← Qwen3-4B)** — both turned into compact STEM **reasoners that drive tools**. The differentiation is the deployment recipe, not the pretraining.

**The deployment recipe (identical for both — the family's identity):**
base/pretrain → **light continued-pretrain** *(capable tier only)* → **SFT** (instructions + reasoning-trace format + **tool-use demos**) → **RLVR-TIR** *(the main event)* → **DPO** polish. Reasoning is the *path* to STEM excellence, not an add-on: SFT → competent assistant; **RLVR → reasoner**.

**Track S — from-scratch 500M (sovereign tier).** Mix-sweep on the 100M rig (directional ~5–6 runs, decided on per-domain **bpb** — STEM benchmarks flat-line below ~500M) → train the 500M on the winning STEM recipe (~600B tok, 32k STEM tokenizer), **including a long-context extension phase** (high-theta RoPE from day one, ~1e6 per the settled 2026-07-04 architecture decisions, + long-doc anneal — epic E10, so the model natively handles the 4k–16k harvested reasoning traces at SFT with no rope-scaling hack; target context set by profiling the trace-length distribution; full attention — SWA gated to 3B-if-ever, see E10) → the shared deployment recipe. Fully owned. ~$1.3k pretrain.

**Track C — continued-pretrained 4B hero (capable tier).** Qwen3-4B-base → **light** continued-pretrain (mostly *verified-synthetic + reasoning + STEM anneal*, general-replay mixed in; ~100–300B — **not** re-showing public data the base already has) → the shared deployment recipe. The hero — capable enough to *actually reason*; edge-deployable (GGUF/4-bit), runs next to StrataDB. **~$15–20k all-in**, most of it RLVR + synthetic, *not* continued-pretrain.

**The tool sandbox + verifier (shared infra — the concrete new build).** A sandboxed executor for **Python (SymPy/NumPy/SciPy)** + **Octave** — *both* the inference-time tool runtime *and* the **RLVR verifier** (run the call → check value by numeric tolerance / symbolic equivalence). Plus **verified-synthetic TIR data**: generate problems + reason→call-tool→use-result→answer traces, **keep only what the sandbox runs correctly** (the genuinely-additive data the 36T base lacks). This *extends* the Phase-11 `MathVerifier`/GRPO to **execute**, not just extract — the executable STEM verifier, now concrete.

**Engineering makes TIR mandatory, not optional (and proves the compact-edge thesis).** Math/code degrade *gracefully* without tools; engineering just **fails**: 10–20 chained arithmetic steps (98% per-step accuracy ⇒ ~67% over 20 steps), property lookups (steam tables, material constants — looked up, never memorized), numerical methods, unit discipline. No 500M–4B model holds 8-sig-fig computation in its weights — and it doesn't need to: **tools convert computation from a *capacity* problem into a *delegation* problem**. The model supplies *what to compute, in what order, and is-it-sane*; the sandbox supplies exactness. That orchestration skill fits in a small model — which is precisely why "compact STEM reasoner at the edge" is coherent. Three concrete additions:
- **Sandbox grows three packages:** **CoolProp** (thermophysical properties), **python-control** (the controls crown jewel — makes essentially every controls/signals textbook problem executable), **pint** (units).
- **Units as a verifier dimension:** run answers through `pint` — wrong-dimension solutions die instantly. Dimensional analysis is engineering's "the code runs": a free, executable correctness signal alongside numeric tolerance / symbolic equivalence.
- **The eng-TIR data mandate:** math TIR traces exist (OpenMathReasoning's 1.7M, Nemotron-Math-v2's with-Python configs — harvested); **engineering TIR traces don't exist anywhere** — nobody has published R1-solves-thermo-with-CoolProp. So the self-generation engine's clearest mandate: problem banks (FE exams, textbook problems, quals — `corpus/seed_index.csv` kind=problems) → open teacher solves *with tools in our sandbox* → verify (value **and** units) → keep. **A corpus nobody else has.**

**Synthetic-data generators — open reasoners only (GLM-5.2 lead for code).** The verified-synthetic TIR/reasoning data is distilled (generate → sandbox-verify → keep correct → own) from *permissively-licensed* teachers: **GLM-5.2** (Z.ai, open weights — current best *open* coder: SWE-Bench Pro 62.1 SoTA-open, Terminal-Bench 2.1 81.0) as the **lead code teacher**; **Qwen2.5-Coder-32B / DeepSeek-V3** (code); **Qwen2.5-Math-72B** (math); **DeepSeek-R1 / QwQ-32B / Qwen3-thinking** (reasoning/TIR traces). **Never GPT/Claude** — their ToS bars using outputs to train competing models (a provable contract breach; open-weights distillation, by contrast, the authors *invite*). Capacity-gated to the 4B hero / 500M (distillation transfers *style not substance* on a tiny student — Phase-11 lesson).

**Reward-hacking defense (lifted from GLM-5.2's coding RL).** Because the sandbox *is* the RLVR reward surface, the policy *will* try to game it (our banked Goodhart lesson). An **LLM judge screens each tool call** for gaming — hard-coded expected answers, no-op/trivial calls, injecting or short-circuiting the verifier — and **returns dummy data instead of reward on suspicion**, killing the exploit while keeping training stable. Standing rule alongside it: **watch real rollouts, not the reward curve**.

**Acceptance:** the 4B hero is best-in-class-for-its-size on the **executable STEM + transfer battery** with **tool-integrated reasoning** (it reasons, calls SymPy/Octave, the sandbox verifies correctness), edge-deployable; the from-scratch 500M is the owned/sovereign counterpart; **the deployment recipe is identical across both** (proving it's a *family*). Winning mix + bpb surface recorded.

## Phase 13 — Family scale-out + fallbacks  ·  ◻  *(deferred until the MVP proves the recipe)*

Once the MVP (500M + 4B hero) lands, expanding the family is mostly **config + compute** on the proven, scale-invariant pipeline — not new engineering:
- **8B ← Qwen3-8B** — same light-continued-pretrain + deployment recipe. ~$13k incremental. The "high-end edge" tier.
- **1B from-scratch** — the second sovereign model, *if* the 500M proves the from-scratch tier worth extending.
- **Documented fallbacks** (only if the open-base path disappoints on quality): heavy continued-pretrain (≥500B), or a **from-scratch 3B @ 2–3T** (~$26–40k; ~600B–1T unique STEM repeated ~3–4 epochs per data-constrained-scaling; needs **multi-node + elastic-for-spot** training — a new infra dimension, deferred until justified).
- **Small-tier fallback (the mirror case):** if the from-scratch **500M** disappoints against a continued-pretrained **Qwen3-0.6B** on the same battery (an eval we should run as a matter of course — it's cheap and keeps us honest), the *shipping* 500M-class model becomes the CPT'd Qwen and the from-scratch line **retreats to the R1/R2 research vehicle** — where it's irreplaceable anyway (trained-in retrieval/sparse attention and the 32k-tokenizer economics only exist from-scratch). Sovereignty stays intact via the research line; the product ships whatever wins the battery.

**Acceptance:** each new family member ships through the **unchanged** recipe (continued-pretrain/pretrain → SFT → RLVR-TIR → DPO → eval → GGUF export → R2 → model card), best-in-class-for-its-size on the STEM + TIR battery. Confirmation runs, not the experiment loop.

## Research tracks R1 + R2 — StrataDB in the inference path  ·  ◻  *(post-MVP; possibly the project's final destination)*

> **Gate: neither track touches the Phase-12 MVP.** Banked 2026-07-01; begins only after the MVP family ships. Two tracks, one primitive: ANN + graph lookup over a StrataDB store — **R1 delegates *facts*** (long-term memory), **R2 delegates *attention state*** (working memory). Systems framing: **Lithos = processor, StrataDB = memory hierarchy, tools = coprocessor.**

**Thesis.** Transformers store facts in weights *by accident*, not by design — FFN layers are literally key-value memories (Geva et al. 2021): a lossy, compressed, **unwritable, unauditable** database paid for in parameters. R1 makes the database explicit and external: **weights hold reasoning, language, and orchestration; facts live in StrataDB, fused into the inference path** — not agentic memory, not prompt-level RAG, but retrieval as an architectural component of the foundation model. This completes the delegation argument that already defines the product: **tools delegate computation (Phase 12); memory-fusion delegates facts; the model keeps judgment.** Endgame framing: the Strata stack completing itself — the model layer and the database layer fuse into one owned system.

**Why the abandoned lineage is right for *us*.** kNN-LM (2020, per-token datastore interpolation — rare-fact wins), **RETRO** (DeepMind 2022, chunked cross-attention over a trillion-token store *trained in from scratch* — 7B matched ~25× larger on knowledge tasks), Meta's **memory layers** (2024, FFNs replaced by explicit sparse KV lookup — beat dense/MoE at fixed FLOPs on factual QA), **Memorizing Transformers** (2022, the closest thing to an inference-time *write* path). Frontier labs walked away because scale was cheaper than plumbing and prompt-RAG got 80% for free. Our economics are inverted on every axis: (a) **a 500M model's scarcest resource is parameters** — facts in weights are wasted capacity; (b) **we own the database** — StrataDB as the model's non-parametric memory is updatable, auditable, provenance-tracked (sovereignty, satisfied in a way weights never can be), and no other small-model builder owns both layers; (c) **edge deployment kills the latency objection** — the datastore is local SSD next to the model, not a network hop (RETRO-style per-chunk retrieval, not per-token, is the cost model); (d) **we pretrain from scratch** — retrieval-augmented architectures work best trained-in from day one, which almost nobody retrofitting a checkpoint can do, and the 100M rig + per-domain bpb harness is exactly the cheap ablation machine this needs.

**Staged plan (each stage gates the next):**
- **R1.1 — read-only fusion, 100M rig.** kNN-LM-style datastore interpolation bolted onto the existing 100M. Decision metric: per-domain bpb, especially **rare-fact perplexity** (where the lineage shows the effect). Weeks, not months; pure ablation.
- **R1.2 — trained-in retrieval, from-scratch 500M.** If R1.1 moves bpb: RETRO-style chunked cross-attention (and/or memory layers as the in-model variant) trained in from the start, StrataDB as the store. This is the sovereign tier's architectural differentiator.
- **R1.3 — the write path (long-horizon).** The model durably writes to its own memory at the foundation level (Memorizing-Transformers direction). Genuinely unsolved; attempt only on a proven read stack.

**Honest risks:** facts and skills don't separate cleanly — "Paris is the capital" participates in reasoning circuits, so there's a knowledge floor in weights to find empirically; retrieval brings its own failure modes (misses, stale/adversarial store content — **the datastore joins the trust boundary**, same threat class as the RLVR reward-hacking defense); and it's a real engineering fork (datastore-during-pretraining, train/test store consistency). Hence the gate and the staging.

---

**R2 — attention state in StrataDB (KV cache as a database + graph problem).** The KV cache is the model's *working memory*, and at long context it **exceeds the model weights** — which means long context **does not fit on edge devices** at all without spilling attention state to SSD. The datacenter world already treats this as a database problem (vLLM PagedAttention = virtual-memory paging; Mooncake = a disaggregated KV-cache store at the center of serving; LMCache = content-addressed KV reuse); the **embedded KV store for edge inference is an empty niche, and we own an embedded Rust DB.** R2 is not an optimization — it's the enabling mechanism for a long-context technical agent on-device.

**Why it's feasible (the latency physics).** Full attention through a storage boundary fails on bandwidth (GPU ~TB/s vs SSD ~GB/s) — but attention mass concentrates in a tiny fraction of keys, so cold attention becomes **top-k retrieval**: hot recent window + attention sinks stay on-GPU; the cold tail is an **ANN query** over StrataDB (Quest / RetrievalAttention: ~1–2% of KV suffices). Frontier proof that this can be *trained in*: **GLM-5.2's DeepSeek Sparse Attention** (lightning indexer, top-2048 per query) — top-k attention learned from scratch at 744B scale.

**The graph layer (StrataDB's differentiator — beyond raw KV).** Attention is *learned associativity*, not just similarity — different heads learn co-reference, syntax, induction. Top-k-by-*similarity* starves the relational heads (the multi-hop failure: "the circuit we discussed before lunch" isn't embedding-similar to its answer). StrataDB's knowledge-graph capability patches exactly this, three ways:
- **Expand, never gate:** retrieve top-k by similarity (the floor), then **one-hop edge expansion** (co-reference, entity link, temporal order, depends-on) to recover related-but-dissimilar keys that stacked attention layers would have found. Edges only *add* candidates — a missing edge costs an enrichment, never a hard failure.
- **Edge-driven prefetch (the latency answer):** vector search can't prefetch (the next query is unknown); a graph gives locality structure — prefetch the current node's neighborhood into RAM ahead of need, turning random SSD reads into predictable ones.
- **Computed edges over extracted ones (the STEM advantage):** open-domain graph construction needs an error-prone extractor; *our* domain's graphs largely **fall out of parsers, deterministically** — code has ASTs/call/import graphs, math has symbol-binding and equation dependency, datasheets have part/pin references. "A calls B" is an attention prior from tree-sitter, not a hallucination.

**Consolidation — the bridge that fuses R1 and R2.** The memory hierarchy is *fidelity* tiers, not just temperature tiers: **hot exact KV (GPU) → cold top-k + graph expansion (SSD) → consolidated graph memory** (entities, relations, claims with provenance — tiny, queryable, auditable). That last tier *is* R1's knowledge store: working memory graduates into long-term memory over time, in one engine — episodic → semantic consolidation. One StrataDB engine serves both tracks; building R2's ANN + graph machinery pays R1's infrastructure bill.

**Sequencing note:** R2 may sensibly *precede* R1 — it's a systems bet (the sparse-attention literature already works) rather than a modeling bet, and KV offload delivers user-visible value (long context on-device) even before memory-fusion proves out. Trained-in beats retrofit here too (DSA/NSA precedent): retrieved tokens carrying edge-type information during training teach the model to *query structurally*. Same gate as R1: post-MVP, nothing touches Phase 12.

## Economics (cost model, $19/hr 8×B200 spot · 40% MFU · ~$733/10²¹ FLOPs)

The decisive lever is **continued-pretrain depth** (the rest is roughly fixed). Per-model training cost ≈ 6·params·tokens × $733/10²¹:

| component | basis | cost |
|---|---|---|
| **Open base pretraining** (Qwen3-4B's 36T) | inherited | **$0** *(≈$633k to replicate — the gift)* |
| From-scratch 500M @ 600B | 1.8e21 | ~$1.3k |
| 4B continued-pretrain — **light** @ 100–300B | 2.4–7.2e21 | ~$1.8–5.3k |
| &nbsp;&nbsp;*(heavy fallback @ 1T)* | 2.4e22 | *~$17.6k* |
| Synthetic generation (verified TIR/reasoning — *the additive data*) | teacher inference | ~$2–4k |
| Reasoning RL (RLVR-TIR) — **the differentiator** | rollout-heavy | ~$2–4k/model |
| Mix-sweeps + ablations (100M rig) | ~6 runs | ~$1.5k |
| Contingency (reruns ~30–40%) + eval + R2 | — | ~$3–5k |

- **MVP (500M + 4B hero, *light* continued-pretrain):** **≈ $20–26k** all-in. *(Heavy-continued fallback pushes it to ~$38k.)*
- **Hero-first staging:** land the 4B (~$15–20k), then the 500M is a ~$2.4k add; 8B later ~$13k.
- **Sensitivities:** continued-pretrain depth (biggest), RLVR depth, synthetic volume, MFU/spot price (on-demand ~doubles training), reruns. Labor isn't priced — it's the solo+AI time.

## Cross-cutting (Part II)

- **Scale-invariance & the shared recipe:** μP/μTransfer + scaling-law fits make the *from-scratch* tier (100M → 500M → 1B) config-only; and the **deployment recipe (SFT → RLVR-TIR → DPO) is identical across the whole family** — from-scratch *and* continued-pretrained. That shared recipe is what *makes* it a family and what makes scale-out cheap (config + compute, not re-engineering).
- **Tool sandbox / TIR (Phase 12):** the sandboxed **Python(SymPy/NumPy/SciPy + CoolProp/python-control/pint) + Octave** executor is shared infra — inference-time tool runtime, RLVR verifier, *and* the generate-then-check for verified-synthetic TIR data. **Sandbox security (isolation, timeouts, resource limits) is the real engineering.** Edge target: the executor + StrataDB ship *on-device* alongside the model.
- **Data-construction toolkit (Phase 10):** the reusable ingestion engine — any source → a verified, canonical dataset. Agents at the meta level only (never the per-doc hot path); **StrataDB is the catalog dogfood target**. An owned foundation in its own right; built emergently from two real adapters, not designed upfront.
- **Docs / model cards / reproducibility / quality gates** — as in Part I; synthetic-data + teacher-provenance disclosures mandatory.

---

## Critical path & parallelization (reconciled)

```
Part I (built):  P0–P5 ─▶ [SKELETON ✅] ─▶ P6 100M (running) ─▶ P7 export ✅ / P8 DDP ✅
                                                   │
Part II:   P9 eval + TIR-verifier ─┬─▶ P10 STEM corpus + mix machinery ──┐
                                   └─▶ P11 post-training ✅ ──────────────┴─▶ P12 family MVP (from-scratch 500M + 4B hero ← Qwen3-4B + TIR) ─▶ P13 scale-out (8B/1B)
```
- **P9 (eval harness) gates all of Part II** — the measuring stick; build first. Domain pivot adds the executable STEM + transfer battery + per-domain bpb; the executable **verifier becomes the RLVR/TIR sandbox** (shared with P12).
- **P10 (STEM corpus)** and **P11 (post-training ✅ on the test bench)** ran in parallel once P9 was green. P10's real build is the **per-domain sub-corpora + mix machinery + verified-synthetic TIR data**.
- **P12 = the MVP family** — from-scratch **500M** (Track S) + the **4B hero** continued-pretrained from Qwen3-4B (Track C) + the **tool sandbox/verifier**, both finished with the shared **SFT → RLVR-TIR → DPO** recipe. **P13** = family scale-out (8B, 1B) + fallbacks.
- Part II builds at small/cheap scale (local 4070/5090 + the 100M rig); only **P12/P13 need secured (spot) cloud compute**.

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
| **Open base already saw our STEM data** (36T base) | Continued-pretrain on public data = *re-weighting, not new knowledge* → keep it **light**; spend on **verified-synthetic + RLVR-TIR** (the additive, deployment-teaching parts the base lacks). |
| **Tool-sandbox security** (executing model-written code) | Hard **isolation + timeouts + resource/network limits**; the sandbox is *both* runtime *and* RLVR verifier, so it must be airtight before any TIR run. |
| **Reasoning won't emerge at small scale** | Emerges with capacity (~4B is near the floor); rely on **distillation from an open reasoning teacher + RLVR**. The from-scratch 500M won't reason — **tools compensate** (it calls SymPy/Octave instead of computing). |
| **Open-base lineage / license** | Qwen3 = **Apache-2.0** (derivatives + commercial OK with attribution; open weights can't be revoked once downloaded). Disclose lineage in the model card. |
| **Spot preemption loses progress** | Durable R2 checkpoints + exact resume already; add the **SIGTERM-checkpoint handler + auto-resume supervisor** before the first spot run → ~zero lost work. |

## Definition of done

- **v0 — Part I (✅ essentially complete):** clone+install · toy trains locally · tokenizer trains from FineWeb-Edu · corpus tokenizes to shards in R2 · 100M trains on 2×H100 (DDP) · metrics logged (JSONL + W&B) · checkpoints resume from R2 · perplexity/export/generation work · provisioning scripts bring up a box one-shot · tests pass · docs explain the workflow.
- **v1 — Part II (the STEM-reasoning era):** a frozen, decontaminated eval harness *with an executable STEM + transfer + TIR battery* · a **constructed STEM corpus** (mixable per-domain slices, empirically-swept) + **verified-synthetic TIR/reasoning data** · the **post-training stack — SFT → RLVR-TIR → DPO — built & validated** (✅ test-bench, Phase 11) · a **tool sandbox** (Python SymPy/NumPy/SciPy + Octave) that is *both* runtime *and* RLVR verifier · the **MVP family**: a *from-scratch* **500M** (fully owned, 32k) + a *continued-pretrained* **4B hero** (← Qwen3-4B, Apache) — both compact STEM **reasoners that drive SymPy/Octave**, best-for-size on the STEM+TIR battery, **edge-deployable** next to StrataDB — produced by **one shared deployment recipe**. Every *deployment* layer owned end to end; the capable tier's general-knowledge base is open (Apache, disclosed).

# Lithos Implementation Plan

Companion to `lithos-prd.md`. This turns the PRD (esp. §19 milestones, §21 build order, §26 resolved decisions) into an ordered, buildable sequence. Each phase is a vertical slice that runs on the local dev GPU (RTX 4070 Super, 12GB) — CPU-capable for the fast unit tests — before anything scales to cloud.

## Guiding strategy

1. **Walking skeleton first.** Get the entire §23 command sequence working end-to-end at *toy/smoke* scale (Phases 0–6) before the first real training run. Correctness and reproducibility before scale (PRD §3.8, §20).
2. **Single-GPU/CPU before distributed.** No FSDP, no multi-GPU complexity until the single-process path is green (§20.13–14). Distributed is Phase 8, after the 100M model already trains.
3. **Every artifact gets a manifest; every module gets a test.** No silent overwrites, no unprovenanced data (§20.5–9).
4. **Each phase has an acceptance gate.** Do not start phase N+1 until phase N's gate is green.

## Locked decisions (from PRD §26)

Local **RTX 4070 Super (12GB)** for dev/smoke + cloud for scale · `nvidia/Nemotron-CC-v2` (+ `Nemotron-CC-Math-v1`) corpus · over-train (~100–200 tok/param) · byte-level BPE 32k · cloud object store + HF Hub artifacts · `uv` · determinism scoped (bitwise CPU, best-effort GPU). Architecture: modernized Llama — GQA-native, optional QK-norm, configurable RoPE theta, SDPA backend, KV cache, depth-scaled init; MoE/MLA/sliding-window deferred behind seams (§6.1).

**Execution targets:** the 4070 runs dev, the toy model, the end-to-end smoke, short single-GPU sanity runs, and local inference (≤~1B in bf16). Cloud runs the over-trained 100M/300M and all 1B + FSDP. The local card is single-GPU, so FSDP (Phase 8) is smoke-tested on a cloud multi-GPU node, not locally. Unit/shape/correctness tests stay CPU-fast so free-runner CI can execute them.

**Inference (§26.8):** build & own the in-repo PyTorch generator; export checkpoints **HF/Qwen3-compatible** as the single hub feeding eval, vLLM (documented cloud serving, not built), and llama.cpp/GGUF (deferred distribution). Local FastAPI `/generate` only — no hosted product (§3.7).

---

## Phase 0 — Repo & tooling skeleton  ·  Milestone 0  ·  size S

**Goal:** an installable package where `pytest` and `ruff check .` run green, with the config system everything else depends on.

**Deliverables**
- `pyproject.toml` (uv-managed) + `uv.lock`, `.python-version` (3.11), `.gitignore` (excludes `runs/`, `data/`, `artifacts/`, `*.bin`, checkpoints), `README.md`.
- Package tree per PRD §5 with `__init__.py` stubs for every subpackage.
- `utils/config.py` — **the foundation.** YAML load → pydantic-validated config objects → CLI dotted-key overrides (`--model.n_layers=12`) → **resolved config dump**. Fail loudly on missing required keys (§20.11). Config composition: a run config references model/data/train sub-configs and merges them.
- `utils/seed.py` (global seeding), `utils/device.py` (device/dtype resolution), `utils/io.py` (atomic writes, sha256, JSON/YAML helpers, **no-clobber guard**), `utils/checks.py` (shape/assert helpers), `train/logging.py` skeleton (run-dir creation, JSONL writer).
- Tooling: `ruff` + `mypy`/`pyright` config in `pyproject.toml`; `tests/` with a trivial passing test.
- **CI** (`.github/workflows/ci.yml`): `ruff check .` + `pytest` (unit/shape/correctness tests — CPU, fast, runs on free GitHub runners). The end-to-end training smoke is device-agnostic and run **locally on the 4070** (CPU-capable as a fallback); it is not part of free-runner CI. Enforces §17.3.

**Dependencies to pin:** `torch`, `numpy`, `safetensors`, `tokenizers`, `datasets`, `huggingface_hub`, `pyarrow`, `polars`, `pydantic`, `rich`/`tqdm`, `zstandard`; dev: `pytest`, `ruff`, `mypy`.

**Tests:** config load + override + resolve; missing-required-key raises; no-clobber guard raises on existing run dir.

**Acceptance:** `uv sync` works · `pytest` green · `ruff check .` clean · README states purpose. (PRD §19 M0)

---

## Phase 1 — Toy transformer (modernized Llama)  ·  Milestone 1  ·  size M

**Goal:** a correct decoder-only model that runs on CPU, with generation and the full shape/correctness test suite.

**Deliverables** (build in PRD §21 order)
- `model/config.py` — `ModelConfig`: `vocab_size`, `pad_vocab_to` (default 128), `seq_len`, `n_layers`, `hidden`, `n_heads`, `n_kv_heads`, `intermediate_size` (default round(8/3·hidden → /256)), `rope_theta`, `qk_norm` (bool), `tie_embeddings`, `dropout`, `rms_eps`, `init_std`, `attn_backend` (`sdpa`|`eager`). Validates head divisibility, `n_heads % n_kv_heads == 0`.
- `model/norm.py` — RMSNorm (configurable eps).
- `model/rope.py` — RoPE with configurable `theta`; applied to q/k; shape-preserving.
- `model/attention.py` — **GQA-native** causal attention (`n_kv_heads`, KV head repeat), **SDPA backend** (`F.scaled_dot_product_attention`) + **eager materialized-mask fallback**, **optional QK-norm**, **KV cache** interface for incremental decode.
- `model/mlp.py` — SwiGLU.
- `model/layers.py` — pre-norm transformer block (norm→attn→residual, norm→mlp→residual).
- `model/transformer.py` — full model: embedding, blocks, final norm, output head (tied/untied), **vocab padding with padding rows masked from loss**, **depth-scaled init** (residual projections ×1/√(2·n_layers)), cross-entropy loss.
- `model/generation.py` — greedy, temperature, top-k, top-p; uses KV cache.
- `configs/model/lithos-toy.yaml` (5–20M params, seq 256–512).

**Key notes:** keep attention/block factored so MoE/MLA/sliding-window slot in later (§6.1 seams). Validate tensor shapes aggressively (§20.4).

**Tests** (PRD §6.3, §17.1): forward/loss shapes · causal mask blocks future tokens · RoPE correctness & shape · RMSNorm · SwiGLU · **GQA==MHA when n_kv_heads==n_heads** · **SDPA==eager** · **KV-cache==full-recompute (fixed seed)** · greedy+sampling generation · **loss ignores vocab padding**.

**Acceptance:** shape tests pass on CPU · greedy generation works · toy model instantiates and runs a forward+backward on CPU. (PRD §19 M1)

---

## Phase 2 — Tokenizer  ·  size S

**Goal:** a versioned byte-level BPE tokenizer trained from a corpus sample, with stable special tokens.

**Deliverables**
- `tokenizer/tokenizer_config.py` — config schema (vocab 32k, special tokens per §7.1, digit-splitting, byte-level pretok).
- `tokenizer/train_tokenizer.py` — trains HF `tokenizers` byte-level BPE; emits model files, tokenizer config, **training manifest** (§7.2: sources, #docs, char count, vocab size, special tokens, normalization, pretok, date, git commit), and a sample tokenization report.
- `tokenizer/inspect_tokenizer.py` — token stats / fertility report.
- `configs/tokenizer/bpe-32k.yaml`; `scripts/train_tokenizer.py`.
- Pull a small `Nemotron-CC-v2` sample (via `datasets` streaming) as training input; decontaminate against eval sets before training.

**Tests** (§7.3): roundtrip for ordinary text / code / math symbols · special-token IDs stable · unusual Unicode doesn't crash · empty string doesn't crash.

**Acceptance:** roundtrip tests pass · manifest + sample report written · special-token IDs pinned.

---

## Phase 3 — Data pipeline v0  ·  Milestone 2  ·  size L

**Goal:** turn raw Nemotron-CC slices into tokenized, packed, resumable training shards with full provenance.

**Deliverables**
- `data/documents.py` — reader for JSONL/JSONL.zst and Parquet; canonical record schema (§8.3); a thin Nemotron-CC ingestion (HF streaming → canonical docs, carrying source/synthetic/quality metadata).
- `data/filters.py` — min/max length, language, repeated-char, duplicate-line, symbol-density, whitespace-only; each filter **records counts** (§8.7, no silent deletion).
- `data/dedup.py` — exact document-hash dedup; line-level optional; **MinHash interface stubbed for later** (§8.8).
- `data/tokenize.py` — docs → token streams using the Phase 2 tokenizer.
- `data/shard.py` — binary shard writer (np.memmap `uint16`/`uint32` + `.idx`), per-shard sha256, manifest entries (§8.5).
- `data/packing.py` — fixed-length sequence packing. **Decide doc-boundary handling** (PRD §27): BOS-separated concatenation with EOS at doc ends; expose optional intra-document attention masking + RoPE position reset as a config flag, defaulting to standard GPT-style bleed for v0. Packing test covers boundaries.
- `data/dataloader.py` — packed dataloader with **resumable, deterministic ordering** (shard index + offset + sampler RNG state checkpointed; §27 / §9.9 gap). Multi-source **mixture sampling** driven by manifest weights.
- `data/manifest.py` — corpus manifest (§8.6) + shard manifests; writes CC-vs-synthetic mixture and license notes (§26.2).
- `scripts/prepare_smoke_data.py`, `scripts/tokenize_corpus.py`; `configs/data/smoke.yaml`; `corpus/recipes/smoke.yaml`.

**Tests** (§17.1): packing (incl. boundary handling) · shard write/read roundtrip · **dataloader determinism & resume-position** (same sequence before/after a simulated resume) · filter-count accounting.

**Acceptance:** can tokenize the smoke corpus · produces fixed-length batches · packing + resume tests pass. (PRD §19 M2)

---

## Phase 4 — Training loop v0 (single process)  ·  Milestone 3  ·  size L

**Goal:** an explicit, readable training loop with logging, checkpointing, and exact resume — no hidden trainer (§20.2).

**Deliverables**
- `train/optim.py` — AdamW (configurable lr/betas/eps/weight-decay/clip); **optimizer states in fp32** under bf16.
- `train/scheduler.py` — linear warmup → cosine decay → min lr.
- `train/loop.py` — the explicit loop (§9.2): construct model · load shards · batch · forward · CE loss · backward · grad clip · opt step · sched step · periodic log/eval/checkpoint · resume · graceful interrupt.
- `train/checkpoint.py` — model weights (safetensors) + optimizer/scheduler/step/tokens/RNG/**dataloader position** (torch) + resolved config + tokenizer & corpus-manifest references (§9.8). **No-clobber** (§20.7).
- `train/logging.py` — run dir (`runs/<ts>_<name>/` with `resolved_config.yaml`, `metrics.jsonl`, `samples/`, `checkpoints/`, `evals/`, `run_manifest.json`); `metrics.jsonl` fields per §9.7. Effective-batch accounting logged (§9.6).
- `train/train.py` + `scripts/train_model.py`; `configs/train/single-gpu-smoke.yaml`.
- Precision: fp32 (CPU) / bf16 (GPU); **gradient accumulation**; **gradient checkpointing flag**; **torch.compile flag** w/ eager fallback (§27).

**Tests / integration** (§17.2): **tiny model overfits tiny dataset** within documented steps · checkpoint save → resume **reproduces step/token count and data position** · metrics written to JSONL · run-dir no-clobber.

**Acceptance:** tiny overfits · resume reproduces expected step count · metrics in JSONL. (PRD §19 M3, §9.10 1–3,5,7)

---

## Phase 5 — Evaluation v0  ·  Milestone 5  ·  size M

**Goal:** internal perplexity + sample generation + the external-harness path, with versioned reports.

**Deliverables**
- `evals/perplexity.py` — validation perplexity; **define the held-out val set** (fixed, decontaminated, constructed from held-out shards; §27 gap); loss-by-source when available (§11.1).
- `evals/generate_samples.py` — fixed prompt set → `samples/`; repetition checks.
- **Core HF/Qwen3 exporter** (`serve/export.py`, brought forward from P7) — minimal export so lm-evaluation-harness runs via `--model hf` with no bespoke wrapper (`vllm` backend available for speed) (§11.2, §26.8). External tasks: hellaswag, arc_easy/challenge, piqa, winogrande, lambada_openai. At smoke scale this validates plumbing only; real numbers come at 100M (P6).
- `evals/report.py` — writes `results.json` + `results.md` + `config.yaml` + `model_reference.json` per eval run (§11.3); md includes model/checkpoint/tokenizer/corpus/benchmark versions/scores/caveats.
- `evals/tasks/`; `configs/eval/base.yaml`, `configs/eval/lm-eval.yaml`; `scripts/run_evals.py`.
- Wire periodic in-loop val-loss into Phase 4 (in-loop ≠ offline harness).

**Tests:** perplexity numerically sane on a tiny fixture · report files written · adapter returns scores on a 1-step dummy.

**Acceptance:** val perplexity reported · ≥3 external tasks run · results saved as JSON+MD · eval config versioned. (PRD §19 M5)

> **GATE — Walking skeleton green.** At this point the full §23 command sequence runs end-to-end at smoke scale (prepare → train tokenizer → tokenize → train → eval → generate) on the local 4070. CI is green (ruff + unit/shape tests, CPU). Only now do we scale to cloud.

---

## Phase 6 — Lithos 100M end-to-end  ·  Milestone 4  ·  size L

**Goal:** first real training run — the integration test of everything above.

**Deliverables**
- `configs/model/lithos-100m.yaml`, `configs/train/100m.yaml`, `configs/data/corpus-v0.1.yaml`, `corpus/recipes/lithos-v0.1.yaml`.
- Real corpus build: Nemotron-CC-v2 web slices + Nemotron-CC-Math-v1, decontaminated against §11 benchmarks; corpus manifest with CC/synthetic/math mixture + license notes.
- Final tokenizer v0.
- Validate the config with a short local 4070 sanity run (hundreds of steps), then the full training run targeting ~10–20B tokens (over-train, §26.3) on a **cloud** GPU; eval report; **model card draft** (`model_cards/lithos-100m-base.md`, §14).

**Acceptance:** 100M trains stably ≥1,000 steps without loss explosion · val perplexity reported · samples saved · model card exists. (PRD §19 M4, §9.10)

---

## Phase 7 — Inference, export & serving  ·  size M

**Goal:** usable local generation and a HF/Qwen3-compatible export that every downstream engine consumes (§26.8). The *core* exporter ships earlier in Phase 5 (to enable `--model hf` eval); this phase is full packaging + serving.

**Deliverables**
- `serve/generate.py` (CLI per §13.1 flags, uses the in-repo KV-cache generator) + `scripts/generate.py`.
- `serve/api.py` — local-only FastAPI `/generate` (unhardened warning) (§13.2).
- `serve/export.py` + `scripts/export_checkpoint.py` — **HF `transformers`-loadable, Qwen3-architecture export** `artifacts/<name>/` (Qwen3Config-compatible config.json, HF-named model.safetensors, tokenizer.json, generation_config.json, model_card.md) (§13.3); publish to HF Hub (§26.5).
- **vLLM serving:** documented command path (`vllm serve <export>`), not built (§3.7); smoke-tested on cloud once an export exists.
- `serve/quantize.py` — stub + `convert_hf_to_gguf.py` llama.cpp/GGUF docs (§13.4, deferred).

**Tests** (§17.2): generate from checkpoint · **exported checkpoint loads in `transformers` and reproduces the in-repo generator's greedy output (fixed seed)** · round-trips through HF Hub.

**Acceptance:** generation works from an exported artifact · export loads drop-in in `transformers` (and therefore vLLM / llama.cpp).

---

## Phase 8 — Distributed / FSDP  ·  Milestone 6  ·  size L

**Goal:** multi-GPU training via torchrun + FSDP, with distributed safety.

> **Note:** the local 4070 is single-GPU — this phase cannot be smoke-tested locally. FSDP/multi-GPU smoke runs on a cloud 2×/8× node.

**Deliverables**
- `train/distributed.py` — torchrun init, rank-aware logging (**rank 0 writes** human-readable + manifests), DDP path then FSDP with transformer-block auto-wrap, mixed precision, sharded checkpoint (or documented single-rank fallback) (§10.2).
- Distributed safety (§10.3): no two ranks writing the same artifact, config-hash consistency check across ranks, fail-fast if a rank dies, eval not duplicated across ranks.
- Multi-GPU smoke (toy/100M); `configs/train/` multi-GPU variants.

**Acceptance:** multi-GPU toy/100M run completes · logs uncorrupted · resume works or limitation documented. (PRD §19 M6)

---

## Phase 9 — Lithos 300M  ·  Milestone 7  ·  size M

GQA enabled (`n_kv_heads < n_heads`); larger corpus recipe; ~30–60B tokens; training run; eval report **comparing to 100M**; model card. **Acceptance:** trains stably · 300M-vs-100M comparison · weaknesses documented. (PRD §19 M7)

---

## Phase 10 — SFT v0  ·  Milestone 8  ·  size M

**Goal:** instruction-following variant on top of a base checkpoint.

**Deliverables**
- Versioned Lithos **chat template** (§12.2) — versioned because it changes behavior.
- `posttrain/datasets.py` (messages format §12.1), `posttrain/collator.py` (**loss masking for non-assistant tokens**, conversation packing), `posttrain/sft.py` (reuses pretrain loop, base-checkpoint reference, SFT eval prompts, SFT run manifest) (§12.3).
- `configs/posttrain/sft-smoke.yaml`; instruct model card.

**Acceptance:** SFT run completes · follows simple instructions better than base · chat template versioned. (PRD §19 M8)

---

## Phase 11 — Lithos 1B  ·  Milestone 9  ·  size L

1B config; FSDP training config; ~50B-token plan (over-train); full run manifest (§15); eval report; model card documenting data/hardware/limitations. **Acceptance:** stable 1B run · resume works · val loss + benchmarks reported · complete model card. (PRD §19 M9)

---

## Cross-cutting (continuous)

- **Docs** (§18): write `docs/{architecture,tokenizer,corpus,pretraining,distributed,evaluation,posttraining,inference,runbooks}.md` as each phase lands — not at the end.
- **Reproducibility** (§15): every real run emits the full run manifest (git commit, resolved config, tokenizer/corpus versions, shard checksums, seed, hardware, lockfile, command).
- **Model cards** (§14): drafted with each model milestone; synthetic-data + code-gap disclosures mandatory (§26.2).
- **Quality gates** before merging major changes: `ruff check .`, `pytest`, optional `mypy lithos` (§17.3), enforced in CI.

## Critical path & parallelization

```
P0 ─▶ P1 ─▶ P3 ─▶ P4 ─▶ P5 ─▶ [SKELETON GATE] ─▶ P6 ─▶ P7
        └▶ P2 ─┘                                   └▶ P8 ─▶ P9 ─▶ P11
                                                            └▶ P10
```
- **P2 (tokenizer)** can run in parallel with **P1 (model)** — they meet at **P3 (data)**.
- **P7 (inference/export)** can begin as soon as **P4** produces a checkpoint; doesn't block P8.
- **P8 (FSDP)** gates the larger models (P9, P11) but not P10 (SFT can start from any base checkpoint, single-GPU at 100M/300M).
- Hard gate: do not begin **P6** (first real run) until the **skeleton gate** after P5 is green.

## Top risks & mitigations

| Risk | Mitigation |
|---|---|
| Loss spikes / instability at scale | QK-norm default-on ≥100M; grad clipping; documented skip-batch/rollback stance (§27); bf16 w/ fp32 optimizer states. |
| Resume silently re-sees data | Dataloader position checkpointed and tested in P3/P4 (§27). |
| Benchmark contamination inflates evals | Decontaminate corpus + val set against §11 tasks before P6 (§8.9). |
| Synthetic-data provenance not disclosed | Mixture recorded in corpus manifest; mandatory model-card disclosure (§26.2). |
| No code data → weak code ability | Documented v0.1 gap; The Stack/StarCoder blend deferred, not silently skipped. |
| Distributed complexity introduced too early | FSDP is Phase 8, after 100M already trains single-GPU (§20.13–14). |
| Determinism over-promised | Scope stated: bitwise CPU, best-effort GPU, not under FSDP (§26.6). |
| Modern refinements break engine compatibility | Export targets the **Qwen3 envelope** (GQA + QK-norm), drop-in to HF/vLLM/llama.cpp; stepping outside it (MLA, sliding-window) needs custom modeling code — a documented, conscious cost (§26.8). |

## Definition of done — v0 (maps to PRD §22)

Achieved when Phases 0–7 are complete: clone+install · toy trains locally · tokenizer trains from a small corpus · corpus tokenizes to shards · 100M trains on one GPU · metrics logged · checkpoints resume · perplexity eval works · generation works · artifact exports · model card produced · tests pass · docs explain the workflow. Phases 8–11 extend v0 toward the full ladder.

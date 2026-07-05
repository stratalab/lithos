# Changelog

All notable changes to Lithos are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

## [Unreleased]

## [0.1.0] — 2026-07-05

First tagged cut. The full model lifecycle is built and validated end-to-end at the
100M shakedown scale.

### Added
- **Model** — modernized-Llama / Qwen3-envelope decoder (RoPE, RMSNorm, SwiGLU, GQA,
  QK-norm) with KV-cache generation.
- **Tokenizer** — owned 32k BPE + evaluation/probe harness.
- **Data pipeline** — canonical document schema + readers; extractors for HTML/CNXML
  (EX-1), PDF via Docling (EX-2), StackExchange dumps, OpenStax, and The-Stack code
  (EX-6); MinHash dedup, decontamination, an owned quality classifier, packing, sharding.
- **Training** — loop, optimizer, scheduler, checkpointing, distributed, JSONL metrics,
  run manifests, optional W&B mirror.
- **Evaluation** — bpb/perplexity, a frozen benchmark battery, ablation, append-only
  scorecards.
- **Post-training** — packed SFT (dual-stream shards), verifier-labeled DPO, and
  GRPO/RLVR with tool-integrated-reasoning rollouts (E1–E8), plus the E1 verifier/sandbox.
- **Serving** — generation + HF/Qwen3 export **and** import (bit-exact logit parity),
  de-risking the Qwen3-4B continued-pretrain hero track.
- **Infra** — pluggable object-store artifact storage (local / S3 / R2 / GCS) and the
  acquisition driver.

### Repo
- Made the repository production-grade: accurate README, `docs/` index, `LICENSE`,
  `CONTRIBUTING`, `Makefile`, pre-commit; CI now gates ruff **and mypy** (both green).

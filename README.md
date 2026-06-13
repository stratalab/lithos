# Lithos

Lithos is a foundation-model engineering project in the Strata ecosystem: a from-scratch
family of small, general-purpose language models built to **own and understand the full
model lifecycle** — tokenizer, corpus, pretraining, evaluation, post-training, inference,
and reproducible artifacts. It is explicitly *not* a frontier-scale effort.

The architecture is a **modernized Llama** decoder-only transformer (RoPE, RMSNorm, SwiGLU,
GQA-native, optional QK-norm, KV cache), trained across a size ladder: toy → 100M → 300M → 1B.

- **Requirements:** [`lithos-prd.md`](lithos-prd.md)
- **Build plan:** [`lithos-implementation-plan.md`](lithos-implementation-plan.md)

## Status

**Phase 0 — repo & tooling skeleton.** Package layout, the configuration system, core
utilities (seeding, device/dtype, atomic I/O, run-dir + JSONL logging), and lint/test/CI are
in place. The model, data pipeline, training loop, evaluation, and inference land in later
phases (see the build plan).

## Requirements

- Python ≥ 3.11 (developed on 3.12)
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
# Install uv: https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the environment and install dependencies.
# On Linux with an NVIDIA GPU, the default torch wheel is CUDA-enabled.
uv sync

# CPU-only environment (CI, or a machine without a GPU):
UV_TORCH_BACKEND=cpu uv sync
```

## Quality gates

```bash
uv run ruff check .
uv run pytest
uv run mypy lithos   # optional
```

## Repository layout

```text
lithos/            # the Python package
  model/           # modernized-Llama transformer (Phase 1)
  tokenizer/       # byte-level BPE 32k tokenizer (Phase 2)
  data/            # documents → filters → tokenized shards → packed loader (Phase 3)
  train/           # optimizer, scheduler, loop, checkpointing, logging (Phases 3–4)
  evals/           # perplexity, samples, lm-eval-harness path (Phase 5)
  posttrain/       # supervised fine-tuning (Phase 10)
  serve/           # generation, FastAPI, HF/Qwen3 export (Phase 7)
  utils/           # config, seed, device, io, checks
configs/           # YAML configs (model / tokenizer / data / train / eval / posttrain)
corpus/            # corpus recipes and manifests
scripts/           # runnable entrypoints
tests/             # unit + integration tests
docs/              # architecture, corpus, training, evaluation, ...
model_cards/       # per-model cards
runs/              # training run outputs (git-ignored)
```

## License

Apache-2.0. Training-data licensing and provenance (incl. the Nemotron-CC synthetic-data
disclosure) are tracked in the PRD (§14, §26.2).

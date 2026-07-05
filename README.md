# Lithos

**Lithos is a from-scratch family of small, general-purpose language models** — the
model foundry of the [Strata](docs/chisel.md) ecosystem. It exists to *own and
understand the full model lifecycle*: tokenizer, corpus, pretraining, evaluation,
post-training (incl. RL), inference, and reproducible artifacts. It is explicitly
**not** a frontier-scale effort — the bet is a compact, cross-domain **STEM reasoner**
(code · math · physics · engineering) that a small team can build, audit, and run on
the edge.

The architecture is a **modernized-Llama / Qwen3-envelope** decoder-only transformer —
RoPE, RMSNorm, SwiGLU, GQA-native, QK-norm, KV cache — trained across a size ladder:

> **100M** (validated) → **500M** flagship → **1B** → **3B** (Qwen3-4B continued-pretrain hero)

The from-scratch models and the Qwen-lineage hero share **one deployment recipe** — the
same train loop, generation, and RLVR driver — verified by bit-exact Qwen3 weight import.

## Status

The **full lifecycle is built and validated end-to-end at 100M scale**, not a skeleton:

- **Model** — the transformer + KV-cache generation.
- **Tokenizer** — an owned 32k BPE + evaluation harness.
- **Data pipeline** — canonical document schema, extractors (HTML/CNXML, PDF/Docling,
  StackExchange, OpenStax, The-Stack code), dedup, decontam, quality classifier,
  packing, sharding.
- **Training** — loop, optimizer, scheduler, checkpointing, distributed, tracking.
- **Evaluation** — bpb/perplexity, a frozen benchmark battery, ablation, scorecards.
- **Post-training** — packed SFT, verifier-labeled DPO, and **GRPO/RLVR with
  tool-integrated-reasoning** (reason → run code in a sandbox → use the result), all
  exercised end-to-end.
- **Serving** — generation + HF/Qwen3 export (bit-exact parity).

~400 tests, green on `ruff` + `mypy` + `pytest`.

## Setup

```bash
# Install uv: https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

# Full dev environment (adds the data/eval extras; on Linux+NVIDIA torch is CUDA-enabled).
make install                    # == uv sync --extra data --extra eval
uv sync --all-extras            # everything, incl. pdf (docling) and serve

# CPU-only (a machine without a GPU):
UV_TORCH_BACKEND=cpu uv sync
```

## Quickstart

```bash
uv run lithos tokenizer --config configs/tokenizer/bpe-32k.yaml
uv run lithos train     --config configs/train/100m.yaml
uv run lithos eval      --config configs/eval/lithos-100m.yaml --checkpoint <ckpt>
uv run lithos sft       --config configs/sft/lithos-100m-packed.yaml
```

`lithos <command>` is the unified entrypoint (`lithos --help` lists them). Distributed
runs still launch the shim directly: `torchrun --nproc_per_node=N scripts/train_model.py …`.

## Storage & secrets

Durable artifacts (shards, checkpoints, exports) live in an object store configured by
`configs/storage.yaml` (local by default). For a cloud bucket, copy `.env.example` to
`.env` (git-ignored) and set the R2/S3 credentials + `LITHOS_STORAGE_BASE_URI`; the
storage layer loads `.env` automatically. Move data with `python scripts/sync.py`. The
end-to-end storage tiering (local HDD → NVMe → R2) is in
[`docs/chisel-lithos-r2-contract.md`](docs/chisel-lithos-r2-contract.md).

## Quality gates

```bash
make check      # ruff + mypy + pytest   (or: make lint / typecheck / test)
```

## Repository layout

```text
lithos/            # the Python package
  model/           # modernized-Llama / Qwen3-envelope transformer + generation
  tokenizer/       # owned 32k BPE tokenizer + eval harness
  data/            # documents → extract → filter → dedup → tokenize → packed shards
  train/           # optimizer, scheduler, loop, checkpointing, distributed, logging
  evals/           # bpb/perplexity, benchmark battery, ablation, scorecards
  posttrain/       # SFT, DPO, GRPO/RLVR, TIR rollout, verifier, sandbox
  serve/           # generation, HF/Qwen3 export/import
  utils/           # config, seed, device, io, storage, checks
configs/           # YAML configs (model / tokenizer / data / train / eval / sft / dpo / grpo)
corpus/            # the Canon (seed_index.csv) + acquisition specs + task banks
scripts/           # runnable entrypoints
tests/             # unit + integration tests
docs/              # see docs/README.md for the index
runs/              # training run outputs (git-ignored)
```

## Documentation

See [`docs/README.md`](docs/README.md) for the full index — the PRD, the build plan,
architecture/design notes, the post-training specs, and the Strata-ecosystem handoffs
(Chisel, the R2 contract, Moho).

## License

[Apache-2.0](LICENSE). Training-data licensing and provenance are tracked in the PRD
([`docs/prd.md`](docs/prd.md)) and the Canon (`corpus/seed_index.csv`).

# Lithos Requirements Document

## 1. Project Summary

Lithos is a foundation model engineering project inside the Strata ecosystem.

The goal is to build a family of small general-purpose language models from scratch, starting with toy and small-scale runs, then progressing to 100M, 300M, and 1B parameter models. Lithos should include the full model lifecycle: tokenizer training, corpus preparation, pretraining, evaluation, post-training, inference, model cards, and reproducibility metadata.

Lithos is not intended to compete with frontier models. The purpose is to own and understand the foundation model stack end to end.

## 2. Core Goals

Lithos must enable the following:

1. Train a decoder-only transformer language model from scratch.
2. Support multiple model sizes: toy, 100M, 300M, and 1B.
3. Build and version tokenizers.
4. Prepare, filter, tokenize, and shard pretraining datasets.
5. Train models on single-GPU and multi-GPU setups.
6. Evaluate base models using perplexity and standard LM benchmarks.
7. Post-train base models into instruction-following variants.
8. Serve models locally for text generation.
9. Produce reproducible artifacts: configs, manifests, logs, checkpoints, eval reports, and model cards.
10. Keep the codebase readable enough that a human can understand the full pipeline.

## 3. Non-Goals

The initial version must not attempt to:

1. Train a frontier-scale model.
2. Build a custom deep learning framework.
3. Build custom CUDA kernels.
4. Implement every distributed training strategy.
5. Create a web crawler from scratch.
6. Store large raw datasets or large checkpoints directly in Git.
7. Build a polished hosted inference product.
8. Optimize for maximum benchmark performance before correctness and reproducibility.

## 4. Technical Stack

### 4.1 Language

Use Python for all training, evaluation, post-training, and initial serving code.

Use Rust only later for performance-sensitive data tooling or Strata runtime integration. Do not use Rust for initial model training.

### 4.2 Core Libraries

Required:

* Python 3.11+
* PyTorch
* NumPy
* safetensors
* tokenizers and/or sentencepiece
* datasets and huggingface_hub, required (corpus is pulled from HF Hub; see §26.2)
* pyarrow and pandas or polars for corpus processing
* tqdm or rich for progress output
* pydantic or dataclasses for config validation
* pytest
* ruff
* mypy or pyright, optional but preferred

Distributed training:

* torch.distributed
* PyTorch FSDP
* torchrun entrypoints

Evaluation:

* internal perplexity evaluator
* integration path for EleutherAI lm-evaluation-harness

Experiment tracking:

* local JSONL logs required
* Weights & Biases or MLflow optional
* all runs must be reproducible without a hosted tracking service

Configuration:

* YAML config files
* CLI override support
* every run must save its resolved config

## 5. Repository Structure

The initial repo should be a single repository named `lithos`.

```text
lithos/
  README.md
  REQUIREMENTS.md
  pyproject.toml
  uv.lock
  .gitignore
  .python-version

  lithos/
    __init__.py

    model/
      __init__.py
      config.py
      transformer.py
      layers.py
      attention.py
      mlp.py
      norm.py
      rope.py
      generation.py

    tokenizer/
      __init__.py
      train_tokenizer.py
      inspect_tokenizer.py
      tokenizer_config.py

    data/
      __init__.py
      documents.py
      filters.py
      dedup.py
      tokenize.py
      shard.py
      packing.py
      dataloader.py
      manifest.py

    train/
      __init__.py
      train.py
      loop.py
      optim.py
      scheduler.py
      checkpoint.py
      distributed.py
      logging.py

    evals/
      __init__.py
      perplexity.py
      generate_samples.py
      lm_eval_adapter.py
      report.py
      tasks/

    posttrain/
      __init__.py
      sft.py
      datasets.py
      collator.py

    serve/
      __init__.py
      generate.py
      api.py
      export.py
      quantize.py

    utils/
      __init__.py
      config.py
      seed.py
      device.py
      io.py
      checks.py

  configs/
    model/
      lithos-toy.yaml
      lithos-100m.yaml
      lithos-300m.yaml
      lithos-1b.yaml
    tokenizer/
      bpe-32k.yaml
    data/
      smoke.yaml
      corpus-v0.1.yaml
    train/
      single-gpu-smoke.yaml
      100m.yaml
      300m.yaml
      1b.yaml
    eval/
      base.yaml
      lm-eval.yaml
    posttrain/
      sft-smoke.yaml

  corpus/
    recipes/
      smoke.yaml
      lithos-v0.1.yaml
    manifests/
      README.md

  scripts/
    prepare_smoke_data.py
    train_tokenizer.py
    tokenize_corpus.py
    train_model.py
    run_evals.py
    generate.py
    export_checkpoint.py

  tests/
    test_model_shapes.py
    test_attention_mask.py
    test_rope.py
    test_tokenizer_roundtrip.py
    test_packing.py
    test_checkpoint_roundtrip.py
    test_generation.py

  docs/
    architecture.md
    corpus.md
    tokenizer.md
    pretraining.md
    evaluation.md
    posttraining.md
    inference.md
    runbooks.md

  model_cards/
    lithos-100m-base.md
    lithos-300m-base.md
    lithos-1b-base.md

  runs/
    README.md
```

## 6. Model Requirements

### 6.1 Architecture

Implement a Llama-style ("modernized Llama") decoder-only transformer.

The Llama backbone — decoder-only, RoPE, pre-norm RMSNorm, SwiGLU, GQA — remains the substrate of essentially all current dense models and the dense layers of MoE models; it has been *extended* (MoE, MLA, sliding-window), not replaced. v0 adopts that backbone plus a few cheap, stability-oriented refinements, and leaves clean seams for later additions without committing to them now.

Required features:

1. Token embedding.
2. Decoder-only causal self-attention.
3. Grouped-query attention (GQA) as a first-class code path. `n_kv_heads` is always configurable: set `n_kv_heads == n_heads` for plain multi-head attention (toy/100M default) and `n_kv_heads < n_heads` for GQA (300M/1B). The model is written around `n_kv_heads` from day one, so enabling GQA is a config change, not a rewrite.
4. Rotary positional embeddings (RoPE) with configurable base/theta, so long-context extension (e.g. YaRN-style scaling) is possible later without changing model code.
5. Optional QK-normalization (RMSNorm on per-head queries and keys before attention), config-gated and default-on for ≥100M runs, to suppress loss spikes at small scale.
6. RMSNorm (pre-norm), with configurable eps.
7. SwiGLU MLP, with configurable intermediate size (default ≈ 8/3 · hidden, rounded to a multiple of 256).
8. Residual connections.
9. Attention backend abstraction: default to `torch.nn.functional.scaled_dot_product_attention` (FlashAttention-class kernels for free) with an eager, materialized-mask fallback for debugging and CPU. Both paths must produce identical outputs in tests.
10. KV cache for incremental decoding, used by generation. A test must assert cached and uncached generation produce identical tokens for a fixed seed.
11. Tied or untied output embeddings, configurable.
12. Configurable vocabulary size; embedding/output matrices padded up to a multiple of 128 for tensor-core efficiency, with padding rows masked out of the loss.
13. Configurable sequence length.
14. Configurable number of layers, hidden size, and attention heads.
15. Dropout, configurable but default 0 for large pretraining runs.
16. Weight initialization documented and deterministic: normal(0, 0.02) for embeddings and linear layers, with residual output projections additionally scaled by 1/sqrt(2 · n_layers) (GPT-2/Llama-style depth scaling); norm weights initialized to 1. The scheme is recorded in the model card and resolved config.

**Explicitly deferred (design seams, not v0 features):** MoE / sparse FFN, MLA (latent KV compression), sliding-window / local-global attention, and attention sinks. The attention and transformer-block modules must be factored so these slot in as opt-in variants later rather than requiring a rewrite. MoE is the one fork worth a conscious scope decision if the goal shifts toward understanding frontier scaling rather than the dense pipeline.

This v0 feature set (GQA + QK-norm + RoPE + SwiGLU + RMSNorm) sits inside the **Qwen3 export envelope** (§26.8), so checkpoints export to HF/vLLM/llama.cpp by weight-renaming; the deferred features step outside it and would require custom modeling code.

### 6.2 Model Configs

Provide configs for:

#### Lithos Toy

Purpose: local correctness and debugging.

Approximate target:

* Parameters: 5M to 20M
* Context length: 256 or 512
* Vocabulary: small tokenizer or 8K to 32K
* Runs on CPU or one consumer GPU

#### Lithos 100M

Purpose: first real training dynamics model.

Approximate target:

* Parameters: around 100M
* Context length: 1024 or 2048
* Vocabulary: 32K
* Runs on one GPU

#### Lithos 300M

Purpose: serious intermediate model.

Approximate target:

* Parameters: around 300M
* Context length: 2048
* Vocabulary: 32K to 50K
* Runs on one or multiple GPUs

#### Lithos 1B

Purpose: first serious foundation model.

Approximate target:

* Parameters: around 1B
* Context length: 2048 initially
* Vocabulary: 32K to 50K
* Runs on multi-GPU setup using FSDP

### 6.3 Shape and Correctness Tests

The codebase must include tests that verify:

1. Forward pass shape correctness.
2. Loss computation shape correctness.
3. Attention mask prevents future-token leakage.
4. RoPE applies correctly and does not change tensor shape unexpectedly.
5. Generation works with greedy decoding and sampling.
6. Checkpoint save/load preserves outputs for a fixed seed.
7. A tiny model can overfit a tiny dataset.
8. GQA (`n_kv_heads < n_heads`) matches MHA reference behavior when `n_kv_heads == n_heads`.
9. SDPA backend and eager fallback produce identical outputs.
10. KV-cache (incremental) and full-recompute generation produce identical tokens for a fixed seed.
11. Loss correctly ignores vocab-padding rows.

Acceptance criterion:

* `pytest` must pass on CPU for shape tests.
* A tiny model must overfit a tiny dataset within a documented number of steps.

## 7. Tokenizer Requirements

### 7.1 Tokenizer Goals

The tokenizer must support general-purpose English-heavy text with code and math exposure.

Initial tokenizer target:

* Byte-level BPE (decided; see §26.4)
* Vocabulary size: 32K
* Special tokens:

  * `<unk>` if required by tokenizer type
  * `<bos>`
  * `<eos>`
  * `<pad>`
  * `<|user|>`
  * `<|assistant|>`
  * `<|system|>`
  * `<|end|>`

### 7.2 Tokenizer Training

The repo must include a tokenizer training script.

Inputs:

* JSONL.zst or plain JSONL documents
* Config file
* Output directory

Outputs:

* tokenizer model files
* tokenizer config
* tokenizer training manifest
* sample tokenization report

The tokenizer manifest must include:

1. Training data sources.
2. Number of documents sampled.
3. Approximate character count.
4. Vocabulary size.
5. Special tokens.
6. Normalization settings.
7. Pre-tokenization settings.
8. Date created.
9. Git commit hash if available.

### 7.3 Tokenizer Tests

Tests must verify:

1. Encode/decode roundtrip for ordinary text.
2. Encode/decode roundtrip for code.
3. Encode/decode roundtrip for math symbols.
4. Special token IDs are stable.
5. Unknown or unusual Unicode does not crash.
6. Empty strings do not crash.

## 8. Corpus and Data Requirements

### 8.1 Data Philosophy

Lithos should not begin by crawling the web. It should assemble and curate a reproducible corpus from open datasets.

The data system must preserve provenance and allow the same corpus version to be recreated.

The v0.1 corpus source is fixed: slices of `nvidia/Nemotron-CC-v2` (plus `Nemotron-CC-Math-v1` for math). See §26.2 for source, license, synthetic-data provenance, and the code gap.

### 8.2 Data Stages

Support the following stages:

```text
raw source
  -> extracted documents
  -> cleaned documents
  -> filtered documents
  -> tokenized shards
  -> packed training sequences
```

### 8.3 Document Format

The canonical intermediate document format is JSONL or JSONL.zst.

Each record should support:

```json
{
  "id": "string",
  "text": "string",
  "source": "string",
  "subset": "string",
  "language": "en",
  "license": "string_or_unknown",
  "metadata": {}
}
```

### 8.4 Curated Corpus Format

For larger corpora, support Parquet as a curated document format.

Parquet schema should include at least:

* id
* text
* source
* subset
* language
* license
* quality_score, optional
* dedup_hash, optional
* metadata, optional

### 8.5 Tokenized Shard Format

The trainer must consume pre-tokenized binary shards.

Initial acceptable formats:

1. NumPy memory-mapped arrays.
2. `.bin/.idx` style shards.
3. safetensors shards, optional.

Each tokenized shard must have a manifest entry:

```json
{
  "shard_id": "shard_000001",
  "path": "tokenized/shard_000001.bin",
  "num_tokens": 123456789,
  "dtype": "uint16_or_uint32",
  "tokenizer": "lithos-bpe-32k-v0.1",
  "source_mixture": {},
  "sha256": "..."
}
```

### 8.6 Corpus Manifest

Every corpus version must include a manifest.

Required fields:

```json
{
  "corpus_name": "lithos-general-corpus",
  "version": "v0.1",
  "created_at": "YYYY-MM-DD",
  "tokenizer": "tokenizer-name-version",
  "num_documents": 0,
  "num_tokens": 0,
  "sources": [],
  "mixture": {},
  "filters": [],
  "dedup": {},
  "decontamination": {},
  "license_notes": [],
  "shards": []
}
```

### 8.7 Data Filtering

Initial filters:

1. Minimum text length.
2. Maximum text length.
3. Language filter.
4. Remove documents with excessive repeated characters.
5. Remove documents with excessive duplicate lines.
6. Remove documents with extreme symbol density.
7. Remove empty or whitespace-only documents.
8. Optional profanity or unsafe-content flags, but do not silently delete without recording filter behavior.

### 8.8 Deduplication

Initial dedup requirements:

1. Exact document hash deduplication.
2. Exact line-level deduplication, optional.
3. Near-dedup with MinHash, optional for v0 but design should allow it.

### 8.9 Benchmark Decontamination

For serious runs, include a decontamination report against benchmark datasets used in evaluation.

Initial support may be simple n-gram overlap detection.

The report must include:

1. Benchmarks checked.
2. Method used.
3. Thresholds.
4. Number of documents flagged.
5. Whether flagged documents were removed or only reported.

## 9. Training Requirements

### 9.1 Training Modes

Support:

1. CPU smoke training.
2. Single-GPU training.
3. Multi-GPU distributed training with torchrun.
4. FSDP-based training for larger models.

### 9.2 Training Loop

The training loop must explicitly implement:

1. Model construction from config.
2. Dataset loading from tokenized shards.
3. Batch creation.
4. Forward pass.
5. Cross-entropy loss.
6. Backward pass.
7. Gradient clipping.
8. Optimizer step.
9. Learning rate scheduler step.
10. Periodic logging.
11. Periodic evaluation.
12. Periodic checkpointing.
13. Resume from checkpoint.
14. Graceful handling of interruption where possible.

### 9.3 Optimizer

Initial optimizer:

* AdamW

Configurable:

* learning rate
* betas
* epsilon
* weight decay
* gradient clipping norm

### 9.4 Scheduler

Support:

1. Linear warmup.
2. Cosine decay.
3. Minimum learning rate.

### 9.5 Precision

Support:

1. fp32 for CPU tests.
2. bf16 for serious GPU training.
3. fp16 optional.

### 9.6 Gradient Accumulation

Must support gradient accumulation to achieve larger effective batch sizes.

The resolved run config must log:

```text
micro_batch_size
gradient_accumulation_steps
world_size
global_batch_size
sequence_length
tokens_per_step
```

### 9.7 Logging

Every training run must create a run directory:

```text
runs/
  2026-06-13_120000_lithos-100m/
    resolved_config.yaml
    metrics.jsonl
    samples/
    checkpoints/
    evals/
    run_manifest.json
```

`metrics.jsonl` must log:

1. step
2. tokens_seen
3. train_loss
4. learning_rate
5. grad_norm
6. throughput_tokens_per_sec
7. gpu_memory_allocated, if available
8. validation_loss, when evaluated
9. timestamp

### 9.8 Checkpointing

Checkpoints must include:

1. model weights
2. optimizer state
3. scheduler state
4. training step
5. tokens seen
6. random number generator state where practical
7. resolved config
8. tokenizer reference
9. corpus manifest reference

Checkpoint formats:

* safetensors for model weights where practical
* PyTorch checkpoint for optimizer/scheduler/training state

### 9.9 Resume Requirements

A resumed run must:

1. Load model state.
2. Load optimizer state.
3. Load scheduler state.
4. Continue step count.
5. Continue token count.
6. Preserve learning rate schedule.
7. Not overwrite previous logs unless explicitly configured.

### 9.10 Training Acceptance Criteria

Before training any 300M+ model, the system must satisfy:

1. Tiny model overfits tiny data.
2. 100M model runs for at least 1,000 steps without loss explosion.
3. Checkpoint resume works.
4. Validation loss is computed.
5. Throughput is logged.
6. Generated samples are saved.
7. Training can be stopped and resumed.

## 10. Distributed Training Requirements

### 10.1 Launch

Distributed training must launch with `torchrun`.

Example:

```bash
torchrun --nproc_per_node=8 scripts/train_model.py --config configs/train/1b.yaml
```

### 10.2 FSDP

The 1B model path must support PyTorch FSDP.

Required:

1. Config flag to enable FSDP.
2. Auto-wrapping policy for transformer blocks.
3. Mixed precision support.
4. Sharded checkpoint support or documented fallback.
5. Rank-aware logging.
6. Only rank 0 writes human-readable summary logs unless configured otherwise.

### 10.3 Distributed Safety

The code must prevent:

1. All ranks writing the same artifact simultaneously.
2. Silent mismatch in configs across ranks.
3. Training continuing if one rank fails.
4. Evaluation being accidentally run redundantly on every rank unless intended.

## 11. Evaluation Requirements

### 11.1 Internal Evaluations

Implement internal evaluation for:

1. Validation perplexity.
2. Loss by source, if source information is available.
3. Generated sample prompts.
4. Repetition checks.
5. Basic instruction prompt smoke tests after SFT.

### 11.2 External Evaluation Harness

Because checkpoints export HF/Qwen3-compatible (§26.8), the EleutherAI lm-evaluation-harness runs directly via `--model hf` — no bespoke adapter needed (the `vllm` backend is available for faster eval). Document the command path.

Initial target tasks:

1. hellaswag
2. arc_easy
3. arc_challenge
4. piqa
5. winogrande
6. lambada_openai
7. mmlu subsets, optional for 300M+
8. gsm8k, optional and expected to be weak for small models

### 11.3 Eval Reports

Every eval run must write:

```text
evals/
  eval-name/
    results.json
    results.md
    config.yaml
    model_reference.json
```

The markdown report should include:

1. model name
2. checkpoint path
3. tokenizer
4. corpus version
5. benchmark versions
6. scores
7. notes and caveats

### 11.4 Evaluation Acceptance Criteria

Before declaring a model milestone complete:

1. Validation perplexity must be reported.
2. At least one standard external benchmark suite must be run.
3. A fixed prompt sample set must be generated.
4. A model card draft must be created.
5. Known weaknesses must be documented.

## 12. Post-Training Requirements

### 12.1 Initial SFT

Implement supervised fine-tuning.

Supported input format:

```json
{
  "messages": [
    {"role": "system", "content": "string"},
    {"role": "user", "content": "string"},
    {"role": "assistant", "content": "string"}
  ]
}
```

### 12.2 Chat Template

Define a Lithos chat template.

Example:

```text
<|system|>
{system}
<|end|>
<|user|>
{user}
<|end|>
<|assistant|>
{assistant}
<|end|>
```

The template must be versioned because changing the template changes model behavior.

### 12.3 SFT Training

The SFT loop may reuse the pretraining loop but must support:

1. Loss masking for non-assistant tokens.
2. Conversation packing.
3. SFT-specific eval prompts.
4. SFT run manifests.
5. Base checkpoint reference.

### 12.4 Preference Tuning

Preference tuning is optional for v0.

Possible future methods:

1. DPO.
2. ORPO.
3. GRPO-style experiments.

Do not implement preference tuning until SFT is stable and evaluated.

## 13. Inference Requirements

**Strategy (see §26.8):** we build and own the in-repo PyTorch generator; all external engines consume one **HF/Qwen3-compatible export**. HuggingFace is the interop + eval hub, vLLM is the documented cloud serving path, llama.cpp/GGUF is the deferred portable-distribution path. v0 builds the generator + a local FastAPI only.

### 13.1 Local Generation

Implement a local generation script.

Required decoding methods:

1. greedy
2. temperature sampling
3. top-k sampling
4. top-p sampling

Required options:

```bash
python scripts/generate.py \
  --checkpoint path/to/checkpoint \
  --tokenizer path/to/tokenizer \
  --prompt "Explain why the sky is blue." \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-p 0.95
```

### 13.2 Minimal API

Optional but desirable:

* FastAPI server for local inference
* `/generate` endpoint
* request/response JSON
* no authentication required for local development
* clear warning that it is not production-hardened

### 13.3 Export

Export must produce a **HuggingFace `transformers`-loadable, Qwen3-architecture-compatible** directory (the interop hub of §26.8) — loadable by `transformers`, vLLM, and (after GGUF conversion) llama.cpp without bespoke code:

```text
artifacts/
  lithos-100m-base/
    config.json            # Qwen3Config-compatible (arch, hidden, heads, kv_heads, qk_norm, rope_theta, ...)
    tokenizer.json
    model.safetensors      # HF/Qwen3 weight naming
    generation_config.json
    model_card.md
```

An export test must round-trip: load the exported checkpoint in `transformers` and confirm logits / greedy output match the in-repo generator for a fixed seed.

### 13.4 Quantization

Quantization is optional for v0.

Add documentation for later export to llama.cpp or other runtimes.

## 14. Model Cards

Each released model must have a model card.

Required sections:

1. Model name.
2. Version.
3. Parameter count.
4. Architecture summary.
5. Tokenizer.
6. Context length.
7. Training corpus summary.
8. Number of training tokens.
9. Training hardware.
10. Training duration.
11. Evaluation results.
12. Intended use.
13. Limitations.
14. Safety considerations.
15. License.
16. Citation or attribution notes.

## 15. Reproducibility Requirements

Every meaningful run must be reproducible from:

1. Git commit hash.
2. Resolved config.
3. Tokenizer version.
4. Corpus manifest.
5. Shard checksums.
6. Random seed.
7. Hardware summary.
8. Software package lockfile.
9. Training command.

Run manifest example:

```json
{
  "run_id": "2026-06-13_120000_lithos-100m",
  "git_commit": "abc123",
  "model_config": "configs/model/lithos-100m.yaml",
  "resolved_config": "runs/.../resolved_config.yaml",
  "tokenizer": "lithos-bpe-32k-v0.1",
  "corpus": "lithos-general-corpus-v0.1",
  "num_parameters": 100000000,
  "sequence_length": 2048,
  "global_batch_size": 512,
  "tokens_seen": 5000000000,
  "hardware": "1xH100",
  "precision": "bf16"
}
```

## 16. CLI Requirements

Provide scripts or CLI commands for the main workflow.

Required commands:

```bash
# Prepare smoke data
python scripts/prepare_smoke_data.py --out data/smoke

# Train tokenizer
python scripts/train_tokenizer.py --config configs/tokenizer/bpe-32k.yaml

# Tokenize corpus
python scripts/tokenize_corpus.py --config configs/data/corpus-v0.1.yaml

# Train model
python scripts/train_model.py --config configs/train/100m.yaml

# Distributed train
torchrun --nproc_per_node=8 scripts/train_model.py --config configs/train/1b.yaml

# Run evals
python scripts/run_evals.py --config configs/eval/base.yaml --checkpoint path/to/checkpoint

# Generate text
python scripts/generate.py --checkpoint path/to/checkpoint --prompt "Hello"
```

## 17. Testing Requirements

### 17.1 Unit Tests

Required unit tests:

1. Model config validation.
2. Transformer forward pass.
3. Attention mask.
4. RoPE.
5. RMSNorm.
6. SwiGLU MLP.
7. Tokenizer roundtrip.
8. Data packing.
9. Dataset shard loading.
10. Checkpoint save/load.
11. Generation.

### 17.2 Integration Tests

Required integration tests:

1. Train toy model for 10 steps.
2. Save checkpoint.
3. Resume checkpoint.
4. Generate text.
5. Run validation loss.
6. Export artifact.

### 17.3 Quality Gates

Before merging major changes:

```bash
ruff check .
pytest
```

Optional:

```bash
mypy lithos
```

## 18. Documentation Requirements

The repo must include docs for:

1. Architecture.
2. Tokenizer.
3. Corpus format.
4. Training.
5. Distributed training.
6. Evaluation.
7. Post-training.
8. Inference.
9. Reproducibility.
10. Common failure modes.

## 19. Development Milestones

### Milestone 0: Repo Skeleton

Deliverables:

1. Repo structure.
2. pyproject.
3. Basic README.
4. Config loader.
5. Test setup.
6. Lint setup.

Acceptance:

* `pytest` runs.
* `ruff check .` runs.
* README explains project purpose.

### Milestone 1: Toy Transformer

Deliverables:

1. Decoder-only transformer implementation.
2. Tiny config.
3. Forward pass.
4. Generation.
5. Unit tests.

Acceptance:

* Shape tests pass.
* Greedy generation works.
* Tiny model can run on CPU.

### Milestone 2: Data Pipeline v0

Deliverables:

1. JSONL document reader.
2. Basic filters.
3. Tokenization script.
4. Binary shard writer.
5. Packed dataloader.

Acceptance:

* Can tokenize smoke corpus.
* Can produce fixed-length training batches.
* Packing tests pass.

### Milestone 3: Training Loop v0

Deliverables:

1. Single-GPU training loop.
2. AdamW optimizer.
3. Cosine LR schedule.
4. Logging.
5. Checkpointing.
6. Resume.

Acceptance:

* Tiny model overfits tiny dataset.
* Checkpoint resume reproduces expected step count.
* Metrics are written to JSONL.

### Milestone 4: Lithos 100M

Deliverables:

1. 100M model config.
2. Tokenizer v0.
3. Corpus v0 smoke or small public corpus.
4. Training run.
5. Eval report.
6. Model card draft.

Acceptance:

* 100M model trains stably.
* Validation perplexity is reported.
* Generated samples are saved.
* Model card exists.

### Milestone 5: Evaluation Harness

Deliverables:

1. Internal perplexity eval.
2. Fixed prompt eval.
3. lm-evaluation-harness adapter or documented integration.
4. Markdown eval report.

Acceptance:

* At least 3 external tasks can run.
* Results are saved as JSON and Markdown.
* Eval config is versioned.

### Milestone 6: FSDP Training

Deliverables:

1. torchrun support.
2. FSDP config.
3. Rank-aware logging.
4. Distributed checkpointing or documented single-rank fallback.
5. Multi-GPU smoke test.

Acceptance:

* Multi-GPU toy/100M run completes.
* Logs are not corrupted by multiple ranks.
* Resume works or limitations are explicitly documented.

### Milestone 7: Lithos 300M

Deliverables:

1. 300M config.
2. Larger corpus recipe.
3. Training run.
4. Eval report.
5. Model card.

Acceptance:

* 300M model trains stably.
* Eval report compares against 100M.
* Known weaknesses are documented.

### Milestone 8: SFT v0

Deliverables:

1. Chat template.
2. SFT dataset loader.
3. Loss masking.
4. SFT training script.
5. SFT eval prompts.
6. Instruct model card.

Acceptance:

* SFT run completes.
* Model follows simple instructions better than base.
* Chat template is versioned.

### Milestone 9: Lithos 1B

Deliverables:

1. 1B model config.
2. FSDP training config.
3. 50B-token training plan.
4. Run manifest.
5. Eval report.
6. Model card.

Acceptance:

* 1B training run is stable.
* Checkpoint resume works.
* Validation loss and benchmark evals are reported.
* Model card documents data, hardware, and limitations.

## 20. Engineering Principles for Codex/Claude Code

When implementing this project, follow these rules:

1. Prefer simple, readable code over clever abstractions.
2. Do not hide the training loop behind a high-level trainer.
3. Keep configs explicit.
4. Validate tensor shapes aggressively.
5. Add tests for every core module.
6. Write manifests for every generated artifact.
7. Never silently overwrite checkpoints or run outputs.
8. Never assume a dataset has a safe license without recording provenance.
9. Keep large artifacts out of Git.
10. Make every script runnable from the repo root.
11. Fail loudly when required config values are missing.
12. Make CPU smoke tests possible.
13. Make single-GPU development possible before multi-GPU training.
14. Do not introduce distributed complexity until the single-GPU path works.
15. Document known limitations honestly.

## 21. Initial Implementation Order

Codex/Claude Code should implement in this order:

1. Repo skeleton and packaging.
2. Config loader.
3. Model config dataclass.
4. RMSNorm.
5. RoPE (configurable base/theta).
6. Causal self-attention — GQA-native (`n_kv_heads`), SDPA backend with eager fallback, optional QK-norm.
7. SwiGLU MLP.
8. Transformer block.
9. Full decoder-only model.
10. Generation utility with KV cache.
11. Shape tests.
12. Tokenizer wrapper.
13. JSONL document loader.
14. Tokenized shard writer.
15. Packed dataloader.
16. Training loop.
17. Checkpointing.
18. Resume logic.
19. Perplexity eval.
20. Sample generation eval.
21. 100M config.
22. FSDP support.
23. lm-evaluation-harness integration.
24. SFT pipeline.
25. Export and model card generation.

## 22. Definition of Done for v0

Lithos v0 is complete when:

1. A user can clone the repo and install dependencies.
2. A toy model can be trained locally.
3. A tokenizer can be trained from a small corpus.
4. A corpus can be tokenized into training shards.
5. A 100M model can be trained on one GPU.
6. Training metrics are logged.
7. Checkpoints can be resumed.
8. Perplexity evaluation works.
9. Text generation works.
10. A model artifact can be exported.
11. A model card can be produced.
12. Tests pass.
13. Documentation explains the full workflow.

## 23. First Command Sequence Target

The following sequence should eventually work:

```bash
git clone https://github.com/strata-labs/lithos.git
cd lithos

uv sync

python scripts/prepare_smoke_data.py --out data/smoke

python scripts/train_tokenizer.py \
  --config configs/tokenizer/bpe-32k.yaml

python scripts/tokenize_corpus.py \
  --config configs/data/smoke.yaml

python scripts/train_model.py \
  --config configs/train/single-gpu-smoke.yaml

python scripts/run_evals.py \
  --config configs/eval/base.yaml \
  --checkpoint runs/latest/checkpoints/latest

python scripts/generate.py \
  --checkpoint runs/latest/checkpoints/latest \
  --prompt "Explain what a language model is in one paragraph."
```

## 24. Naming Convention

Use the following naming convention:

```text
Lithos {size} {variant}
```

Examples:

```text
Lithos 100M Base
Lithos 300M Base
Lithos 1B Base
Lithos 1B Instruct
```

Artifact names:

```text
lithos-100m-base
lithos-300m-base
lithos-1b-base
lithos-1b-instruct
```

## 25. Long-Term Extensions

Possible future extensions:

1. Better dataset quality classifiers.
2. MinHash near-deduplication.
3. Benchmark decontamination improvements.
4. GQA support.
5. FlashAttention integration.
6. torch.compile support.
7. Preference tuning.
8. Quantized inference.
9. llama.cpp export.
10. Model routing between Lithos variants.
11. Data mixture ablation framework.
12. Synthetic data generation framework.
13. Long-context extension.

## 26. Resolved Decisions (v0.2)

These were open in the first PRD pass and are now decided. Defaults remain adjustable, but implementation should assume them.

### 26.1 Compute — local RTX 4070 Super (dev) + cloud (scale)

Local dev machine: a single **RTX 4070 Super (12GB)** — used for development, the toy model, the end-to-end smoke, short single-GPU sanity runs, and local inference (fits models up to ~1B in bf16). **Cloud** (flexible credits, portable design, no host-specific assumptions) runs the real over-trained 100M/300M runs and all of the 1B (training + FSDP). The local card is single-GPU, so the **FSDP/multi-GPU path cannot be smoke-tested locally** — that runs on a cloud 2×/8× node. Code stays CPU-capable for fast unit/shape tests and CI (§20.12), but the *local smoke loop uses the GPU*, not CPU. Record actual hardware per run in the run manifest (§15).

### 26.2 Corpus — Nemotron-CC-v2 slices

v0.1 corpus is built from slices of `nvidia/Nemotron-CC-v2` (NVIDIA-curated Common Crawl, 2024–2025 snapshots; ≈3.36T English CC tokens + ≈1.26T English synthetic-rephrased tokens via Qwen3-30B-A3B). Parquet on HF Hub, pulled via `datasets` streaming. Implications:

- **Provenance / disclosure:** a meaningful fraction is model-generated synthetic text. Record the CC-vs-synthetic split in the corpus manifest (§8.6 `mixture`) and disclose it in model cards (§14).
- **License:** NVIDIA Open Data License Agreement — use limited to fair uses "such as model training." Training is explicitly permitted; record in §20.8 provenance and the model-card license section. Do not redistribute raw slices.
- **Quality filtering:** lean on the dataset's own quality metadata for selection rather than building heavy filters first; §8.7 filters become a light secondary pass.
- **Code/math gap:** Nemotron-CC-v2 is English web + synthetic prose, not code. To honor the §7.1 code/math goal, blend in `nvidia/Nemotron-CC-Math-v1` (≈133B tokens, quality buckets e.g. `4plus`) for math; defer a code source (e.g. The Stack v2 / StarCoder slices) as a later mixture addition. v0.1 may ship web+math only, with the code gap documented.

### 26.3 Token budget — over-train

Target ≈100–200 tokens/param: 100M → ~10–20B, 300M → ~30–60B, 1B → ~50B+. Record actual tokens/param per run in the run manifest. Consistent with §15 and Milestone 9 numbers.

### 26.4 Tokenizer — byte-level BPE, 32k

HF `tokenizers` byte-level BPE, vocab 32k, **no `<unk>`** (byte-level covers all inputs). Special tokens per §7.1. Split runs of digits; keep byte-level pre-tokenization so code/math/Unicode round-trip cleanly. Train on a decontaminated sample of the §26.2 corpus.

### 26.5 Artifact storage — cloud object store + HF Hub

Shards and checkpoints live in a cloud object store (S3/GCS-compatible), keyed by manifest + sha256; released model artifacts (§13.3) publish to HF Hub. Git holds only code, configs, manifests, and cards. (Adjustable to local disk for early milestones.)

### 26.6 Determinism scope

Bitwise reproducible on CPU smoke tests; best-effort on single GPU (seeded RNG, cuDNN deterministic flags where affordable, documented nondeterministic ops); explicitly **not** bitwise under FSDP. §15 "reproducible from seed" is scoped accordingly.

### 26.7 Dependency manager — uv

Pin `uv` (`uv.lock`); drop the poetry alternative in §5.

### 26.8 Inference & deployment — Qwen3-compatible export as the hub

Three engines, one export; they are not alternatives:

- **Build & own:** the in-repo PyTorch generator (`model/generation.py` + KV cache) — the correctness ground truth and v0 default; drives `scripts/generate.py` and a local-only FastAPI `/generate` (§13.2; no hosted product per §3.7).
- **Interop hub:** export checkpoints in **HuggingFace `transformers`-loadable, Qwen3-architecture form** (Llama + GQA + QK-norm — the envelope that preserves our refinements while loading drop-in). This single format is the on-ramp to eval and every serving engine.
- **Serving:** **vLLM** is the documented cloud serving path (`vllm serve <export>`, OpenAI-compatible, continuous batching) — leveraged, not built; used when throughput / multi-user is needed.
- **Portable distribution:** **llama.cpp / GGUF** (quantized, CPU/Mac/edge) — deferred (§13.4/§25), reachable via `convert_hf_to_gguf.py` from the Qwen3-compatible export.

**Constraint:** keep the model within the Qwen3 export envelope so export is weight-renaming, not re-implementation. Stepping outside it (e.g. MLA, sliding-window) requires custom modeling code (`trust_remote_code`) and forfeits drop-in vLLM/GGUF support — a conscious, documented cost.

## 27. Proposed Engineering Addenda (pending sign-off)

Completeness items to thread into the existing sections once §26 is settled:

- **Dataloader resume position (§9.9).** Resume currently restores optimizer/step/token count but not data order/position — a resumed run would re-see data. Add resumable, deterministic data ordering (shard index + offset + sampler RNG in the checkpoint).
- **Document-boundary handling in packed sequences (§6/§8).** Decide cross-document attention masking vs BOS-separated bleed, and whether RoPE positions reset at doc boundaries; the packing test must cover it.
- **Validation set definition (§11).** Define how the held-out val set is constructed, kept fixed across runs, and decontaminated — currently perplexity is required but the set is undefined.
- **Gradient (activation) checkpointing (§9).** Needed to fit 300M/1B and longer context; add as a config flag.
- **torch.compile (move up from §25).** Large near-free speedup; opt-in flag with eager fallback.
- **Mixed-precision details (§9.5).** Specify fp32 optimizer states / master weights under bf16, and fp16 loss-scaling behavior with grad accumulation.
- **Loss-spike handling (§9).** At minimum document the stance (skip-batch-on-spike / checkpoint rollback) for 1B runs.
- **CI (§17.3).** GitHub Actions job running `ruff check .`, `pytest`, and the CPU end-to-end smoke so the gates are enforced, not just documented.
- **Software pinning (§15.8).** Pin CUDA/PyTorch versions alongside the lockfile for real reproducibility.

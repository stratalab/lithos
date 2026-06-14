"""Evaluation configuration (PRD §11.3)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from lithos.evals.benchmarks import BATTERY_VERSION, DEFAULT_TASKS


class BenchmarkConfig(BaseModel):
    """Frozen, versioned external-benchmark battery (lm-eval-harness)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    battery_version: str = BATTERY_VERSION
    tasks: list[str] = Field(default_factory=lambda: list(DEFAULT_TASKS))
    num_fewshot: int = 0
    limit: int | None = None  # cap examples/task for smoke; None = full battery
    batch_size: str | int = "auto"
    dtype: str = "bfloat16"


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "base"
    tokenizer_path: str
    val_corpus_manifest: str | None = None
    eval_batches: int = 50
    batch_size: int = 8
    prompts: list[str] = Field(default_factory=list)
    sample_max_new_tokens: int = 64
    greedy: bool = False
    output_dir: str = "runs/eval"
    export_dir: str | None = None

    # Phase 9 — the measuring stick.
    benchmarks: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    scorecard_path: str | None = None  # append a comparable result row here if set
    data_recipe: str | None = None  # provenance for the scorecard (which corpus/recipe)

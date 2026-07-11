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


class TIRBatteryConfig(BaseModel):
    """The TIR tool-uplift battery (docs/eval-tir-battery-plan.md).

    Scores each post-cutoff problem twice (tools off vs on) and reports the verified
    solve-rate difference per difficulty tier. Greedy by default (``temperature=0``);
    ``>0`` is a labelled ``maj@k`` extension, not the honest pass@1 headline.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    battery_version: str = "tir-v1"
    task_bank: str | None = None  # kind=problems JSONL (Chisel-produced)
    cutoff_year: int | None = None  # eval on > cutoff (post-cutoff hold-out); None = all
    levels: list[str] | None = None  # optional difficulty-ladder filter
    max_new_tokens: int = 512
    max_tool_calls: int = 4
    tool_timeout_s: float = 5.0
    temperature: float = 0.0  # greedy (honest pass@1); >0 needs a labelled maj@k
    result_token_cap: int = 256
    transcript_sample: int = 12  # rollouts persisted for spot-checking (§0.8)
    limit: int | None = None  # cap tasks for smoke; None = full pool


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
    tir: TIRBatteryConfig = Field(default_factory=TIRBatteryConfig)  # E8 tool-uplift
    scorecard_path: str | None = None  # append a comparable result row here if set
    data_recipe: str | None = None  # provenance for the scorecard (which corpus/recipe)

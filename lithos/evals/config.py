"""Evaluation configuration (PRD §11.3)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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

"""Training run configuration (PRD §9).

Composes the model config with optimizer, schedule, data, and loop settings into
one validated object that is saved verbatim with every run (PRD §15).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lithos.model.config import ModelConfig


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0


class ScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warmup_steps: int = 100
    max_steps: int = 1000
    min_lr_ratio: float = 0.1  # min lr = lr * min_lr_ratio


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # "packed": corpus_manifest is a tokenized-shard manifest (pretraining).
    # "sft":    corpus_manifest is a messages-JSONL file, rendered at load time.
    kind: Literal["packed", "sft", "dpo"] = "packed"
    corpus_manifest: str
    seq_len: int
    val_corpus_manifest: str | None = None
    tokenizer_path: str | None = None  # required for kind="sft" (renders messages)

    @model_validator(mode="after")
    def _require_tokenizer_for_rendered(self) -> DataConfig:
        if self.kind in ("sft", "dpo") and not self.tokenizer_path:
            raise ValueError(f"data.tokenizer_path is required when data.kind={self.kind!r}")
        return self


class WandbConfig(BaseModel):
    """Optional Weights & Biases mirror of ``metrics.jsonl`` (rank-0 only).

    Disabled by default; the local JSONL stays the canonical record either way.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    project: str = "lithos"
    entity: str | None = None  # team/user; None -> wandb default
    mode: Literal["online", "offline", "disabled"] = "online"
    group: str | None = None  # None -> run_name (groups resumed segments together)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None


class TrainConfig(BaseModel):
    # protected_namespaces=() so a field named ``model`` is allowed.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    run_name: str
    runs_dir: str = "runs"
    seed: int = 0
    device: str = "auto"
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    # Weight-only init from a checkpoint dir (fine-tuning/SFT): loads model weights,
    # then starts fresh optimizer + schedule from step 0 (distinct from resume_from).
    init_from: str | None = None
    dpo_beta: float = 0.1  # DPO reference-deviation strength (only the DPO trainer reads this)

    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    grad_checkpointing: bool = False
    compile: bool = False

    log_interval: int = 10
    eval_interval: int = 0  # 0 -> no in-loop eval
    eval_steps: int = 50
    checkpoint_interval: int = 0  # 0 -> only the final checkpoint

    model: ModelConfig
    data: DataConfig
    optim: OptimConfig = Field(default_factory=OptimConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    @property
    def global_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def tokens_per_step(self) -> int:
        return self.global_batch_size * self.data.seq_len

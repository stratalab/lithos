"""Tokenizer configuration and construction (PRD §7.1, §26.4).

Byte-level BPE, 32k vocab, **no** ``<unk>`` — a full 256-byte alphabet makes
every input encodable, so the tokenizer is lossless. Special tokens occupy fixed
low IDs (their order here == their IDs) so they stay stable across retraining.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

# Fixed order -> fixed IDs (PRD §7.1, §7.3.4).
DEFAULT_SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]


class TokenizerConfig(BaseModel):
    """Byte-level BPE tokenizer definition."""

    model_config = ConfigDict(extra="forbid")

    name: str = "lithos-bpe-32k"
    version: str = "v0.1"
    vocab_size: int = 32000
    special_tokens: list[str] = Field(default_factory=lambda: list(DEFAULT_SPECIAL_TOKENS))
    individual_digits: bool = True
    add_prefix_space: bool = False
    min_frequency: int = 2

    @property
    def full_name(self) -> str:
        return f"{self.name}-{self.version}"


class DataSourceSpec(BaseModel):
    """Where tokenizer training text comes from: an HF dataset or local JSONL."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["hf", "jsonl"] = "hf"
    dataset: str | None = None
    config_name: str | None = None
    split: str = "train"
    paths: list[str] = Field(default_factory=list)
    text_field: str = "text"
    max_documents: int = 200_000


class TokenizerTrainConfig(BaseModel):
    """Full training-run config (tokenizer + data source + output)."""

    model_config = ConfigDict(extra="forbid")

    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    data: DataSourceSpec
    output_dir: str = "artifacts/tokenizer"
    report_samples: list[str] = Field(default_factory=list)


def build_tokenizer(cfg: TokenizerConfig) -> Tokenizer:
    """Construct an untrained byte-level BPE tokenizer from ``cfg``."""
    tok = Tokenizer(models.BPE(unk_token=None))
    pretoks: list = []
    if cfg.individual_digits:
        pretoks.append(pre_tokenizers.Digits(individual_digits=True))
    pretoks.append(pre_tokenizers.ByteLevel(add_prefix_space=cfg.add_prefix_space, use_regex=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence(pretoks)
    tok.decoder = decoders.ByteLevel()
    return tok

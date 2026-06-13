"""Tests for lithos.model.config — defaults, derived properties, validation."""

import pytest
from lithos.model import ModelConfig
from lithos.model.config import default_intermediate_size
from pydantic import ValidationError


def test_defaults_resolve():
    cfg = ModelConfig(vocab_size=100, n_layers=2, hidden=64, n_heads=8)
    assert cfg.n_kv_heads == 8  # defaults to MHA
    assert cfg.head_dim == 8
    assert cfg.intermediate_size == default_intermediate_size(64)
    assert cfg.padded_vocab_size == 128  # rounded up to pad_vocab_to


def test_padding_is_noop_when_already_aligned():
    cfg = ModelConfig(vocab_size=256, n_layers=1, hidden=32, n_heads=4)
    assert cfg.padded_vocab_size == 256


def test_gqa_group_count():
    cfg = ModelConfig(vocab_size=64, n_layers=1, hidden=64, n_heads=8, n_kv_heads=2)
    assert cfg.n_kv_groups == 4


def test_hidden_must_be_divisible_by_heads():
    with pytest.raises(ValidationError):
        ModelConfig(vocab_size=64, n_layers=1, hidden=30, n_heads=4)


def test_heads_must_be_divisible_by_kv_heads():
    with pytest.raises(ValidationError):
        ModelConfig(vocab_size=64, n_layers=1, hidden=64, n_heads=8, n_kv_heads=3)


def test_head_dim_must_be_even_for_rope():
    with pytest.raises(ValidationError):
        ModelConfig(vocab_size=64, n_layers=1, hidden=12, n_heads=4)  # head_dim=3


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(vocab_size=64, n_layers=1, hidden=64, n_heads=4, bogus=1)

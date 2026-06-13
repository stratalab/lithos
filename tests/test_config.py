"""Tests for lithos.utils.config — merge, overrides, includes, validation."""

import pytest
from lithos.utils.config import (
    ConfigError,
    apply_overrides,
    deep_merge,
    load_and_validate,
    load_config,
    parse_override,
    save_resolved_config,
)
from lithos.utils.io import read_yaml
from pydantic import BaseModel


def test_deep_merge_is_recursive_and_pure():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    over = {"a": {"y": 20, "z": 3}}
    assert deep_merge(base, over) == {"a": {"x": 1, "y": 20, "z": 3}, "b": 1}
    # Inputs are not mutated.
    assert base == {"a": {"x": 1, "y": 2}, "b": 1}


def test_parse_override_infers_types():
    assert parse_override("model.n_layers=12") == {"model": {"n_layers": 12}}
    assert parse_override("train.lr=3e-4") == {"train": {"lr": 3e-4}}
    assert parse_override("flag=true") == {"flag": True}
    assert parse_override("name=lithos-toy") == {"name": "lithos-toy"}


def test_parse_override_rejects_missing_equals():
    with pytest.raises(ConfigError):
        parse_override("no-equals-sign")


def test_apply_overrides_merges_into_data():
    data = {"model": {"n_layers": 4}}
    out = apply_overrides(data, ["model.n_layers=8", "model.hidden=256"])
    assert out == {"model": {"n_layers": 8, "hidden": 256}}


def test_load_config_resolves_includes(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nb: 2\n")
    (tmp_path / "child.yaml").write_text("includes: base.yaml\nb: 20\nc: 30\n")
    assert load_config(tmp_path / "child.yaml") == {"a": 1, "b": 20, "c": 30}


def test_load_config_applies_overrides(tmp_path):
    (tmp_path / "c.yaml").write_text("a: 1\n")
    out = load_config(tmp_path / "c.yaml", overrides=["a=2", "d.e=5"])
    assert out == {"a": 2, "d": {"e": 5}}


def test_load_config_missing_file_fails_loudly(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


class _ModelCfg(BaseModel):
    n_layers: int
    hidden: int = 128


def test_load_and_validate_ok(tmp_path):
    (tmp_path / "m.yaml").write_text("n_layers: 6\nhidden: 256\n")
    cfg = load_and_validate(tmp_path / "m.yaml", _ModelCfg)
    assert cfg.n_layers == 6
    assert cfg.hidden == 256


def test_missing_required_key_fails_loudly(tmp_path):
    (tmp_path / "m.yaml").write_text("hidden: 256\n")  # n_layers missing
    with pytest.raises(ConfigError):
        load_and_validate(tmp_path / "m.yaml", _ModelCfg)


def test_save_resolved_config_roundtrip(tmp_path):
    cfg = _ModelCfg(n_layers=3)
    out = tmp_path / "resolved.yaml"
    save_resolved_config(cfg, out)
    assert read_yaml(out) == {"n_layers": 3, "hidden": 128}

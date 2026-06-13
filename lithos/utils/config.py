"""Configuration loading: YAML + includes -> CLI overrides -> pydantic validation.

Design goals (PRD §4.1, §20.3, §20.11):

- YAML config files, with optional ``includes:`` composition.
- CLI dotted-key overrides (``model.n_layers=12``).
- Validation against a pydantic model; missing required keys fail loudly.
- Every resolved config can be saved verbatim for reproducibility (PRD §15).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, TypeVar

import yaml

from lithos.utils.io import read_yaml, write_yaml

__all__ = [
    "ConfigError",
    "apply_overrides",
    "deep_merge",
    "load_and_validate",
    "load_config",
    "parse_override",
    "save_resolved_config",
]

INCLUDES_KEY = "includes"

T = TypeVar("T")


class ConfigError(Exception):
    """Raised for malformed configs, bad overrides, or validation failures."""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins). Pure — inputs unchanged."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _coerce_scalar(raw: str) -> Any:
    """Interpret an override RHS as int/float/bool/null/list/str.

    Uses YAML scalar rules, with a ``float()`` fallback for forms the YAML 1.1
    resolver misses (e.g. ``3e-4`` or ``.5``), so CLI overrides like
    ``train.lr=3e-4`` coerce to a number rather than a string.
    """
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    if isinstance(value, str):
        try:
            return float(raw)
        except ValueError:
            return value
    return value


def parse_override(override: str) -> dict[str, Any]:
    """Parse ``a.b.c=value`` into a nested dict ``{'a': {'b': {'c': value}}}``."""
    if "=" not in override:
        raise ConfigError(f"Invalid override {override!r}: expected 'dotted.key=value'.")
    dotted, raw = override.split("=", 1)
    dotted = dotted.strip()
    if not dotted:
        raise ConfigError(f"Invalid override {override!r}: empty key.")
    value = _coerce_scalar(raw.strip())
    keys = dotted.split(".")
    node: dict[str, Any] = {}
    cursor = node
    for k in keys[:-1]:
        cursor[k] = {}
        cursor = cursor[k]
    cursor[keys[-1]] = value
    return node


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply a list of ``dotted.key=value`` override strings to ``data``."""
    result = data
    for ov in overrides:
        result = deep_merge(result, parse_override(ov))
    return result


def _load_with_includes(path: Any, seen: frozenset[Path]) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if resolved in seen:
        raise ConfigError(f"Circular include detected at {resolved}.")
    if not resolved.exists():
        raise ConfigError(f"Config file not found: {resolved}")
    seen = seen | {resolved}
    raw = read_yaml(resolved) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping: {resolved}")
    includes = raw.pop(INCLUDES_KEY, []) or []
    if isinstance(includes, str):
        includes = [includes]
    merged: dict[str, Any] = {}
    for inc in includes:
        merged = deep_merge(merged, _load_with_includes(resolved.parent / inc, seen))
    return deep_merge(merged, raw)


def load_config(path: Any, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML (resolving ``includes:``) and apply CLI overrides -> plain dict."""
    data = _load_with_includes(path, frozenset())
    if overrides:
        data = apply_overrides(data, list(overrides))
    return data


def load_and_validate(path: Any, model_cls: type[T], overrides: list[str] | None = None) -> T:
    """Load + validate into a pydantic model ``model_cls``; raise ConfigError on failure."""
    data = load_config(path, overrides)
    try:
        from pydantic import ValidationError
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ConfigError("pydantic is required for config validation.") from e
    try:
        return model_cls(**data)
    except ValidationError as e:
        raise ConfigError(f"Invalid config {Path(path)}:\n{e}") from e


def save_resolved_config(config: Any, path: Any) -> Path:
    """Dump a resolved config (pydantic model or dict) to YAML for reproducibility."""
    if hasattr(config, "model_dump"):
        data = config.model_dump(mode="json")
    elif isinstance(config, dict):
        data = config
    else:
        raise ConfigError(f"Cannot serialize config of type {type(config)!r}.")
    return write_yaml(path, data)

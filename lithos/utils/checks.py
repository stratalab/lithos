"""Aggressive runtime checks — shapes and invariants fail loudly (PRD §20.4, §20.11)."""

from __future__ import annotations

from typing import Any

__all__ = ["check_divisible", "check_shape", "require"]


def require(condition: Any, message: str) -> None:
    """Raise ``ValueError(message)`` unless ``condition`` is truthy."""
    if not condition:
        raise ValueError(message)


def check_divisible(value: int, divisor: int, name: str = "value") -> None:
    """Require ``value`` to be a nonzero-divisor multiple of ``divisor``."""
    require(
        divisor != 0 and value % divisor == 0,
        f"{name}={value} must be divisible by {divisor}.",
    )


def check_shape(tensor: Any, expected: tuple[int | None, ...], name: str = "tensor") -> None:
    """Assert ``tensor.shape`` matches ``expected``; ``None`` entries are wildcards."""
    shape = tuple(tensor.shape)
    if len(shape) != len(expected):
        raise ValueError(
            f"{name} has rank {len(shape)} {shape}, expected rank {len(expected)} {expected}."
        )
    for axis, (got, want) in enumerate(zip(shape, expected, strict=True)):
        if want is not None and got != want:
            raise ValueError(
                f"{name} axis {axis} is {got}, expected {want} (full shape {shape} vs {expected})."
            )

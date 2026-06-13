"""Device and precision resolution (PRD §9.5).

Torch is imported lazily inside the functions so importing this module is cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["DTYPES", "resolve_device", "resolve_dtype"]

# Friendly precision names -> torch dtype attribute names.
DTYPES = {
    "fp32": "float32",
    "float32": "float32",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
}


def resolve_device(prefer: str | None = None) -> str:
    """Resolve a device string. ``prefer`` (other than ``"auto"``) wins; else auto-detect."""
    import torch

    if prefer and prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str) -> torch.dtype:
    """Map a friendly precision name (``bf16``/``fp16``/``fp32``) to a torch dtype."""
    import torch

    key = name.lower()
    if key not in DTYPES:
        raise ValueError(f"Unknown dtype {name!r}; expected one of {sorted(DTYPES)}.")
    return getattr(torch, DTYPES[key])

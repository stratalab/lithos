"""Deterministic seeding across Python, NumPy, and PyTorch (PRD §15, §26.6).

Bitwise determinism is only guaranteed on CPU; GPU is best-effort (PRD §26.6).
Torch and NumPy are imported lazily so this module stays importable (and fast)
in minimal environments.
"""

from __future__ import annotations

import os
import random

__all__ = ["seed_everything"]


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and (if installed) PyTorch RNGs; return ``seed``.

    With ``deterministic=True`` also request best-effort deterministic kernels.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ModuleNotFoundError:
        pass

    return seed

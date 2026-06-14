"""Distributed training helpers (PRD §10).

DDP, not FSDP: on big-memory GPUs (e.g. 192GB B200, 80GB H100) the whole model +
optimizer fit on one device through ~7B, so plain data-parallel suffices and is
far simpler. Launch with ``torchrun --nproc_per_node=N scripts/train_model.py``.

When not launched under torchrun (no ``RANK`` in the env), everything degrades to
a single process (world_size=1), so single-GPU and CPU runs are unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from lithos.utils.device import resolve_device


@dataclass(frozen=True)
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    device: str
    backend: str | None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def setup_distributed(prefer_device: str = "auto") -> DistInfo:
    """Initialize the process group if under torchrun; else single-process."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if prefer_device != "cpu" and torch.cuda.is_available():
            backend = "nccl"
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            backend = "gloo"
            device = "cpu"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        return DistInfo(rank, local_rank, world_size, device, backend)
    return DistInfo(0, 0, 1, resolve_device(prefer_device), None)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: float, device: str) -> float:
    """Average a scalar across ranks (no-op when not distributed)."""
    if not dist.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / dist.get_world_size())

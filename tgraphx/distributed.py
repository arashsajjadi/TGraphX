"""Lightweight distributed-training helpers for TGraphX.

Goals
-----
* Be safe to import in any environment, including CPU-only and
  single-process runs.
* Never call ``torch.distributed.init_process_group`` automatically.
* Provide rank-zero conveniences (printing, decorators) so user training
  scripts don't have to special-case distributed mode.

Honesty
-------
TGraphX does **not** ship a distributed training framework.  Multi-GPU
and DDP are the user's responsibility.  These helpers exist so that
existing training utilities (`fit`, `train_epoch`, `evaluate`, logging)
behave reasonably whether the user is running single-GPU, single-CPU,
or multi-process DDP.

Usage::

    from tgraphx.distributed import (
        is_distributed_available_and_initialized, get_rank, get_world_size,
        is_rank_zero, rank_zero_print, rank_zero_only,
    )

    rank_zero_print(f"world_size={get_world_size()}")

    @rank_zero_only
    def maybe_save_checkpoint(...):
        ...
"""
from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

import torch

__all__ = [
    "is_distributed_available_and_initialized",
    "get_rank",
    "get_world_size",
    "is_rank_zero",
    "rank_zero_print",
    "rank_zero_only",
    "barrier",
]


F = TypeVar("F", bound=Callable[..., Any])


def is_distributed_available_and_initialized() -> bool:
    """``True`` if torch.distributed is importable AND initialized."""
    if not hasattr(torch, "distributed"):
        return False
    dist = torch.distributed
    if not dist.is_available():
        return False
    try:
        return bool(dist.is_initialized())
    except Exception:  # extremely defensive
        return False


def get_rank(default: int = 0) -> int:
    """Return distributed rank, or ``default`` (0) when not initialized."""
    if not is_distributed_available_and_initialized():
        return default
    return int(torch.distributed.get_rank())


def get_world_size(default: int = 1) -> int:
    """Return distributed world size, or ``default`` (1) when not initialized."""
    if not is_distributed_available_and_initialized():
        return default
    return int(torch.distributed.get_world_size())


def is_rank_zero() -> bool:
    """Convenience: ``get_rank() == 0`` (always ``True`` outside DDP)."""
    return get_rank() == 0


def rank_zero_print(*args: Any, **kwargs: Any) -> None:
    """``print(...)`` only on rank 0; no-op elsewhere."""
    if is_rank_zero():
        print(*args, **kwargs)


def rank_zero_only(fn: F) -> F:
    """Decorate ``fn`` to execute only on rank 0; other ranks return ``None``.

    Useful for logging / checkpointing helpers that should not be invoked
    by every worker.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if is_rank_zero():
            return fn(*args, **kwargs)
        return None

    return wrapper  # type: ignore[return-value]


def barrier() -> None:
    """Distributed barrier; no-op when not initialized."""
    if is_distributed_available_and_initialized():
        torch.distributed.barrier()

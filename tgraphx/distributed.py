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
import os
from typing import Any, Callable, Dict, Optional, TypeVar

import torch

__all__ = [
    "is_distributed_available_and_initialized",
    "is_distributed",
    "get_rank",
    "get_world_size",
    "is_rank_zero",
    "rank_zero_print",
    "rank_zero_only",
    "barrier",
    "detect_distributed_environment",
    "rank_seed",
    "distributed_device",
    "maybe_wrap_ddp",
    "shard_indices",
    "write_distributed_run_summary",
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


def is_distributed() -> bool:
    """Alias for :func:`is_distributed_available_and_initialized`."""
    return is_distributed_available_and_initialized()


# ── Environment detection ────────────────────────────────────────────────────


def detect_distributed_environment() -> Dict[str, Any]:
    """Inspect environment variables that hint at distributed launch.

    Reads ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``,
    ``MASTER_PORT`` and the SLURM-style fallbacks ``SLURM_PROCID`` /
    ``SLURM_NTASKS``.  Does **not** call ``init_process_group``.

    Returns:
        Dict with keys:
        ``initialized`` (bool) — is ``torch.distributed`` already up?
        ``rank``, ``local_rank``, ``world_size`` (ints; defaults 0/0/1).
        ``backend`` (str | None) — only meaningful if initialized.
        ``master_addr``, ``master_port`` (str | None).
        ``launcher`` (str) — one of ``"none"``, ``"torchrun"``,
        ``"slurm"``, ``"manual"``.
    """
    env = os.environ
    initialized = is_distributed_available_and_initialized()
    rank = int(env.get("RANK", env.get("SLURM_PROCID", 0)))
    local_rank = int(env.get("LOCAL_RANK", env.get("SLURM_LOCALID", 0)))
    world_size = int(env.get("WORLD_SIZE", env.get("SLURM_NTASKS", 1)))
    backend: Optional[str] = None
    if initialized:
        try:
            backend = str(torch.distributed.get_backend())
            rank = int(torch.distributed.get_rank())
            world_size = int(torch.distributed.get_world_size())
        except Exception:
            pass
    if "RANK" in env or "WORLD_SIZE" in env:
        launcher = "torchrun"
    elif "SLURM_PROCID" in env:
        launcher = "slurm"
    elif initialized:
        launcher = "manual"
    else:
        launcher = "none"
    return {
        "initialized": bool(initialized),
        "rank": int(rank),
        "local_rank": int(local_rank),
        "world_size": int(world_size),
        "backend": backend,
        "master_addr": env.get("MASTER_ADDR"),
        "master_port": env.get("MASTER_PORT"),
        "launcher": launcher,
    }


def rank_seed(base_seed: int, rank: Optional[int] = None) -> int:
    """Return a per-rank deterministic seed derived from ``base_seed``.

    Useful for ensuring each rank has its own RNG stream while remaining
    reproducible across runs.

    Args:
        base_seed: Base seed shared across all ranks.
        rank: Override rank (defaults to :func:`get_rank`).

    Returns:
        Integer seed suitable for ``torch.manual_seed`` /
        ``torch.Generator.manual_seed``.
    """
    r = get_rank() if rank is None else int(rank)
    # Combine via a stable mixing function that is deterministic and
    # gives well-distributed seeds even for adjacent ranks.
    mixed = (int(base_seed) * 1_000_003 + r * 16_777_619) & 0x7FFFFFFF
    return int(mixed)


def distributed_device(local_rank: Optional[int] = None) -> torch.device:
    """Pick a sensible device for the current rank.

    Returns ``cuda:local_rank`` when CUDA is available, else CPU.

    Args:
        local_rank: Override (defaults to ``LOCAL_RANK`` env or 0).
    """
    if local_rank is None:
        env = detect_distributed_environment()
        local_rank = int(env["local_rank"])
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(local_rank) % max(torch.cuda.device_count(), 1)}")
    return torch.device("cpu")


# ── DDP helpers ──────────────────────────────────────────────────────────────


def maybe_wrap_ddp(
    model: torch.nn.Module,
    device_ids: Optional[list] = None,
    **kwargs: Any,
) -> torch.nn.Module:
    """Wrap ``model`` in ``DistributedDataParallel`` when a process group is up.

    No-op when not running distributed.  Raises a clear error if
    ``torch.distributed`` is unavailable.  Other kwargs are forwarded to
    ``DistributedDataParallel`` (e.g. ``find_unused_parameters``).
    """
    if not is_distributed_available_and_initialized():
        return model
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch.nn.parallel.DistributedDataParallel unavailable") from exc
    return DDP(model, device_ids=device_ids, **kwargs)


def shard_indices(
    indices: torch.Tensor,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    drop_last: bool = False,
) -> torch.Tensor:
    """Return the slice of ``indices`` owned by ``rank``.

    Distributes ``indices`` across ranks contiguously (``world_size``
    near-equal chunks).  When ``drop_last=False`` (default) the last
    rank may receive a slightly larger or smaller chunk.

    Args:
        indices: ``LongTensor[K]`` of global indices.
        rank: Defaults to :func:`get_rank`.
        world_size: Defaults to :func:`get_world_size`.
        drop_last: When ``True``, truncate ``indices`` to a multiple of
            ``world_size`` so every rank sees the same count.

    Returns:
        ``LongTensor`` slice for this rank.
    """
    r = get_rank() if rank is None else int(rank)
    w = get_world_size() if world_size is None else int(world_size)
    if w < 1:
        raise ValueError("world_size must be >= 1")
    K = int(indices.numel())
    if drop_last:
        K = (K // w) * w
        indices = indices[:K]
    base, rem = divmod(K, w)
    start = r * base + min(r, rem)
    end = start + base + (1 if r < rem else 0)
    return indices[start:end]


# ── Distributed run summary writer ──────────────────────────────────────────


@rank_zero_only
def write_distributed_run_summary(path: str, **fields: Any) -> str:
    """Write ``distributed_run_summary.json`` from rank 0 only.

    Captures world_size, rank, backend, device, base_seed, and any
    additional caller-provided fields.  Other ranks no-op.

    Args:
        path: Output JSON path.
        **fields: Arbitrary extra fields (e.g. ``base_seed``,
            ``model_name``, ``dataset``).

    Returns:
        The path on rank 0; ``None`` on other ranks (decorator return).
    """
    import json
    import tempfile
    from pathlib import Path

    env = detect_distributed_environment()
    payload = {**env, **fields}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, default=str)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, str(p))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(p)

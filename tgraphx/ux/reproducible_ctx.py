"""Reproducible run context manager."""
from __future__ import annotations

import contextlib
import os
import platform
from typing import Any, Dict, Iterator, Optional

import torch


@contextlib.contextmanager
def reproducible(
    seed: int = 42,
    deterministic: bool = True,
    warn_only: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Context manager that seeds RNGs and records reproducibility state.

    On enter:
      - Calls ``tgraphx.reproducibility.set_seed(seed, deterministic=deterministic,
        warn_only=warn_only)``.
      - Yields a state dict containing seed, torch version, CUDA availability,
        backend settings, and the platform.

    On exit:
      - The current PyTorch settings are NOT rolled back automatically — set_seed
        often involves global state (cuDNN flags) that other code may rely on.
      - Use ``tgx.reproducibility_state()`` after exit to inspect current state.

    CUDA caveat:
      Determinism on CUDA is best-effort. Some PyTorch ops (e.g. adaptive_avg_pool2d
      backward) have no deterministic implementation; ``warn_only=True`` allows
      them with a warning.

    Args:
        seed: RNG seed (random, numpy, torch CPU+CUDA).
        deterministic: Pass through to :func:`tgraphx.set_seed`.
        warn_only: Pass through to :func:`tgraphx.set_seed`.

    Yields:
        A dict with keys: seed, torch_version, cuda_available, device_count,
        platform, deterministic, warn_only.
    """
    from ..reproducibility import set_seed
    set_seed(seed, deterministic=deterministic, warn_only=warn_only)
    state = reproducibility_state()
    state["seed"] = seed
    state["deterministic"] = deterministic
    state["warn_only"] = warn_only
    try:
        yield state
    finally:
        # Intentionally do not restore prior backend flags; reproducible() is a
        # one-way seed/configure call. Use reproducibility_state() to inspect.
        pass


@contextlib.contextmanager
def seeded(seed: int = 42) -> Iterator[Dict[str, Any]]:
    """Lightweight alias for ``reproducible(seed=seed, deterministic=False)``.

    Faster than full deterministic mode; useful when exact CUDA reproducibility
    is not required.
    """
    with reproducible(seed=seed, deterministic=False) as state:
        yield state


def reproducibility_state() -> Dict[str, Any]:
    """Return the current reproducibility / determinism state."""
    state: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": (torch.cuda.device_count() if torch.cuda.is_available() else 0),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    if hasattr(torch.backends, "cudnn"):
        state["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        state["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    return state

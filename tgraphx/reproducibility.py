"""Reproducibility utilities for TGraphX.

Provides a clean, composable reproducibility layer on top of PyTorch's
own seeding and determinism controls.

Important caveats
-----------------
- Hardware-independent bit-exact numerical identity is **not guaranteed**.
  CPU and GPU use different kernel implementations and may produce slightly
  different floating-point results even for the same seed.
- ``deterministic=True`` improves same-device reproducibility at the cost
  of performance (some CUDA operations fall back to slower implementations).
- CUBLAS workspaces: for full CUDA determinism you may also need to set
  ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` **before** launching the Python
  process.  This cannot be done from Python after CUDA is initialised.
- Python string/bytes hashing is randomised by ``PYTHONHASHSEED``
  (default in Python 3.3+), but integer and tuple-of-integer hashing
  is deterministic.  TGraphX's internal WL kernel uses tuple-of-integer
  keys and is therefore not affected by ``PYTHONHASHSEED``.

Stability: Beta (v0.4.1+).
"""
from __future__ import annotations

import contextlib
import random
import sys
import warnings
from typing import Any, Dict, Optional

import torch

__all__ = [
    "set_seed",
    "make_generator",
    "seed_worker",
    "reproducibility_report",
    "deterministic_mode",
]


# ── Public API ───────────────────────────────────────────────────────────────


def set_seed(
    seed: int,
    deterministic: bool = False,
    benchmark: Optional[bool] = None,
    warn_only: bool = True,
) -> Dict[str, Any]:
    """Set all relevant RNG seeds for reproducible experiments.

    Sets ``random``, ``numpy`` (if installed), ``torch`` (CPU), and
    ``torch.cuda`` (all devices, if CUDA is available).

    Args:
        seed: Integer seed value.
        deterministic: When ``True``:
            - enables ``torch.backends.cudnn.deterministic = True``;
            - disables ``torch.backends.cudnn.benchmark`` unless
              *benchmark* is explicitly set;
            - attempts ``torch.use_deterministic_algorithms(True,
              warn_only=warn_only)`` (warns for non-deterministic ops
              instead of raising unless ``warn_only=False``).
            Note: this reduces GPU throughput.  Use only when
            reproducibility matters more than speed.
        benchmark: Override for ``torch.backends.cudnn.benchmark``.
            When ``None`` (default), the flag is set to ``not deterministic``.
        warn_only: Passed to ``torch.use_deterministic_algorithms`` when
            ``deterministic=True``.  If ``False``, non-deterministic
            operations raise ``RuntimeError``.

    Returns:
        Dict describing the resulting reproducibility state (useful for
        logging or storing in run metadata).
    """
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    _benchmark = bool(not deterministic) if benchmark is None else bool(benchmark)
    backend_settings: Dict[str, Any] = {}

    if deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = _benchmark
            backend_settings["cudnn.deterministic"] = True
            backend_settings["cudnn.benchmark"] = _benchmark
        except AttributeError:
            backend_settings["cudnn"] = "not available"

        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            backend_settings["use_deterministic_algorithms"] = True
            backend_settings["warn_only"] = warn_only
        except TypeError:
            # Older PyTorch without warn_only argument.
            try:
                torch.use_deterministic_algorithms(True)
                backend_settings["use_deterministic_algorithms"] = True
            except RuntimeError as e:
                warnings.warn(
                    f"torch.use_deterministic_algorithms(True) raised: {e}. "
                    "Some operations may not be deterministic.",
                    stacklevel=2,
                )
    else:
        backend_settings["deterministic_mode"] = False

    return {
        "seed": seed,
        "deterministic": deterministic,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "backend_settings": backend_settings,
    }


def make_generator(
    seed: int,
    device: str = "cpu",
) -> torch.Generator:
    """Return a seeded ``torch.Generator``.

    Args:
        seed: Integer seed value.
        device: ``"cpu"`` (default) or ``"cuda"`` if available.  CPU
            generators are always supported; CUDA generators require
            an available GPU.

    Returns:
        A ``torch.Generator`` with the given seed set.

    Note:
        CPU generators are used in most TGraphX sampling utilities
        (random walks, negative sampling, etc.) to avoid cross-device
        seed contamination.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def seed_worker(worker_id: int) -> None:
    """DataLoader worker initialisation function for reproducibility.

    Usage::

        from tgraphx.reproducibility import seed_worker
        from tgraphx import GraphDataLoader

        g = torch.Generator()
        g.manual_seed(0)
        loader = GraphDataLoader(
            dataset, batch_size=32, shuffle=True,
            worker_init_fn=seed_worker, generator=g,
        )

    The worker seed is derived deterministically from PyTorch's
    initial seed so that different workers get different but
    reproducible seeds.

    Args:
        worker_id: Worker index (passed automatically by PyTorch).
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    try:
        import numpy as np
        np.random.seed(worker_seed)
    except ImportError:
        pass


def reproducibility_report() -> Dict[str, Any]:
    """Return a snapshot of the current reproducibility-relevant state.

    Returns:
        JSON-serialisable dict containing:
          - ``torch_version``
          - ``cuda_available``
          - ``cudnn_deterministic``
          - ``cudnn_benchmark``
          - ``use_deterministic_algorithms`` (if detectable)
          - ``python_hash_seed`` (``PYTHONHASHSEED`` env var if set)
          - ``cuda_device_count``
    """
    import os

    report: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
        "use_deterministic_algorithms": None,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
    }

    try:
        report["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        report["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except AttributeError:
        pass

    try:
        report["use_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    except AttributeError:
        pass

    return report


@contextlib.contextmanager
def deterministic_mode(seed: int = 0, warn_only: bool = True):
    """Context manager that enables deterministic mode and restores state.

    Usage::

        from tgraphx.reproducibility import deterministic_mode

        with deterministic_mode(seed=42):
            # reproducible computation
            output = model(x, edge_index)

    On exit, the context manager attempts to restore:
      - ``torch.backends.cudnn.deterministic``
      - ``torch.backends.cudnn.benchmark``
      - deterministic algorithm mode (best-effort; older PyTorch may
        not support querying the current state).

    Args:
        seed: Seed to set at context entry.
        warn_only: Passed to ``set_seed``; when ``True``, non-deterministic
            ops warn instead of raising.

    Yields:
        The reproducibility state dict returned by :func:`set_seed`.
    """
    # Snapshot existing state.
    prev: Dict[str, Any] = {}
    try:
        prev["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        prev["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    except AttributeError:
        pass
    try:
        prev["use_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    except AttributeError:
        pass

    state = set_seed(seed, deterministic=True, warn_only=warn_only)
    try:
        yield state
    finally:
        # Restore previous state.
        try:
            if "cudnn_deterministic" in prev:
                torch.backends.cudnn.deterministic = prev["cudnn_deterministic"]
            if "cudnn_benchmark" in prev:
                torch.backends.cudnn.benchmark = prev["cudnn_benchmark"]
        except AttributeError:
            pass
        try:
            if "use_deterministic_algorithms" in prev:
                torch.use_deterministic_algorithms(
                    prev["use_deterministic_algorithms"], warn_only=True
                )
        except (AttributeError, TypeError):
            try:
                if "use_deterministic_algorithms" in prev:
                    torch.use_deterministic_algorithms(prev["use_deterministic_algorithms"])
            except (AttributeError, RuntimeError):
                pass

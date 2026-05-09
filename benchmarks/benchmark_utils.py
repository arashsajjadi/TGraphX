"""Shared utilities for TGraphX benchmarks.

Provides consistent metadata for all benchmark JSON outputs.
"""
from __future__ import annotations

import platform
import sys
import time
from typing import Any, Dict, Optional

import torch

try:
    import tgraphx
    _VERSION = tgraphx.__version__
except Exception:
    _VERSION = "unknown"


def benchmark_metadata(
    benchmark_name: str,
    seed: int = 42,
    algorithm: Optional[str] = None,
    method: Optional[str] = None,
    device: Optional[str] = None,
    limitations: Optional[str] = None,
    status: str = "ok",
) -> Dict[str, Any]:
    """Return a standard metadata dict for benchmark JSON outputs.

    Args:
        benchmark_name: Name of the benchmark.
        seed: Random seed used.
        algorithm: Algorithm name (for RL/evolution benchmarks).
        method: Generator method name (for generation benchmarks).
        device: Device string. Defaults to 'cuda' if available else 'cpu'.
        limitations: Optional limitation string.
        status: 'ok' or 'error'.

    Returns:
        Dict with package_version, benchmark, seed, device, status, limitations, python, torch.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    meta: Dict[str, Any] = {
        "package_version": _VERSION,
        "benchmark": benchmark_name,
        "seed": seed,
        "device": device,
        "status": status,
        "limitations": limitations or "CPU-only small-scale; no SOTA claims; Experimental stability",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": torch.__version__,
    }
    if algorithm is not None:
        meta["algorithm"] = algorithm
    if method is not None:
        meta["method"] = method
    return meta


def enrich_output(
    output: Dict[str, Any],
    benchmark_name: str,
    seed: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Merge standard benchmark metadata into an existing output dict.

    Args:
        output: Existing benchmark output dict.
        benchmark_name: Name of the benchmark.
        seed: Random seed.
        **kwargs: Extra kwargs forwarded to benchmark_metadata.

    Returns:
        Dict with standard metadata merged (metadata takes priority for version/device/etc.).
    """
    meta = benchmark_metadata(benchmark_name, seed=seed, **kwargs)
    # Merge: put metadata at front, preserve existing keys unless overridden by meta
    merged = {**output}
    for k, v in meta.items():
        if k not in merged:
            merged[k] = v
    return merged

"""Performance utilities for TGraphX.

Provides:
- ``env_report``              — Python/PyTorch/hardware environment dict
- ``estimate_message_memory`` — peak message-buffer memory estimate

Design rules
------------
* Optional deps (psutil, pynvml) are imported lazily per call — never at
  module load time.
* No file writes unless explicitly requested.
* No profiling or monitoring is enabled by default.
* All functions degrade gracefully when optional packages are absent.
* CPU, CUDA, MPS, Linux, Windows, macOS all supported.
"""
from __future__ import annotations

import platform
import sys
from typing import Any, Dict, Optional, Tuple, Union

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def env_report(
    include_hardware: bool = False,
    include_sensors: bool = False,
) -> Dict[str, Any]:
    """Return a snapshot of the runtime environment.

    Args:
        include_hardware: If ``True``, include CPU count, RAM totals, and
            process memory.  Requires ``psutil`` for RAM fields; those
            fields are ``None`` when ``psutil`` is absent.
        include_sensors:  If ``True``, add GPU utilization and temperature
            if ``pynvml`` is available; otherwise those fields are ``None``.

    Returns:
        Dictionary with the keys below.  Never raises due to missing
        optional packages.

    Always-present keys
    -------------------
    python, os, torch, tgraphx, cuda_available, cuda_device,
    mps_available, recommended_device

    With include_hardware=True
    --------------------------
    cpu_count, ram_total_gb, ram_avail_gb, process_ram_mb,
    cuda_mem_total_mb, cuda_mem_free_mb, psutil_available

    With include_sensors=True
    -------------------------
    gpu_util_pct, gpu_temp_c, pynvml_available
    """
    info: Dict[str, Any] = {}

    # ── Core ─────────────────────────────────────────────────────────────────
    info["python"] = sys.version.split()[0]
    info["os"] = f"{platform.system()} {platform.release()} {platform.machine()}"

    info["torch"] = torch.__version__

    try:
        import tgraphx as _tgx
        info["tgraphx"] = _tgx.__version__
    except Exception:
        info["tgraphx"] = "unknown"

    # ── CUDA ─────────────────────────────────────────────────────────────────
    cuda = torch.cuda.is_available()
    info["cuda_available"] = cuda
    info["cuda_device"] = None
    if cuda:
        try:
            info["cuda_device"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

    # ── MPS (Apple Silicon) ───────────────────────────────────────────────────
    _mps = getattr(getattr(torch, "backends", None), "mps", None)
    info["mps_available"] = bool(_mps is not None and getattr(_mps, "is_available", lambda: False)())

    # ── Recommended device ────────────────────────────────────────────────────
    if cuda:
        info["recommended_device"] = "cuda"
    elif info["mps_available"]:
        info["recommended_device"] = "mps"
    else:
        info["recommended_device"] = "cpu"

    # ── Hardware (optional) ───────────────────────────────────────────────────
    if include_hardware:
        import os as _os
        info["cpu_count"] = _os.cpu_count()

        # psutil
        try:
            import psutil
            info["psutil_available"] = True
            vm = psutil.virtual_memory()
            info["ram_total_gb"] = round(vm.total / 1024**3, 2)
            info["ram_avail_gb"] = round(vm.available / 1024**3, 2)
            try:
                info["process_ram_mb"] = round(psutil.Process().memory_info().rss / 1024**2, 1)
            except Exception:
                info["process_ram_mb"] = None
        except ImportError:
            info["psutil_available"] = False
            info["ram_total_gb"] = None
            info["ram_avail_gb"] = None
            info["process_ram_mb"] = None

        # CUDA memory
        if cuda:
            try:
                info["cuda_mem_total_mb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**2, 1
                )
                info["cuda_mem_free_mb"] = round(
                    (torch.cuda.get_device_properties(0).total_memory
                     - torch.cuda.memory_allocated(0)) / 1024**2, 1
                )
            except Exception:
                info["cuda_mem_total_mb"] = None
                info["cuda_mem_free_mb"] = None
        else:
            info["cuda_mem_total_mb"] = None
            info["cuda_mem_free_mb"] = None

    # ── Sensors (optional) ────────────────────────────────────────────────────
    if include_sensors:
        try:
            import pynvml
            pynvml.nvmlInit()
            info["pynvml_available"] = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            try:
                info["gpu_util_pct"] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except Exception:
                info["gpu_util_pct"] = None
            try:
                info["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                info["gpu_temp_c"] = None
        except Exception:
            info["pynvml_available"] = False
            info["gpu_util_pct"] = None
            info["gpu_temp_c"] = None

    return info


# ─────────────────────────────────────────────────────────────────────────────

def estimate_message_memory(
    num_edges: int,
    out_shape: Union[int, Tuple[int, ...]],
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Estimate the peak size of the per-edge message buffer ``[E, *out_shape]``.

    This is the largest intermediate tensor in the message-passing forward
    pass (before scatter-aggregation).  Actual peak usage may be higher
    due to intermediate convolution outputs, but this is a useful floor.

    Args:
        num_edges:  Number of edges (E).
        out_shape:  Per-node output shape, e.g. ``64`` or ``(64, 8, 8)``
                    or ``(64, 4, 8, 8)``.
        dtype:      Element dtype (default ``torch.float32``).

    Returns:
        Dict with ``bytes_per_edge``, ``total_bytes``, ``total_mb``,
        and a human-readable ``note``.
    """
    if isinstance(out_shape, int):
        out_shape = (out_shape,)
    out_shape = tuple(out_shape)

    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    elems_per_edge = 1
    for d in out_shape:
        elems_per_edge *= d

    bpe = elems_per_edge * dtype_bytes
    total_bytes = num_edges * bpe
    total_mb = total_bytes / 1024**2

    size_str = f"{total_mb:.1f} MB" if total_mb >= 1 else f"{total_bytes / 1024:.1f} KB"
    note = (
        f"Message buffer [E={num_edges}, *{out_shape}] of dtype={dtype} "
        f"≈ {size_str}.  Actual peak may exceed this by 2–3× during forward."
    )

    return {
        "num_edges": num_edges,
        "out_shape": out_shape,
        "dtype": str(dtype),
        "bytes_per_edge": bpe,
        "total_bytes": total_bytes,
        "total_mb": round(total_mb, 3),
        "note": note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Device selection helper (re-exported convenience)
# ─────────────────────────────────────────────────────────────────────────────

def recommended_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    r = env_report()["recommended_device"]
    return torch.device(r)


__all__ = ["env_report", "estimate_message_memory", "recommended_device"]

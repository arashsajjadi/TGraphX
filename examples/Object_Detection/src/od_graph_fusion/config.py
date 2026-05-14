"""Configuration loading and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file as a plain dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p) as f:
        return yaml.safe_load(f)


def resolve_device(spec: str, allow_cpu: bool = True) -> str:
    """Resolve device spec to a concrete torch device string.

    "auto" → "cuda" if available, else "mps" if available, else "cpu".
    If CUDA is available but "cpu" was explicitly requested and allow_cpu=False,
    a warning is printed so the caller can flag the downgrade.
    """
    import torch
    if spec in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except AttributeError:
            pass
        return "cpu"
    if spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' requested but torch.cuda.is_available()=False. "
            "Use device='auto' or device='cpu'."
        )
    if spec == "cpu" and not allow_cpu:
        import torch
        if torch.cuda.is_available():
            import warnings
            warnings.warn(
                "Training on CPU but CUDA is available. Pass allow_cpu=True to silence.",
                stacklevel=2,
            )
    return spec


def device_audit(requested: str, resolved: str) -> dict:
    """Return a dict describing the device environment for logging."""
    import torch
    info = {
        "requested_device": requested,
        "resolved_device": resolved,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_count": 0,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def project_root() -> Path:
    """Return the project root (Object_Detection folder)."""
    here = Path(__file__).resolve()
    # src/od_graph_fusion/config.py → .../Object_Detection
    return here.parents[2]


def run_dir(config: Dict[str, Any], run_name: Optional[str] = None) -> Path:
    """Return the per-run output directory and create it."""
    name = run_name or config.get("run_name", "run")
    base = Path(config.get("output", {}).get("run_dir_base", "runs"))
    if not base.is_absolute():
        base = project_root() / base
    out = base / name
    out.mkdir(parents=True, exist_ok=True)
    return out

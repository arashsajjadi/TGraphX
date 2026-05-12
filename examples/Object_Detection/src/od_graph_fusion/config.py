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


def resolve_device(spec: str) -> str:
    """Resolve ``"auto"`` to "cuda" or "cpu" based on torch.cuda."""
    import torch
    if spec in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


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

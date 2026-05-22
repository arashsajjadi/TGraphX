"""Run metadata collection and saving for full reproducibility."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return bool(out.strip())
    except Exception:
        return False


def _cpu_info() -> str:
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["name"]      = torch.cuda.get_device_name(0)
            info["count"]     = torch.cuda.device_count()
            info["cuda_ver"]  = torch.version.cuda or "unknown"
            info["vram_gb"]   = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except Exception:
        pass
    return info


def _config_hash(config: Dict) -> str:
    s = json.dumps(config, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def collect_run_metadata(
    config:       Dict[str, Any],
    run_dir:      Path,
    model:        Optional[Any] = None,
    extra:        Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect and return a complete metadata dictionary for the current run."""
    import torch

    meta: Dict[str, Any] = {
        "start_time":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "start_ts":      time.time(),
        "run_dir":       str(run_dir),
        "python":        sys.version,
        "torch":         torch.__version__,
        "platform":      platform.platform(),
        "cpu":           _cpu_info(),
        "cpu_count":     os.cpu_count(),
        "gpu":           _gpu_info(),
        "git_hash":      _git_hash(),
        "git_dirty":     _git_dirty(),
        "config":        config,
        "config_hash":   _config_hash(config),
        "seed":          config.get("seed", 0),
    }

    if model is not None:
        try:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            meta["model_params"] = n_params
        except Exception:
            pass

    if extra:
        meta.update(extra)

    return meta


def save_run_metadata(
    config:   Dict[str, Any],
    run_dir:  Path,
    model:    Optional[Any] = None,
    extra:    Optional[Dict[str, Any]] = None,
) -> Path:
    """Save run metadata JSON to *run_dir*/metadata.json."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = collect_run_metadata(config, run_dir, model, extra)
    out  = run_dir / "metadata.json"
    out.write_text(json.dumps(meta, indent=2, default=str))
    return out

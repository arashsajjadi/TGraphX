"""Checkpoint save / load with full reproducibility metadata."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _system_info() -> Dict[str, str]:
    info: Dict[str, str] = {
        "python":   sys.version,
        "torch":    torch.__version__,
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        info["cuda"]     = torch.version.cuda or "n/a"
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def save_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    step:      int,
    games:     int,
    config:    Dict[str, Any],
    path:      str | Path,
    scaler:    Optional[Any] = None,
    scheduler: Optional[Any] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step":      step,
        "games":     games,
        "config":    config,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "system":    _system_info(),
        "git_hash":  _git_hash(),
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    torch.save(ckpt, path)
    # Also keep a "latest" symlink
    latest = path.parent / "latest.pt"
    latest.unlink(missing_ok=True)
    latest.symlink_to(path.name)


def load_checkpoint(
    path:      str | Path,
    model:     Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device:    str = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if model is not None:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt

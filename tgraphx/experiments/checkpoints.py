"""Checkpoint helpers used by the experiment runner."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a checkpoint atomically (write to ``.tmp`` then rename)."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": {k: float(v) for k, v in (metrics or {}).items()},
        "extra": dict(extra or {}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Any = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint and restore model/optimizer state in-place.

    Returns the full payload dict so callers can inspect ``epoch`` /
    ``metrics`` / ``extra``.
    """
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover  (older torch)
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(
            f"{path}: not a TGraphX checkpoint payload (missing 'model_state_dict')."
        )
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload

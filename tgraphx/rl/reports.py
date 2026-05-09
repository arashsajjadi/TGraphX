"""Report writers for graph RL.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["write_graph_rl_env_report", "write_graph_rl_training_report"]

_MAX_ROWS = 500


def _atomic_write(path: str, payload: Dict[str, Any]) -> str:
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


def _cap(lst: list) -> list:
    if len(lst) > _MAX_ROWS:
        return lst[:_MAX_ROWS] + [f"... ({len(lst) - _MAX_ROWS} more)"]
    return lst


def _safe_str(v: Any) -> Any:
    import torch
    if isinstance(v, torch.Tensor):
        return {"shape": list(v.shape), "dtype": str(v.dtype)}
    return v


def write_graph_rl_env_report(
    path: str,
    env_name: str,
    episode_returns: List[float],
    success_rates: List[float],
    episode_lengths: List[int],
    action_distribution: Optional[Dict[str, int]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Write an RL environment evaluation report.

    Args:
        path: Output file path.
        env_name: Name of the environment.
        episode_returns: Per-episode total returns.
        success_rates: Per-episode success flags (0/1).
        episode_lengths: Per-episode step counts.
        action_distribution: Optional action count histogram.
        extra: Optional extra fields.

    Returns:
        Absolute path to written file.
    """
    payload: Dict[str, Any] = {
        "report_type": "rl_env",
        "env_name": str(env_name),
        "episode_returns": _cap(episode_returns),
        "mean_return": sum(episode_returns) / len(episode_returns) if episode_returns else 0.0,
        "success_rate": sum(success_rates) / len(success_rates) if success_rates else 0.0,
        "episode_lengths": _cap(episode_lengths),
        "mean_length": sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0,
    }
    if action_distribution:
        payload["action_distribution"] = action_distribution
    if extra:
        payload["extra"] = {k: _safe_str(v) for k, v in extra.items()}
    return _atomic_write(path, payload)


def write_graph_rl_training_report(
    path: str,
    algorithm: str,
    loss_curves: Dict[str, List[float]],
    return_curves: List[float],
    entropy: Optional[List[float]] = None,
    kl: Optional[List[float]] = None,
    td_error: Optional[List[float]] = None,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Write an RL training report.

    Args:
        path: Output file path.
        algorithm: Algorithm name.
        loss_curves: Dict of loss name -> list of values per update.
        return_curves: Per-episode total returns.
        entropy: Optional per-update entropy values.
        kl: Optional per-update approximate KL values.
        td_error: Optional per-update TD errors.
        config: Optional training config dict.
        extra: Optional extra fields.

    Returns:
        Absolute path to written file.
    """
    payload: Dict[str, Any] = {
        "report_type": "rl_training",
        "algorithm": str(algorithm),
        "loss_curves": {k: _cap(v) for k, v in loss_curves.items()},
        "return_curves": _cap(return_curves),
        "mean_return": sum(return_curves) / len(return_curves) if return_curves else 0.0,
    }
    if entropy is not None:
        payload["entropy"] = _cap(entropy)
    if kl is not None:
        payload["approx_kl"] = _cap(kl)
    if td_error is not None:
        payload["td_error"] = _cap(td_error)
    if config is not None:
        payload["config"] = {k: _safe_str(v) for k, v in config.items()}
    if extra is not None:
        payload["extra"] = {k: _safe_str(v) for k, v in extra.items()}
    return _atomic_write(path, payload)

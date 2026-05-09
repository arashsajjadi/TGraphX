"""Report writers for graph generation artifacts.

All functions write atomic JSON files with NO raw tensors.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "write_graph_generation_report",
    "write_generation_metrics_report",
    "write_neural_generation_report",
    "write_sequence_model_report",
]

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


def _cap_rows(data: Any, max_rows: int = _MAX_ROWS) -> Any:
    if isinstance(data, list) and len(data) > max_rows:
        return data[:max_rows] + [f"... ({len(data) - max_rows} more)"]
    return data


def _safe_str(v: Any) -> Any:
    """Ensure JSON-safe value."""
    import torch  # local import to avoid circular dependency
    if isinstance(v, torch.Tensor):
        return {"shape": list(v.shape), "dtype": str(v.dtype)}
    return v


def write_graph_generation_report(
    path: str,
    generator_name: str,
    seed: Optional[int],
    params: Dict[str, Any],
    graph_stats: Dict[str, Any],
    feature_shapes: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Write a graph generation report artifact.

    Args:
        path: Output file path.
        generator_name: Name of the generator used.
        seed: Random seed used.
        params: Generator parameters (JSON-safe).
        graph_stats: Summary statistics (num_nodes, num_edges, density, etc.).
        feature_shapes: Dict of feature name -> shape list.
        extra: Optional extra fields.

    Returns:
        Absolute path to the written file.
    """
    payload: Dict[str, Any] = {
        "report_type": "graph_generation",
        "generator": str(generator_name),
        "seed": seed,
        "params": {k: _safe_str(v) for k, v in params.items()},
        "graph_stats": {k: _safe_str(v) for k, v in graph_stats.items()},
        "feature_shapes": feature_shapes,
    }
    if extra:
        payload["extra"] = {k: _safe_str(v) for k, v in extra.items()}
    return _atomic_write(path, payload)


def write_generation_metrics_report(
    path: str,
    validity: float,
    uniqueness: float,
    novelty: float,
    diversity: float,
    degree_dist: Optional[float] = None,
    motif_dist: Optional[float] = None,
    spectral_dist: Optional[float] = None,
    mmd_degree: Optional[float] = None,
    mmd_clustering: Optional[float] = None,
    constraint_rates: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Write a generation metrics report artifact.

    Args:
        path: Output file path.
        validity: Fraction of valid graphs.
        uniqueness: Fraction of unique graphs.
        novelty: Fraction novel vs reference.
        diversity: Average pairwise diversity.
        degree_dist: Degree distribution distance (optional).
        motif_dist: Motif distribution distance (optional).
        spectral_dist: Spectral distance (optional).
        mmd_degree: MMD on degrees (optional).
        mmd_clustering: MMD on clustering coefficients (optional).
        constraint_rates: Per-constraint satisfaction rates.
        extra: Optional extra fields.

    Returns:
        Absolute path to the written file.
    """
    payload: Dict[str, Any] = {
        "report_type": "generation_metrics",
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "diversity": diversity,
    }
    if degree_dist is not None:
        payload["degree_distribution_distance"] = degree_dist
    if motif_dist is not None:
        payload["motif_distribution_distance"] = motif_dist
    if spectral_dist is not None:
        payload["spectral_distance"] = spectral_dist
    if mmd_degree is not None:
        payload["mmd_degree"] = mmd_degree
    if mmd_clustering is not None:
        payload["mmd_clustering"] = mmd_clustering
    if constraint_rates:
        payload["constraint_rates"] = constraint_rates
    if extra:
        payload["extra"] = {k: _safe_str(v) for k, v in extra.items()}
    return _atomic_write(path, payload)


def write_neural_generation_report(
    path: str,
    loss_curves: List[float],
    validity_metrics: Dict[str, float],
    sampled_stats: List[Dict[str, Any]],
    gradient_health: Dict[str, Any],
    model_name: str = "neural_generator",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Write a neural generation training report.

    Args:
        path: Output file path.
        loss_curves: List of loss values per epoch.
        validity_metrics: Dict of metric name -> value.
        sampled_stats: List of graph summary dicts (NO raw tensors).
        gradient_health: Dict with gradient norm stats.
        model_name: Model name.
        extra: Optional extra fields.

    Returns:
        Absolute path to the written file.
    """
    payload: Dict[str, Any] = {
        "report_type": "neural_generation",
        "model": str(model_name),
        "loss_curves": _cap_rows(loss_curves),
        "validity_metrics": validity_metrics,
        "sampled_stats": _cap_rows(sampled_stats),
        "gradient_health": gradient_health,
    }
    if extra:
        payload["extra"] = {k: _safe_str(v) for k, v in extra.items()}
    return _atomic_write(path, payload)


def write_sequence_model_report(
    path: str,
    loss_curves: List[float],
    accuracy: Optional[List[float]],
    generated_examples: List[Dict[str, Any]],
    model_name: str = "sequence_model",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Write a sequence model generation report.

    Args:
        path: Output file path.
        loss_curves: List of loss per epoch.
        accuracy: Optional list of accuracy per epoch.
        generated_examples: List of example graph summary dicts (NO raw tensors).
        model_name: Model name.
        extra: Optional extra fields.

    Returns:
        Absolute path to the written file.
    """
    payload: Dict[str, Any] = {
        "report_type": "sequence_model",
        "model": str(model_name),
        "loss_curves": _cap_rows(loss_curves),
    }
    if accuracy is not None:
        payload["accuracy"] = _cap_rows(accuracy)
    payload["generated_examples"] = _cap_rows(generated_examples)
    if extra:
        payload["extra"] = {k: _safe_str(v) for k, v in extra.items()}
    return _atomic_write(path, payload)

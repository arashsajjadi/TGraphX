"""KG dashboard artifact writers.

All writers use atomic temp-file-then-rename writes and produce JSON-safe
output.  Tensor data is never included in reports — only summaries/scalars.

Stability: Beta.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "write_kg_summary",
    "write_kg_evaluation_report",
    "write_kg_training_report",
    "write_kg_model_report",
    "write_kg_gnn_report",
    "write_temporal_kg_report",
    "write_kg_reasoning_report",
    "write_kg_benchmark_report",
    "write_kg_multimodal_feature_report",
]


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


def write_kg_summary(path: str, summary: Dict[str, Any]) -> str:
    """Write ``kg_summary.json``."""
    return _atomic_write(path, summary)


def write_kg_evaluation_report(path: str, report: Dict[str, Any]) -> str:
    """Write ``kg_evaluation_report.json`` with filtered ranking metrics."""
    return _atomic_write(path, report)


def write_kg_training_report(path: str, report: Dict[str, Any]) -> str:
    """Write ``kg_training_report.json`` with loss curves and val metrics."""
    return _atomic_write(path, report)


def write_kg_model_report(path: str, report: Dict[str, Any]) -> str:
    """Write ``kg_model_report.json`` with model type and parameter count."""
    return _atomic_write(path, report)


def write_kg_gnn_report(path: str, report: Dict[str, Any]) -> str:
    """Write ``kg_gnn_report.json`` with RGCN/CompGCN status and metrics."""
    return _atomic_write(path, report)


def write_temporal_kg_report(path: str, report: Dict[str, Any]) -> str:
    """Write ``temporal_kg_report.json`` with time range and split info."""
    return _atomic_write(path, report)


def write_kg_reasoning_report(path: str, report: Dict[str, Any]) -> str:
    """Write ``kg_reasoning_report.json`` with rule support/confidence."""
    return _atomic_write(path, report)


def write_kg_benchmark_report(path: str, report: Dict[str, Any]) -> str:
    """Write ``kg_benchmark_report.json`` for a benchmark run."""
    return _atomic_write(path, report)


def write_kg_multimodal_feature_report(path: str, report: Dict[str, Any]) -> str:
    """Write ``kg_multimodal_feature_report.json`` with entity type and modality info."""
    return _atomic_write(path, report)

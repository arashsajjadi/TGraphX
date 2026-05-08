"""Mining report writers — atomic JSON artifact helpers.

These functions write structured JSON artifacts that the TGraphX
dashboard and downstream tooling can read.

All functions write to **explicit user-provided paths** only.
They never create directories outside the given path.
They use atomic writes (write to a temp file, then rename/replace).

Stability: Beta (v0.3.2+).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "write_graph_mining_summary",
    "write_motif_summary",
    "write_link_prediction_summary",
    "write_anomaly_summary",
    "write_prototype_membership_report",
]


def _atomic_write(path: str, payload: Dict[str, Any]) -> str:
    """Write JSON atomically (write to temp then rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Ensure JSON-serializable.
    text = json.dumps(payload, indent=2, default=str)
    # Atomic: write to a sibling temp file, then rename.
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


def write_graph_mining_summary(
    path: str,
    summary: Dict[str, Any],
) -> str:
    """Write ``graph_mining_summary.json``.

    Expected keys (all optional): ``num_nodes``, ``num_edges``,
    ``density``, ``degree_statistics``, ``num_connected_components``,
    ``motif_counts``, ``community_summary``, ``warnings``.

    Args:
        path: Output file path.
        summary: Dict from :func:`~tgraphx.mining.structural.graph_summary`
            or a custom dict.

    Returns:
        Resolved path string.
    """
    return _atomic_write(path, summary)


def write_motif_summary(
    path: str,
    summary: Dict[str, Any],
) -> str:
    """Write ``motif_summary.json``.

    Expected keys: ``triangles``, ``wedges``, ``mean_clustering_coefficient``.

    Args:
        path: Output file path.
        summary: Dict from :func:`~tgraphx.mining.motifs.motif_counts`.

    Returns:
        Resolved path string.
    """
    return _atomic_write(path, summary)


def write_link_prediction_summary(
    path: str,
    pairs: Any,
    scores: Dict[str, Any],
) -> str:
    """Write ``link_prediction_summary.json``.

    Args:
        path: Output file path.
        pairs: Candidate edge pairs (will be serialized as lists).
        scores: Dict of ``{scorer_name: score_list}`` from the classical
            link prediction scoring functions.

    Returns:
        Resolved path string.
    """
    if hasattr(pairs, "tolist"):
        pairs = pairs.tolist()
    serializable_scores = {
        k: (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in scores.items()
    }
    payload = {"pairs": pairs, "scores": serializable_scores}
    return _atomic_write(path, payload)


def write_anomaly_summary(
    path: str,
    method: str,
    node_scores: Any,
    top_k: int = 20,
    threshold: Optional[float] = None,
    graph_scores: Optional[Any] = None,
) -> str:
    """Write ``anomaly_summary.json``.

    Args:
        path: Output file path.
        method: Name of the scoring method.
        node_scores: ``FloatTensor[N]`` or list of node anomaly scores.
        top_k: Number of top-anomalous nodes to record.
        threshold: Optional decision threshold.
        graph_scores: Optional graph-level scores.

    Returns:
        Resolved path string.
    """
    if hasattr(node_scores, "tolist"):
        scores_list = node_scores.tolist()
    else:
        scores_list = list(node_scores)

    sorted_pairs = sorted(
        enumerate(scores_list), key=lambda x: -x[1],
    )[:top_k]
    payload: Dict[str, Any] = {
        "method": method,
        "num_nodes": len(scores_list),
        "top_anomalous_nodes": [
            {"node_id": int(nid), "score": round(float(s), 6)}
            for nid, s in sorted_pairs
        ],
    }
    if threshold is not None:
        payload["threshold"] = float(threshold)
        payload["num_flagged"] = sum(1 for s in scores_list if s > threshold)
    if graph_scores is not None:
        if hasattr(graph_scores, "tolist"):
            graph_scores = graph_scores.tolist()
        payload["graph_anomaly_scores"] = [round(float(s), 6) for s in graph_scores]
    return _atomic_write(path, payload)


def write_prototype_membership_report(
    path: str,
    report: Dict[str, Any],
) -> str:
    """Write ``prototype_membership_report.json``.

    Args:
        path: Output file path.
        report: Dict from :meth:`~tgraphx.mining.prototype.MembershipEvaluator.evaluate`.

    Returns:
        Resolved path string.
    """
    return _atomic_write(path, report)

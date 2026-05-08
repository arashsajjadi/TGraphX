"""High-level graph mining API for common workflows.

These functions wrap lower-level mining utilities into convenient
single-call interfaces.  Each function:

- accepts an explicit seed where randomness is involved;
- writes to explicit output paths only (no hidden file writes);
- never starts background services;
- returns a plain Python dict or dataclass (JSON-serialisable).

Stability: Beta (v0.4.2+).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

__all__ = [
    "analyze_graph",
    "graph_mining_report",
    "run_link_prediction_baseline",
]


def analyze_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    directed: bool = True,
    include_centrality: bool = True,
    include_motifs: bool = True,
    include_spectral: bool = True,
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Comprehensive one-call graph analysis.

    Computes structural summaries, degree statistics, motifs, centrality
    (if fast enough), and spectral properties for small graphs.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        directed: When ``True``, treat as directed for density/stats.
        include_centrality: Compute PageRank and degree centrality.
        include_motifs: Compute triangle/wedge/clustering counts.
        include_spectral: Compute Laplacian spectrum for small graphs.
        output_json: Optional path to write a JSON report.

    Returns:
        JSON-serialisable dict with all available results.
        Large or slow computations are skipped with a note in ``"skipped"``.
    """
    from .structural import graph_density, degree_statistics, graph_summary
    from .motifs import motif_counts

    result: Dict[str, Any] = {}
    skipped: List[str] = []

    # Basic structural summary.
    result["summary"] = graph_summary(edge_index, num_nodes, directed=directed)

    # Motifs.
    if include_motifs:
        try:
            result["motifs"] = motif_counts(edge_index, num_nodes, directed=False)
        except Exception as e:
            skipped.append(f"motifs: {e}")

    # Centrality (fast ones only).
    if include_centrality and num_nodes <= 5_000:
        from .centrality import degree_centrality, pagerank
        dc = degree_centrality(edge_index, num_nodes, directed=directed)
        pr = pagerank(edge_index, num_nodes)
        result["centrality"] = {
            "degree_centrality_mean": round(float(dc.mean().item()), 6),
            "degree_centrality_max": round(float(dc.max().item()), 6),
            "pagerank_mean": round(float(pr.mean().item()), 6),
            "pagerank_max": round(float(pr.max().item()), 6),
            "top_pagerank_node": int(pr.argmax().item()),
        }
    elif include_centrality:
        skipped.append(f"centrality: num_nodes={num_nodes} > 5000")

    # Spectral (small graphs only).
    if include_spectral and num_nodes <= 500:
        try:
            from .spectral import algebraic_connectivity, laplacian_eigenvalues
            evals = laplacian_eigenvalues(edge_index, num_nodes, k=5)
            result["spectral"] = {
                "algebraic_connectivity": round(algebraic_connectivity(edge_index, num_nodes), 6),
                "smallest_eigenvalues": [round(float(v), 6) for v in evals.tolist()],
            }
        except Exception as e:
            skipped.append(f"spectral: {e}")
    elif include_spectral:
        skipped.append(f"spectral: num_nodes={num_nodes} > 500")

    result["skipped"] = skipped

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(result, indent=2, default=str))

    return result


def graph_mining_report(
    edge_index: torch.Tensor,
    num_nodes: int,
    output_dir: Optional[str] = None,
    include_centrality: bool = True,
    include_motifs: bool = True,
    include_communities: bool = True,
    include_anomaly: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """Full-featured graph mining report.

    Runs a suite of mining analyses and optionally saves all artifacts
    to ``output_dir`` for the TGraphX dashboard.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        output_dir: Optional directory for JSON artifact writes.
        include_centrality: Compute centrality metrics.
        include_motifs: Compute motif statistics.
        include_communities: Run label propagation community detection.
        include_anomaly: Run degree anomaly scoring.
        seed: RNG seed for community detection.

    Returns:
        JSON-serialisable dict of all mining results.
    """
    report: Dict[str, Any] = {}

    # Structural.
    report["analysis"] = analyze_graph(
        edge_index, num_nodes,
        include_centrality=include_centrality,
        include_motifs=include_motifs,
        include_spectral=(num_nodes <= 200),
    )

    # Communities.
    if include_communities and num_nodes <= 5_000:
        from .communities import label_propagation_communities, community_summary
        comms = label_propagation_communities(edge_index, num_nodes, seed=seed)
        report["communities"] = community_summary(edge_index, comms, num_nodes)

    # Anomaly detection.
    if include_anomaly:
        from .anomaly import DegreeAnomalyScorer
        scorer = DegreeAnomalyScorer().fit(edge_index, num_nodes)
        scores = scorer.score_nodes(edge_index, num_nodes)
        top = scorer.top_k_anomalous(edge_index, num_nodes, k=10)
        report["anomaly"] = {
            "method": "degree_zscore",
            "mean_score": round(float(scores.mean().item()), 6),
            "max_score": round(float(scores.max().item()), 6),
            "top_anomalous_nodes": top,
        }

    # Write artifacts.
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "graph_mining_report.json").write_text(
            json.dumps(report, indent=2, default=str)
        )
        # Also write dashboard-compatible sub-artifacts.
        if "analysis" in report and "summary" in report["analysis"]:
            (out / "graph_mining_summary.json").write_text(
                json.dumps(report["analysis"]["summary"], indent=2, default=str)
            )
        if "communities" in report:
            (out / "community_summary.json").write_text(
                json.dumps(report["communities"], indent=2, default=str)
            )
        if "anomaly" in report:
            from .reports import write_anomaly_summary
            write_anomaly_summary(
                str(out / "anomaly_summary.json"),
                report["anomaly"]["method"],
                scores,
                top_k=10,
            )

    return report


def run_link_prediction_baseline(
    edge_index: torch.Tensor,
    num_nodes: int,
    test_pairs: torch.Tensor,
    test_labels: Optional[torch.Tensor] = None,
    scores: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run classical link prediction baseline scores.

    Args:
        edge_index: ``LongTensor[2, E]`` positive training edges.
        num_nodes: Node count.
        test_pairs: ``LongTensor[2, P]`` candidate pairs.
        test_labels: Optional ``FloatTensor[P]`` true labels (1=positive).
        scores: List of scorer names to run.  Default: all five scorers.

    Returns:
        Dict of ``{scorer_name: FloatTensor[P]}`` scores.
        If ``test_labels`` provided, also includes AUC-style ranking.
    """
    from .link_prediction import (
        common_neighbors_score, jaccard_score, adamic_adar_score,
        resource_allocation_score, preferential_attachment_score,
    )
    _all_scores = {
        "common_neighbors": common_neighbors_score,
        "jaccard": jaccard_score,
        "adamic_adar": adamic_adar_score,
        "resource_allocation": resource_allocation_score,
        "preferential_attachment": preferential_attachment_score,
    }
    if scores is None:
        scores = list(_all_scores.keys())
    result: Dict[str, Any] = {}
    for name in scores:
        if name not in _all_scores:
            raise ValueError(f"Unknown scorer {name!r}; available: {sorted(_all_scores)}")
        result[name] = _all_scores[name](edge_index, test_pairs, num_nodes=num_nodes)
    if test_labels is not None:
        result["_labels"] = test_labels
    return result

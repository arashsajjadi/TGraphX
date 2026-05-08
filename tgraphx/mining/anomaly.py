"""Graph anomaly detection baselines.

Classical, non-trainable anomaly scorers based on structural properties.
These are **baselines**, not state-of-the-art anomaly detection systems.
They are useful for identifying obviously anomalous nodes or graphs in
exploratory graph mining.

Stability: Experimental (v0.3.2+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

__all__ = [
    "DegreeAnomalyScorer",
    "EgoDensityAnomalyScorer",
    "graph_level_anomaly_scores",
]


def _degrees(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Total degree (in + out)."""
    deg = torch.zeros(num_nodes, dtype=torch.float)
    if edge_index.numel():
        ones = torch.ones(edge_index.size(1), dtype=torch.float)
        deg.scatter_add_(0, edge_index[0], ones)
        deg.scatter_add_(0, edge_index[1], ones)
    return deg


class DegreeAnomalyScorer:
    """Score nodes by how anomalous their degree is relative to the graph.

    Uses a robust z-score based on median and MAD (median absolute
    deviation) so that a few extreme hubs do not distort the baseline.

    Stability: Experimental.
    """

    def __init__(self) -> None:
        self._median: Optional[float] = None
        self._mad: Optional[float] = None
        self._fitted = False

    def fit(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> "DegreeAnomalyScorer":
        """Fit the scorer on the training graph.

        Args:
            edge_index: ``LongTensor[2, E]``.
            num_nodes: Node count.

        Returns:
            self (fluent API).
        """
        deg = _degrees(edge_index, num_nodes)
        self._median = float(deg.median().item())
        self._mad = float((deg - self._median).abs().median().item())
        self._fitted = True
        return self

    def score_nodes(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Return anomaly z-scores for each node.

        Higher score = more anomalous.

        Args:
            edge_index: ``LongTensor[2, E]``.
            num_nodes: Node count.

        Returns:
            ``FloatTensor[num_nodes]``, non-negative.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before score_nodes().")
        deg = _degrees(edge_index, num_nodes)
        mad = max(self._mad, 1e-8)
        scores = (deg - self._median).abs() / mad
        return scores.to(torch.float)

    def top_k_anomalous(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Return the top-k most anomalous nodes and their scores.

        Args:
            edge_index: ``LongTensor[2, E]``.
            num_nodes: Node count.
            k: Number of top anomalous nodes.

        Returns:
            Dict with keys ``"node_ids"`` and ``"scores"`` (plain Python lists).
        """
        scores = self.score_nodes(edge_index, num_nodes)
        k_actual = min(k, num_nodes)
        top_scores, top_idx = scores.topk(k_actual)
        return {
            "node_ids": top_idx.tolist(),
            "scores": [round(float(s), 6) for s in top_scores.tolist()],
        }


class EgoDensityAnomalyScorer:
    """Score nodes by deviation of ego-network density from graph mean.

    The ego-network of node v is the induced subgraph over v and its
    direct neighbours.  High ego-density anomaly = the ego-net is much
    denser or sparser than average.

    Stability: Experimental.
    """

    def __init__(self, min_ego_size: int = 2) -> None:
        self._mean_density: Optional[float] = None
        self._std_density: Optional[float] = None
        self._fitted = False
        self.min_ego_size = min_ego_size

    def _ego_density(
        self,
        v: int,
        adj: List[set],
        num_nodes: int,
    ) -> float:
        nbrs = adj[v]
        ego_nodes = {v} | nbrs
        n = len(ego_nodes)
        if n < self.min_ego_size:
            return 0.0
        # Count edges in ego subgraph.
        ego_list = list(ego_nodes)
        ego_set = ego_nodes
        edges = 0
        for u in ego_list:
            for w in adj[u]:
                if w in ego_set:
                    edges += 1
        max_edges = n * (n - 1)
        if max_edges == 0:
            return 0.0
        return float(edges) / float(max_edges)

    def fit(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> "EgoDensityAnomalyScorer":
        """Fit on a training graph."""
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, E]")
        adj: List[set] = [set() for _ in range(num_nodes)]
        if edge_index.numel():
            src = edge_index[0].cpu().tolist()
            dst = edge_index[1].cpu().tolist()
            for u, v in zip(src, dst):
                adj[u].add(v)
                adj[v].add(u)
        densities = [self._ego_density(v, adj, num_nodes) for v in range(num_nodes)]
        t = torch.tensor(densities, dtype=torch.float)
        self._mean_density = float(t.mean().item())
        self._std_density = float(t.std().item()) if num_nodes > 1 else 1.0
        self._adj = adj
        self._fitted = True
        return self

    def score_nodes(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Return ego-density anomaly z-scores for each node."""
        if not self._fitted:
            raise RuntimeError("Call fit() before score_nodes().")
        adj: List[set] = [set() for _ in range(num_nodes)]
        if edge_index.numel():
            src = edge_index[0].cpu().tolist()
            dst = edge_index[1].cpu().tolist()
            for u, v in zip(src, dst):
                adj[u].add(v)
                adj[v].add(u)
        densities = torch.tensor(
            [self._ego_density(v, adj, num_nodes) for v in range(num_nodes)],
            dtype=torch.float,
        )
        std = max(self._std_density, 1e-8)
        return ((densities - self._mean_density) / std).abs()


def graph_level_anomaly_scores(
    graphs: list,
    method: str = "degree_histogram",
) -> torch.Tensor:
    """Score each graph by its distance from the dataset centroid.

    Higher score = further from the average graph in feature space.

    Args:
        graphs: List of graph dicts or TGraphX :class:`~tgraphx.Graph`
            objects.
        method: ``"degree_histogram"`` (default) or ``"wl"``.

    Returns:
        ``FloatTensor[G]`` of anomaly scores, non-negative.
    """
    if method == "degree_histogram":
        from .kernels import degree_histogram_features
        feat = degree_histogram_features(graphs)
    elif method == "wl":
        from .kernels import wl_graph_features
        feat, _ = wl_graph_features(graphs)
    else:
        raise ValueError(f"method must be 'degree_histogram' or 'wl'; got {method!r}")

    if feat.size(0) == 0:
        return torch.zeros(0, dtype=torch.float)
    centroid = feat.mean(dim=0, keepdim=True)
    dists = (feat - centroid).norm(dim=1)
    return dists

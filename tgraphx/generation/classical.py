"""Feature-aware and extended classical graph generators.

Builds on existing ``tgraphx.mining.generators`` — does NOT duplicate them.

All functions return ``GeneratedGraph`` instances with optional tensor features.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from tgraphx.mining.generators import (
    erdos_renyi_graph,
    barabasi_albert_graph,
    synthetic_anomaly_graph,
    motif_injected_graph,
)
from .data_model import GeneratedGraph

__all__ = [
    "FeatureAwareERGraph",
    "FeatureAwareBAGraph",
    "TemporalEvolvingGraph",
    "TypedGeneratedGraph",
    "AnomalyInjectedGraph",
    "MotifInjectedGraph",
]

_MAX_NODES = 10_000


def _make_rng(seed: Optional[int]) -> torch.Generator:
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(int(seed))
    return g


def FeatureAwareERGraph(
    n: int,
    p: float,
    node_feature_dim: int = 8,
    edge_feature_dim: Optional[int] = None,
    directed: bool = False,
    seed: Optional[int] = None,
) -> GeneratedGraph:
    """Erdős-Rényi graph with normal-distributed node/edge features.

    Calls the existing ``erdos_renyi_graph`` and attaches tensor features.

    Args:
        n: Number of nodes.
        p: Edge probability.
        node_feature_dim: Feature dimensionality for nodes (produces [N, F]).
        edge_feature_dim: Feature dimensionality for edges. None means no edge features.
        directed: Whether to generate a directed graph.
        seed: Optional RNG seed.

    Returns:
        GeneratedGraph with node_features [N, node_feature_dim].
    """
    if n > _MAX_NODES:
        raise ValueError(f"n={n} > {_MAX_NODES}")
    rng = _make_rng(seed)
    edge_index, num_nodes = erdos_renyi_graph(n, p, directed=directed, seed=seed)
    num_edges = int(edge_index.shape[1])

    node_features = torch.randn(num_nodes, node_feature_dim, generator=rng)
    edge_features: Optional[torch.Tensor] = None
    if edge_feature_dim is not None and num_edges > 0:
        edge_features = torch.randn(num_edges, edge_feature_dim, generator=rng)

    return GeneratedGraph(
        edge_index=edge_index,
        num_nodes=num_nodes,
        directed=directed,
        node_features=node_features,
        edge_features=edge_features,
        metadata={"generator": "ER", "n": n, "p": p},
    )


def FeatureAwareBAGraph(
    n: int,
    m: int,
    node_feature_dim: int = 8,
    edge_feature_dim: Optional[int] = None,
    seed: Optional[int] = None,
) -> GeneratedGraph:
    """Barabási-Albert graph with normal-distributed node/edge features.

    Calls the existing ``barabasi_albert_graph`` and attaches tensor features.

    Args:
        n: Number of nodes.
        m: Number of edges per new node.
        node_feature_dim: Feature dimensionality for nodes.
        edge_feature_dim: Feature dimensionality for edges. None means no edge features.
        seed: Optional RNG seed.

    Returns:
        GeneratedGraph with node_features [N, node_feature_dim].
    """
    if n > _MAX_NODES:
        raise ValueError(f"n={n} > {_MAX_NODES}")
    rng = _make_rng(seed)
    edge_index, num_nodes = barabasi_albert_graph(n, m, seed=seed)
    num_edges = int(edge_index.shape[1])

    node_features = torch.randn(num_nodes, node_feature_dim, generator=rng)
    edge_features: Optional[torch.Tensor] = None
    if edge_feature_dim is not None and num_edges > 0:
        edge_features = torch.randn(num_edges, edge_feature_dim, generator=rng)

    return GeneratedGraph(
        edge_index=edge_index,
        num_nodes=num_nodes,
        directed=False,
        node_features=node_features,
        edge_features=edge_features,
        metadata={"generator": "BA", "n": n, "m": m},
    )


def TemporalEvolvingGraph(
    n: int,
    steps: int,
    edge_add_prob: float = 0.1,
    edge_remove_prob: float = 0.05,
    seed: Optional[int] = None,
) -> GeneratedGraph:
    """A graph that evolves over ``steps`` discrete time steps.

    At each step, edges are randomly added/removed. The final graph holds
    all surviving edges with timestamps indicating when they were added.

    Args:
        n: Number of nodes (fixed throughout).
        steps: Number of evolution steps.
        edge_add_prob: Probability of adding each potential edge per step.
        edge_remove_prob: Probability of removing each existing edge per step.
        seed: Optional RNG seed.

    Returns:
        GeneratedGraph with ``timestamps`` on edges (float, step when added).
    """
    if n > _MAX_NODES:
        raise ValueError(f"n={n} > {_MAX_NODES}")
    rng = _make_rng(seed)

    # Current edge set: maps (src, dst) -> timestamp
    edge_map: Dict[tuple, float] = {}

    for step in range(steps):
        # Add edges
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (i, j) not in edge_map:
                    if torch.rand(1, generator=rng).item() < edge_add_prob:
                        edge_map[(i, j)] = float(step)

        # Remove edges
        to_remove = [
            k for k in list(edge_map.keys())
            if torch.rand(1, generator=rng).item() < edge_remove_prob
        ]
        for k in to_remove:
            del edge_map[k]

    if edge_map:
        edges_list = list(edge_map.keys())
        ts_list = list(edge_map.values())
        src_arr = [e[0] for e in edges_list]
        dst_arr = [e[1] for e in edges_list]
        edge_index = torch.tensor([src_arr, dst_arr], dtype=torch.long)
        timestamps = torch.tensor(ts_list, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        timestamps = torch.zeros(0, dtype=torch.float)

    return GeneratedGraph(
        edge_index=edge_index,
        num_nodes=n,
        directed=True,
        timestamps=timestamps,
        metadata={"generator": "TemporalEvolving", "n": n, "steps": steps},
    )


def TypedGeneratedGraph(
    n: int,
    node_types_list: List[int],
    type_edge_probs: Dict[tuple, float],
    node_feature_dims_by_type: Dict[int, int],
    seed: Optional[int] = None,
) -> GeneratedGraph:
    """Graph with multiple node types and type-conditioned edge probabilities.

    Different node types can have different feature dimensions. Node features
    are stacked to the maximum feature dim (zero-padded for smaller types).

    Args:
        n: Number of nodes.
        node_types_list: List of length n assigning a type to each node.
            Types must be non-negative integers.
        type_edge_probs: Dict mapping (type_a, type_b) -> edge probability.
        node_feature_dims_by_type: Dict mapping type -> feature dimension.
        seed: Optional RNG seed.

    Returns:
        GeneratedGraph with node_types [N] and node_features [N, max_feat_dim].
    """
    if n > _MAX_NODES:
        raise ValueError(f"n={n} > {_MAX_NODES}")
    if len(node_types_list) != n:
        raise ValueError(
            f"node_types_list length {len(node_types_list)} != n={n}"
        )

    rng = _make_rng(seed)
    node_types_t = torch.tensor(node_types_list, dtype=torch.long)

    max_feat_dim = max(node_feature_dims_by_type.values()) if node_feature_dims_by_type else 1
    node_features = torch.zeros(n, max_feat_dim)
    for i, nt in enumerate(node_types_list):
        fdim = node_feature_dims_by_type.get(nt, max_feat_dim)
        feat = torch.randn(fdim, generator=rng)
        node_features[i, :fdim] = feat

    src_list: List[int] = []
    dst_list: List[int] = []
    for i in range(n):
        for j in range(i + 1, n):
            ti = node_types_list[i]
            tj = node_types_list[j]
            p = type_edge_probs.get((ti, tj), type_edge_probs.get((tj, ti), 0.1))
            if torch.rand(1, generator=rng).item() < p:
                src_list.extend([i, j])
                dst_list.extend([j, i])

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return GeneratedGraph(
        edge_index=edge_index,
        num_nodes=n,
        directed=False,
        node_features=node_features,
        node_types=node_types_t,
        metadata={
            "generator": "TypedGeneratedGraph",
            "n": n,
            "node_types": node_types_list,
        },
    )


def AnomalyInjectedGraph(
    base_graph: GeneratedGraph,
    anomaly_fraction: float = 0.05,
    anomaly_type: str = "random_edges",
    seed: Optional[int] = None,
) -> GeneratedGraph:
    """Inject structural anomalies into a base graph.

    Anomaly types:
        - ``"random_edges"`` — add random extra edges between random nodes
        - ``"clique"`` — add a small clique of anomalous nodes
        - ``"star"`` — add a hub node connecting to many random nodes

    Anomaly labels are stored in metadata as ``"anomaly_labels"``
    (per-node BoolTensor).

    Args:
        base_graph: Source GeneratedGraph.
        anomaly_fraction: Fraction of nodes involved in anomalies.
        anomaly_type: Type of structural anomaly.
        seed: Optional RNG seed.

    Returns:
        New GeneratedGraph with injected anomalies.
    """
    rng = _make_rng(seed)
    g = base_graph.clone()
    n = g.num_nodes
    n_anomaly = max(1, int(n * anomaly_fraction))

    anomaly_labels = torch.zeros(n, dtype=torch.bool)

    def _extend_edges(g: GeneratedGraph, new_src: list, new_dst: list) -> None:
        """Add new edges and extend edge_features/edge_weight if present."""
        if not new_src:
            return
        new_edges = torch.tensor([new_src, new_dst], dtype=torch.long)
        n_new = new_edges.shape[1]
        g.edge_index = torch.cat([g.edge_index, new_edges], dim=1)
        if g.edge_features is not None:
            feat_shape = list(g.edge_features.shape[1:])
            padding = torch.zeros(n_new, *feat_shape, dtype=g.edge_features.dtype)
            g.edge_features = torch.cat([g.edge_features, padding], dim=0)
        if g.edge_weight is not None:
            padding_w = torch.zeros(n_new, dtype=g.edge_weight.dtype)
            g.edge_weight = torch.cat([g.edge_weight, padding_w], dim=0)

    if anomaly_type == "random_edges":
        # Use existing synthetic_anomaly_graph logic as reference
        perm = torch.randperm(n, generator=rng)
        anom_nodes = perm[:n_anomaly].tolist()
        for ni in anom_nodes:
            anomaly_labels[ni] = True

        new_src = []
        new_dst = []
        for _ in range(n_anomaly * 2):
            s = anom_nodes[int(torch.randint(n_anomaly, (1,), generator=rng).item())]
            d = int(torch.randint(n, (1,), generator=rng).item())
            if s != d:
                new_src.append(s)
                new_dst.append(d)

        _extend_edges(g, new_src, new_dst)

    elif anomaly_type == "clique":
        perm = torch.randperm(n, generator=rng)
        anom_nodes = perm[:n_anomaly].tolist()
        for ni in anom_nodes:
            anomaly_labels[ni] = True

        new_src, new_dst = [], []
        for i, u in enumerate(anom_nodes):
            for v in anom_nodes[i + 1:]:
                new_src.extend([u, v])
                new_dst.extend([v, u])

        _extend_edges(g, new_src, new_dst)

    elif anomaly_type == "star":
        # Hub node
        hub = int(torch.randint(n, (1,), generator=rng).item())
        anomaly_labels[hub] = True
        spokes = torch.randperm(n, generator=rng)[:n_anomaly].tolist()
        for s in spokes:
            anomaly_labels[s] = True

        new_src, new_dst = [], []
        for s in spokes:
            if s != hub:
                new_src.extend([hub, s])
                new_dst.extend([s, hub])

        _extend_edges(g, new_src, new_dst)

    else:
        raise ValueError(
            f"Unknown anomaly_type={anomaly_type!r}. "
            f"Choose from 'random_edges', 'clique', 'star'."
        )

    g.metadata["anomaly_labels"] = anomaly_labels
    g.metadata["anomaly_type"] = anomaly_type
    g.metadata["anomaly_fraction"] = anomaly_fraction
    return g


def MotifInjectedGraph(
    base_graph: GeneratedGraph,
    motif_type: str = "triangle",
    motif_count: int = 5,
    seed: Optional[int] = None,
) -> GeneratedGraph:
    """Inject motif subgraphs into a base graph.

    Motif types:
        - ``"triangle"`` — inject triangles (cliques of 3)
        - ``"path"`` — inject paths of length 3
        - ``"cycle"`` — inject cycles of length 4

    Motif node labels stored in metadata as ``"motif_labels"`` (LongTensor [N],
    -1 = not in motif, 0..k = motif ID).

    Args:
        base_graph: Source GeneratedGraph.
        motif_type: Type of motif to inject.
        motif_count: How many motifs to inject.
        seed: Optional RNG seed.

    Returns:
        New GeneratedGraph with injected motifs.
    """
    rng = _make_rng(seed)
    g = base_graph.clone()
    n = g.num_nodes

    motif_labels = torch.full((n,), -1, dtype=torch.long)

    new_src, new_dst = [], []

    for mid in range(motif_count):
        if motif_type == "triangle":
            if n < 3:
                break
            nodes = torch.randperm(n, generator=rng)[:3].tolist()
            for ni in nodes:
                motif_labels[ni] = mid
            a, b, c = nodes
            new_src.extend([a, b, b, c, a, c])
            new_dst.extend([b, a, c, b, c, a])

        elif motif_type == "path":
            if n < 4:
                break
            nodes = torch.randperm(n, generator=rng)[:4].tolist()
            for ni in nodes:
                motif_labels[ni] = mid
            a, b, c, d = nodes
            new_src.extend([a, b, b, c, c, d])
            new_dst.extend([b, a, c, b, d, c])

        elif motif_type == "cycle":
            if n < 4:
                break
            nodes = torch.randperm(n, generator=rng)[:4].tolist()
            for ni in nodes:
                motif_labels[ni] = mid
            a, b, c, d = nodes
            new_src.extend([a, b, b, c, c, d, d, a])
            new_dst.extend([b, a, c, b, d, c, a, d])

        else:
            raise ValueError(
                f"Unknown motif_type={motif_type!r}. "
                f"Choose from 'triangle', 'path', 'cycle'."
            )

    if new_src:
        new_edges = torch.tensor([new_src, new_dst], dtype=torch.long)
        n_new = new_edges.shape[1]
        g.edge_index = torch.cat([g.edge_index, new_edges], dim=1)
        if g.edge_features is not None:
            feat_shape = list(g.edge_features.shape[1:])
            padding = torch.zeros(n_new, *feat_shape, dtype=g.edge_features.dtype)
            g.edge_features = torch.cat([g.edge_features, padding], dim=0)
        if g.edge_weight is not None:
            padding_w = torch.zeros(n_new, dtype=g.edge_weight.dtype)
            g.edge_weight = torch.cat([g.edge_weight, padding_w], dim=0)

    g.metadata["motif_labels"] = motif_labels
    g.metadata["motif_type"] = motif_type
    g.metadata["motif_count"] = motif_count
    return g

"""Structural and positional encodings for graph neural networks.

These utilities compute node-level structural features that can be
used as positional encodings in graph transformers, or concatenated
to node features in standard GNN architectures.

All encodings are deterministic for a given graph (given a seed where
randomness is involved).

Stability: Beta (v0.4.3+).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "degree_encoding",
    "random_walk_structural_encoding",
    "shortest_path_anchor_encoding",
    "centrality_encoding",
    "community_encoding",
    "StructuralEncodingModule",
    "attach_structural_encodings",
]


def degree_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_degree: int = 128,
    normalize: bool = True,
) -> torch.Tensor:
    """Per-node degree feature (in-degree and out-degree).

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        max_degree: Values clipped to ``max_degree`` before normalisation.
        normalize: When ``True``, divide by ``max_degree``.

    Returns:
        ``FloatTensor[N, 2]`` columns: [out_degree, in_degree] (normalised).
    """
    device = edge_index.device
    out_d = torch.zeros(num_nodes, dtype=torch.float, device=device)
    in_d = torch.zeros(num_nodes, dtype=torch.float, device=device)
    if edge_index.numel():
        ones = torch.ones(edge_index.size(1), dtype=torch.float, device=device)
        out_d.scatter_add_(0, edge_index[0], ones)
        in_d.scatter_add_(0, edge_index[1], ones)
    out_d = out_d.clamp(max=float(max_degree))
    in_d = in_d.clamp(max=float(max_degree))
    if normalize:
        out_d = out_d / float(max_degree)
        in_d = in_d / float(max_degree)
    return torch.stack([out_d, in_d], dim=1)


def random_walk_structural_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    walk_length: int = 8,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Random-walk landing probabilities as structural encoding (RRWP).

    For each node v: ``p_k[v] = prob. of landing on v after k steps
    starting from v``.  Approximated via vectorised power of the row-normalised
    adjacency matrix.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        walk_length: Number of walk steps (= encoding dimension).
        seed: Unused (for API consistency); RRWP is deterministic.

    Returns:
        ``FloatTensor[N, walk_length]``.

    Notes:
        Dense N×N computation; guarded to ``num_nodes <= 2 000``.
    """
    if num_nodes > 2_000:
        raise ValueError(
            f"random_walk_structural_encoding: num_nodes={num_nodes} > 2000. "
            "Use a sparse/sampling-based approximation for large graphs."
        )
    device = edge_index.device
    N = num_nodes
    # Row-normalised adjacency.
    deg = torch.zeros(N, dtype=torch.float, device=device)
    if edge_index.numel():
        ones = torch.ones(edge_index.size(1), dtype=torch.float, device=device)
        deg.scatter_add_(0, edge_index[0], ones)
    D_inv = 1.0 / deg.clamp(min=1.0)
    # Sparse RW matrix: no dense allocation for propagation.
    # We compute RW diagonal (landing prob) iteratively using message passing.
    # P = D^{-1} A, landing prob at v after k steps = P^k[v,v].
    # Use a basis-vector approach: for each node, run k steps of propagation.
    # Approximation: use mean landing prob via vectorised power.
    # For exact: compute dense P^k. Acceptable for N <= 2000.
    A = torch.zeros(N, N, dtype=torch.float, device=device)
    if edge_index.numel():
        A[edge_index[0], edge_index[1]] = 1.0
    P = A * D_inv.unsqueeze(1)  # row-normalised: P[i,j] = A[i,j] / deg[i]
    enc = torch.zeros(N, walk_length, dtype=torch.float, device=device)
    Pk = torch.eye(N, dtype=torch.float, device=device)
    for k in range(walk_length):
        Pk = Pk @ P
        enc[:, k] = Pk.diagonal()
    return enc


def shortest_path_anchor_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_anchors: int = 16,
    seed: int = 0,
    max_dist: float = 1e6,
) -> torch.Tensor:
    """Shortest-path distance encoding to random anchor nodes.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        num_anchors: Number of anchor nodes.
        seed: RNG seed for anchor selection.
        max_dist: Distance for unreachable nodes.

    Returns:
        ``FloatTensor[N, num_anchors]``.
    """
    from .paths import batched_shortest_path_length
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    anchors = torch.randperm(num_nodes, generator=gen)[:num_anchors]
    # [num_anchors, N]
    dists = batched_shortest_path_length(edge_index, anchors, num_nodes)
    dists = dists.t()  # [N, num_anchors]
    # Replace inf with max_dist, normalise by max_dist.
    dists = dists.clamp(max=float(max_dist))
    dists = dists / float(max_dist)
    return dists


def centrality_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    include: Optional[List[str]] = None,
) -> torch.Tensor:
    """Centrality features as node structural encoding.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        include: List of centrality names to include.
            Options: ``"degree"``, ``"pagerank"``, ``"eigenvector"``,
            ``"katz"``.  When ``None``, includes all four.

    Returns:
        ``FloatTensor[N, len(include)]``.
    """
    from .centrality import (
        degree_centrality, pagerank, eigenvector_centrality, katz_centrality,
    )
    if include is None:
        include = ["degree", "pagerank", "eigenvector", "katz"]
    _available = {
        "degree":      lambda: degree_centrality(edge_index, num_nodes),
        "pagerank":    lambda: pagerank(edge_index, num_nodes),
        "eigenvector": lambda: eigenvector_centrality(edge_index, num_nodes),
        "katz":        lambda: katz_centrality(edge_index, num_nodes),
    }
    cols = []
    for name in include:
        if name not in _available:
            raise ValueError(f"Unknown centrality: {name!r}. Choose from {sorted(_available)}.")
        cols.append(_available[name]())
    if not cols:
        return torch.zeros(num_nodes, 0, dtype=torch.float)
    return torch.stack(cols, dim=1)


def community_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_clusters: Optional[int] = None,
    seed: int = 0,
) -> torch.Tensor:
    """One-hot community assignment encoding.

    Uses spectral clustering or label propagation to assign communities,
    then encodes as a one-hot (or soft) feature.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        num_clusters: Number of communities.  When ``None``, uses
            ``min(5, num_nodes // 2)``.
        seed: Seed for clustering.

    Returns:
        ``FloatTensor[N, num_clusters]`` one-hot community encoding.
    """
    if num_clusters is None:
        num_clusters = max(2, min(5, num_nodes // 2))
    if num_nodes > 500:
        from .communities import label_propagation_communities
        labels = label_propagation_communities(edge_index, num_nodes, seed=seed)
    else:
        from .spectral import spectral_clustering
        labels = spectral_clustering(edge_index, num_nodes, num_clusters, seed=seed)
    # Map labels to [0, K).
    K = int(labels.max().item()) + 1
    enc = torch.zeros(num_nodes, K, dtype=torch.float, device=edge_index.device)
    enc[torch.arange(num_nodes), labels.to(edge_index.device)] = 1.0
    return enc


class StructuralEncodingModule(nn.Module):
    """Learnable projection of structural encodings.

    Projects a fixed-size structural encoding into a target dimension
    via a learned linear layer (with optional nonlinearity).

    Args:
        in_dim: Structural encoding dimension (computed externally).
        out_dim: Projected dimension.
        dropout: Dropout probability.
        use_activation: When ``True``, applies ``ReLU`` after projection.

    Stability: Experimental.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        use_activation: bool = True,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_activation = use_activation

    def forward(self, enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc: ``FloatTensor[N, in_dim]``.

        Returns:
            ``FloatTensor[N, out_dim]``.
        """
        out = self.proj(enc.float())
        if self.use_activation:
            out = torch.relu(out)
        return self.dropout(out)


def attach_structural_encodings(
    node_features: torch.Tensor,
    structural_enc: torch.Tensor,
    mode: str = "concat",
) -> torch.Tensor:
    """Attach structural encodings to node features.

    For **vector** node features ``[N, D]``: concatenates along feature dim.
    For **spatial/volumetric** node features (dim > 2): only ``"side"`` mode
    is supported (returns the structural encoding separately, not concatenated).

    Args:
        node_features: ``Tensor[N, *]``.
        structural_enc: ``FloatTensor[N, S]``.
        mode: ``"concat"`` (default for 2D) or ``"side"`` (always valid).

    Returns:
        Augmented features or structural encoding (depending on mode and shape).

    Notes:
        For spatial/volumetric features, use ``"side"`` mode and pass the
        structural encoding as a separate input to your model.
    """
    if mode == "side":
        return structural_enc
    if node_features.dim() != 2:
        raise ValueError(
            f"attach_structural_encodings: node_features has dim={node_features.dim()} "
            f"(shape {tuple(node_features.shape)}).  Use mode='side' for spatial/volumetric "
            f"features; only vector [N, D] features can be concatenated."
        )
    return torch.cat([node_features.float(), structural_enc.to(node_features.device)], dim=1)

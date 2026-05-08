"""Graph similarity and distance measures in pure PyTorch.

These are approximate structural similarity measures suitable for
exploratory graph analysis and mining.  They are **not** exact graph
isomorphism tests.

Stability: Beta (v0.3.2+).
"""
from __future__ import annotations

from typing import List, Optional

import torch

__all__ = [
    "degree_histogram_distance",
    "wl_feature_similarity",
    "graph_feature_cosine_similarity",
    "pairwise_graph_similarity",
]


def _l1(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().sum().item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = float(a.norm().item())
    nb = float(b.norm().item())
    if na < 1e-12 or nb < 1e-12:
        return 1.0 if na < 1e-12 and nb < 1e-12 else 0.0
    return float((a * b).sum().item()) / (na * nb)


def degree_histogram_distance(
    g1_edge_index: torch.Tensor,
    g1_num_nodes: int,
    g2_edge_index: torch.Tensor,
    g2_num_nodes: int,
    method: str = "l1",
) -> float:
    """L1 or L2 distance between normalised degree histograms.

    Args:
        g1_edge_index, g1_num_nodes: First graph.
        g2_edge_index, g2_num_nodes: Second graph.
        method: ``"l1"`` or ``"l2"``.

    Returns:
        Non-negative float.  0 = identical degree histograms.
    """
    from .kernels import degree_histogram_features
    graphs = [
        {"edge_index": g1_edge_index, "num_nodes": g1_num_nodes},
        {"edge_index": g2_edge_index, "num_nodes": g2_num_nodes},
    ]
    feat = degree_histogram_features(graphs)
    a, b = feat[0], feat[1]
    # Pad to same length.
    L = max(a.shape[0], b.shape[0])
    a = torch.nn.functional.pad(a, (0, L - a.shape[0]))
    b = torch.nn.functional.pad(b, (0, L - b.shape[0]))
    if method == "l1":
        return float(_l1(a, b))
    if method == "l2":
        return float((a - b).pow(2).sum().sqrt().item())
    raise ValueError(f"method must be 'l1' or 'l2'; got {method!r}")


def wl_feature_similarity(
    g1_edge_index: torch.Tensor,
    g1_num_nodes: int,
    g2_edge_index: torch.Tensor,
    g2_num_nodes: int,
    num_iterations: int = 3,
    method: str = "cosine",
) -> float:
    """Cosine or normalised dot-product similarity of WL feature vectors.

    Args:
        g1_edge_index, g1_num_nodes: First graph.
        g2_edge_index, g2_num_nodes: Second graph.
        num_iterations: WL rounds.
        method: ``"cosine"`` (default) or ``"dot"`` (raw dot product).

    Returns:
        Float.  1.0 = identical WL features; 0.0 = orthogonal.
    """
    from .kernels import wl_graph_features
    graphs = [
        {"edge_index": g1_edge_index, "num_nodes": g1_num_nodes},
        {"edge_index": g2_edge_index, "num_nodes": g2_num_nodes},
    ]
    feat, _ = wl_graph_features(graphs, num_iterations)
    a, b = feat[0], feat[1]
    if method == "cosine":
        return _cosine(a, b)
    if method == "dot":
        return float((a * b).sum().item())
    raise ValueError(f"method must be 'cosine' or 'dot'; got {method!r}")


def graph_feature_cosine_similarity(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
) -> torch.Tensor:
    """Pairwise cosine similarity between two sets of graph feature vectors.

    Args:
        features_a: ``FloatTensor[A, D]`` — feature matrix for set A.
        features_b: ``FloatTensor[B, D]`` — feature matrix for set B.

    Returns:
        ``FloatTensor[A, B]`` of cosine similarities.
    """
    if features_a.dim() != 2 or features_b.dim() != 2:
        raise ValueError("features_a and features_b must be 2-D")
    if features_a.size(1) != features_b.size(1):
        raise ValueError(
            f"Feature dimension mismatch: {features_a.size(1)} vs {features_b.size(1)}"
        )
    a_norm = features_a / (features_a.norm(dim=1, keepdim=True).clamp(min=1e-12))
    b_norm = features_b / (features_b.norm(dim=1, keepdim=True).clamp(min=1e-12))
    return a_norm @ b_norm.t()


def pairwise_graph_similarity(
    graphs: list,
    method: str = "wl",
    num_iterations: int = 3,
) -> torch.Tensor:
    """Symmetric pairwise similarity matrix ``[G, G]``.

    Args:
        graphs: List of graph dicts or TGraphX :class:`~tgraphx.Graph`
            objects (``edge_index`` + ``num_nodes``).
        method: ``"wl"`` (WL kernel, default) or ``"degree"`` (degree histogram).
        num_iterations: WL iterations (only for ``method="wl"``).

    Returns:
        ``FloatTensor[G, G]``, symmetric, diagonal ≈ 1.
    """
    if method == "wl":
        from .kernels import wl_kernel_matrix
        return wl_kernel_matrix(graphs, num_iterations=num_iterations, normalize=True)
    if method == "degree":
        from .kernels import degree_histogram_features
        feat = degree_histogram_features(graphs)
        return graph_feature_cosine_similarity(feat, feat)
    raise ValueError(f"method must be 'wl' or 'degree'; got {method!r}")

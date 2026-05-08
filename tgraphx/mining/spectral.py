"""Spectral graph analysis utilities.

These functions compute Laplacian-based graph properties useful for
spectral clustering, positional encodings, and graph signal processing.

All computations are exact for small graphs.  Size guards prevent
accidental O(N²) operations on large graphs.

Stability: Beta (v0.4.2+).
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch

__all__ = [
    "graph_laplacian",
    "normalized_laplacian",
    "laplacian_eigenvalues",
    "fiedler_vector",
    "algebraic_connectivity",
    "laplacian_eigvec_positional_encoding",
    "spectral_clustering",
    "spectral_distance",
    "dirichlet_energy",
]

_MAX_NODES_EIGEN = 2_000
_MAX_NODES_SPECTRAL_CLUSTER = 500


def _degree_vector(edge_index: torch.Tensor, num_nodes: int, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    device = edge_index.device
    deg = torch.zeros(num_nodes, dtype=torch.float, device=device)
    if edge_index.numel():
        w = weight.to(dtype=torch.float, device=device) if weight is not None else torch.ones(
            edge_index.size(1), dtype=torch.float, device=device)
        deg.scatter_add_(0, edge_index[0], w)
    return deg


def graph_laplacian(
    edge_index: torch.Tensor,
    num_nodes: int,
    weight: Optional[torch.Tensor] = None,
    max_nodes: int = _MAX_NODES_EIGEN,
) -> torch.Tensor:
    """Return the combinatorial Laplacian matrix ``L = D - A``.

    Args:
        edge_index: ``LongTensor[2, E]`` (treated as undirected).
        num_nodes: Node count.
        weight: Optional ``FloatTensor[E]`` edge weights.
        max_nodes: Dense-allocation guard.

    Returns:
        ``FloatTensor[N, N]`` Laplacian matrix.
    """
    if num_nodes > max_nodes:
        raise ValueError(f"graph_laplacian: num_nodes={num_nodes} > max_nodes={max_nodes}. "
                         "Use a sparse representation for large graphs.")
    N = num_nodes
    device = edge_index.device
    L = torch.zeros(N, N, dtype=torch.float, device=device)
    if edge_index.numel():
        # Make symmetric.
        ei_sym = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        if weight is not None:
            w_sym = torch.cat([weight, weight], dim=0).to(dtype=torch.float, device=device)
        else:
            w_sym = torch.ones(ei_sym.size(1), dtype=torch.float, device=device)
        # A matrix.
        for k in range(ei_sym.size(1)):
            i, j = int(ei_sym[0, k]), int(ei_sym[1, k])
            L[i, j] -= w_sym[k].item()
        # D matrix.
        deg = _degree_vector(ei_sym, N, w_sym)
        L.diagonal().copy_(deg)
    return L


def normalized_laplacian(
    edge_index: torch.Tensor,
    num_nodes: int,
    weight: Optional[torch.Tensor] = None,
    max_nodes: int = _MAX_NODES_EIGEN,
) -> torch.Tensor:
    """Return the symmetric normalised Laplacian ``L_norm = D^{-1/2} L D^{-1/2}``.

    Eigenvalues of L_norm are in [0, 2].

    Args:
        edge_index: ``LongTensor[2, E]`` (treated as undirected).
        num_nodes: Node count.
        weight: Optional edge weights.
        max_nodes: Dense-allocation guard.

    Returns:
        ``FloatTensor[N, N]`` symmetric normalised Laplacian.
    """
    L = graph_laplacian(edge_index, num_nodes, weight, max_nodes)
    N = num_nodes
    device = L.device
    deg = L.diagonal().clone()
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[deg == 0] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ L @ D_inv_sqrt


def laplacian_eigenvalues(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: Optional[int] = None,
    normalized: bool = True,
    max_nodes: int = _MAX_NODES_EIGEN,
) -> torch.Tensor:
    """Return the (k smallest) Laplacian eigenvalues.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        k: Number of smallest eigenvalues to return.  When ``None``,
            returns all N eigenvalues.
        normalized: Use normalised Laplacian.
        max_nodes: Dense-allocation guard.

    Returns:
        ``FloatTensor[k or N]`` non-negative eigenvalues in ascending order.
    """
    if normalized:
        L = normalized_laplacian(edge_index, num_nodes, max_nodes=max_nodes)
    else:
        L = graph_laplacian(edge_index, num_nodes, max_nodes=max_nodes)
    evals = torch.linalg.eigvalsh(L)
    evals = evals.clamp(min=0.0)  # numerical stability
    if k is not None:
        evals = evals[:min(int(k), evals.numel())]
    return evals


def fiedler_vector(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_nodes: int = _MAX_NODES_EIGEN,
) -> Tuple[torch.Tensor, float]:
    """Return the Fiedler vector (second smallest Laplacian eigenvector).

    The Fiedler vector can be used for spectral bisection.

    Args:
        edge_index: ``LongTensor[2, E]`` (treated as undirected).
        num_nodes: Node count.
        max_nodes: Dense-allocation guard.

    Returns:
        ``(fiedler_vector, algebraic_connectivity)`` tuple.
        ``fiedler_vector`` is ``FloatTensor[N]``.
    """
    L = graph_laplacian(edge_index, num_nodes, max_nodes=max_nodes)
    evals, evecs = torch.linalg.eigh(L)
    evals = evals.clamp(min=0.0)
    # Second smallest eigenvalue = algebraic connectivity (λ₂).
    if evals.numel() < 2:
        return torch.zeros(num_nodes, dtype=torch.float), 0.0
    return evecs[:, 1], float(evals[1].item())


def algebraic_connectivity(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_nodes: int = _MAX_NODES_EIGEN,
) -> float:
    """Return the algebraic connectivity (second smallest Laplacian eigenvalue).

    Zero iff the graph is disconnected.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        max_nodes: Dense guard.

    Returns:
        Non-negative float.
    """
    _, lam2 = fiedler_vector(edge_index, num_nodes, max_nodes)
    return lam2


def laplacian_eigvec_positional_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 8,
    max_nodes: int = _MAX_NODES_EIGEN,
    normalized: bool = True,
) -> torch.Tensor:
    """Return the k smallest Laplacian eigenvectors as positional encodings.

    A common positional encoding for Graph Transformers (Dwivedi et al.).

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        k: Number of eigenvectors to return.
        max_nodes: Dense guard.
        normalized: Use normalised Laplacian.

    Returns:
        ``FloatTensor[N, k]`` — k smallest Laplacian eigenvectors,
        excluding the trivial zero eigenvector.  If the graph has fewer
        than k+1 eigenvectors, the output is zero-padded.
    """
    if normalized:
        L = normalized_laplacian(edge_index, num_nodes, max_nodes=max_nodes)
    else:
        L = graph_laplacian(edge_index, num_nodes, max_nodes=max_nodes)
    evals, evecs = torch.linalg.eigh(L)
    evals = evals.clamp(min=0.0)
    # Skip the first eigenvector (constant, zero eigenvalue).
    evecs = evecs[:, 1:]
    k_actual = min(k, evecs.size(1))
    enc = torch.zeros(num_nodes, k, dtype=torch.float, device=edge_index.device)
    enc[:, :k_actual] = evecs[:, :k_actual]
    return enc


def spectral_clustering(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_clusters: int,
    max_nodes: int = _MAX_NODES_SPECTRAL_CLUSTER,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Simple spectral clustering using the k smallest Laplacian eigenvectors.

    Uses normalised Laplacian + k-means on the eigenvectors.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        num_clusters: Number of clusters.
        max_nodes: Dense guard.
        seed: RNG seed for k-means initialisation.

    Returns:
        ``LongTensor[N]`` cluster assignments in ``[0, num_clusters)``.
    """
    if num_nodes > max_nodes:
        raise ValueError(f"spectral_clustering: num_nodes={num_nodes} > max_nodes={max_nodes}.")
    k = min(num_clusters, num_nodes)
    enc = laplacian_eigvec_positional_encoding(
        edge_index, num_nodes, k=k, max_nodes=max_nodes,
    )
    # Simple Lloyd k-means on CPU.
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(int(seed))
    # Random initialisation.
    perm = torch.randperm(num_nodes, generator=rng)[:k]
    centroids = enc[perm].clone()
    labels = torch.zeros(num_nodes, dtype=torch.long)
    for _ in range(50):
        # Assignment.
        dists = torch.cdist(enc, centroids)  # [N, k]
        labels = dists.argmin(dim=1)
        # Update.
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, dtype=torch.long)
        for n in range(num_nodes):
            c = int(labels[n].item())
            new_centroids[c] += enc[n]
            counts[c] += 1
        for c in range(k):
            if counts[c] > 0:
                new_centroids[c] /= float(counts[c])
            else:
                new_centroids[c] = enc[int(torch.randint(num_nodes, (1,), generator=rng).item())]
        if torch.allclose(new_centroids, centroids, atol=1e-6):
            break
        centroids = new_centroids
    return labels


def spectral_distance(
    edge_index_1: torch.Tensor,
    num_nodes_1: int,
    edge_index_2: torch.Tensor,
    num_nodes_2: int,
    k: int = 10,
    method: str = "l2",
) -> float:
    """Distance between two graphs based on their Laplacian spectra.

    Args:
        edge_index_1, num_nodes_1: First graph.
        edge_index_2, num_nodes_2: Second graph.
        k: Number of eigenvalues to compare.
        method: ``"l2"`` or ``"l1"``.

    Returns:
        Non-negative float.  0 = identical spectra.
    """
    ev1 = laplacian_eigenvalues(edge_index_1, num_nodes_1, k=k)
    ev2 = laplacian_eigenvalues(edge_index_2, num_nodes_2, k=k)
    # Pad to same length.
    L = max(ev1.numel(), ev2.numel())
    ev1_pad = torch.zeros(L, dtype=torch.float)
    ev2_pad = torch.zeros(L, dtype=torch.float)
    ev1_pad[:ev1.numel()] = ev1
    ev2_pad[:ev2.numel()] = ev2
    if method == "l2":
        return float((ev1_pad - ev2_pad).pow(2).sum().sqrt().item())
    return float((ev1_pad - ev2_pad).abs().sum().item())


def dirichlet_energy(
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> float:
    """Dirichlet energy: ``Σ_{(u,v)} ||x_u - x_v||²``.

    Measures the smoothness of a node feature signal on the graph.
    Low energy = smooth signal.

    Args:
        node_features: ``FloatTensor[N, D]``.
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.

    Returns:
        Non-negative float.
    """
    if edge_index.numel() == 0:
        return 0.0
    src, dst = edge_index[0], edge_index[1]
    diff = node_features[src].float() - node_features[dst].float()
    return float(diff.pow(2).sum().item())

"""Opt-in learned / differentiable graph construction helpers.

These utilities bridge rule-based graph construction (the main TGraphX
approach) and soft adjacency-based graph learning.

The graph topology is **still user-controlled by default**.  These helpers
provide building blocks for models that want to learn edge importance or
dynamically construct adjacency from node embeddings.

APIs
----
soft_adjacency_from_embeddings(z, temperature)
    Differentiable [N, N] pairwise similarity matrix from embeddings.

top_k_edges_from_scores(scores, k, ...)
    Non-differentiable discrete top-k edge selection.
    Returns ``edge_index [2, E]`` + ``edge_scores [E]``.

build_knn_graph_from_embeddings(z, k, ...)
    Convenience wrapper: L2-based kNN using only torch.cdist.
    Identical output to ``build_knn_graph(coords, k)`` but accepts the
    embedding tensor directly.

EdgeScorer
    Learnable MLP that produces per-edge importance scores from
    concatenated source/destination embeddings.

Warnings
--------
* ``soft_adjacency_from_embeddings`` computes an **O(N²)** matrix.
  A ``UserWarning`` is emitted for N > 5 000.
* ``top_k_edges_from_scores`` selects edges **non-differentiably** via
  ``torch.topk``.  Gradients cannot flow through the edge-selection step.
  Use ``soft_adjacency_from_embeddings`` when differentiability matters.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_LARGE_N_WARN = 5_000

__all__ = [
    "soft_adjacency_from_embeddings",
    "top_k_edges_from_scores",
    "build_knn_graph_from_embeddings",
    "EdgeScorer",
]


def soft_adjacency_from_embeddings(
    z: torch.Tensor,
    temperature: float = 1.0,
    similarity: str = "cosine",
) -> torch.Tensor:
    """Compute a differentiable soft adjacency matrix from node embeddings.

    .. warning::
        Allocates an **O(N²)** matrix.  A ``UserWarning`` is emitted for
        N > 5 000.

    The resulting matrix ``A[i, j]`` is a continuous score in ``[0, 1]``
    that can be used as soft edge weights.  Gradients flow through ``A``
    back to ``z``.

    Args:
        z: ``[N, D]`` node embeddings.
        temperature: Softmax / sigmoid temperature.  Higher values produce
            more uniform adjacency; lower values produce sparser, more
            peaked adjacency.
        similarity: ``"cosine"`` (default) or ``"dot"``.

    Returns:
        ``[N, N]`` soft adjacency matrix with values in ``(0, 1)``.

    Example::

        z = model.encode(x, edge_index)   # [N, D]
        A = soft_adjacency_from_embeddings(z, temperature=0.5)
        # Use A as soft edge weights or to select top-k edges.
    """
    if z.dim() != 2:
        raise ValueError(f"z must be 2-D [N, D]; got {tuple(z.shape)}")
    N = z.size(0)
    if N > _LARGE_N_WARN:
        warnings.warn(
            f"soft_adjacency_from_embeddings: N={N} > {_LARGE_N_WARN}. "
            f"Allocates an O(N²) matrix ({N}×{N} floats).",
            stacklevel=2,
        )
    if similarity == "cosine":
        z_norm = F.normalize(z, p=2, dim=-1)
        scores = torch.mm(z_norm, z_norm.t())  # [N, N], in [-1, 1]
    elif similarity == "dot":
        scores = torch.mm(z, z.t()) / (temperature + 1e-8)
        return torch.sigmoid(scores)
    else:
        raise ValueError(f"similarity must be 'cosine' or 'dot'; got {similarity!r}")
    return torch.sigmoid(scores / (temperature + 1e-8))


def top_k_edges_from_scores(
    scores: torch.Tensor,
    k: int,
    self_loops: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select the top-``k`` outgoing edges per node from a score matrix.

    .. warning::
        This operation is **not differentiable** — gradients cannot flow
        through the edge-selection step (``torch.topk`` is discrete).
        Use ``soft_adjacency_from_embeddings`` when gradient flow matters.

    .. warning::
        Allocates an O(N²) scores matrix.  Use for moderate N only.

    Args:
        scores: ``[N, N]`` pairwise edge score matrix.
        k: Number of outgoing edges to keep per node.
        self_loops: If ``False`` (default), set the diagonal to ``-inf``
            before top-k so self-edges are excluded.

    Returns:
        Tuple of:
        - ``edge_index`` — ``[2, N*k]`` LongTensor.
        - ``edge_scores`` — ``[N*k]`` FloatTensor of selected scores.

    Example::

        A = soft_adjacency_from_embeddings(z)
        ei, es = top_k_edges_from_scores(A, k=5)
        # ei usable as edge_index in any TGraphX layer.
    """
    if scores.dim() != 2 or scores.size(0) != scores.size(1):
        raise ValueError(f"scores must be [N, N]; got {tuple(scores.shape)}")
    N = scores.size(0)
    if k >= N:
        raise ValueError(f"k={k} must be < N={N}")
    with torch.no_grad():
        s = scores.clone()
        if not self_loops:
            s.fill_diagonal_(float("-inf"))
        top_vals, top_idx = torch.topk(s, k, dim=1)  # [N, k]
        src = torch.arange(N, device=scores.device).repeat_interleave(k)
        dst = top_idx.reshape(-1)
        edge_scores = top_vals.reshape(-1)
    return torch.stack([src, dst], dim=0), edge_scores


def build_knn_graph_from_embeddings(
    z: torch.Tensor,
    k: int,
    directed: bool = False,
    self_loops: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Build a kNN ``edge_index`` from node embeddings via L2 distance.

    Convenience wrapper that calls :func:`~tgraphx.build_knn_graph` with
    the embedding matrix as coordinates.  **Not differentiable** — the
    graph topology is constructed with ``torch.no_grad()``.

    Args:
        z: ``[N, D]`` node embeddings.
        k: Number of nearest neighbours per node (self excluded).
        directed: If ``False`` (default), include both ``(u→v)`` and
            ``(v→u)`` for every kNN pair.
        self_loops: If ``True`` (default), add one ``i→i`` per node.
        normalize: If ``True`` (default), L2-normalise embeddings before
            computing distances (cosine kNN).

    Returns:
        ``edge_index [2, E]`` LongTensor.
    """
    from .graph_builders import build_knn_graph
    if z.dim() != 2:
        raise ValueError(f"z must be 2-D [N, D]; got {tuple(z.shape)}")
    with torch.no_grad():
        coords = F.normalize(z, p=2, dim=-1) if normalize else z
        return build_knn_graph(coords.detach(), k=k, directed=directed,
                               self_loops=self_loops)


class EdgeScorer(nn.Module):
    """Learnable MLP edge scorer from source/destination embeddings.

    Computes a scalar importance score ``s_ij`` for each edge ``(i, j)``
    from the concatenation ``[z_i ‖ z_j]``.  Gradients flow from the
    scores back to the embedding ``z``.

    The edge-selection step (converting scores to a discrete topology) is
    **not differentiable**.  Use the scores as continuous edge weights, or
    pass them to ``top_k_edges_from_scores`` after detaching.

    Example::

        scorer = EdgeScorer(in_dim=64, hidden_dim=32)
        scores = scorer(z, edge_index)  # [E] per-edge logits
        weights = torch.sigmoid(scores) # [E] soft weights → pass to layer
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        out_dim: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            in_dim: Node embedding dimension.
            hidden_dim: MLP hidden dimension.
            out_dim: Output dimension per edge (1 = scalar score).
            dropout: Dropout on hidden layer (0 = no dropout).
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-edge scores.

        Args:
            z: ``[N, D]`` node embeddings.
            edge_index: ``[2, E]`` LongTensor.

        Returns:
            ``[E, out_dim]`` score tensor (squeeze to ``[E]`` when
            ``out_dim=1``).
        """
        if z.dim() != 2:
            raise ValueError(f"z must be [N, D]; got {tuple(z.shape)}")
        src = edge_index[0]
        dst = edge_index[1]
        pair = torch.cat([z[src], z[dst]], dim=-1)  # [E, 2*D]
        scores = self.mlp(pair)  # [E, out_dim]
        if scores.size(-1) == 1:
            return scores.squeeze(-1)
        return scores

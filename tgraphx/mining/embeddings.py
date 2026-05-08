"""Graph embedding extraction utilities.

These helpers extract node and graph-level embeddings from trained
TGraphX GNN models, respecting ``no_grad`` and device placement.

Stability: Beta (v0.4.2+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "extract_node_embeddings",
    "extract_graph_embeddings",
    "embedding_similarity_matrix",
    "embedding_pairwise_distances",
    "embedding_nearest_neighbors",
]


def extract_node_embeddings(
    model: nn.Module,
    edge_index: torch.Tensor,
    node_features: torch.Tensor,
    num_nodes: Optional[int] = None,
    device: Optional[torch.device] = None,
    no_grad: bool = True,
) -> torch.Tensor:
    """Extract node embeddings from a trained GNN model.

    Calls ``model(node_features, edge_index)`` with optional ``no_grad``
    context and returns the output tensor.

    Args:
        model: A ``nn.Module`` with signature
            ``forward(node_features, edge_index, ...) -> FloatTensor[N, D]``.
        edge_index: ``LongTensor[2, E]``.
        node_features: ``Tensor[N, *]``.
        num_nodes: Optional; inferred from ``node_features.size(0)``.
        device: Optional device for input tensors.
        no_grad: When ``True`` (default), wraps in ``torch.no_grad()``.

    Returns:
        ``FloatTensor[N, D]`` detached from the autograd graph.
    """
    model.eval()
    if device is not None:
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        model = model.to(device)

    def _run():
        return model(node_features, edge_index)

    if no_grad:
        with torch.no_grad():
            out = _run()
    else:
        out = _run()

    return out.detach().float()


def extract_graph_embeddings(
    model: nn.Module,
    graphs: List[Any],
    pooling: str = "mean",
    device: Optional[torch.device] = None,
    no_grad: bool = True,
) -> torch.Tensor:
    """Extract graph-level embeddings by pooling node embeddings.

    Args:
        model: GNN model with ``forward(node_features, edge_index)`` signature.
        graphs: List of TGraphX :class:`~tgraphx.Graph` objects or dicts
            with ``node_features`` and ``edge_index`` keys.
        pooling: ``"mean"``, ``"max"``, or ``"sum"``.
        device: Optional device.
        no_grad: Use ``torch.no_grad()`` (default: ``True``).

    Returns:
        ``FloatTensor[G, D]`` — one embedding per graph.
    """
    model.eval()
    all_embs: List[torch.Tensor] = []

    for g in graphs:
        if hasattr(g, "node_features"):
            nf = g.node_features
            ei = g.edge_index
        else:
            nf = g["node_features"]
            ei = g["edge_index"]

        if device is not None:
            nf = nf.to(device)
            ei = ei.to(device)
            model = model.to(device)

        def _run():
            return model(nf, ei)

        if no_grad:
            with torch.no_grad():
                node_emb = _run()
        else:
            node_emb = _run()

        node_emb = node_emb.detach().float()
        if pooling == "mean":
            g_emb = node_emb.mean(dim=0)
        elif pooling == "max":
            g_emb = node_emb.max(dim=0).values
        elif pooling == "sum":
            g_emb = node_emb.sum(dim=0)
        else:
            raise ValueError(f"pooling must be 'mean', 'max', or 'sum'; got {pooling!r}")
        all_embs.append(g_emb)

    if not all_embs:
        return torch.zeros((0, 1), dtype=torch.float)
    return torch.stack(all_embs, dim=0)


def embedding_similarity_matrix(
    embeddings: torch.Tensor,
    method: str = "cosine",
) -> torch.Tensor:
    """Compute a pairwise similarity matrix.

    Args:
        embeddings: ``FloatTensor[N, D]``.
        method: ``"cosine"`` or ``"dot"``.

    Returns:
        ``FloatTensor[N, N]`` symmetric similarity matrix.
    """
    if embeddings.dim() != 2:
        raise ValueError(f"embeddings must be 2-D; got shape {tuple(embeddings.shape)}")
    emb = embeddings.float()
    if method == "cosine":
        norms = emb.norm(dim=1, keepdim=True).clamp(min=1e-12)
        emb_norm = emb / norms
        return emb_norm @ emb_norm.t()
    if method == "dot":
        return emb @ emb.t()
    raise ValueError(f"method must be 'cosine' or 'dot'; got {method!r}")


def embedding_pairwise_distances(
    embeddings: torch.Tensor,
    method: str = "euclidean",
) -> torch.Tensor:
    """Compute a pairwise distance matrix.

    Args:
        embeddings: ``FloatTensor[N, D]``.
        method: ``"euclidean"`` (L2) or ``"cosine"`` (1 - cosine similarity).

    Returns:
        ``FloatTensor[N, N]`` non-negative symmetric distance matrix.
    """
    emb = embeddings.float()
    if method == "euclidean":
        return torch.cdist(emb, emb, p=2)
    if method == "cosine":
        return (1.0 - embedding_similarity_matrix(emb, "cosine")).clamp(min=0.0)
    raise ValueError(f"method must be 'euclidean' or 'cosine'; got {method!r}")


def embedding_nearest_neighbors(
    query_embeddings: torch.Tensor,
    key_embeddings: torch.Tensor,
    k: int,
    method: str = "cosine",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the k nearest neighbours for each query.

    Args:
        query_embeddings: ``FloatTensor[Q, D]``.
        key_embeddings: ``FloatTensor[K, D]``.
        k: Number of neighbours.
        method: ``"cosine"`` or ``"euclidean"``.

    Returns:
        ``(indices, scores)`` — two tensors of shape ``[Q, k]``.
        Indices are into ``key_embeddings``.
        For cosine: higher score = more similar.
        For euclidean: lower score = closer.
    """
    q = query_embeddings.float()
    ke = key_embeddings.float()
    k_actual = min(int(k), ke.size(0))
    if method == "cosine":
        q_norm = q / q.norm(dim=1, keepdim=True).clamp(min=1e-12)
        k_norm = ke / ke.norm(dim=1, keepdim=True).clamp(min=1e-12)
        sim = q_norm @ k_norm.t()  # [Q, K]
        scores, idx = sim.topk(k_actual, dim=1, largest=True, sorted=True)
        return idx, scores
    if method == "euclidean":
        dist = torch.cdist(q, ke, p=2)  # [Q, K]
        scores, idx = dist.topk(k_actual, dim=1, largest=False, sorted=True)
        return idx, scores
    raise ValueError(f"method must be 'cosine' or 'euclidean'; got {method!r}")

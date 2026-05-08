"""Random walk generation for graph mining.

These utilities generate random walks over a graph for use in
embedding training (e.g., Node2Vec, DeepWalk-style), feature extraction,
and graph mining.

**This is walk generation only** — no embedding model is included.

Dead ends (nodes with no outgoing edges) are handled by staying in
place (repeating the current node) for the remainder of the walk.

Stability: Beta (v0.3.2+).
"""
from __future__ import annotations

from typing import List, Optional

import torch

__all__ = [
    "random_walks",
    "generate_random_walks",
]


def _build_csr(
    edge_index: torch.Tensor, num_nodes: int,
) -> tuple:
    """Build a CSR-style adjacency for O(1) random access."""
    if edge_index.numel() == 0:
        row_ptr = torch.zeros(num_nodes + 1, dtype=torch.long)
        col_idx = torch.zeros(0, dtype=torch.long)
        return row_ptr, col_idx

    # Sort edges by source node.
    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)
    order = torch.argsort(src, stable=True)
    src = src[order]
    dst = dst[order]

    # Build row pointer.
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    ones = torch.ones(src.size(0), dtype=torch.long)
    row_ptr[1:].scatter_add_(0, src, ones)
    row_ptr = torch.cumsum(row_ptr, dim=0)
    return row_ptr, dst


def random_walks(
    edge_index: torch.Tensor,
    start_nodes: torch.Tensor,
    walk_length: int,
    num_nodes: Optional[int] = None,
    seed: Optional[int] = None,
    p: float = 1.0,
    q: float = 1.0,
) -> torch.Tensor:
    """Generate random walks starting from ``start_nodes``.

    Dead-end handling: if a node has no outgoing edges, the walk stays
    at that node for the remaining steps (the node id is repeated).

    Args:
        edge_index: ``LongTensor[2, E]`` (directed).
        start_nodes: ``LongTensor[W]`` — starting node for each walk.
        walk_length: Number of steps (output length = walk_length + 1
            including the start node).
        num_nodes: Optional node count; inferred when ``None``.
        seed: Optional RNG seed; no global RNG side effects.
        p: Return parameter (Node2Vec).  ``p=1, q=1`` gives uniform
            random walks.  Full biased Node2Vec is not implemented —
            set ``p=q=1`` for unbiased walks.
        q: In-out parameter (Node2Vec).  See above.

    Returns:
        ``LongTensor[W, walk_length + 1]`` of node ids.  All valid in
        ``[0, num_nodes)``.

    Notes:
        - Biased Node2Vec walks (``p ≠ 1`` or ``q ≠ 1``) require access
          to the previous node at each step.  This implementation
          supports biased walks but is CPU-only for the biased path.
        - CUDA tensors for ``edge_index`` are moved to CPU internally.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E]; got {tuple(edge_index.shape)}")
    if start_nodes.dim() != 1:
        raise ValueError(f"start_nodes must be 1-D; got shape {tuple(start_nodes.shape)}")
    if walk_length < 0:
        raise ValueError(f"walk_length must be non-negative; got {walk_length}")
    if num_nodes is None:
        num_nodes = (
            max(int(edge_index.max().item()), int(start_nodes.max().item())) + 1
            if edge_index.numel() and start_nodes.numel()
            else int(start_nodes.max().item()) + 1 if start_nodes.numel() else 0
        )
    if start_nodes.numel() and int(start_nodes.max().item()) >= num_nodes:
        raise ValueError("start_nodes contains out-of-range node ids")

    W = int(start_nodes.size(0))
    row_ptr, col_idx = _build_csr(edge_index.cpu(), num_nodes)
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(int(seed))

    walks = torch.empty(W, walk_length + 1, dtype=torch.long)
    start_cpu = start_nodes.to(torch.long).cpu()
    walks[:, 0] = start_cpu

    unbiased = (abs(p - 1.0) < 1e-9 and abs(q - 1.0) < 1e-9)

    for step in range(1, walk_length + 1):
        for wi in range(W):
            cur = int(walks[wi, step - 1].item())
            start_ptr = int(row_ptr[cur].item())
            end_ptr = int(row_ptr[cur + 1].item())
            deg = end_ptr - start_ptr
            if deg == 0:
                walks[wi, step] = cur  # dead-end: stay in place
                continue
            if unbiased:
                idx = int(torch.randint(deg, (1,), generator=rng).item())
                walks[wi, step] = int(col_idx[start_ptr + idx].item())
            else:
                # Biased walk: use p and q relative to previous node.
                prev = int(walks[wi, step - 2].item()) if step > 1 else -1
                probs = torch.ones(deg, dtype=torch.float)
                row_ptr_cpu = row_ptr
                col_idx_cpu = col_idx
                for k in range(deg):
                    nxt = int(col_idx_cpu[start_ptr + k].item())
                    if nxt == prev:
                        probs[k] = 1.0 / p
                    else:
                        # Check if prev is connected to nxt.
                        prev_start = int(row_ptr_cpu[prev].item()) if prev >= 0 else -1
                        prev_end = int(row_ptr_cpu[prev + 1].item()) if prev >= 0 else -1
                        is_nbr = False
                        if prev >= 0:
                            for ki in range(prev_end - prev_start):
                                if int(col_idx_cpu[prev_start + ki].item()) == nxt:
                                    is_nbr = True
                                    break
                        probs[k] = 1.0 if is_nbr else 1.0 / q
                idx = int(torch.multinomial(probs, 1, generator=rng).item())
                walks[wi, step] = int(col_idx[start_ptr + idx].item())

    return walks


def generate_random_walks(
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
    num_walks_per_node: int = 10,
    walk_length: int = 20,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate multiple random walks starting from every node.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Optional node count.
        num_walks_per_node: Number of walks per node.
        walk_length: Steps per walk.
        seed: Optional RNG seed.

    Returns:
        ``LongTensor[N * num_walks_per_node, walk_length + 1]``.
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 0
    if num_nodes == 0:
        return torch.zeros((0, walk_length + 1), dtype=torch.long)
    start = torch.arange(num_nodes, dtype=torch.long).repeat(num_walks_per_node)
    return random_walks(
        edge_index, start, walk_length, num_nodes=num_nodes, seed=seed,
    )

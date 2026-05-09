"""Hypergraph utilities and incidence-based analysis.

A hypergraph generalises a graph by allowing edges (hyperedges) to
connect any number of nodes.  Represented as an incidence matrix
``H[N, M]`` where ``H[i, e] = 1`` iff node ``i`` belongs to hyperedge ``e``.

Stability: Experimental (v0.4.4+).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

__all__ = [
    "Hypergraph",
    "incidence_to_bipartite_graph",
    "clique_expansion",
    "star_expansion",
    "hypergraph_density",
]


class Hypergraph:
    """Hypergraph represented by an incidence list.

    Args:
        num_nodes: Number of nodes.
        hyperedges: List of lists; each inner list is a hyperedge (set of
            node IDs).  Duplicate nodes within a hyperedge are deduplicated.

    Stability: Experimental.
    """

    def __init__(
        self,
        num_nodes: int,
        hyperedges: List[List[int]],
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.hyperedges: List[List[int]] = [sorted(set(he)) for he in hyperedges]
        for he in self.hyperedges:
            for v in he:
                if not (0 <= v < num_nodes):
                    raise ValueError(f"Node id {v} out of range [0, {num_nodes})")

    @property
    def num_hyperedges(self) -> int:
        return len(self.hyperedges)

    def node_hyperdegree(self) -> torch.Tensor:
        """Number of hyperedges containing each node.

        Returns:
            ``LongTensor[N]``.
        """
        deg = torch.zeros(self.num_nodes, dtype=torch.long)
        for he in self.hyperedges:
            for v in he:
                deg[v] += 1
        return deg

    def hyperedge_degree(self) -> torch.Tensor:
        """Cardinality (size) of each hyperedge.

        Returns:
            ``LongTensor[M]``.
        """
        return torch.tensor([len(he) for he in self.hyperedges], dtype=torch.long)

    def incidence_matrix(self) -> torch.Tensor:
        """Dense incidence matrix ``H[N, M]``.

        H[i, e] = 1 if node i belongs to hyperedge e, else 0.

        Returns:
            ``FloatTensor[N, M]``.

        Raises:
            ValueError: If the matrix would exceed 10 000 entries without
                an explicit guard override.
        """
        N, M = self.num_nodes, self.num_hyperedges
        if N * M > 100_000:
            raise ValueError(
                f"incidence_matrix: N×M = {N}×{M} = {N*M} > 100 000. "
                "Use incidence_to_bipartite_graph for large hypergraphs."
            )
        H = torch.zeros(N, M, dtype=torch.float)
        for e, he in enumerate(self.hyperedges):
            for v in he:
                H[v, e] = 1.0
        return H

    def density(self) -> float:
        """Fraction of (node, hyperedge) pairs that are active."""
        return hypergraph_density(self)

    def summary(self) -> Dict[str, float]:
        """Return a JSON-serialisable summary dict."""
        node_deg = self.node_hyperdegree()
        edge_deg = self.hyperedge_degree()
        return {
            "num_nodes": self.num_nodes,
            "num_hyperedges": self.num_hyperedges,
            "density": round(self.density(), 6),
            "mean_node_degree": round(float(node_deg.float().mean().item()), 4),
            "max_node_degree": int(node_deg.max().item()) if node_deg.numel() else 0,
            "mean_hyperedge_size": round(float(edge_deg.float().mean().item()), 4) if edge_deg.numel() else 0.0,
            "max_hyperedge_size": int(edge_deg.max().item()) if edge_deg.numel() else 0,
        }


def incidence_to_bipartite_graph(
    hg: "Hypergraph",
) -> Tuple[torch.Tensor, int]:
    """Convert hypergraph to bipartite graph representation.

    Left nodes (IDs 0..N-1) represent original nodes; right nodes
    (IDs N..N+M-1) represent hyperedges.  An edge connects node ``v``
    to hyperedge-node ``N + e`` iff ``v ∈ hyperedge[e]``.

    Args:
        hg: :class:`Hypergraph` instance.

    Returns:
        ``(edge_index, total_nodes)`` where ``edge_index`` is
        ``LongTensor[2, 2*sum_of_hyperedge_sizes]`` (undirected).
    """
    N = hg.num_nodes
    src_list: List[int] = []
    dst_list: List[int] = []
    for e, he in enumerate(hg.hyperedges):
        hyperedge_node = N + e
        for v in he:
            src_list.extend([v, hyperedge_node])
            dst_list.extend([hyperedge_node, v])
    total_nodes = N + hg.num_hyperedges
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long), total_nodes
    return torch.tensor([src_list, dst_list], dtype=torch.long), total_nodes


def clique_expansion(
    hg: "Hypergraph",
) -> Tuple[torch.Tensor, int]:
    """Convert hypergraph to graph via clique expansion.

    Each hyperedge becomes a clique (complete graph) over its members.
    Parallel edges are deduplicated.

    Args:
        hg: :class:`Hypergraph`.

    Returns:
        ``(edge_index, num_nodes)`` of the expanded graph.
    """
    N = hg.num_nodes
    edges: set = set()
    for he in hg.hyperedges:
        nodes = he
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                edges.add((u, v))
                edges.add((v, u))
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long), N
    src_list = [e[0] for e in edges]
    dst_list = [e[1] for e in edges]
    return torch.tensor([src_list, dst_list], dtype=torch.long), N


def star_expansion(
    hg: "Hypergraph",
) -> Tuple[torch.Tensor, int]:
    """Convert hypergraph to graph via star expansion.

    Each hyperedge becomes a star — all nodes connect to an auxiliary
    hyperedge-node.  Equivalent to the bipartite representation but
    with directed edges pointing from original nodes to hyperedge-nodes
    only (for a compact representation).

    Returns:
        ``(edge_index, total_nodes)`` — same as
        :func:`incidence_to_bipartite_graph`.
    """
    return incidence_to_bipartite_graph(hg)


def hypergraph_density(hg: "Hypergraph") -> float:
    """Fraction of (node, hyperedge) pairs that are active."""
    N = hg.num_nodes
    M = hg.num_hyperedges
    if N == 0 or M == 0:
        return 0.0
    total_active = sum(len(he) for he in hg.hyperedges)
    return float(total_active) / float(N * M)

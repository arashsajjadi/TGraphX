"""Tests for tgraphx.algorithms — v0.3.2 beta primitives.

Covers connectivity (connected_components, is_connected, count) and
traversal (bfs_layers, bfs_edges, shortest_path_length).
"""
from __future__ import annotations

import pytest
import torch

from tgraphx.algorithms import (
    bfs_edges,
    bfs_layers,
    connected_components,
    is_connected,
    number_connected_components,
    shortest_path_length,
    weakly_connected_components,
)


# ── connectivity ─────────────────────────────────────────────────────────────


class TestConnectedComponents:
    def test_two_disjoint_paths(self):
        # Components: {0, 1, 2} and {3, 4}.
        edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]], dtype=torch.long)
        labels = connected_components(edge_index, num_nodes=5)
        assert labels.tolist() == [0, 0, 0, 1, 1]
        assert number_connected_components(edge_index, 5) == 2

    def test_one_connected_graph(self):
        # 0-1-2-0 triangle.
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        labels = connected_components(edge_index, 3)
        assert labels.tolist() == [0, 0, 0]
        assert is_connected(edge_index, 3)
        assert number_connected_components(edge_index, 3) == 1

    def test_isolated_nodes(self):
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        labels = connected_components(edge_index, num_nodes=4)
        # 4 isolated nodes ⇒ 4 components, deterministic 0..3.
        assert labels.tolist() == [0, 1, 2, 3]
        assert number_connected_components(edge_index, 4) == 4
        assert not is_connected(edge_index, 4)

    def test_single_node_is_connected(self):
        assert is_connected(torch.zeros((2, 0), dtype=torch.long), num_nodes=1)

    def test_empty_graph_not_connected(self):
        assert not is_connected(torch.zeros((2, 0), dtype=torch.long), num_nodes=0)
        assert number_connected_components(
            torch.zeros((2, 0), dtype=torch.long), num_nodes=0,
        ) == 0

    def test_directed_treated_as_undirected(self):
        # Edge 0→1 only (not 1→0).
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        labels = connected_components(edge_index, num_nodes=2)
        assert labels.tolist() == [0, 0]

    def test_weakly_connected_alias(self):
        edge_index = torch.tensor([[0, 1, 2, 4], [1, 2, 3, 5]], dtype=torch.long)
        a = connected_components(edge_index, 6)
        b = weakly_connected_components(edge_index, 6)
        assert torch.equal(a, b)

    def test_compact_label_ids(self):
        # Components are returned as contiguous [0, K).
        edge_index = torch.tensor([[1, 2, 5], [2, 3, 6]], dtype=torch.long)
        labels = connected_components(edge_index, num_nodes=7)
        unique = torch.unique(labels)
        # Labels must form a contiguous range starting at 0.
        assert unique.tolist() == list(range(unique.numel()))

    def test_validates_edge_index_shape(self):
        with pytest.raises(ValueError, match="\\[2, E\\]"):
            connected_components(torch.zeros(3, 4, dtype=torch.long), num_nodes=4)

    def test_validates_edge_id_range(self):
        bad = torch.tensor([[0, 5], [1, 2]], dtype=torch.long)
        with pytest.raises(ValueError, match="out of range"):
            connected_components(bad, num_nodes=4)

    def test_inferred_num_nodes(self):
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        labels = connected_components(edge_index)  # infers num_nodes=3
        assert labels.numel() == 3


# ── BFS layers ───────────────────────────────────────────────────────────────


class TestBFSLayers:
    def test_chain(self):
        # 0 → 1 → 2 → 3
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        layers = bfs_layers(edge_index, source=0, num_nodes=4)
        assert [l.tolist() for l in layers] == [[0], [1], [2], [3]]

    def test_branching(self):
        # 0 → {1, 2}; 1 → 3
        edge_index = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long)
        layers = bfs_layers(edge_index, source=0, num_nodes=4)
        assert layers[0].tolist() == [0]
        assert sorted(layers[1].tolist()) == [1, 2]
        assert layers[2].tolist() == [3]

    def test_isolated_source(self):
        layers = bfs_layers(torch.zeros((2, 0), dtype=torch.long),
                            source=0, num_nodes=3)
        assert [l.tolist() for l in layers] == [[0]]

    def test_max_hops(self):
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        layers = bfs_layers(edge_index, source=0, num_nodes=5, max_hops=2)
        assert sum(l.numel() for l in layers) == 3  # source + 2 hops

    def test_invalid_source(self):
        with pytest.raises(ValueError, match="source"):
            bfs_layers(torch.zeros((2, 0), dtype=torch.long),
                       source=10, num_nodes=4)


# ── BFS edges ────────────────────────────────────────────────────────────────


class TestBFSEdges:
    def test_chain_edges(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        bfs = bfs_edges(edge_index, source=0, num_nodes=4)
        # Expect exactly the chain.
        assert bfs.tolist() == [[0, 1, 2], [1, 2, 3]]

    def test_no_edges_returns_empty(self):
        bfs = bfs_edges(torch.zeros((2, 0), dtype=torch.long),
                        source=0, num_nodes=3)
        assert bfs.shape == (2, 0)

    def test_branching_count(self):
        # 0 → {1, 2}; 1 → 3
        edge_index = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long)
        bfs = bfs_edges(edge_index, source=0, num_nodes=4)
        # Exactly 3 reachable non-source nodes ⇒ 3 BFS edges.
        assert bfs.size(1) == 3
        # Each child node appears once (BFS tree property).
        assert sorted(bfs[1].tolist()) == [1, 2, 3]

    def test_no_revisit(self):
        # Cycle: 0 → 1 → 2 → 0
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        bfs = bfs_edges(edge_index, source=0, num_nodes=3)
        # Only 2 BFS edges — node 0 is the source and is not revisited.
        assert bfs.size(1) == 2

    def test_max_hops(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        bfs = bfs_edges(edge_index, source=0, num_nodes=4, max_hops=1)
        # With 1 hop only, BFS tree edges = 1.
        assert bfs.size(1) == 1


# ── Shortest path length ─────────────────────────────────────────────────────


class TestShortestPathLength:
    def test_chain(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        d = shortest_path_length(edge_index, source=0, num_nodes=4)
        assert d.tolist() == [0, 1, 2, 3]

    def test_unreachable_node(self):
        # 0 → 1 ; 2 isolated (in directed sense)
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        d = shortest_path_length(edge_index, source=0, num_nodes=3)
        assert d[0].item() == 0
        assert d[1].item() == 1
        assert d[2].item() == -1

    def test_self_distance(self):
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        d = shortest_path_length(edge_index, source=0, num_nodes=3)
        assert d[0].item() == 0
        assert d[1].item() == -1
        assert d[2].item() == -1

    def test_branching_distances(self):
        # 0 → {1, 2}; 1 → 3
        edge_index = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long)
        d = shortest_path_length(edge_index, source=0, num_nodes=4)
        assert d.tolist() == [0, 1, 1, 2]

    def test_directed_one_way(self):
        # 0 → 1, 1 → 2, no reverse edges.  From 2 nothing is reachable.
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        d = shortest_path_length(edge_index, source=2, num_nodes=3)
        assert d[2].item() == 0
        assert d[0].item() == -1
        assert d[1].item() == -1


# ── Edge cases and robustness ─────────────────────────────────────────────────


class TestEdgeCasesAndValidation:
    def test_self_loop_edges_do_not_crash_bfs(self):
        # Self-loops should be ignored by BFS (node already visited).
        edge_index = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.long)
        layers = bfs_layers(edge_index, source=0, num_nodes=3)
        all_nodes = set()
        for l in layers:
            all_nodes.update(l.tolist())
        assert 0 in all_nodes and 1 in all_nodes and 2 in all_nodes

    def test_self_loop_edges_do_not_crash_components(self):
        edge_index = torch.tensor([[0, 1, 1], [1, 2, 1]], dtype=torch.long)  # node 1 self-loop
        labels = connected_components(edge_index, num_nodes=3)
        assert labels.tolist() == [0, 0, 0]

    def test_duplicate_edges_do_not_duplicate_components(self):
        # Duplicate edge 0→1 should not change component structure.
        edge_index = torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long)
        labels = connected_components(edge_index, num_nodes=3)
        assert labels.tolist() == [0, 0, 0]

    def test_bfs_out_of_range_raises(self):
        edge_index = torch.tensor([[0, 1], [1, 5]], dtype=torch.long)
        with pytest.raises(ValueError, match="out of range"):
            bfs_layers(edge_index, source=0, num_nodes=3)

    def test_shortest_path_out_of_range_raises(self):
        edge_index = torch.tensor([[0, 1], [1, 5]], dtype=torch.long)
        with pytest.raises(ValueError, match="out of range"):
            shortest_path_length(edge_index, source=0, num_nodes=3)

    def test_large_sparse_graph_smoke(self):
        # 1000-node ring, both directions.
        N = 1000
        src = torch.arange(N)
        dst = (src + 1) % N
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src]),
        ], dim=0)
        labels = connected_components(edge_index, num_nodes=N)
        assert int(number_connected_components(edge_index, N)) == 1
        d = shortest_path_length(edge_index, source=0, num_nodes=N)
        # Max distance in a ring is N//2.
        assert d.min().item() == 0
        assert d.max().item() == N // 2

    @pytest.mark.skipif(
        not pytest.importorskip("networkx", reason="networkx not installed"),
        reason="networkx not installed",
    )
    def test_parity_with_networkx_components(self):
        """Optional parity test if networkx is installed."""
        import networkx as nx
        G = nx.Graph()
        edges = [(0, 1), (1, 2), (4, 5)]
        G.add_edges_from(edges)
        # All 7 nodes must be present in both graphs for a fair comparison.
        for n in range(7):
            G.add_node(n)
        edge_index = torch.tensor(
            [[u for u, v in edges] + [v for u, v in edges],
             [v for u, v in edges] + [u for u, v in edges]],
            dtype=torch.long,
        )
        labels = connected_components(edge_index, num_nodes=7)
        nx_count = nx.number_connected_components(G)
        tgx_count = int(labels.max().item()) + 1
        assert tgx_count == nx_count

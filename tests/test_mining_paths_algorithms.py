"""Tests for tgraphx.mining.paths — graph path algorithms."""
import math
import pytest
import torch
from tgraphx.mining import (
    bfs_order, dfs_order, multi_source_bfs, reachable_nodes,
    dijkstra_shortest_path, batched_shortest_path_length,
    all_pairs_shortest_path_length, reconstruct_path,
    minimum_spanning_tree, maximum_spanning_tree,
    cut_size, normalized_cut, conductance, volume, boundary_edges,
    write_path_summary,
)


def _chain(N=4):
    src = list(range(N-1)) + list(range(1, N))
    dst = list(range(1, N)) + list(range(N-1))
    return torch.tensor([src, dst], dtype=torch.long), N


def _star(N=5):
    src = [0]*(N-1) + list(range(1, N))
    dst = list(range(1, N)) + [0]*(N-1)
    return torch.tensor([src, dst], dtype=torch.long), N


class TestBFSOrder:
    def test_chain_bfs(self):
        ei, N = _chain()
        order = bfs_order(ei, 0, N)
        assert order[0].item() == 0
        assert sorted(order.tolist()) == list(range(N))

    def test_disconnected_partial(self):
        ei = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        order = bfs_order(ei, 0, 4)
        # Start at 0; only reaches 0 and 1.
        assert set(order.tolist()) == {0, 1}

    def test_invalid_start(self):
        ei, N = _chain()
        with pytest.raises(ValueError, match="start"):
            bfs_order(ei, N + 5, N)

    def test_single_node(self):
        ei = torch.zeros((2, 0), dtype=torch.long)
        order = bfs_order(ei, 0, 1)
        assert order.tolist() == [0]

    def test_star_hub_first(self):
        ei, N = _star()
        order = bfs_order(ei, 0, N)
        assert order[0].item() == 0
        assert order.size(0) == N


class TestDFSOrder:
    def test_chain_dfs(self):
        ei, N = _chain()
        order = dfs_order(ei, 0, N)
        assert order[0].item() == 0
        assert sorted(order.tolist()) == list(range(N))

    def test_disconnected(self):
        ei = torch.tensor([[0], [1]], dtype=torch.long)
        order = dfs_order(ei, 0, 4)
        # Directed; 0->1 but 2,3 unreachable.
        assert set(order.tolist()) == {0, 1}


class TestMultiSourceBFS:
    def test_two_sources(self):
        ei, N = _chain()
        sources = torch.tensor([0, 3], dtype=torch.long)
        dist = multi_source_bfs(ei, sources, N)
        # node 0: dist 0, node 1: dist 1 from 0, node 3: dist 0, node 2: dist 1 from 3
        assert dist[0].item() == 0
        assert dist[3].item() == 0
        assert dist[1].item() == 1
        assert dist[2].item() == 1

    def test_unreachable_minus_one(self):
        ei = torch.tensor([[0], [1]], dtype=torch.long)
        sources = torch.tensor([0], dtype=torch.long)
        dist = multi_source_bfs(ei, sources, 4, directed=True)
        assert dist[2].item() == -1
        assert dist[3].item() == -1


class TestDijkstra:
    def test_chain_unit_weights(self):
        ei, N = _chain()
        dist, pred = dijkstra_shortest_path(ei, 0, N)
        assert dist.tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_weighted_graph(self):
        # Triangle: 0-1 weight 1, 1-2 weight 1, 0-2 weight 10.
        ei = torch.tensor([[0, 1, 0, 1, 2, 2], [1, 0, 2, 2, 0, 1]], dtype=torch.long)
        w = torch.tensor([1.0, 1.0, 10.0, 1.0, 10.0, 1.0])
        dist, pred = dijkstra_shortest_path(ei, 0, 3, edge_weight=w)
        assert abs(float(dist[0]) - 0.0) < 1e-6
        assert abs(float(dist[1]) - 1.0) < 1e-6
        assert abs(float(dist[2]) - 2.0) < 1e-6  # via 0→1→2

    def test_unreachable_inf(self):
        ei = torch.tensor([[0], [1]], dtype=torch.long)
        dist, _ = dijkstra_shortest_path(ei, 0, 4, directed=True)
        assert dist[0].item() == 0.0
        assert dist[1].item() == 1.0
        assert math.isinf(float(dist[2]))
        assert math.isinf(float(dist[3]))

    def test_source_zero_dist(self):
        ei, N = _chain()
        dist, _ = dijkstra_shortest_path(ei, 2, N)
        assert dist[2].item() == 0.0

    def test_negative_weight_raises(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        w = torch.tensor([1.0, -1.0])
        with pytest.raises(ValueError, match="non-negative"):
            dijkstra_shortest_path(ei, 0, 2, edge_weight=w)

    def test_reconstruct_path(self):
        ei, N = _chain()
        dist, pred = dijkstra_shortest_path(ei, 0, N)
        path = reconstruct_path(0, 3, pred)
        assert path == [0, 1, 2, 3]

    def test_reconstruct_path_unreachable(self):
        ei = torch.zeros((2, 0), dtype=torch.long)
        _, pred = dijkstra_shortest_path(ei, 0, 4)
        path = reconstruct_path(0, 3, pred)
        assert path == []

    def test_reconstruct_path_self(self):
        ei, N = _chain()
        _, pred = dijkstra_shortest_path(ei, 0, N)
        path = reconstruct_path(0, 0, pred)
        assert path == [0]


class TestAllPairsSP:
    def test_chain_symmetric(self):
        ei, N = _chain()
        D = all_pairs_shortest_path_length(ei, N)
        assert D.shape == (N, N)
        assert torch.allclose(D, D.t())  # undirected → symmetric

    def test_chain_diagonal_zero(self):
        ei, N = _chain()
        D = all_pairs_shortest_path_length(ei, N)
        assert (D.diagonal() == 0).all()

    def test_size_guard(self):
        ei = torch.zeros((2, 0), dtype=torch.long)
        with pytest.raises(ValueError, match="max_nodes"):
            all_pairs_shortest_path_length(ei, 1001, max_nodes=1000)


class TestSpanningTree:
    def test_triangle_mst_two_edges(self):
        # Triangle K3: 3 edges, MST should pick 2 cheapest.
        ei = torch.tensor([[0,1,2,1,2,0],[1,2,0,0,1,2]], dtype=torch.long)
        w = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        mst_ei, mst_w, total = minimum_spanning_tree(ei, 3, edge_weight=w)
        # MST should have 2 undirected edges (= 4 directed edge_index entries).
        assert mst_ei.size(1) == 4
        assert abs(total - 3.0) < 1e-6  # edges of weight 1+2

    def test_mst_unit_weights(self):
        ei, N = _chain()
        mst_ei, _, total = minimum_spanning_tree(ei, N)
        # Chain is already a spanning tree; MST = all edges.
        assert abs(total - float(N - 1)) < 1e-6

    def test_maxst_total_ge_mst_total(self):
        ei = torch.tensor([[0,1,2,1,2,0],[1,2,0,0,1,2]], dtype=torch.long)
        w = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        _, _, min_total = minimum_spanning_tree(ei, 3, w)
        _, _, max_total = maximum_spanning_tree(ei, 3, w)
        assert max_total >= min_total

    def test_empty_graph(self):
        ei = torch.zeros((2, 0), dtype=torch.long)
        mst_ei, _, _ = minimum_spanning_tree(ei, 3)
        assert mst_ei.size(1) == 0

    def test_disconnected_forest(self):
        # Two disjoint edges: 0-1 and 2-3.
        ei = torch.tensor([[0,1,2,3],[1,0,3,2]], dtype=torch.long)
        mst_ei, _, _ = minimum_spanning_tree(ei, 4)
        assert mst_ei.size(1) == 4  # 2 undirected edges (forest)


class TestCutsAndMetrics:
    def test_cut_size_chain(self):
        ei, N = _chain()
        subset = torch.tensor([0, 1])
        # Both directed crossings (1→2 and 2→1) for the undirected 1-2 edge.
        assert cut_size(ei, N, subset) == 2

    def test_cut_size_empty_subset(self):
        ei, N = _chain()
        assert cut_size(ei, N, torch.zeros(0, dtype=torch.long)) == 0

    def test_cut_size_full_subset(self):
        ei, N = _chain()
        assert cut_size(ei, N, torch.arange(N)) == 0

    def test_volume_star(self):
        ei, N = _star()
        # Hub has degree N-1.
        vol_hub = volume(ei, N, torch.tensor([0]))
        assert vol_hub == N - 1

    def test_conductance_two_cliques(self):
        # Two cliques {0,1,2} and {3,4,5} connected by one edge 2-3.
        src = [0,1,2,3,4,1,2,4,5,2]
        dst = [1,2,3,4,5,0,1,3,4,3]
        ei = torch.tensor([src+dst, dst+src], dtype=torch.long)
        ei = torch.unique(ei, dim=1)
        N = 6
        subset = torch.tensor([0, 1, 2])
        c = conductance(ei, N, subset)
        assert c >= 0.0 and c <= 1.0

    def test_normalized_cut_two_cliques(self):
        src = [0,1,2,3,4,1,2,4,5]
        dst = [1,2,0,4,5,0,1,3,4]
        ei = torch.tensor([src+dst, dst+src], dtype=torch.long)
        ei = torch.unique(ei, dim=1)
        labels = torch.tensor([0,0,0,1,1,1])
        nc = normalized_cut(ei, 6, labels)
        assert nc >= 0.0

    def test_boundary_edges_shape(self):
        ei, N = _chain()
        subset = torch.tensor([0, 1])
        be = boundary_edges(ei, N, subset)
        assert be.shape[0] == 2
        assert be.shape[1] > 0

    def test_write_path_summary(self, tmp_path):
        p = write_path_summary(
            str(tmp_path / "algo.json"),
            source=0, num_reachable=4, mean_distance=1.5,
        )
        import json; d = json.loads(open(p).read())
        assert d["source"] == 0
        assert d["mean_distance"] == 1.5

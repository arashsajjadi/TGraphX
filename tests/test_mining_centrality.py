"""Tests for tgraphx.mining.centrality — centrality algorithms."""
import math
import pytest
import torch
from tgraphx.mining import (
    degree_centrality, pagerank, personalized_pagerank, hits,
    katz_centrality, closeness_centrality, harmonic_centrality,
    betweenness_centrality, eigenvector_centrality, k_core_numbers,
    in_degree_centrality, out_degree_centrality,
)


def _chain(N=4):
    src = list(range(N-1)) + list(range(1, N))
    dst = list(range(1, N)) + list(range(N-1))
    return torch.tensor([src, dst], dtype=torch.long), N


def _star(N=5):
    src = [0]*(N-1) + list(range(1, N))
    dst = list(range(1, N)) + [0]*(N-1)
    return torch.tensor([src, dst], dtype=torch.long), N


def _triangle():
    ei = torch.tensor([[0,1,2,1,2,0],[1,2,0,0,1,2]], dtype=torch.long)
    return ei, 3


class TestDegreeCentrality:
    def test_star_hub_highest(self):
        ei, N = _star()
        dc = degree_centrality(ei, N, directed=False)
        assert float(dc[0]) == float(dc.max())

    def test_range_zero_one(self):
        ei, N = _chain()
        dc = degree_centrality(ei, N)
        assert (dc >= 0).all() and (dc <= 1).all()

    def test_isolated_zero(self):
        ei = torch.zeros((2, 0), dtype=torch.long)
        dc = degree_centrality(ei, 5)
        assert dc.sum() == 0.0

    def test_complete_all_one(self):
        from tgraphx.mining import complete_graph
        ei, N = complete_graph(4)
        dc = degree_centrality(ei, N, directed=False)
        assert all(abs(float(v) - 1.0) < 1e-5 for v in dc.tolist())


class TestPageRank:
    def test_sums_to_one(self):
        ei, N = _chain()
        pr = pagerank(ei, N)
        assert abs(float(pr.sum()) - 1.0) < 1e-4

    def test_non_negative(self):
        ei, N = _chain()
        pr = pagerank(ei, N)
        assert (pr >= 0).all()

    def test_uniform_on_symmetric_complete(self):
        from tgraphx.mining import complete_graph
        ei, N = complete_graph(4)
        pr = pagerank(ei, N)
        assert pr.std().item() < 0.01  # all equal for symmetric complete graph

    def test_empty_graph(self):
        ei = torch.zeros((2, 0), dtype=torch.long)
        pr = pagerank(ei, 4)
        # All equal: 1/N
        assert pr.shape == (4,)
        assert (pr >= 0).all()

    def test_deterministic(self):
        ei, N = _chain()
        pr1 = pagerank(ei, N)
        pr2 = pagerank(ei, N)
        assert torch.equal(pr1, pr2)


class TestPersonalisedPageRank:
    def test_teleport_to_seed(self):
        """PPR from node 0 should assign higher weight to node 0's side."""
        ei, N = _chain(6)
        pers = torch.zeros(6)
        pers[0] = 1.0
        ppr = personalized_pagerank(ei, N, pers)
        assert float(ppr[0]) >= float(ppr[N-1])

    def test_sums_to_one(self):
        ei, N = _chain()
        pers = torch.ones(N)
        ppr = personalized_pagerank(ei, N, pers)
        assert abs(float(ppr.sum()) - 1.0) < 1e-4


class TestHITS:
    def test_shapes(self):
        ei, N = _chain()
        h, a = hits(ei, N)
        assert h.shape == (N,) and a.shape == (N,)

    def test_non_negative(self):
        ei, N = _chain()
        h, a = hits(ei, N)
        assert (h >= 0).all() and (a >= 0).all()

    def test_empty_graph(self):
        ei = torch.zeros((2, 0), dtype=torch.long)
        h, a = hits(ei, 3)
        assert h.shape == (3,) and a.shape == (3,)


class TestKatzCentrality:
    def test_non_negative(self):
        ei, N = _chain()
        k = katz_centrality(ei, N)
        assert (k >= 0).all()

    def test_higher_degree_higher_katz(self):
        # In a chain, middle nodes should have >= katz as leaves.
        ei, N = _chain(5)
        k = katz_centrality(ei, N)
        assert float(k[0]) <= float(k[2])  # leaf <= middle


class TestClosenessCentrality:
    def test_hand_computed_triangle(self):
        """All K3 nodes have closeness = 1 (distance to each other node = 1)."""
        ei, N = _triangle()
        cc = closeness_centrality(ei, N)
        for v in cc.tolist():
            assert abs(v - 1.0) < 1e-5

    def test_path_center_highest(self):
        ei, N = _chain(5)  # 0-1-2-3-4
        cc = closeness_centrality(ei, N)
        # Node 2 (center) has highest closeness.
        assert int(cc.argmax().item()) == 2

    def test_size_guard(self):
        ei = torch.zeros((2, 0), dtype=torch.long)
        with pytest.raises(ValueError, match="max_nodes_exact"):
            closeness_centrality(ei, 2001, max_nodes_exact=2000)


class TestHarmonicCentrality:
    def test_non_negative(self):
        ei, N = _chain()
        hc = harmonic_centrality(ei, N)
        assert (hc >= 0).all()

    def test_star_hub_highest(self):
        ei, N = _star()
        hc = harmonic_centrality(ei, N)
        assert int(hc.argmax().item()) == 0


class TestBetweennessCentrality:
    def test_star_hub_highest(self):
        """Hub node 0 lies on all shortest paths between leaves."""
        ei, N = _star()
        bc = betweenness_centrality(ei, N)
        assert int(bc.argmax().item()) == 0

    def test_path_middle_highest(self):
        ei, N = _chain(5)
        bc = betweenness_centrality(ei, N)
        # Middle node 2 has highest betweenness in a 5-node path.
        assert int(bc.argmax().item()) == 2

    def test_triangle_all_zero(self):
        """In K3, no node lies strictly between two others (only direct edges)."""
        ei, N = _triangle()
        bc = betweenness_centrality(ei, N)
        assert bc.max().item() < 1e-6

    def test_size_guard(self):
        ei = torch.zeros((2, 0), dtype=torch.long)
        with pytest.raises(ValueError, match="max_nodes_exact"):
            betweenness_centrality(ei, 501, max_nodes_exact=500)


class TestEigenvectorCentrality:
    def test_non_negative(self):
        ei, N = _chain()
        ec = eigenvector_centrality(ei, N)
        assert (ec >= 0).all()

    def test_star_hub_highest(self):
        ei, N = _star()
        ec = eigenvector_centrality(ei, N)
        assert int(ec.argmax().item()) == 0


class TestKCoreNumbers:
    def test_path_core_one(self):
        """In a path graph all nodes have k-core = 1 (or 0 for isolated)."""
        ei, N = _chain()
        kc = k_core_numbers(ei, N)
        assert kc.max().item() == 1

    def test_triangle_core_two(self):
        """In K3 all nodes have k-core = 2."""
        ei, N = _triangle()
        kc = k_core_numbers(ei, N)
        assert kc.min().item() == 2 and kc.max().item() == 2

    def test_isolated_core_zero(self):
        ei = torch.zeros((2, 0), dtype=torch.long)
        kc = k_core_numbers(ei, 5)
        assert kc.max().item() == 0

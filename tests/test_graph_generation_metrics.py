"""Tests for graph generation metrics."""
import pytest
import torch

from tgraphx.generation.data_model import GeneratedGraph
from tgraphx.generation.metrics import (
    graph_wl_hash,
    validity_score,
    uniqueness_score,
    novelty_score,
    diversity_score,
    degree_distribution_distance,
    mmd_degree,
    mmd_clustering,
    constraint_satisfaction_rate,
    spectral_distance,
)


def _path_graph(n) -> GeneratedGraph:
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    ei = torch.tensor([src, dst], dtype=torch.long)
    return GeneratedGraph(edge_index=ei, num_nodes=n)


def _complete_graph(n) -> GeneratedGraph:
    src = [i for i in range(n) for j in range(n) if i != j]
    dst = [j for i in range(n) for j in range(n) if i != j]
    ei = torch.tensor([src, dst], dtype=torch.long)
    return GeneratedGraph(edge_index=ei, num_nodes=n)


def _empty_graph(n) -> GeneratedGraph:
    return GeneratedGraph(
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        num_nodes=n,
    )


class TestUniqueness:
    def test_identical_set_uniqueness_low(self):
        g = _path_graph(5)
        graphs = [g.clone() for _ in range(5)]
        score = uniqueness_score(graphs)
        assert score == 1.0 / 5  # All same WL hash

    def test_all_different_uniqueness_one(self):
        graphs = [_path_graph(n) for n in range(3, 8)]
        score = uniqueness_score(graphs)
        assert score == 1.0

    def test_empty_list(self):
        assert uniqueness_score([]) == 0.0


class TestNovelty:
    def test_generated_subset_of_reference(self):
        ref = [_path_graph(n) for n in range(3, 8)]
        gen = [_path_graph(n) for n in [3, 4, 5]]
        score = novelty_score(gen, ref)
        assert score == 0.0

    def test_no_overlap(self):
        ref = [_path_graph(n) for n in range(3, 6)]
        gen = [_complete_graph(n) for n in range(3, 6)]
        score = novelty_score(gen, ref)
        assert score == 1.0


class TestValidity:
    def test_always_true_constraint(self):
        graphs = [_path_graph(n) for n in range(3, 8)]
        score = validity_score(graphs, lambda g: True)
        assert score == 1.0

    def test_always_false_constraint(self):
        graphs = [_path_graph(n) for n in range(3, 8)]
        score = validity_score(graphs, lambda g: False)
        assert score == 0.0


class TestDegreeDistanceDistance:
    def test_identical_distributions_zero(self):
        graphs = [_path_graph(5) for _ in range(3)]
        d = degree_distribution_distance(graphs, graphs, method="l1")
        assert d == pytest.approx(0.0, abs=1e-5)

    def test_l2_method(self):
        graphs_a = [_path_graph(5)]
        graphs_b = [_complete_graph(4)]
        d = degree_distribution_distance(graphs_a, graphs_b, method="l2")
        assert d >= 0.0

    def test_js_method_nonneg(self):
        graphs_a = [_path_graph(5)]
        graphs_b = [_empty_graph(5)]
        d = degree_distribution_distance(graphs_a, graphs_b, method="js")
        assert d >= 0.0


class TestMMD:
    def test_mmd_degree_nonneg(self):
        graphs_a = [_path_graph(5)]
        graphs_b = [_complete_graph(4)]
        d = mmd_degree(graphs_a, graphs_b)
        assert d >= 0.0

    def test_mmd_degree_symmetric(self):
        graphs_a = [_path_graph(5)]
        graphs_b = [_complete_graph(4)]
        d_ab = mmd_degree(graphs_a, graphs_b)
        d_ba = mmd_degree(graphs_b, graphs_a)
        # MMD^2 is symmetric by definition
        assert abs(d_ab - d_ba) < 0.5  # approximately symmetric

    def test_mmd_clustering_nonneg(self):
        graphs_a = [_path_graph(5)]
        graphs_b = [_complete_graph(4)]
        d = mmd_clustering(graphs_a, graphs_b)
        assert d >= 0.0


class TestConstraintSatisfactionRate:
    def test_hand_computed(self):
        # All graphs satisfy max_nodes=10 (all have 5 nodes)
        graphs = [_path_graph(5) for _ in range(5)]
        rates = constraint_satisfaction_rate(graphs, {"max_nodes": 10})
        assert rates["max_nodes"] == pytest.approx(1.0)
        assert rates["overall"] == pytest.approx(1.0)

    def test_violation_rate(self):
        # mix of graphs
        graphs = [_path_graph(3), _path_graph(5), _path_graph(7)]
        rates = constraint_satisfaction_rate(graphs, {"max_nodes": 4})
        # Only path_graph(3) satisfies max_nodes=4
        assert rates["overall"] == pytest.approx(1.0 / 3, abs=0.01)


class TestSpectralDistance:
    def test_symmetric(self):
        graphs_a = [_path_graph(4)]
        graphs_b = [_complete_graph(3)]
        d_ab = spectral_distance(graphs_a, graphs_b)
        d_ba = spectral_distance(graphs_b, graphs_a)
        assert abs(d_ab - d_ba) < 0.1

    def test_nonneg(self):
        graphs_a = [_path_graph(4)]
        graphs_b = [_complete_graph(3)]
        d = spectral_distance(graphs_a, graphs_b)
        assert d >= 0.0

    def test_zero_for_identical(self):
        graphs = [_path_graph(5)]
        d = spectral_distance(graphs, graphs)
        assert d == pytest.approx(0.0, abs=1e-4)

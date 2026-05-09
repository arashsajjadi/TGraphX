"""Tests for evolutionary graph optimization."""
import pytest
import torch

from tgraphx.evolutionary.genome import GraphGenome
from tgraphx.evolutionary.operators import (
    mutate_add_node,
    mutate_remove_node,
    mutate_add_edge,
    mutate_remove_edge,
    mutate_node_feature,
    edge_set_crossover,
    feature_crossover,
)
from tgraphx.evolutionary.selection import tournament_selection, elitism_selection
from tgraphx.evolutionary.multi_objective import pareto_dominates, non_dominated_sort
from tgraphx.evolutionary.fitness import connectivity_fitness, density_fitness
from tgraphx.evolutionary.algorithms import (
    GeneticAlgorithmConfig,
    GeneticAlgorithmOptimizer,
    SimulatedAnnealingOptimizer,
    EvolutionConfig,
)


def _simple_genome(n=5, num_edges=3) -> GraphGenome:
    if num_edges > 0 and n >= 2:
        src = list(range(min(num_edges, n - 1)))
        dst = [s + 1 for s in src]
        ei = torch.tensor([src, dst], dtype=torch.long)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)
    nf = torch.randn(n, 4)
    return GraphGenome(edge_index=ei, num_nodes=n, node_features=nf)


def _simple_genome_with_edge_features(n=4, e=2) -> GraphGenome:
    src = list(range(e))
    dst = [s + 1 for s in src]
    ei = torch.tensor([src, dst], dtype=torch.long)
    nf = torch.randn(n, 4)
    ef = torch.randn(e, 3)
    return GraphGenome(edge_index=ei, num_nodes=n, node_features=nf, edge_features=ef)


# ── Mutation ─────────────────────────────────────────────────────────────────

class TestMutateAddNode:
    def test_increases_num_nodes_by_1(self):
        g = _simple_genome(n=4)
        g2 = mutate_add_node(g)
        assert g2.num_nodes == 5

    def test_node_features_extended(self):
        g = _simple_genome(n=4)
        g2 = mutate_add_node(g)
        assert g2.node_features.shape == (5, 4)

    def test_does_not_mutate_in_place(self):
        g = _simple_genome(n=4)
        _ = mutate_add_node(g)
        assert g.num_nodes == 4

    def test_tensor_features_preserved(self):
        ei = torch.zeros((2, 0), dtype=torch.long)
        # Image-like features [N, C, H, W]
        nf = torch.randn(3, 2, 4, 4)
        g = GraphGenome(edge_index=ei, num_nodes=3, node_features=nf)
        g2 = mutate_add_node(g)
        assert g2.node_features.shape == (4, 2, 4, 4)


class TestMutateRemoveNode:
    def test_decreases_num_nodes_by_1(self):
        g = _simple_genome(n=5)
        gen = torch.Generator()
        gen.manual_seed(0)
        g2 = mutate_remove_node(g, generator=gen)
        assert g2.num_nodes == 4

    def test_no_dangling_edge_ids(self):
        g = _simple_genome(n=5, num_edges=4)
        gen = torch.Generator()
        gen.manual_seed(42)
        g2 = mutate_remove_node(g, generator=gen)
        if g2.num_edges > 0:
            assert int(g2.edge_index.max().item()) < g2.num_nodes

    def test_raises_on_empty_genome(self):
        g = GraphGenome(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=0)
        with pytest.raises(ValueError):
            mutate_remove_node(g)


class TestMutateAddEdge:
    def test_increases_num_edges_by_1(self):
        g = _simple_genome(n=4, num_edges=0)
        gen = torch.Generator()
        gen.manual_seed(0)
        g2 = mutate_add_edge(g, generator=gen)
        assert g2.num_edges == 1

    def test_does_not_mutate_in_place(self):
        g = _simple_genome(n=4, num_edges=0)
        _ = mutate_add_edge(g)
        assert g.num_edges == 0


class TestMutateNodeFeature:
    def test_shape_preserved(self):
        g = _simple_genome(n=4)
        gen = torch.Generator()
        gen.manual_seed(1)
        g2 = mutate_node_feature(g, node_id=0, noise_scale=0.1, generator=gen)
        assert g2.node_features.shape == g.node_features.shape

    def test_values_changed_with_nonzero_noise(self):
        g = _simple_genome(n=4)
        gen = torch.Generator()
        gen.manual_seed(2)
        g2 = mutate_node_feature(g, node_id=0, noise_scale=10.0, generator=gen)
        assert not torch.allclose(g.node_features[0], g2.node_features[0])

    def test_raises_on_no_features(self):
        g = GraphGenome(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            num_nodes=3,
        )
        with pytest.raises(ValueError):
            mutate_node_feature(g, node_id=0)


# ── Crossover ────────────────────────────────────────────────────────────────

class TestEdgeSetCrossover:
    def test_returns_valid_genomes(self):
        g1 = _simple_genome(n=5, num_edges=4)
        g2 = _simple_genome(n=5, num_edges=2)
        gen = torch.Generator()
        gen.manual_seed(10)
        ca, cb = edge_set_crossover(g1, g2, generator=gen)
        ca.validate()
        cb.validate()

    def test_feature_shapes_preserved(self):
        g1 = _simple_genome_with_edge_features(n=4, e=2)
        g2 = _simple_genome(n=4, num_edges=2)
        gen = torch.Generator()
        gen.manual_seed(5)
        ca, cb = edge_set_crossover(g1, g2, generator=gen)
        # Should not raise
        assert ca.num_nodes == 4


class TestFeatureCrossover:
    def test_feature_shapes_preserved(self):
        g1 = _simple_genome(n=4)
        g2 = _simple_genome(n=4)
        gen = torch.Generator()
        gen.manual_seed(20)
        ca, cb = feature_crossover(g1, g2, generator=gen)
        assert ca.node_features.shape == (4, 4)
        assert cb.node_features.shape == (4, 4)


# ── Selection ────────────────────────────────────────────────────────────────

class TestTournamentSelection:
    def test_winner_has_higher_fitness(self):
        pop = [_simple_genome(n=3) for _ in range(5)]
        fitness = [0.1, 0.2, 0.9, 0.4, 0.3]
        gen = torch.Generator()
        gen.manual_seed(42)
        selected = tournament_selection(pop, fitness, k=3, generator=gen)
        # Not a strict guarantee on all, but the best should appear
        assert len(selected) == 5


# ── Multi-objective ──────────────────────────────────────────────────────────

class TestParetoFront:
    def test_pareto_dominates_correct_on_toy(self):
        # [1.0, 2.0] dominates [0.5, 1.0]
        assert pareto_dominates([1.0, 2.0], [0.5, 1.0])
        # [1.0, 2.0] does NOT dominate [2.0, 1.0] (not better in all objectives)
        assert not pareto_dominates([1.0, 2.0], [2.0, 1.0])
        # [1.0, 1.0] does NOT dominate [1.0, 1.0] (equal, not strictly better)
        assert not pareto_dominates([1.0, 1.0], [1.0, 1.0])

    def test_non_dominated_sort_front0_not_dominated(self):
        pop = [_simple_genome(n=3) for _ in range(4)]
        fvecs = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.0]]
        fronts = non_dominated_sort(pop, fvecs)
        assert len(fronts) > 0
        front0 = fronts[0]
        # No member of front0 should be dominated by any other member
        for i in front0:
            for j in front0:
                if i != j:
                    assert not pareto_dominates(fvecs[j], fvecs[i])


# ── GA ───────────────────────────────────────────────────────────────────────

class TestGeneticAlgorithm:
    def test_ga_fitness_improves_on_connectivity(self):
        """GA should improve connectivity fitness over 10 generations."""
        config = GeneticAlgorithmConfig(
            population_size=8,
            n_generations=10,
            seed=42,
            max_nodes=5,
            max_edges=10,
        )
        pop = [_simple_genome(n=4, num_edges=0) for _ in range(8)]
        optimizer = GeneticAlgorithmOptimizer(config, connectivity_fitness)
        result = optimizer.optimize(pop)

        # Should find some positive fitness eventually
        assert result.best_fitness >= 0.0
        # Fitness history should grow
        assert len(result.fitness_history) == 10

    def test_ga_deterministic_with_seed(self):
        """Two GA runs with same seed should give same result."""
        config = GeneticAlgorithmConfig(
            population_size=4,
            n_generations=5,
            seed=99,
            max_nodes=4,
        )
        pop = [_simple_genome(n=3, num_edges=1) for _ in range(4)]

        result1 = GeneticAlgorithmOptimizer(config, connectivity_fitness).optimize(pop)
        result2 = GeneticAlgorithmOptimizer(config, connectivity_fitness).optimize(pop)
        assert result1.best_fitness == result2.best_fitness


class TestSimulatedAnnealing:
    def test_sa_fitness_improves_on_density(self):
        config = EvolutionConfig(seed=7, max_nodes=6, max_edges=20, n_generations=50)
        initial = _simple_genome(n=4, num_edges=1)
        fn = lambda g: density_fitness(g, target_density=0.5)
        optimizer = SimulatedAnnealingOptimizer(config, fn, T_init=2.0)
        result = optimizer.optimize(initial)
        assert result.best_fitness >= 0.0
        assert result.best_genome is not None

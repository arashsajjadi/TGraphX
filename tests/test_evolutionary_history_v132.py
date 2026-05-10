"""Regression tests for EvolutionResult.history (v1.3.2 bugfix).

Bug fixed: EvolutionResult had no `history` attribute, causing AttributeError
when users accessed `result.history` after optimization.
"""
from __future__ import annotations
import json
import pytest
import torch

from tgraphx.evolutionary import (
    GraphGenome, GeneticAlgorithmOptimizer, GeneticAlgorithmConfig,
    NSGAIIOptimizer, EvolutionConfig, connectivity_fitness,
)


def _genome(seed=0):
    torch.manual_seed(seed)
    return GraphGenome(edge_index=torch.randint(0, 8, (2, 10)), num_nodes=8)


def _ga_result(n_gen=5):
    cfg = GeneticAlgorithmConfig(population_size=6, n_generations=n_gen, seed=0)
    return GeneticAlgorithmOptimizer(cfg, connectivity_fitness).optimize([_genome(i) for i in range(6)])


class TestGAHistory:
    def test_history_attribute_exists(self):
        r = _ga_result()
        assert hasattr(r, "history")

    def test_history_length_equals_n_generations(self):
        n = 7
        r = _ga_result(n)
        assert len(r.history) == n

    def test_history_entries_are_dicts(self):
        r = _ga_result()
        for entry in r.history:
            assert isinstance(entry, dict)

    def test_history_has_required_keys(self):
        r = _ga_result()
        for entry in r.history:
            assert "generation" in entry
            assert "best_fitness" in entry

    def test_history_generation_index_correct(self):
        r = _ga_result(4)
        assert r.history[0]["generation"] == 0
        assert r.history[3]["generation"] == 3

    def test_best_fitness_unchanged(self):
        r = _ga_result()
        assert isinstance(r.best_fitness, float)
        assert r.best_fitness > 0

    def test_colab_exact_repro(self):
        """Exact snippet from the Colab bug report."""
        config = GeneticAlgorithmConfig(population_size=10, n_generations=15, seed=42)
        pop = [_genome(i) for i in range(10)]
        result = GeneticAlgorithmOptimizer(config, connectivity_fitness).optimize(pop)
        # These were the two print statements that triggered the bug:
        _ = result.best_fitness          # must not raise
        _ = len(result.history)          # must not raise (was AttributeError)
        assert len(result.history) > 0


class TestGASummaryToDict:
    def test_summary_returns_string(self):
        r = _ga_result()
        s = r.summary()
        assert isinstance(s, str)
        assert "Best fitness" in s

    def test_to_dict_json_serializable(self):
        r = _ga_result()
        d = r.to_dict()
        json.dumps(d)

    def test_to_dict_has_expected_keys(self):
        r = _ga_result()
        d = r.to_dict()
        assert "best_fitness" in d
        assert "n_generations" in d
        assert "fitness_history" in d


class TestNSGAIIHistory:
    def _nsga_result(self, n_gen=6, single_fn=True):
        cfg = EvolutionConfig(population_size=6, n_generations=n_gen, seed=0)
        fn = connectivity_fitness if single_fn else [connectivity_fitness, connectivity_fitness]
        return NSGAIIOptimizer(cfg, fn).optimize([_genome(i) for i in range(6)])

    def test_history_attribute_exists(self):
        r = self._nsga_result()
        assert hasattr(r, "history")

    def test_history_length_equals_n_generations(self):
        n = 8
        r = self._nsga_result(n)
        assert len(r.history) == n

    def test_history_has_generation_key(self):
        r = self._nsga_result()
        for entry in r.history:
            assert "generation" in entry

    def test_pareto_front_still_present(self):
        r = self._nsga_result()
        assert r.pareto_front is not None

    def test_single_callable_accepted(self):
        """NSGAIIOptimizer must accept a single callable (not just a list)."""
        cfg = EvolutionConfig(population_size=4, n_generations=3, seed=0)
        r = NSGAIIOptimizer(cfg, connectivity_fitness).optimize([_genome(i) for i in range(4)])
        assert len(r.history) == 3

    def test_list_of_fns_still_works(self):
        """Passing a list of functions must still work."""
        cfg = EvolutionConfig(population_size=4, n_generations=3, seed=0)
        r = NSGAIIOptimizer(cfg, [connectivity_fitness, connectivity_fitness]).optimize(
            [_genome(i) for i in range(4)]
        )
        assert len(r.history) == 3

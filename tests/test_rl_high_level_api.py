"""Tests for the high-level RL, generation, and evolutionary APIs."""
from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch


# ---------------------------------------------------------------------------
# list_graph_rl_algorithms
# ---------------------------------------------------------------------------

def test_list_graph_rl_algorithms_has_expected_keys():
    from tgraphx.rl import list_graph_rl_algorithms
    algs = list_graph_rl_algorithms()
    for key in ("dqn", "td3", "sac", "reinforce", "random", "ppo", "a2c", "actor_critic"):
        assert key in algs, f"'{key}' missing from list_graph_rl_algorithms()"


def test_list_graph_rl_algorithms_info_structure():
    from tgraphx.rl import list_graph_rl_algorithms
    algs = list_graph_rl_algorithms()
    for name, info in algs.items():
        assert "action_type" in info
        assert "stability" in info
        assert "description" in info


# ---------------------------------------------------------------------------
# list_graph_generation_methods
# ---------------------------------------------------------------------------

def test_list_graph_generation_methods_has_expected_keys():
    from tgraphx.generation import list_graph_generation_methods
    methods = list_graph_generation_methods()
    for key in ("erdos_renyi", "barabasi_albert", "path", "cycle", "star"):
        assert key in methods, f"'{key}' missing from list_graph_generation_methods()"


# ---------------------------------------------------------------------------
# list_evolutionary_optimizers
# ---------------------------------------------------------------------------

def test_list_evolutionary_optimizers_has_expected_keys():
    from tgraphx.evolutionary import list_evolutionary_optimizers
    optimizers = list_evolutionary_optimizers()
    for key in ("ga", "sa", "nsga2", "hill_climbing", "random_search"):
        assert key in optimizers, f"'{key}' missing from list_evolutionary_optimizers()"


# ---------------------------------------------------------------------------
# run_graph_rl
# ---------------------------------------------------------------------------

def test_run_graph_rl_random():
    from tgraphx.rl import run_graph_rl
    result = run_graph_rl(env="graph_navigation", algorithm="random", episodes=5, seed=42)
    assert "episode_returns" in result.metrics
    assert len(result.metrics["episode_returns"]) == 5
    assert "mean_return" in result.metrics


def test_run_graph_rl_dqn():
    from tgraphx.rl import run_graph_rl
    result = run_graph_rl(env="graph_navigation", algorithm="dqn", episodes=5, seed=42)
    assert "episode_returns" in result.metrics
    assert len(result.metrics["episode_returns"]) == 5


def test_run_graph_rl_reinforce():
    from tgraphx.rl import run_graph_rl
    result = run_graph_rl(env="graph_navigation", algorithm="reinforce", episodes=5, seed=42)
    assert "episode_returns" in result.metrics


def test_run_graph_rl_ppo():
    from tgraphx.rl import run_graph_rl
    result = run_graph_rl(env="graph_navigation", algorithm="ppo", episodes=5, seed=42)
    assert "episode_returns" in result.metrics


def test_run_graph_rl_actor_critic():
    from tgraphx.rl import run_graph_rl
    result = run_graph_rl(env="graph_navigation", algorithm="actor_critic", episodes=3, seed=42)
    assert "episode_returns" in result.metrics


def test_run_graph_rl_a2c():
    from tgraphx.rl import run_graph_rl
    result = run_graph_rl(env="graph_navigation", algorithm="a2c", episodes=3, seed=42)
    assert "episode_returns" in result.metrics


def test_run_graph_rl_invalid_algorithm_raises():
    from tgraphx.rl import run_graph_rl
    with pytest.raises(ValueError, match="Unknown algorithm"):
        run_graph_rl(env="graph_navigation", algorithm="foobar", episodes=3)


def test_run_graph_rl_result_metrics_keys():
    from tgraphx.rl import run_graph_rl
    result = run_graph_rl(env="graph_navigation", algorithm="random", episodes=3, seed=1)
    assert "episode_returns" in result.metrics
    assert "mean_return" in result.metrics
    assert "success_rate" in result.metrics
    assert "algorithm" in result.metrics
    assert "environment" in result.metrics


def test_run_graph_rl_dashboard_dir():
    from tgraphx.rl import run_graph_rl
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_graph_rl(
            env="graph_navigation", algorithm="random",
            episodes=3, seed=42, dashboard_dir=tmpdir,
        )
        assert result.report_path is not None
        assert os.path.isfile(result.report_path)
        with open(result.report_path) as f:
            data = json.load(f)
        assert "metrics" in data
        assert "config" in data


def test_run_graph_rl_deterministic_same_seed():
    from tgraphx.rl import run_graph_rl
    result1 = run_graph_rl(env="graph_navigation", algorithm="random", episodes=5, seed=0)
    result2 = run_graph_rl(env="graph_navigation", algorithm="random", episodes=5, seed=0)
    assert result1.metrics["episode_returns"] == result2.metrics["episode_returns"]


# ---------------------------------------------------------------------------
# run_graph_generation
# ---------------------------------------------------------------------------

def test_run_graph_generation_erdos_renyi():
    from tgraphx.generation import run_graph_generation
    result = run_graph_generation(method="erdos_renyi", num_graphs=4, num_nodes=10, seed=42)
    assert len(result.graphs) == 4
    assert "validity" in result.metrics
    assert "uniqueness" in result.metrics


def test_run_graph_generation_barabasi_albert():
    from tgraphx.generation import run_graph_generation
    result = run_graph_generation(method="barabasi_albert", num_graphs=4, num_nodes=10, m=2, seed=42)
    assert len(result.graphs) == 4


def test_run_graph_generation_invalid_method_raises():
    from tgraphx.generation import run_graph_generation
    with pytest.raises(ValueError, match="Unknown generation method"):
        run_graph_generation(method="not_a_method", num_graphs=4, num_nodes=10)


def test_run_graph_generation_path():
    from tgraphx.generation import run_graph_generation
    result = run_graph_generation(method="path", num_graphs=3, num_nodes=8, seed=0)
    assert len(result.graphs) == 3


def test_run_graph_generation_metrics_keys():
    from tgraphx.generation import run_graph_generation
    result = run_graph_generation(method="erdos_renyi", num_graphs=4, num_nodes=8, seed=0)
    assert "validity" in result.metrics
    assert "uniqueness" in result.metrics
    assert "num_graphs" in result.metrics


def test_run_graph_generation_dashboard_dir():
    from tgraphx.generation import run_graph_generation
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_graph_generation(
            method="erdos_renyi", num_graphs=3, num_nodes=8,
            seed=42, dashboard_dir=tmpdir,
        )
        assert result.report_path is not None
        assert os.path.isfile(result.report_path)
        with open(result.report_path) as f:
            data = json.load(f)
        assert "metrics" in data


# ---------------------------------------------------------------------------
# run_evolutionary_optimization
# ---------------------------------------------------------------------------

def test_run_evolutionary_optimization_ga():
    from tgraphx.evolutionary import run_evolutionary_optimization
    result = run_evolutionary_optimization(
        algorithm="ga", objective="connectivity",
        population_size=5, generations=3, num_nodes=8, seed=42,
    )
    assert result.best_genome is not None
    assert isinstance(result.best_fitness, float)


def test_run_evolutionary_optimization_nsga2():
    from tgraphx.evolutionary import run_evolutionary_optimization
    result = run_evolutionary_optimization(
        algorithm="nsga2", objective=["connectivity", "density"],
        population_size=4, generations=3, num_nodes=6, seed=42,
    )
    assert result.best_genome is not None


def test_run_evolutionary_optimization_invalid_algorithm_raises():
    from tgraphx.evolutionary import run_evolutionary_optimization
    with pytest.raises(ValueError, match="Unknown evolutionary algorithm"):
        run_evolutionary_optimization(algorithm="bad_algo", generations=3)


def test_run_evolutionary_optimization_sa():
    from tgraphx.evolutionary import run_evolutionary_optimization
    result = run_evolutionary_optimization(
        algorithm="sa", objective="density",
        generations=5, num_nodes=6, seed=0,
    )
    assert isinstance(result.best_fitness, float)


def test_run_evolutionary_optimization_hill_climbing():
    from tgraphx.evolutionary import run_evolutionary_optimization
    result = run_evolutionary_optimization(
        algorithm="hill_climbing", objective="clustering",
        generations=4, num_nodes=6, seed=1,
    )
    assert result.best_genome is not None


def test_run_evolutionary_optimization_random_search():
    from tgraphx.evolutionary import run_evolutionary_optimization
    result = run_evolutionary_optimization(
        algorithm="random_search", objective="connectivity",
        generations=5, num_nodes=6, seed=2,
    )
    assert isinstance(result.best_fitness, float)


def test_run_evolutionary_optimization_dashboard_dir():
    from tgraphx.evolutionary import run_evolutionary_optimization
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_evolutionary_optimization(
            algorithm="ga", objective="connectivity",
            population_size=4, generations=3, num_nodes=6,
            seed=0, dashboard_dir=tmpdir,
        )
        assert result.report_path is not None
        assert os.path.isfile(result.report_path)
        with open(result.report_path) as f:
            data = json.load(f)
        assert "config" in data
        assert "best_fitness" in data

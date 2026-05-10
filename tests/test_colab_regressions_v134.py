"""Colab regression tests (v1.3.4).

One test per public snippet that was reported as broken from a Colab / PyPI install.
These tests pin the exact user-facing code patterns so they can never silently regress.

All tests must:
- Use only public APIs (importable after `pip install tgraphx`).
- Run on CPU without network.
- Be fast (< 30 seconds each).
- Fail with a clear message if the bug reappears.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


# ── Bug A — Easy Mode deterministic reproducibility ───────────────────────────


class TestReproducibilityDetministicCPU:
    """Same seed + deterministic=True + device=cpu must give identical final loss."""

    def _run_once(self):
        import tgraphx as tgx
        from tgraphx.reproducibility import set_seed

        set_seed(42, deterministic=True)
        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=64, node_shape=(4, 4, 4), num_classes=3,
            num_edges=200, seed=42,
        )
        r = tgx.easy.train_node_classifier(
            data, epochs=2, batch_size=16, fanouts=[5, 3],
            verbose=False, seed=42, deterministic=True, device="cpu",
        )
        return r

    def test_deterministic_cpu_exact_match(self):
        r1 = self._run_once()
        r2 = self._run_once()
        diff = abs(r1.metrics["loss"] - r2.metrics["loss"])
        assert diff < 1e-7, f"CPU deterministic loss diff too large: {diff:.2e}"

    def test_synthetic_data_reproducible(self):
        import tgraphx as tgx
        d1 = tgx.easy.synthetic_tensor_node_classification(num_nodes=32, seed=7)
        d2 = tgx.easy.synthetic_tensor_node_classification(num_nodes=32, seed=7)
        assert torch.equal(d1.node_features, d2.node_features)
        assert torch.equal(d1.edge_index, d2.edge_index)

    def test_set_seed_returns_state_dict(self):
        from tgraphx.reproducibility import set_seed
        state = set_seed(42, deterministic=True)
        assert isinstance(state, dict)
        assert state["seed"] == 42
        assert state["deterministic"] is True
        assert "torch_version" in state

    def test_train_stores_reproducibility_state(self):
        r = self._run_once()
        state = r.config.get("reproducibility_state", {})
        assert state.get("seed") == 42
        assert state.get("deterministic") is True

    def test_exact_user_colab_snippet(self):
        """Exact snippet from the reproducibility bug report."""
        import tgraphx as tgx
        from tgraphx.reproducibility import set_seed

        results = []
        for _ in range(2):
            set_seed(42, deterministic=True)
            data = tgx.easy.synthetic_tensor_node_classification(
                num_nodes=64, node_shape=(4, 4, 4), num_classes=3,
                num_edges=200, seed=42,
            )
            r = tgx.easy.train_node_classifier(
                data, epochs=2, batch_size=16, fanouts=[5, 3],
                verbose=False, seed=42, deterministic=True, device="cpu",
            )
            results.append(r.metrics["loss"])

        diff = abs(results[0] - results[1])
        assert diff < 1e-7, diff


# ── Bug B — NSGA-II composite_fitness misuse ──────────────────────────────────


class TestNSGAIIOptimizer:
    """NSGAIIOptimizer requires a list of objectives, not composite_fitness directly."""

    def _genome(self, seed=0):
        from tgraphx.evolutionary import GraphGenome
        torch.manual_seed(seed)
        return GraphGenome(edge_index=torch.randint(0, 8, (2, 10)), num_nodes=8)

    def test_nsga_with_list_of_objectives(self):
        from tgraphx.evolutionary import NSGAIIOptimizer, EvolutionConfig, connectivity_fitness
        objectives = [connectivity_fitness, connectivity_fitness]
        config = EvolutionConfig(population_size=6, n_generations=4, seed=0)
        pop = [self._genome(i) for i in range(6)]
        r = NSGAIIOptimizer(config, objectives).optimize(pop)
        assert r.pareto_front is not None
        assert r.best_fitness >= 0.0
        assert len(r.history) == 4

    def test_ga_history_works(self):
        from tgraphx.evolutionary import (
            GeneticAlgorithmOptimizer, GeneticAlgorithmConfig, connectivity_fitness,
        )
        config = GeneticAlgorithmConfig(population_size=6, n_generations=5, seed=42)
        pop = [self._genome(i) for i in range(6)]
        r = GeneticAlgorithmOptimizer(config, connectivity_fitness).optimize(pop)
        assert len(r.history) == 5
        assert "generation" in r.history[0]
        assert "best_fitness" in r.history[0]

    def test_nsga_history_works(self):
        from tgraphx.evolutionary import NSGAIIOptimizer, EvolutionConfig, connectivity_fitness
        config = EvolutionConfig(population_size=4, n_generations=3, seed=0)
        pop = [self._genome(i) for i in range(4)]
        r = NSGAIIOptimizer(config, [connectivity_fitness]).optimize(pop)
        assert len(r.history) == 3

    def test_evolution_result_to_dict_json_serializable(self):
        from tgraphx.evolutionary import (
            GeneticAlgorithmOptimizer, GeneticAlgorithmConfig, connectivity_fitness,
        )
        config = GeneticAlgorithmConfig(population_size=4, n_generations=3, seed=0)
        pop = [self._genome(i) for i in range(4)]
        r = GeneticAlgorithmOptimizer(config, connectivity_fitness).optimize(pop)
        d = r.to_dict()
        json.dumps(d)  # must not raise


# ── Bug C — GraphML exception handling ───────────────────────────────────────


class TestGraphMLExceptionHandling:
    """ValueError from write_graphml must be formatted with str(e), not e[:120]."""

    def test_str_exception_not_slice(self, tmp_path):
        """Ensures str(e)[:120] works; e[:120] would TypeError in real code."""
        from tgraphx import Graph
        from tgraphx.io import write_graphml

        x_spatial = torch.randn(4, 3, 8, 8)
        g = Graph(node_features=x_spatial,
                  edge_index=torch.tensor([[0, 1], [1, 2]]))

        caught = None
        try:
            write_graphml(g, tmp_path / "test.graphml", include_tensor_features=True)
        except ValueError as e:
            caught = e
            msg = str(e)[:120]  # must not raise TypeError
            assert "multi-dimensional" in msg or "rank" in msg or "shape" in msg or "tensor" in msg.lower()

        assert caught is not None, "Expected ValueError for [N,C,H,W] tensor features"

    def test_1d_features_round_trip(self, tmp_path):
        """1-D node features must round-trip through GraphML without error."""
        from tgraphx import Graph
        from tgraphx.io import write_graphml, read_graphml

        x = torch.tensor([[0.5], [1.0], [0.25]])
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        g = Graph(node_features=x, edge_index=ei)
        path = tmp_path / "roundtrip.graphml"
        write_graphml(g, path, include_tensor_features=True)
        g2 = read_graphml(path)
        assert g2.num_nodes == 3
        assert g2.num_edges == 2


# ── Bug D — motif_profile / graph_summary import ─────────────────────────────


class TestMiningImports:
    def test_motif_profile_importable(self):
        from tgraphx.mining import motif_profile
        assert callable(motif_profile)

    def test_graph_summary_importable(self):
        from tgraphx.mining import graph_summary
        assert callable(graph_summary)

    def test_graph_summary_returns_expected_keys(self):
        from tgraphx.mining import graph_summary
        ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        s = graph_summary(ei, num_nodes=5)
        for key in ("num_nodes", "num_edges"):
            assert key in s, f"Missing key: {key}"
        assert s["num_nodes"] == 5

    def test_motif_profile_returns_dict_with_triangles(self):
        from tgraphx.mining import motif_profile
        # Triangle graph
        ei = torch.tensor([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]], dtype=torch.long)
        m = motif_profile(ei, num_nodes=3)
        assert isinstance(m, dict)
        # There is at least one triangle
        assert m.get("triangles", 0) > 0 or "triangles" in str(m)

    def test_degree_statistics_importable(self):
        from tgraphx.mining import degree_statistics
        ei = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=torch.long)
        s = degree_statistics(ei, num_nodes=5)
        assert isinstance(s, dict)

    def test_notebook_20_smoke(self):
        """Exact imports from notebook 20."""
        from tgraphx.mining import motif_profile, graph_summary
        from tgraphx.mining.matching_coloring import enumerate_maximal_cliques
        import torch
        N = 5
        ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        summary = graph_summary(ei, num_nodes=N)
        assert isinstance(summary, dict)
        profile = motif_profile(ei, num_nodes=N)
        assert isinstance(profile, dict)
        cliques = enumerate_maximal_cliques(ei, num_nodes=N, max_nodes=50)
        assert isinstance(cliques, list)


# ── Bug E — wl_subtree_kernel import ─────────────────────────────────────────


class TestWLSubtreeKernel:
    def test_importable(self):
        from tgraphx.mining import wl_subtree_kernel
        assert callable(wl_subtree_kernel)

    def test_identical_graphs_higher_than_different(self):
        from tgraphx.mining import wl_subtree_kernel
        ei_ring5 = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        ei_ring3 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        k_self = wl_subtree_kernel(ei_ring5, 5, ei_ring5, 5, h=3)
        k_cross = wl_subtree_kernel(ei_ring5, 5, ei_ring3, 3, h=3)
        assert k_self >= k_cross

    def test_symmetry(self):
        from tgraphx.mining import wl_subtree_kernel
        ei_a = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        ei_b = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        k_ab = wl_subtree_kernel(ei_a, 3, ei_b, 3, h=2)
        k_ba = wl_subtree_kernel(ei_b, 3, ei_a, 3, h=2)
        assert abs(k_ab - k_ba) < 1e-6

    def test_normalize_finite(self):
        from tgraphx.mining import wl_subtree_kernel
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        k = wl_subtree_kernel(ei, 2, ei, 2, h=2, normalize=True)
        assert math.isfinite(k)

    def test_notebook_21_smoke(self):
        """Exact import from notebook 21."""
        from tgraphx.mining.kernels import wl_subtree_kernel
        # Must work through kernels module too via alias; check from mining
        from tgraphx.mining import wl_subtree_kernel as wl_sk
        ei5 = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        ei3 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        k = wl_sk(ei5, 5, ei3, 3, h=3)
        assert k >= 0


# ── Bug F — centrality_summary import ────────────────────────────────────────


class TestCentralitySummary:
    def test_importable(self):
        from tgraphx.mining import centrality_summary
        assert callable(centrality_summary)

    def test_star_center_is_top(self):
        from tgraphx.mining import centrality_summary
        # Star graph: node 0 is center
        star_ei = torch.tensor([[0, 0, 0, 0, 1, 2, 3, 4],
                                 [1, 2, 3, 4, 0, 0, 0, 0]], dtype=torch.long)
        cs = centrality_summary(star_ei, num_nodes=5)
        assert cs["top_degree_nodes"][0][0] == 0, "Star center must be top"

    def test_returns_expected_keys(self):
        from tgraphx.mining import centrality_summary
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        cs = centrality_summary(ei, num_nodes=3)
        for key in ("num_nodes", "num_edges", "top_degree_nodes"):
            assert key in cs

    def test_notebook_22_smoke(self):
        """Exact imports from notebook 22."""
        from tgraphx.mining import degree_statistics, centrality_summary
        star_ei = torch.tensor([[0, 0, 0, 0, 1, 2, 3, 4],
                                  [1, 2, 3, 4, 0, 0, 0, 0]], dtype=torch.long)
        ds = degree_statistics(star_ei, num_nodes=5)
        cs = centrality_summary(star_ei, num_nodes=5)
        assert isinstance(ds, dict)
        assert isinstance(cs, dict)


# ── Bug G — Package-level benchmark suite ────────────────────────────────────


class TestPackageBenchmarkSuite:
    def test_importable_from_package(self):
        from tgraphx.benchmarks import run_v13_benchmark_suite
        assert callable(run_v13_benchmark_suite)

    def test_runs_outside_repo_tree(self, tmp_path):
        """Simulate running from outside repo: use package API, not repo script."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "from tgraphx.benchmarks import run_v13_benchmark_suite;"
             "data = run_v13_benchmark_suite(small=True, return_dict=True);"
             "assert 'benchmarks' in data;"
             "print('OK', len(data['benchmarks']))"],
            capture_output=True, text=True, timeout=120,
            cwd=str(tmp_path),  # Run from an empty temp dir, not repo
        )
        assert result.returncode == 0, f"STDERR: {result.stderr[:300]}"
        assert "OK" in result.stdout

    def test_returns_expected_schema(self):
        from tgraphx.benchmarks import run_v13_benchmark_suite
        data = run_v13_benchmark_suite(small=True, return_dict=True)
        assert "suite" in data
        assert "benchmarks" in data
        for row in data["benchmarks"]:
            assert "name" in row
            assert "status" in row

    def test_json_serializable(self):
        from tgraphx.benchmarks import run_v13_benchmark_suite
        data = run_v13_benchmark_suite(small=True, return_dict=True)
        json.dumps(data, default=str)

    def test_python_m_cli(self, tmp_path):
        """python -m tgraphx.benchmarks.run_v13_benchmark_suite --small --json"""
        import subprocess, sys
        out_file = tmp_path / "out.json"
        result = subprocess.run(
            [sys.executable, "-m", "tgraphx.benchmarks.run_v13_benchmark_suite",
             "--small", "--out", str(out_file)],
            capture_output=True, text=True, timeout=120,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, f"STDERR: {result.stderr[:300]}"
        data = json.loads(out_file.read_text())
        assert "benchmarks" in data


# ── Bug H — ConvMessagePassing / NeighborLoader seed-node loss ───────────────


class TestConvMessagePassingNeighborLoader:
    def test_seed_logits_no_unsafe_slicing(self):
        """batch.seed_logits(logits) must work; logits[:batch_size] would be wrong."""
        from tgraphx import Graph, NeighborLoader
        import torch.nn as nn

        torch.manual_seed(0)
        N, D = 50, 8
        x = torch.randn(N, D)
        ei = torch.randint(0, N, (2, 200))
        y = torch.randint(0, 3, (N,))
        g = Graph(node_features=x, edge_index=ei, y=y)

        loader = NeighborLoader(g, fanouts=[5, 3], batch_size=8, seed=0)
        linear = nn.Linear(D, 3)
        opt = torch.optim.Adam(linear.parameters(), lr=1e-2)

        batch = next(iter(loader))
        logits = linear(batch.node_features)
        # Correct API: batch.seed_logits extracts supervision-node logits.
        seed_logits = batch.seed_logits(logits)
        assert seed_logits.shape[0] == batch.batch_size
        loss = F.cross_entropy(seed_logits, batch.seed_y)
        loss.backward()
        assert torch.isfinite(loss)


# ── Bug I — Feature-aware KG regression (v1.3.1 fix) ────────────────────────


class TestFeatureAwareKGRegression:
    def test_entity_features_score_triples(self):
        """Exact snippet from v1.3.1 Colab bug report."""
        from tgraphx.kg import KnowledgeGraph, TransEModel

        torch.manual_seed(0)
        N_e, N_r, N_t = 10, 3, 30
        heads = torch.randint(0, N_e, (N_t,))
        rels = torch.randint(0, N_r, (N_t,))
        tails = torch.randint(0, N_e, (N_t,))
        entity_features = {"visual": torch.randn(N_e, 32)}

        kg = KnowledgeGraph.from_hrt(
            heads, rels, tails,
            num_entities=N_e, num_relations=N_r,
            entity_features=entity_features,
        )

        model = TransEModel(N_e, N_r, embedding_dim=16, entity_feature_dim=32)
        triples = torch.stack([heads, rels, tails], dim=1)
        scores = model.score_triples(triples, entity_features=kg.entity_features["visual"])
        assert scores.shape == (N_t,)
        scores.mean().backward()
        assert model.entity_proj.proj.weight.grad is not None

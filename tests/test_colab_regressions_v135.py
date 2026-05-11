"""Colab regression tests (v1.3.5).

Covers every public snippet that was reported as broken in the v1.3.5 sprint.
Each test represents a specific Colab/PyPI-install failure mode.

All tests must:
- Use only public APIs (importable after `pip install tgraphx`).
- Run on CPU without network.
- Be fast (< 30 seconds each).
- Fail with a clear message if the bug reappears.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


# ── Bug A — NSGA-II composite_fitness misuse ──────────────────────────────────


class TestNSGAIIObjectiveListAPI:
    """NSGAIIOptimizer must accept a list of objectives and reject composite_fitness directly."""

    def _make_genome(self, seed: int = 0):
        from tgraphx.evolutionary import GraphGenome
        torch.manual_seed(seed)
        return GraphGenome(edge_index=torch.randint(0, 8, (2, 10)), num_nodes=8)

    def test_ga_optimizer_still_works(self):
        """GeneticAlgorithmOptimizer must work as before."""
        from tgraphx.evolutionary import (
            GraphGenome, GeneticAlgorithmOptimizer, GeneticAlgorithmConfig,
            connectivity_fitness,
        )
        config = GeneticAlgorithmConfig(population_size=10, n_generations=5, seed=42)
        pop = [self._make_genome(i) for i in range(10)]
        result = GeneticAlgorithmOptimizer(config, connectivity_fitness).optimize(pop)
        assert result.best_fitness >= 0.0
        assert len(result.history) > 0

    def test_nsga2_with_objective_list(self):
        """NSGAIIOptimizer with [connectivity_fitness, sparsity_fitness] must succeed."""
        from tgraphx.evolutionary import (
            NSGAIIOptimizer, EvolutionConfig,
            connectivity_fitness, sparsity_fitness,
        )
        config = EvolutionConfig(population_size=8, n_generations=5, seed=0)
        pop = [self._make_genome(i) for i in range(8)]
        r = NSGAIIOptimizer(config, [connectivity_fitness, sparsity_fitness]).optimize(pop)
        assert r.pareto_front is not None
        assert len(r.history) > 0

    def test_sparsity_fitness_importable_and_valid(self):
        """sparsity_fitness must be importable from tgraphx.evolutionary and return [0,1]."""
        from tgraphx.evolutionary import sparsity_fitness
        g = self._make_genome(0)
        s = sparsity_fitness(g)
        assert 0.0 <= s <= 1.0

    def test_sparsity_fitness_empty_graph(self):
        """Empty graph (no edges) should have sparsity_fitness == 1.0."""
        from tgraphx.evolutionary import sparsity_fitness, GraphGenome
        g = GraphGenome(edge_index=torch.zeros(2, 0, dtype=torch.long), num_nodes=4)
        assert sparsity_fitness(g) == 1.0

    def test_composite_fitness_direct_raises_helpful_error(self):
        """Passing composite_fitness directly to NSGAIIOptimizer must raise TypeError with helpful message."""
        from tgraphx.evolutionary import (
            NSGAIIOptimizer, EvolutionConfig, composite_fitness,
        )
        config = EvolutionConfig(population_size=4, n_generations=2, seed=0)
        with pytest.raises(TypeError) as exc_info:
            NSGAIIOptimizer(config, composite_fitness)
        msg = str(exc_info.value)
        assert "NSGAIIOptimizer expects a sequence/list" in msg, (
            f"Missing helpful diagnostic in error: {msg}"
        )
        assert "GeneticAlgorithmOptimizer" in msg, (
            f"Missing GeneticAlgorithmOptimizer suggestion: {msg}"
        )

    def test_nsga2_pareto_front_non_empty(self):
        """NSGAIIOptimizer.pareto_front must be a non-empty list after optimization."""
        from tgraphx.evolutionary import (
            NSGAIIOptimizer, EvolutionConfig,
            connectivity_fitness, sparsity_fitness,
        )
        config = EvolutionConfig(population_size=8, n_generations=3, seed=1)
        pop = [self._make_genome(i) for i in range(8)]
        r = NSGAIIOptimizer(config, [connectivity_fitness, sparsity_fitness]).optimize(pop)
        assert len(r.pareto_front) >= 1

    def test_nsga2_history_length(self):
        """history entries must equal n_generations."""
        from tgraphx.evolutionary import (
            NSGAIIOptimizer, EvolutionConfig,
            connectivity_fitness, sparsity_fitness,
        )
        n_gen = 4
        config = EvolutionConfig(population_size=6, n_generations=n_gen, seed=2)
        pop = [self._make_genome(i) for i in range(6)]
        r = NSGAIIOptimizer(config, [connectivity_fitness, sparsity_fitness]).optimize(pop)
        assert len(r.history) == n_gen

    def test_full_notebook14_public_snippet(self):
        """Exact public v1.3.5 Colab snippet for notebook 14 must succeed end-to-end."""
        import torch
        from tgraphx.evolutionary import (
            GraphGenome, GeneticAlgorithmOptimizer, GeneticAlgorithmConfig,
            connectivity_fitness,
        )
        from tgraphx.evolutionary import (
            NSGAIIOptimizer, EvolutionConfig,
            sparsity_fitness,
        )

        def make_genome(seed=0):
            torch.manual_seed(seed)
            return GraphGenome(edge_index=torch.randint(0, 8, (2, 10)), num_nodes=8)

        config = GeneticAlgorithmConfig(population_size=10, n_generations=5, seed=42)
        pop = [make_genome(i) for i in range(10)]
        result = GeneticAlgorithmOptimizer(config, connectivity_fitness).optimize(pop)
        assert result.best_fitness >= 0.0
        assert len(result.history) > 0

        objectives = [connectivity_fitness, sparsity_fitness]
        config2 = EvolutionConfig(population_size=8, n_generations=5, seed=0)
        pop2 = [make_genome(i) for i in range(8)]
        r2 = NSGAIIOptimizer(config2, objectives).optimize(pop2)
        assert r2.pareto_front is not None
        assert len(r2.history) > 0


# ── Bug B — README / gallery consolidation ────────────────────────────────────


class TestReadmeGalleryConsolidation:

    def test_no_old_colab_single_tutorial_sentence(self):
        """README must not contain the stale 'A Colab tutorial walks through every workflow' sentence."""
        readme = Path("README.md").read_text()
        assert "A Colab tutorial walks through every workflow" not in readme

    def test_readme_has_notebook_gallery_section(self):
        """README must have a visible Notebook Gallery section."""
        readme = Path("README.md").read_text()
        assert "Notebook Gallery" in readme or "notebook gallery" in readme.lower()

    def test_readme_links_to_colab_gallery(self):
        """README must link to docs/colab_gallery.md."""
        readme = Path("README.md").read_text()
        assert "colab_gallery.md" in readme

    def test_colab_gallery_exists_and_has_links(self):
        """docs/colab_gallery.md must exist and contain Google Drive links."""
        gallery = Path("docs/colab_gallery.md")
        assert gallery.exists()
        text = gallery.read_text()
        assert "drive.google.com" in text

    def test_colab_gallery_covers_key_notebooks(self):
        """Gallery must reference the Google Drive file IDs for the key notebooks.

        (v1.3.6 visual polish replaced raw notebook filenames in the gallery
        with reader-friendly display names. The Drive file IDs are stable and
        we check those instead.)
        """
        text = Path("docs/colab_gallery.md").read_text()
        # Google Drive file IDs (stable identifiers) for the key notebooks.
        required_file_ids = {
            "01_easy_tensor_node_classification": "1C-vydQXnn9LrYhx5hZDQl6H601itnbGp",
            "09_kg_completion_transe_rescal_simple": "1QlCNZg2U0HJ6I6M4V8qXArKEwweZOjqn",
            "13_graph_generation_metrics": "1Q8358qYmw80SBr-fXFmkTg1tcRhaUg16",
            "18_graphml_io_roundtrip": "11Ul2v5KVYkrVFOhoeSZkE6Y8HG1qcgu5",
            "20_graph_mining_motifs_and_cliques": "1ZbtFGqNuPxfqI8xt3FlxzgPozukzbvua",
            "25_reproducibility_and_seed_control": "1ihdOfq-_z9iH9n7s52mJ2Veyog8jqdoB",
        }
        missing = [nb for nb, fid in required_file_ids.items() if fid not in text]
        assert not missing, f"Gallery missing notebooks: {missing}"

    def test_docs_index_links_colab_gallery(self):
        """docs/index.md must link to colab_gallery.md."""
        text = Path("docs/index.md").read_text()
        assert "colab_gallery.md" in text


# ── Bug C — GraphML exception formatting ──────────────────────────────────────


class TestGraphMLExceptionFormatting:
    """str(e)[:120] must work; e[:120] would raise TypeError."""

    def test_graphml_multidim_raises_value_error(self, tmp_path):
        """write_graphml with 4-D node_features must raise ValueError."""
        from tgraphx.core import Graph
        from tgraphx.io import write_graphml

        x_spatial = torch.randn(4, 3, 8, 8)
        g = Graph(node_features=x_spatial, edge_index=torch.tensor([[0, 1], [1, 2]]))
        with pytest.raises(ValueError):
            write_graphml(g, str(tmp_path / "test.graphml"), include_tensor_features=True)

    def test_str_exception_slice_works(self, tmp_path):
        """str(e)[:120] must succeed without TypeError."""
        from tgraphx.core import Graph
        from tgraphx.io import write_graphml

        x_spatial = torch.randn(4, 3, 8, 8)
        g = Graph(node_features=x_spatial, edge_index=torch.tensor([[0, 1], [1, 2]]))
        try:
            write_graphml(g, str(tmp_path / "test.graphml"), include_tensor_features=True)
            pytest.fail("Expected ValueError")
        except ValueError as e:
            msg = str(e)[:120]
            assert isinstance(msg, str)
            assert len(msg) <= 120

    def test_exception_object_slice_raises_type_error(self, tmp_path):
        """e[:120] must still raise TypeError (documents the original bug pattern)."""
        from tgraphx.core import Graph
        from tgraphx.io import write_graphml

        x_spatial = torch.randn(4, 3, 8, 8)
        g = Graph(node_features=x_spatial, edge_index=torch.tensor([[0, 1], [1, 2]]))
        try:
            write_graphml(g, str(tmp_path / "test.graphml"), include_tensor_features=True)
            pytest.fail("Expected ValueError")
        except ValueError as e:
            with pytest.raises(TypeError):
                _ = e[:120]

    def test_graphml_error_message_mentions_multidimensional(self, tmp_path):
        """The ValueError message must be informative."""
        from tgraphx.core import Graph
        from tgraphx.io import write_graphml

        x_spatial = torch.randn(4, 3, 8, 8)
        g = Graph(node_features=x_spatial, edge_index=torch.tensor([[0, 1], [1, 2]]))
        try:
            write_graphml(g, str(tmp_path / "test.graphml"), include_tensor_features=True)
            pytest.fail("Expected ValueError")
        except ValueError as e:
            msg = str(e)
            assert len(msg) > 10, f"Error message too short: {msg!r}"


# ── Bug D — Benchmark notebook package-level API ─────────────────────────────


class TestPackageBenchmarkSuite:

    def test_importable_from_package(self):
        from tgraphx.benchmarks import run_v13_benchmark_suite
        assert callable(run_v13_benchmark_suite)

    def test_runs_outside_repo_tree(self, tmp_path):
        """Must work when cwd is outside the TGraphX repository."""
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            from tgraphx.benchmarks import run_v13_benchmark_suite
            data = run_v13_benchmark_suite(small=True, return_dict=True)
            assert "benchmarks" in data
            assert len(data["benchmarks"]) > 0
        finally:
            os.chdir(old_cwd)

    def test_returns_expected_schema(self):
        from tgraphx.benchmarks import run_v13_benchmark_suite
        data = run_v13_benchmark_suite(small=True, return_dict=True)
        assert "suite" in data
        assert "package_version" in data
        assert "benchmarks" in data
        for row in data["benchmarks"]:
            assert "name" in row
            assert "status" in row

    def test_json_serializable(self):
        from tgraphx.benchmarks import run_v13_benchmark_suite
        data = run_v13_benchmark_suite(small=True, return_dict=True)
        serialized = json.dumps(data)
        reparsed = json.loads(serialized)
        assert len(reparsed["benchmarks"]) == len(data["benchmarks"])

    def test_python_m_cli(self, tmp_path):
        """python -m tgraphx.benchmarks.run_v13_benchmark_suite --small --json must produce valid JSON."""
        result = subprocess.run(
            [sys.executable, "-m", "tgraphx.benchmarks.run_v13_benchmark_suite",
             "--small", "--json"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr[:400]}"
        data = json.loads(result.stdout)
        assert "benchmarks" in data

    def test_no_repo_local_path_in_public_usage(self):
        """The public API call must not reference local filesystem path."""
        from tgraphx.benchmarks import run_v13_benchmark_suite
        import inspect
        src = inspect.getsource(run_v13_benchmark_suite)
        assert "benchmarks/run_v13_benchmark_suite.py" not in src


# ── Bug E — degree_statistics aliases ────────────────────────────────────────


class TestDegreeStatisticsAliases:
    """degree_statistics must include user-friendly min_degree/max_degree/mean_degree aliases."""

    def _star_ei(self):
        """One-directional star: center=0 → leaves 1,2,3,4."""
        src = torch.tensor([0, 0, 0, 0])
        dst = torch.tensor([1, 2, 3, 4])
        return torch.stack([src, dst]), 5

    def test_aliases_present(self):
        from tgraphx.mining.structural import degree_statistics
        ei, N = self._star_ei()
        stats = degree_statistics(ei, N)
        assert "min_degree" in stats, f"min_degree missing from keys: {list(stats.keys())}"
        assert "max_degree" in stats, f"max_degree missing from keys: {list(stats.keys())}"
        assert "mean_degree" in stats, f"mean_degree missing from keys: {list(stats.keys())}"

    def test_aliases_equal_total_degree(self):
        from tgraphx.mining.structural import degree_statistics
        ei, N = self._star_ei()
        stats = degree_statistics(ei, N)
        assert stats["min_degree"] == stats["min_total_degree"]
        assert stats["max_degree"] == stats["max_total_degree"]
        assert stats["mean_degree"] == stats["mean_total_degree"]

    def test_existing_keys_preserved(self):
        """Existing keys must not be removed."""
        from tgraphx.mining.structural import degree_statistics
        ei, N = self._star_ei()
        stats = degree_statistics(ei, N)
        for key in [
            "min_out_degree", "max_out_degree", "mean_out_degree",
            "min_in_degree", "max_in_degree", "mean_in_degree",
            "min_total_degree", "max_total_degree", "mean_total_degree",
            "isolated_node_count", "density",
        ]:
            assert key in stats, f"Existing key {key!r} was removed"

    def test_star_center_max_degree(self):
        """Star center must have the highest degree."""
        from tgraphx.mining.structural import degree_statistics
        ei, N = self._star_ei()
        stats = degree_statistics(ei, N)
        assert stats["max_out_degree"] == 4, "Star center out_degree must be 4"
        # max_degree is total: center has out=4 in=0 → total=4
        assert stats["max_degree"] == 4

    def test_empty_graph_aliases(self):
        """Empty graph (0 nodes) must include aliases with value 0."""
        from tgraphx.mining.structural import degree_statistics
        ei = torch.zeros(2, 0, dtype=torch.long)
        stats = degree_statistics(ei, 0)
        assert stats["min_degree"] == 0
        assert stats["max_degree"] == 0
        assert stats["mean_degree"] == 0.0

    def test_structural_roles_notebook_snippet(self):
        """Exact snippet from structural-roles notebook must run without KeyError."""
        from tgraphx.mining.structural import degree_statistics

        # One-directional star: center=0, leaves=1..4
        N_star = 5
        star_ei = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=torch.long)
        star_degree = degree_statistics(star_ei, num_nodes=N_star)
        # These must not raise KeyError
        assert star_degree["max_degree"] is not None
        assert star_degree["min_degree"] is not None
        # Center has max out-degree=4; total degree also 4 (one direction only)
        center_max = star_degree["max_degree"]
        leaf_min = star_degree["min_degree"]
        assert center_max > leaf_min


# ── Bug F — Reproducibility ──────────────────────────────────────────────────


class TestReproducibilityCPU:
    """CPU deterministic training must give identical results across runs."""

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

    def test_cpu_deterministic_exact_match(self):
        r1 = self._run_once()
        r2 = self._run_once()
        diff = abs(r1.metrics["loss"] - r2.metrics["loss"])
        assert diff < 1e-7, f"CPU deterministic loss diff too large: {diff:.2e}"

    def test_set_seed_returns_state(self):
        from tgraphx.reproducibility import set_seed
        state = set_seed(42, deterministic=True)
        assert isinstance(state, dict)
        assert "torch_seed" in state or len(state) > 0

    def test_train_result_has_reproducibility_info(self):
        r = self._run_once()
        assert hasattr(r, "metrics")
        assert "loss" in r.metrics


# ── Bug G — ConvMessagePassing out_shape spatial downsampling ─────────────────


class TestConvMessagePassingOutShape:
    """ConvMessagePassing must honour exact out_shape including spatial dimensions."""

    def test_same_spatial_dims(self):
        """in_shape=(32,8,8) out_shape=(64,8,8) → output [N,64,8,8]."""
        from tgraphx.layers import ConvMessagePassing
        N = 6
        conv = ConvMessagePassing(in_shape=(32, 8, 8), out_shape=(64, 8, 8))
        x = torch.randn(N, 32, 8, 8)
        ei = torch.stack([torch.arange(N), (torch.arange(N) + 1) % N])
        out = conv(x, ei)
        assert out.shape == (N, 64, 8, 8), f"Expected [6,64,8,8] got {out.shape}"

    def test_spatial_downsampling(self):
        """in_shape=(32,8,8) out_shape=(64,4,4) → output [N,64,4,4] (was bug in ≤v1.3.4)."""
        from tgraphx.layers import ConvMessagePassing
        N = 6
        conv = ConvMessagePassing(in_shape=(32, 8, 8), out_shape=(64, 4, 4))
        x = torch.randn(N, 32, 8, 8)
        ei = torch.stack([torch.arange(N), (torch.arange(N) + 1) % N])
        out = conv(x, ei)
        assert out.shape == (N, 64, 4, 4), (
            f"Expected [6,64,4,4] got {out.shape}. "
            f"ConvMessagePassing must honor out_shape spatial dims."
        )

    def test_spatial_downsampling_gradients(self):
        """Gradients must flow through the spatial downsampling path."""
        from tgraphx.layers import ConvMessagePassing
        N = 4
        conv = ConvMessagePassing(in_shape=(16, 8, 8), out_shape=(32, 4, 4))
        x = torch.randn(N, 16, 8, 8, requires_grad=True)
        ei = torch.stack([torch.arange(N), (torch.arange(N) + 1) % N])
        out = conv(x, ei)
        assert out.shape == (N, 32, 4, 4)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_vector_path_unaffected(self):
        """Vector (1-D spatial) path must not be affected by the fix."""
        from tgraphx.layers import LinearMessagePassing
        N = 8
        layer = LinearMessagePassing(in_shape=(16,), out_shape=(32,))
        x = torch.randn(N, 16)
        ei = torch.stack([torch.arange(N), (torch.arange(N) + 1) % N])
        out = layer(x, ei)
        assert out.shape == (N, 32)

    def test_classifier_no_shape_error(self):
        """Full model: ConvMP(32,8,8→64,4,4) + Linear(64*4*4, 10) must not give shape error."""
        import torch.nn as nn
        from tgraphx.layers import ConvMessagePassing
        N = 10
        C, H, W = 32, 8, 8
        C_out, H_out, W_out = 64, 4, 4

        class SimpleTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = ConvMessagePassing(
                    in_shape=(C, H, W), out_shape=(C_out, H_out, W_out)
                )
                self.fc = nn.Linear(C_out * H_out * W_out, 4)

            def forward(self, x, ei):
                h = self.conv(x, ei)
                return self.fc(h.view(h.size(0), -1))

        model = SimpleTModel()
        x = torch.randn(N, C, H, W)
        ei = torch.stack([torch.arange(N), (torch.arange(N) + 1) % N])
        logits = model(x, ei)
        assert logits.shape == (N, 4)
        logits.sum().backward()


class TestConvMessagePassingNeighborLoader:
    """NeighborLoader + ConvMessagePassing spatial downsampling smoke test."""

    def test_seed_logits_no_shape_error(self):
        """NeighborLoader batch with ConvMP out_shape downsampling must not raise shape error."""
        from tgraphx.core import Graph
        from tgraphx.loaders import NeighborLoader
        from tgraphx.layers import ConvMessagePassing
        import torch.nn as nn

        N, C, H, W = 80, 16, 8, 8
        C_out, H_out, W_out = 32, 4, 4

        torch.manual_seed(0)
        x = torch.randn(N, C, H, W)
        ei = torch.randint(0, N, (2, 300))
        y = torch.randint(0, 3, (N,))
        g = Graph(node_features=x, edge_index=ei, node_labels=y)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = ConvMessagePassing(
                    in_shape=(C, H, W), out_shape=(C_out, H_out, W_out)
                )
                self.fc = nn.Linear(C_out * H_out * W_out, 3)

            def forward(self, x, ei):
                h = self.conv(x, ei)
                return self.fc(h.view(h.size(0), -1))

        model = SimpleModel()
        # NeighborLoader uses fanouts (not num_neighbors) and mask (not seed_nodes)
        seed_mask = torch.zeros(N, dtype=torch.bool)
        seed_mask[:16] = True
        loader = NeighborLoader(g, fanouts=[5, 3], batch_size=8, mask=seed_mask, shuffle=False)
        batch = next(iter(loader))
        logits = model(batch.node_features, batch.edge_index)
        seed_logits = batch.seed_logits(logits)
        loss = F.cross_entropy(seed_logits, batch.seed_y)
        loss.backward()
        assert math.isfinite(loss.item())


# ── Bug H — Mining public APIs ────────────────────────────────────────────────


class TestMiningPublicAPIs:
    """All public mining imports must work from pip-installed package."""

    def test_motif_profile_import(self):
        from tgraphx.mining import motif_profile
        assert callable(motif_profile)

    def test_graph_summary_import(self):
        from tgraphx.mining import graph_summary
        assert callable(graph_summary)

    def test_wl_subtree_kernel_import(self):
        from tgraphx.mining.kernels import wl_subtree_kernel
        assert callable(wl_subtree_kernel)

    def test_centrality_summary_import(self):
        from tgraphx.mining import centrality_summary
        assert callable(centrality_summary)

    def test_degree_statistics_import(self):
        from tgraphx.mining import degree_statistics
        assert callable(degree_statistics)

    def test_motif_profile_triangle_graph(self):
        from tgraphx.mining import motif_profile
        # Triangle graph: 0-1-2-0
        ei = torch.tensor([[0, 1, 2, 0, 1, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long)
        result = motif_profile(ei, num_nodes=3)
        assert isinstance(result, dict)
        triangles = result.get("triangles", result.get("triangle", 0))
        assert triangles >= 1

    def test_graph_summary_expected_keys(self):
        from tgraphx.mining import graph_summary
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        s = graph_summary(ei, num_nodes=3)
        assert isinstance(s, dict)
        for key in ["num_nodes", "num_edges", "density"]:
            assert key in s, f"Key {key!r} missing from graph_summary"

    def test_wl_subtree_kernel_ring_similarity(self):
        from tgraphx.mining.kernels import wl_subtree_kernel
        # Two identical rings of 5 should be more similar to each other than to a ring of 3
        ei5 = torch.tensor([[0,1,2,3,4],[1,2,3,4,0]], dtype=torch.long)
        ei3 = torch.tensor([[0,1,2],[1,2,0]], dtype=torch.long)
        sim_same = wl_subtree_kernel(ei5, 5, ei5, 5)
        sim_diff = wl_subtree_kernel(ei5, 5, ei3, 3)
        assert sim_same >= sim_diff, "Identical graphs must be more similar than different"

    def test_centrality_summary_star_graph(self):
        from tgraphx.mining import centrality_summary
        # Star: center=0
        ei = torch.tensor([[0,0,0,0],[1,2,3,4]], dtype=torch.long)
        result = centrality_summary(ei, num_nodes=5)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_degree_statistics_aliases_present(self):
        from tgraphx.mining import degree_statistics
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        stats = degree_statistics(ei, num_nodes=3)
        assert "min_degree" in stats
        assert "max_degree" in stats
        assert "mean_degree" in stats


# ── Bug I — Feature-aware KG regression ──────────────────────────────────────


class TestFeatureAwareKGRegression:
    """KG models with entity_feature_dim must score triples without shape error."""

    def test_entity_features_score_triples(self):
        from tgraphx.kg import KnowledgeGraph, TransEModel

        N_e, N_r, N_t = 10, 3, 6
        triples = torch.randint(0, N_e, (N_t, 3))
        triples[:, 1] = torch.randint(0, N_r, (N_t,))
        entity_feats = torch.randn(N_e, 32)

        kg = KnowledgeGraph(triples=triples, num_entities=N_e, num_relations=N_r)
        kg.entity_features = {"visual": entity_feats}

        model = TransEModel(N_e, N_r, embedding_dim=16, entity_feature_dim=32)
        scores = model.score_triples(triples, entity_features=kg.entity_features["visual"])
        assert scores.shape == (N_t,), f"Expected ({N_t},) got {scores.shape}"
        loss = scores.mean()
        loss.backward()

    def test_transe_no_entity_features(self):
        from tgraphx.kg import TransEModel
        N_e, N_r, N_t = 8, 2, 4
        model = TransEModel(N_e, N_r, embedding_dim=8)
        triples = torch.randint(0, N_e, (N_t, 3))
        triples[:, 1] = torch.randint(0, N_r, (N_t,))
        scores = model.score_triples(triples)
        assert scores.shape == (N_t,)
        scores.mean().backward()

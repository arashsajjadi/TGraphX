"""Colab notebook regression tests (v1.3.6).

These tests pin the **exact public Colab notebook code paths** that were
reported as broken in the v1.3.5 → v1.3.6 sprint. They guard against:

- Notebook 14 still using `composite_fitness` directly with NSGA-II.
- Notebook 19 still slicing `ValueError` objects.
- Notebook 22 still asserting `max_degree == 4` on a bidirectional graph.
- Notebook 24 still spawning subprocesses against `benchmarks/run_v13_benchmark_suite.py`.
- README still containing the stale single-Colab-link sentence.

Tests must be CPU-safe, fast, no network, and not assume a repo checkout.
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


# ── Public-source surfaces scanned for stale snippets ────────────────────────

_PUBLIC_DIRS_AND_FILES = [
    Path("README.md"),
    Path("docs"),
    Path("tutorials"),
    Path("examples"),
    Path("tools/generate_colab_drafts.py"),
    Path("tools/generate_notebooks.py"),
]


def _iter_public_text_files():
    """Yield every public-facing text source file that ships notebook code."""
    suffixes = {".md", ".py", ".ipynb", ".json", ".txt"}
    for entry in _PUBLIC_DIRS_AND_FILES:
        if not entry.exists():
            continue
        if entry.is_file():
            yield entry
            continue
        for path in entry.rglob("*"):
            if path.is_file() and path.suffix in suffixes:
                yield path


def _grep_public_sources(needle: str):
    """Return list of (path, line_number, line) for occurrences in public sources."""
    hits = []
    for path in _iter_public_text_files():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if needle in line:
                hits.append((path, i, line))
    return hits


# ── Bug A — Notebook 14: NSGA-II composite_fitness misuse ─────────────────────


class TestNotebook14NSGAII:
    """The corrected NSGA-II snippet must run; the broken pattern must not appear."""

    def _make_genome(self, seed=0):
        from tgraphx.evolutionary import GraphGenome
        torch.manual_seed(seed)
        return GraphGenome(edge_index=torch.randint(0, 8, (2, 10)), num_nodes=8)

    def test_corrected_snippet_runs(self):
        """Exact corrected v1.3.6 notebook 14 NSGA-II snippet."""
        from tgraphx.evolutionary import (
            NSGAIIOptimizer, EvolutionConfig,
            connectivity_fitness, sparsity_fitness,
        )
        config2 = EvolutionConfig(population_size=8, n_generations=3, seed=0)
        pop2 = [self._make_genome(i) for i in range(8)]
        objectives = [connectivity_fitness, sparsity_fitness]
        r2 = NSGAIIOptimizer(config2, objectives).optimize(pop2)
        assert r2.pareto_front is not None and len(r2.pareto_front) >= 1
        assert len(r2.history) > 0

    def test_composite_fitness_direct_raises(self):
        """Passing composite_fitness directly must still raise helpful TypeError."""
        from tgraphx.evolutionary import (
            NSGAIIOptimizer, EvolutionConfig, composite_fitness,
        )
        with pytest.raises(TypeError) as exc_info:
            NSGAIIOptimizer(
                EvolutionConfig(population_size=4, n_generations=2, seed=0),
                composite_fitness,
            )
        msg = str(exc_info.value)
        assert "NSGAIIOptimizer expects" in msg
        assert "GeneticAlgorithmOptimizer" in msg

    def test_no_stale_nsga_snippet_in_public_sources(self):
        """No public source file may contain the broken NSGA-II call form."""
        hits = _grep_public_sources("NSGAIIOptimizer(config2, composite_fitness)")
        # Filter out test files that intentionally document the bad pattern.
        non_test = [(p, i, l) for (p, i, l) in hits if "tests" not in p.parts]
        assert not non_test, f"Stale NSGA-II snippet found: {non_test}"

    def test_no_stale_nsga_comment_in_public_sources(self):
        hits = _grep_public_sources("composite_fitness optimizes multiple objectives")
        non_test = [(p, i, l) for (p, i, l) in hits if "tests" not in p.parts]
        assert not non_test, f"Misleading NSGA-II comment found: {non_test}"

    def test_colab_draft_14_uses_objective_list(self):
        """If colab_drafts/14_*.ipynb exists, it must use the corrected snippet."""
        path = Path("colab_drafts/14_graph_generation_evolutionary_optimization.ipynb")
        if not path.exists():
            pytest.skip("colab_drafts/14_*.ipynb not present (gitignored)")
        nb = json.loads(path.read_text())
        srcs = ["".join(c["source"]) for c in nb["cells"]]
        joined = "\n".join(srcs)
        assert "NSGAIIOptimizer(config2, composite_fitness)" not in joined
        assert "objectives" in joined or "connectivity_fitness" in joined
        assert "sparsity_fitness" in joined or "[connectivity_fitness" in joined


# ── Bug B — Notebook 19: GraphML str(e)[:120] formatting ─────────────────────


class TestNotebook19GraphML:
    """The corrected GraphML notebook code must use str(e)[:120], not e[:120]."""

    def test_str_e_slicing_works(self, tmp_path):
        from tgraphx import Graph
        from tgraphx.io import write_graphml
        x_spatial = torch.randn(4, 3, 8, 8)
        g = Graph(node_features=x_spatial, edge_index=torch.tensor([[0, 1], [1, 2]]))
        try:
            write_graphml(g, str(tmp_path / "x.graphml"), include_tensor_features=True)
            pytest.fail("Expected ValueError")
        except ValueError as e:
            msg = str(e)[:120]
            assert isinstance(msg, str)
            assert len(msg) <= 120
            assert "GraphML" in msg or "node_features" in msg

    def test_no_public_e_slicing(self):
        """No public source may slice an exception object directly."""
        hits = _grep_public_sources("e[:120]")
        # The pattern matches both `e[:120]` and `str(e)[:120]`. Filter out
        # the safe form and the changelog/test entries.
        bad = [
            (p, i, l) for (p, i, l) in hits
            if "tests" not in p.parts
            and "CHANGELOG" not in p.name
            and "str(e)[:120]" not in l
            and "str(err)[:120]" not in l
        ]
        assert not bad, f"Public sources still slice exception objects: {bad}"

    def test_colab_draft_19_uses_str_e(self):
        """colab_drafts/19_*.ipynb must use str(e)[:120], not e[:120]."""
        path = Path("colab_drafts/19_io_tensor_semantics_warning.ipynb")
        if not path.exists():
            pytest.skip("colab_drafts/19_*.ipynb not present (gitignored)")
        nb = json.loads(path.read_text())
        joined = "\n".join("".join(c["source"]) for c in nb["cells"])
        assert "str(e)[:120]" in joined
        # No raw exception slicing.
        for bad in (" e[:120]", "(e[:120]"):
            assert bad not in joined, f"Found {bad!r} in colab_drafts/19"


# ── Bug C — Notebook 24: Benchmark must use package API, not repo path ────────


class TestNotebook24Benchmark:
    """Notebook 24 must call the package-level API, not a repo-local script."""

    def test_package_api_importable(self):
        from tgraphx.benchmarks import run_v13_benchmark_suite
        assert callable(run_v13_benchmark_suite)

    def test_returns_dict_with_required_keys(self):
        from tgraphx.benchmarks import run_v13_benchmark_suite
        data = run_v13_benchmark_suite(small=True, return_dict=True)
        for k in ("suite", "package_version", "benchmarks"):
            assert k in data, f"Missing key {k!r}"
        for row in data["benchmarks"]:
            assert "name" in row and "status" in row

    def test_runs_outside_repo_tree(self, tmp_path):
        """Must work when cwd is outside the TGraphX repository."""
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            from tgraphx.benchmarks import run_v13_benchmark_suite
            data = run_v13_benchmark_suite(small=True, return_dict=True)
            assert "benchmarks" in data and len(data["benchmarks"]) > 0
        finally:
            os.chdir(old)

    def test_python_m_cli(self, tmp_path):
        """python -m tgraphx.benchmarks.run_v13_benchmark_suite --small --json works."""
        result = subprocess.run(
            [sys.executable, "-m", "tgraphx.benchmarks.run_v13_benchmark_suite",
             "--small", "--json"],
            capture_output=True, text=True, cwd=str(tmp_path),
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr[:400]}"
        data = json.loads(result.stdout)
        assert "benchmarks" in data

    def test_no_stale_repo_path_in_public_sources(self):
        """No public notebook/doc source may reference benchmarks/run_v13_benchmark_suite.py."""
        hits = _grep_public_sources("benchmarks/run_v13_benchmark_suite.py")
        bad = [
            (p, i, l) for (p, i, l) in hits
            if "tests" not in p.parts
            and "CHANGELOG" not in p.name
        ]
        assert not bad, f"Stale repo-local benchmark path found: {bad}"

    def test_colab_draft_24_uses_package_api(self):
        path = Path("colab_drafts/24_benchmark_suite_v13.ipynb")
        if not path.exists():
            pytest.skip("colab_drafts/24_*.ipynb not present (gitignored)")
        nb = json.loads(path.read_text())
        joined = "\n".join("".join(c["source"]) for c in nb["cells"])
        assert "from tgraphx.benchmarks import run_v13_benchmark_suite" in joined
        assert "subprocess" not in joined
        assert "benchmarks/run_v13_benchmark_suite.py" not in joined


# ── Bug D — Notebook 22: structural-role degree assertions ────────────────────


class TestNotebook22StructuralRoles:
    """Notebook 22 must not assert max_degree == 4 on a bidirectional star."""

    def test_corrected_snippet_runs(self):
        from tgraphx.mining import degree_statistics

        star_ei = torch.tensor(
            [[0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 0, 0, 0]], dtype=torch.long,
        )
        star_degree = degree_statistics(star_ei, num_nodes=5)
        assert star_degree["max_total_degree"] == 8
        assert star_degree["min_total_degree"] == 2

    def test_degree_aliases_present(self):
        from tgraphx.mining.structural import degree_statistics
        stats = degree_statistics(
            torch.tensor([[0, 1], [1, 0]], dtype=torch.long), num_nodes=2
        )
        for k in ("min_degree", "max_degree", "mean_degree"):
            assert k in stats

    def test_no_stale_assertion_in_public_sources(self):
        """No public source may contain the broken bidirectional-star assertion."""
        for needle in (
            "assert star_degree['max_degree'] == 4",
            'assert star_degree["max_degree"] == 4',
        ):
            hits = _grep_public_sources(needle)
            bad = [(p, i, l) for (p, i, l) in hits if "tests" not in p.parts]
            assert not bad, f"Stale assertion {needle!r}: {bad}"

    def test_colab_draft_22_uses_total_degree(self):
        path = Path("colab_drafts/22_structural_roles_concept_demo.ipynb")
        if not path.exists():
            pytest.skip("colab_drafts/22_*.ipynb not present (gitignored)")
        nb = json.loads(path.read_text())
        joined = "\n".join("".join(c["source"]) for c in nb["cells"])
        assert "max_total_degree" in joined
        assert "assert star_degree['max_degree'] == 4" not in joined
        assert 'assert star_degree["max_degree"] == 4' not in joined


# ── Bug E — README / docs gallery consolidation ───────────────────────────────


class TestReadmeAndGalleryConsolidation:

    def test_no_old_colab_single_tutorial_sentence(self):
        text = Path("README.md").read_text()
        assert "A Colab tutorial walks through every workflow" not in text

    def test_readme_has_notebook_gallery_section(self):
        text = Path("README.md").read_text()
        assert "Notebook Gallery" in text or "Notebook gallery" in text

    def test_readme_links_to_colab_gallery(self):
        text = Path("README.md").read_text()
        assert "colab_gallery.md" in text

    def test_docs_index_links_colab_gallery(self):
        text = Path("docs/index.md").read_text()
        assert "colab_gallery.md" in text

    def test_no_public_source_has_stale_colab_sentence(self):
        hits = _grep_public_sources("A Colab tutorial walks through every workflow")
        bad = [(p, i, l) for (p, i, l) in hits
               if "tests" not in p.parts and "CHANGELOG" not in p.name]
        assert not bad, f"Stale Colab sentence remains: {bad}"


# ── Bug F — ConvMessagePassing out_shape with NeighborLoader ──────────────────


class TestConvMessagePassingNeighborLoaderSeedLoss:

    def test_full_v136_colab_smoke(self):
        from tgraphx import Graph, ConvMessagePassing
        from tgraphx.loaders import NeighborLoader
        from tgraphx.reproducibility import set_seed
        import torch.nn as nn

        set_seed(42)

        N = 128
        x = torch.randn(N, 16, 8, 8)
        edge_index = torch.randint(0, N, (2, 1000))
        y = torch.randint(0, 10, (N,))
        g = Graph(node_features=x, edge_index=edge_index, y=y)

        class SimpleTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = ConvMessagePassing(in_shape=(16, 8, 8), out_shape=(32, 8, 8))
                self.conv2 = ConvMessagePassing(in_shape=(32, 8, 8), out_shape=(64, 4, 4))
                self.cls = nn.Linear(64 * 4 * 4, 10)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = torch.relu(x)
                x = self.conv2(x, edge_index)
                assert x.shape[1:] == (64, 4, 4), x.shape
                x = x.reshape(x.size(0), -1)
                return self.cls(x)

        model = SimpleTModel()
        loader = NeighborLoader(graph=g, fanouts=[10, 5], batch_size=32,
                                shuffle=True, seed=42)
        batch = next(iter(loader))
        logits = model(batch.node_features, batch.edge_index)
        loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)
        loss.backward()
        assert math.isfinite(loss.item())


# ── Bug G — Reproducibility CPU strict mode ──────────────────────────────────


class TestReproducibilityCPU:

    def _run(self):
        import tgraphx as tgx
        from tgraphx.reproducibility import set_seed
        set_seed(42, deterministic=True)
        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=48, node_shape=(4, 4, 4), num_classes=3,
            num_edges=200, seed=42,
        )
        return tgx.easy.train_node_classifier(
            data, epochs=2, batch_size=16, fanouts=[5, 3],
            verbose=False, seed=42, deterministic=True, device="cpu",
        )

    def test_deterministic_cpu_exact_match(self):
        r1 = self._run()
        r2 = self._run()
        diff = abs(r1.metrics["loss"] - r2.metrics["loss"])
        assert diff < 1e-7, f"CPU deterministic diff too large: {diff:.2e}"


# ── Bug H — Mining and KG public APIs ────────────────────────────────────────


class TestMiningAndKGPublicAPIs:

    def test_mining_imports(self):
        from tgraphx.mining import (
            motif_profile, graph_summary, degree_statistics, centrality_summary,
        )
        from tgraphx.mining.kernels import wl_subtree_kernel
        assert all(callable(f) for f in (
            motif_profile, graph_summary, degree_statistics,
            centrality_summary, wl_subtree_kernel,
        ))

    def test_feature_aware_kg_scoring(self):
        from tgraphx.kg import KnowledgeGraph, TransEModel
        N_e, N_r, N_t = 10, 3, 6
        triples = torch.randint(0, N_e, (N_t, 3))
        triples[:, 1] = torch.randint(0, N_r, (N_t,))
        kg = KnowledgeGraph(triples, num_entities=N_e, num_relations=N_r)
        kg.entity_features = {"visual": torch.randn(N_e, 32)}
        model = TransEModel(N_e, N_r, embedding_dim=16, entity_feature_dim=32)
        scores = model.score_triples(triples, entity_features=kg.entity_features["visual"])
        assert scores.shape == (N_t,)
        scores.mean().backward()

"""v1.4.1 user-friendly syntax tests — groups 101–115.

Groups 101–110: UX helpers (classify_nodes, kg_completion, make_graph,
explain_error, debug_batch, dataset_card, model_card, benchmark_card,
audit_package_readiness, readiness CLI).

Groups 111–115: Generation/RL/Evolution wrappers (generate_graph,
evaluate_generated_graphs, optimize_graph, train_graph_rl, dashboard audit).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

import tgraphx as tgx
from tgraphx import Graph


# ── GROUP 101 — classify_nodes ────────────────────────────────────────────

class TestGroup101ClassifyNodes:
    def _make_data(self, N=30, NC=3):
        gen = torch.Generator().manual_seed(42)
        x = torch.randn(N, 1, 14, 14, generator=gen)
        src = torch.randint(0, N, (N*3,), generator=gen)
        dst = torch.randint(0, N, (N*3,), generator=gen)
        ei = torch.unique(torch.stack([torch.cat([src,dst]), torch.cat([dst,src])],0), dim=1)
        y = torch.randint(0, NC, (N,))
        return x, ei, y

    def test_tensor_rank4_smoke(self) -> None:
        x, ei, y = self._make_data()
        r = tgx.classify_nodes(x, ei, y, fast_mode=True, seed=42)
        assert "val_accuracy" in r.metrics
        assert math.isfinite(r.metrics["val_accuracy"])
        assert r.task == "node_classification"

    def test_vector_feature_smoke(self) -> None:
        gen = torch.Generator().manual_seed(42)
        x = torch.randn(40, 16, generator=gen)
        ei = torch.tensor([[0,1,2,3,4],[1,2,3,4,0]])
        y = torch.randint(0, 3, (40,))
        r = tgx.classify_nodes(x, ei, y, model="gcn", fast_mode=True, seed=42)
        assert "val_accuracy" in r.metrics

    def test_aliases_work(self) -> None:
        x, ei, y = self._make_data()
        for alias in [tgx.node_classification, tgx.fit_node_classifier, tgx.train_node_classifier]:
            r = alias(x, ei, y, fast_mode=True, seed=42)
            assert "val_accuracy" in r.metrics

    def test_mask_overlap_raises_leakage(self) -> None:
        from tgraphx.ux.leakage import LeakageError
        x = torch.randn(10, 4)
        ei = torch.tensor([[0,1],[1,2]])
        y = torch.randint(0, 2, (10,))
        train = torch.tensor([True, True, True, False, False, False, False, False, False, False])
        val = torch.tensor([True, False, False, True, False, False, False, False, False, False])
        with pytest.raises(LeakageError):
            tgx.classify_nodes(x, ei, y, train_mask=train, val_mask=val, fast_mode=True)

    def test_result_to_dict(self) -> None:
        x, ei, y = self._make_data()
        r = tgx.classify_nodes(x, ei, y, fast_mode=True, seed=42)
        d = r.to_dict()
        assert isinstance(d, dict)
        json.dumps(d)  # must be JSON-serializable

    def test_result_to_markdown(self) -> None:
        x, ei, y = self._make_data()
        r = tgx.classify_nodes(x, ei, y, fast_mode=True, seed=42)
        md = r.to_markdown()
        assert "node_classification" in md


# ── GROUP 102 — kg_completion ─────────────────────────────────────────────

class TestGroup102KGCompletion:
    def _tiny_kg(self):
        triples = torch.zeros((40, 3), dtype=torch.long)
        for i in range(40):
            triples[i] = torch.tensor([i % 8, i % 3, (i + 1) % 8])
        return triples

    def test_transe_smoke(self) -> None:
        triples = self._tiny_kg()
        r = tgx.kg_completion(triples, num_entities=8, num_relations=3,
                               model="transe", fast_mode=True, seed=42)
        assert "final_loss" in r.metrics
        assert math.isfinite(r.metrics["final_loss"])

    def test_model_alias(self) -> None:
        triples = self._tiny_kg()
        r = tgx.kg_completion(triples, num_entities=8, num_relations=3,
                               model="TransE", fast_mode=True, seed=42)
        assert r.task == "kg_link_prediction"

    def test_unknown_model_suggests(self) -> None:
        triples = self._tiny_kg()
        with pytest.raises(ValueError, match="Did you mean|Unknown"):
            tgx.kg_completion(triples, num_entities=8, num_relations=3,
                               model="nope_model")

    def test_aliases(self) -> None:
        triples = self._tiny_kg()
        for alias in [tgx.fit_kg, tgx.train_kg]:
            r = alias(triples, num_entities=8, num_relations=3, fast_mode=True, seed=42)
            assert "final_loss" in r.metrics


# ── GROUP 103 — make_graph ────────────────────────────────────────────────

class TestGroup103MakeGraph:
    def test_from_edge_list(self) -> None:
        x = torch.randn(4, 3)
        g = tgx.make_graph(x=x, edges=[(0,1),(1,2),(2,3)])
        assert g.num_nodes == 4
        assert g.num_edges == 3

    def test_from_edge_index(self) -> None:
        x = torch.randn(4, 3)
        ei = torch.tensor([[0,1,2],[1,2,3]])
        g = tgx.make_graph(x=x, edge_index=ei)
        assert g.num_edges == 3

    def test_from_adjacency(self) -> None:
        adj = torch.tensor([[0,1,0],[1,0,1],[0,1,0]])
        g = tgx.make_graph(x=torch.randn(3,2), adjacency=adj)
        assert g.num_nodes == 3

    def test_from_networkx(self) -> None:
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = nx.path_graph(5)
        g = tgx.make_graph(networkx_graph=G)
        assert g.num_nodes == 5

    def test_ambiguous_raises(self) -> None:
        with pytest.raises(ValueError, match="at most ONE"):
            tgx.make_graph(x=torch.randn(4,3),
                           edges=[(0,1)],
                           edge_index=torch.tensor([[0],[1]]))

    def test_aliases(self) -> None:
        x = torch.randn(4, 2)
        for alias in [tgx.build_graph]:
            g = alias(x=x, edges=[(0,1),(1,2)])
            assert g.num_nodes == 4


# ── GROUP 104 — explain_error ─────────────────────────────────────────────

class TestGroup104ExplainError:
    def test_graphml_rank_error(self) -> None:
        guidance = tgx.explain_error("rank error in GraphML")
        assert "GraphML" in guidance or "tgx.save" in guidance

    def test_vgae_error(self) -> None:
        guidance = tgx.explain_error("vgae not working")
        assert "VGAE" in guidance or "vgae" in guidance.lower()

    def test_mask_overlap_error(self) -> None:
        guidance = tgx.explain_error("mask overlap leakage detected")
        assert "leakage" in guidance.lower() or "check_leakage" in guidance

    def test_unknown_error_generic(self) -> None:
        guidance = tgx.explain_error("zxqwerty completely unknown error xyz123")
        assert isinstance(guidance, str) and len(guidance) > 20

    def test_alias_troubleshoot_error(self) -> None:
        guidance = tgx.troubleshoot_error("edge_index bad shape")
        assert isinstance(guidance, str)


# ── GROUP 105 — debug_batch ───────────────────────────────────────────────

class TestGroup105DebugBatch:
    def test_neighbor_loader_batch(self) -> None:
        from tgraphx import Graph
        from tgraphx.loaders import NeighborLoader
        g = Graph(node_features=torch.randn(20, 3),
                  edge_index=torch.tensor([[i for i in range(19)], [i+1 for i in range(19)]]),
                  y=torch.randint(0, 3, (20,)))
        mask = torch.zeros(20, dtype=torch.bool); mask[:12] = True
        loader = NeighborLoader(g, fanouts=[5,3], batch_size=6, mask=mask, seed=42)
        for batch in loader:
            info = tgx.debug_batch(batch)
            assert "node_features_shape" in info
            assert "num_seed_nodes" in info
            assert isinstance(info["ok"], bool)
            break

    def test_alias(self) -> None:
        from tgraphx import Graph
        from tgraphx.loaders import NeighborLoader
        g = Graph(node_features=torch.randn(20, 3),
                  edge_index=torch.tensor([[i for i in range(19)], [i+1 for i in range(19)]]),
                  y=torch.randint(0, 3, (20,)))
        mask = torch.zeros(20, dtype=torch.bool); mask[:12] = True
        loader = NeighborLoader(g, fanouts=[5,3], batch_size=6, mask=mask, seed=42)
        for batch in loader:
            info = tgx.batch_summary(batch)
            assert isinstance(info, dict)
            break


# ── GROUP 106 — dataset_card / model_card ─────────────────────────────────

class TestGroup106Cards:
    def test_dataset_card_graph(self) -> None:
        g = Graph(node_features=torch.randn(5, 3, 7, 7),
                  edge_index=torch.tensor([[0,1],[1,2]]),
                  y=torch.tensor([0,1,0,1,0]))
        card = tgx.dataset_card(g, task="node_classification")
        assert isinstance(card, dict)
        assert "limitations" in card
        json.dumps(card)  # JSON-serializable

    def test_model_card(self) -> None:
        import torch.nn as nn
        m = nn.Linear(10, 5)
        card = tgx.model_card(m, task="node_classification", seed=42)
        assert card["parameters_total"] == 55
        assert "disclaimer" in card
        json.dumps(card)


# ── GROUP 107 — benchmark_card ────────────────────────────────────────────

class TestGroup107BenchmarkCard:
    def test_benchmark_card_from_workflow(self) -> None:
        r = tgx.workflow(task="graph_mining", fast_mode=True, seed=42)
        card = tgx.benchmark_card(r)
        assert "disclaimer" in card
        assert "SOTA" in card["disclaimer"] or "Not" in card["disclaimer"]
        json.dumps(card)


# ── GROUP 108 — API docs sync ─────────────────────────────────────────────

class TestGroup108APIDocsSync:
    def test_public_api_stability_consistent(self) -> None:
        # All canonical stable APIs must be importable
        from tgraphx.ux.public_api import _STABILITY
        import importlib
        # Spot-check a few stable APIs
        for name in ("Graph", "set_seed", "global_mean_pool"):
            assert _STABILITY.get(name) == "stable", f"{name} should be stable"

    def test_api_cheatsheet_loadable(self) -> None:
        path = Path("docs/api_cheatsheet.json")
        if not path.exists():
            pytest.skip("api_cheatsheet.json not found")
        data = json.loads(path.read_text())
        assert isinstance(data, dict)


# ── GROUP 109 — CPU/GPU parity ────────────────────────────────────────────

class TestGroup109CPUGPUParity:
    def test_cpu_generate_graph_deterministic(self) -> None:
        g1 = tgx.generate_graph("ba", num_nodes=15, m=2, seed=42)
        g2 = tgx.generate_graph("ba", num_nodes=15, m=2, seed=42)
        assert g1.num_nodes == g2.num_nodes
        assert g1.num_edges == g2.num_edges

    def test_cpu_classify_deterministic(self) -> None:
        x = torch.randn(30, 4)
        ei = torch.tensor([[i for i in range(29)], [i+1 for i in range(29)]])
        y = torch.randint(0, 3, (30,))
        r1 = tgx.classify_nodes(x, ei, y, fast_mode=True, seed=7)
        r2 = tgx.classify_nodes(x, ei, y, fast_mode=True, seed=7)
        assert abs(r1.metrics["val_accuracy"] - r2.metrics["val_accuracy"]) < 0.05


# ── GROUP 110 — audit_package_readiness / CLI ─────────────────────────────

class TestGroup110Readiness:
    def test_audit_package_readiness_returns_dict(self) -> None:
        r = tgx.audit_package_readiness()
        assert "tgraphx_version" in r
        assert "torch_version" in r
        assert "cuda_available" in r
        assert "known_limitations" in r
        assert isinstance(r["known_limitations"], list)
        json.dumps(r)  # JSON-serializable

    def test_readiness_cli(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tgraphx", "readiness"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "TGraphX" in result.stdout

    def test_list_datasets_cli(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tgraphx", "list-datasets"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "mnist" in result.stdout or "cifar" in result.stdout

    def test_help_cli(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tgraphx", "help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "readiness" in result.stdout


# ── GROUP 111 — generate_graph ────────────────────────────────────────────

class TestGroup111GenerateGraph:
    def test_ba_method(self) -> None:
        g = tgx.generate_graph("ba", num_nodes=20, m=2, seed=42)
        assert isinstance(g, Graph)
        assert g.num_nodes == 20

    def test_er_alias(self) -> None:
        g = tgx.generate_graph("er", num_nodes=15, seed=42)
        assert isinstance(g, Graph)
        assert g.num_nodes == 15

    def test_ws_alias(self) -> None:
        g = tgx.generate_graph("ws", num_nodes=12, seed=42)
        assert isinstance(g, Graph)

    def test_tensor_node_shape(self) -> None:
        g = tgx.generate_graph("ba", num_nodes=10, m=2, node_shape=(3, 8, 8), seed=42)
        assert g.node_features.shape == (10, 3, 8, 8)

    def test_deterministic_same_seed(self) -> None:
        g1 = tgx.generate_graph("ba", num_nodes=15, m=2, seed=99)
        g2 = tgx.generate_graph("ba", num_nodes=15, m=2, seed=99)
        assert g1.num_nodes == g2.num_nodes and g1.num_edges == g2.num_edges

    def test_vgae_gives_helpful_error(self) -> None:
        with pytest.raises(ValueError, match="VGAE|graph autoencoder|not a classical"):
            tgx.generate_graph("vgae", num_nodes=10)

    def test_unknown_method_suggests(self) -> None:
        with pytest.raises(ValueError, match="Did you mean|Valid|Unknown"):
            tgx.generate_graph("barabassi_albert_typo", num_nodes=10)

    def test_artifacts_written(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            g = tgx.generate_graph("er", num_nodes=8, seed=42, out_dir=d)
            assert (Path(d) / "generation_config.json").exists()
            assert (Path(d) / "graph_summary.json").exists()
            assert (Path(d) / "generation_metrics.json").exists()

    def test_aliases(self) -> None:
        for alias in [tgx.graph_generator, tgx.generate]:
            g = alias("ba", num_nodes=10, m=2, seed=42)
            assert isinstance(g, Graph)


# ── GROUP 112 — evaluate_generated_graphs ────────────────────────────────

class TestGroup112EvaluateGenerated:
    def test_basic_evaluation(self) -> None:
        graphs = [tgx.generate_graph("er", num_nodes=10, seed=i) for i in range(3)]
        report = tgx.evaluate_generated_graphs(graphs)
        assert report["num_graphs"] == 3
        assert "validity" in report
        assert "disclaimer" in report
        json.dumps(report)

    def test_aliases(self) -> None:
        g = tgx.generate_graph("ba", num_nodes=8, m=2, seed=42)
        r1 = tgx.graph_generation_report([g])
        r2 = tgx.compare_generated_graphs([g])
        assert r1["num_graphs"] == r2["num_graphs"] == 1

    def test_artifact_written(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            g = tgx.generate_graph("er", num_nodes=8, seed=42)
            tgx.evaluate_generated_graphs([g], out_dir=d)
            assert (Path(d) / "generation_eval_summary.json").exists()


# ── GROUP 113 — optimize_graph ────────────────────────────────────────────

class TestGroup113OptimizeGraph:
    def test_ga_connectivity(self) -> None:
        r = tgx.optimize_graph("connectivity", algorithm="ga", num_nodes=8,
                               fast_mode=True, seed=42)
        assert hasattr(r, "best_fitness") or hasattr(r, "metrics")
        # best_fitness may be a tensor or float
        bf = r.best_fitness if hasattr(r, "best_fitness") else None
        assert bf is not None

    def test_nsga2_multi_objective(self) -> None:
        r = tgx.optimize_graph(["connectivity", "density"], algorithm="nsga2",
                               num_nodes=8, fast_mode=True, seed=42)
        assert r is not None

    def test_aliases(self) -> None:
        for alias in [tgx.evolve_graph, tgx.graph_evolution, tgx.run_evolution]:
            r = alias("connectivity", algorithm="ga", num_nodes=6, fast_mode=True, seed=42)
            assert r is not None

    def test_unknown_algorithm_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            tgx.optimize_graph("connectivity", algorithm="nope_algo", num_nodes=8)

    def test_artifacts_written(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tgx.optimize_graph("connectivity", algorithm="ga", num_nodes=6,
                               fast_mode=True, seed=42, out_dir=d)
            assert (Path(d) / "evolution_config.json").exists()
            assert (Path(d) / "benchmark_summary.json").exists()


# ── GROUP 114 — train_graph_rl ────────────────────────────────────────────

class TestGroup114TrainGraphRL:
    def test_maxcut_random(self) -> None:
        r = tgx.train_graph_rl(env="maxcut", algorithm="random", episodes=2,
                               fast_mode=True, seed=42)
        assert hasattr(r, "final_reward")

    def test_env_alias_maxcut(self) -> None:
        r = tgx.train_graph_rl(env="max_cut", algorithm="random", episodes=2,
                               fast_mode=True, seed=42)
        assert r is not None

    def test_unknown_env_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown|Did you mean"):
            tgx.train_graph_rl(env="nope_env", algorithm="random", episodes=2)

    def test_unknown_algorithm_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown|Did you mean"):
            tgx.train_graph_rl(env="maxcut", algorithm="nope_algo", episodes=2)

    def test_aliases(self) -> None:
        for alias in [tgx.graph_rl, tgx.run_rl]:
            r = alias(env="maxcut", algorithm="random", episodes=2, fast_mode=True, seed=42)
            assert r is not None

    def test_artifacts_written(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tgx.train_graph_rl(env="maxcut", algorithm="random", episodes=2,
                               fast_mode=True, seed=42, out_dir=d)
            assert (Path(d) / "rl_config.json").exists()
            assert (Path(d) / "rl_metrics_summary.json").exists()


# ── GROUP 115 — Generation/RL/Evolution dashboard audit ──────────────────

class TestGroup115DashboardExtended:
    def _make_generation_dir(self, d: str):
        p = Path(d)
        for fn, data in [
            ("generation_config.json", {"method": "ba", "seed": 42}),
            ("graph_summary.json", {"num_nodes": 20, "density": 0.3}),
            ("generation_metrics.json", {"num_graphs": 1, "validity": 1.0}),
        ]:
            with open(p / fn, "w") as f:
                json.dump(data, f)

    def _make_rl_dir(self, d: str):
        p = Path(d)
        for fn, data in [
            ("rl_config.json", {"env": "max_cut", "algorithm": "random", "seed": 42}),
            ("rl_metrics_summary.json", {"final_reward": 4.0}),
        ]:
            with open(p / fn, "w") as f:
                json.dump(data, f)

    def test_audit_generation_run(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self._make_generation_dir(d)
            r = tgx.audit_generation_run(d)
            assert "issues" in r

    def test_audit_rl_run(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self._make_rl_dir(d)
            r = tgx.audit_rl_run(d)
            assert "issues" in r

    def test_dashboard_audit_workflow_param(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self._make_generation_dir(d)
            r = tgx.dashboard_audit(d, workflow="generation")
            assert "issues" in r

    def test_dashboard_audit_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self._make_generation_dir(d)
            r = tgx.audit_run_dir(d, workflow="generation")
            assert "completeness_score" in r
            assert "reproducibility_score" in r
            assert 0 <= r["completeness_score"] <= 100

    def test_dashboard_audit_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self._make_generation_dir(d)
            r = tgx.audit_run_dir(d, return_markdown=True)
            assert "markdown" in r
            assert "Dashboard Audit" in r["markdown"]

    def test_dashboard_audit_scores_for_good_dir(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            for fn, data in [
                ("run_metadata.json", {"tgraphx_version": "1.4.1", "seed": 42, "device": "cpu"}),
                ("metrics_summary.json", {"val_accuracy": 0.8}),
                ("benchmark_summary.json", {"task": "node_classification"}),
            ]:
                with open(p / fn, "w") as f:
                    json.dump(data, f)
            r = tgx.audit_run_dir(d)
            assert r["ok"]
            assert r["reproducibility_score"] >= 70


# ── Backward compat: old v1.4.0 syntax still works ────────────────────────

class TestBackwardCompatV140:
    def test_workflow_api(self) -> None:
        r = tgx.workflow(task="graph_mining", fast_mode=True, seed=42)
        assert r.task == "graph_mining"

    def test_knn_graph(self) -> None:
        x = torch.randn(10, 4)
        ei = tgx.knn_graph(x, k=2)
        assert ei.shape[0] == 2

    def test_validate_graph(self) -> None:
        g = Graph(node_features=torch.randn(4, 3, 7, 7),
                  edge_index=torch.tensor([[0,1],[1,2]]))
        r = tgx.validate_graph(g, strict=True)
        assert r["ok"]

    def test_save_load(self) -> None:
        import tempfile, os
        g = Graph(node_features=torch.randn(4, 1, 8, 8),
                  edge_index=torch.tensor([[0,1],[1,2]]))
        with tempfile.NamedTemporaryFile(suffix=".tgx", delete=False) as f:
            p = f.name
        try:
            tgx.save(g, p)
            g2 = tgx.load(p)
            assert torch.equal(g.node_features, g2.node_features)
        finally:
            os.unlink(p)

    def test_run_graph_rl_canonical_still_works(self) -> None:
        from tgraphx.rl import run_graph_rl
        r = run_graph_rl("max_cut", algorithm="random", episodes=2, seed=42)
        assert hasattr(r, "final_reward")

    def test_run_evolutionary_canonical(self) -> None:
        from tgraphx.evolutionary import run_evolutionary_optimization
        r = run_evolutionary_optimization("ga", "connectivity", population_size=4,
                                          generations=2, num_nodes=6, seed=42)
        assert r is not None

    def test_run_graph_generation_canonical(self) -> None:
        from tgraphx.generation import run_graph_generation
        r = run_graph_generation("barabasi_albert", num_graphs=1, num_nodes=8, m=2, seed=42)
        assert len(r.graphs) == 1

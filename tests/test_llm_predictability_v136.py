"""LLM-predictability regression tests (v1.3.6).

These tests pin the public API surface that LLMs commonly generate after
`pip install tgraphx`. Each test either:
  - Runs the LLM-natural snippet successfully, OR
  - Asserts that the snippet fails with a helpful, documented error message.

Categories: top-level imports, KG, RL, generation, mining, evolutionary,
IO, NeighborLoader, reproducibility.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


# ── Top-level imports ────────────────────────────────────────────────────────


class TestTopLevelImports:

    def test_graph(self):
        from tgraphx import Graph
        assert Graph is not None

    def test_knowledge_graph_top_level(self):
        """LLM-natural: from tgraphx import KnowledgeGraph"""
        from tgraphx import KnowledgeGraph
        assert KnowledgeGraph is not None

    def test_kg_trainer_top_level(self):
        """LLM-natural: from tgraphx import KGTrainer"""
        from tgraphx import KGTrainer
        assert KGTrainer is not None


# ── KG predictability ────────────────────────────────────────────────────────


class TestKGImportPaths:

    def test_canonical_kg_import(self):
        from tgraphx.kg import KnowledgeGraph, TransEModel, KGTrainer, KGTrainingConfig
        assert all(x is not None for x in
                   (KnowledgeGraph, TransEModel, KGTrainer, KGTrainingConfig))

    def test_models_knowledge_graph_compat_shim(self):
        """LLM-natural: from tgraphx.models.knowledge_graph import TransEModel"""
        from tgraphx.models.knowledge_graph import (
            TransEModel, DistMultModel, ComplExModel, RotatEModel,
            RESCALModel, SimplEModel, KnowledgeGraph, KGTrainer,
        )
        assert all(x is not None for x in (
            TransEModel, DistMultModel, ComplExModel, RotatEModel,
            RESCALModel, SimplEModel, KnowledgeGraph, KGTrainer,
        ))


class TestKGLLMSnippet:
    """Exact maintainer-reported LLM snippet must run end-to-end."""

    def test_full_llm_kg_snippet(self):
        from tgraphx import KnowledgeGraph, KGTrainer
        from tgraphx.models.knowledge_graph import TransEModel

        num_entities = 50
        num_relations = 5
        triples = torch.randint(0, num_entities, (200, 3))
        triples[:, 1] = torch.randint(0, num_relations, (200,))

        kg = KnowledgeGraph(triples, num_entities=num_entities, num_relations=num_relations)

        model = TransEModel(num_entities, num_relations, embedding_dim=8)
        trainer = KGTrainer(model, kg, lr=0.005)

        history = trainer.fit(epochs=2, batch_size=64)
        assert "loss_history" in history
        assert len(history["loss_history"]) == 2

        ev = trainer.evaluate()
        assert isinstance(ev, dict)

    def test_kg_trainer_canonical_form_still_works(self):
        from tgraphx.kg import KGTrainer, KGTrainingConfig, TransEModel
        triples = torch.randint(0, 30, (80, 3))
        triples[:, 1] = torch.randint(0, 4, (80,))
        config = KGTrainingConfig(num_epochs=2, batch_size=32, lr=0.005, seed=0)
        model = TransEModel(30, 4, embedding_dim=8)
        trainer = KGTrainer(model, config, triples)
        result = trainer.train()
        assert "final_loss" in result

    def test_kg_trainer_rejects_both_config_and_kwargs(self):
        from tgraphx.kg import KGTrainer, KGTrainingConfig, TransEModel
        triples = torch.randint(0, 10, (20, 3))
        triples[:, 1] = torch.randint(0, 2, (20,))
        config = KGTrainingConfig(num_epochs=1, batch_size=8, seed=0)
        with pytest.raises(TypeError):
            KGTrainer(TransEModel(10, 2, embedding_dim=4), config, triples, lr=0.01)

    def test_kg_trainer_rejects_bad_config_type(self):
        from tgraphx.kg import KGTrainer, TransEModel
        with pytest.raises(TypeError):
            KGTrainer(TransEModel(10, 2, embedding_dim=4), config=42)

    def test_kg_trainer_fit_overrides_epochs(self):
        from tgraphx import KnowledgeGraph, KGTrainer
        from tgraphx.kg import TransEModel
        triples = torch.randint(0, 20, (40, 3))
        triples[:, 1] = torch.randint(0, 3, (40,))
        kg = KnowledgeGraph(triples, num_entities=20, num_relations=3)
        model = TransEModel(20, 3, embedding_dim=4)
        trainer = KGTrainer(model, kg, lr=0.01, num_epochs=1)
        r = trainer.fit(epochs=3, batch_size=16)
        assert len(r["loss_history"]) == 3


# ── RL predictability ────────────────────────────────────────────────────────


class TestRLImportPaths:

    def test_run_graph_rl_import(self):
        from tgraphx.rl import run_graph_rl
        assert callable(run_graph_rl)

    def test_graph_max_cut_env_import(self):
        """LLM-natural: from tgraphx.rl import GraphMaxCutEnv"""
        from tgraphx.rl import GraphMaxCutEnv
        assert GraphMaxCutEnv is not None


class TestGraphMaxCutEnv:
    """The LLM-natural snippet must run end-to-end."""

    def test_env_construction(self):
        from tgraphx.rl import GraphMaxCutEnv
        env = GraphMaxCutEnv(num_nodes=10, edge_density=0.3, seed=42)
        assert env.num_nodes == 10
        assert env.action_space == 2

    def test_env_reset_and_step(self):
        from tgraphx.rl import GraphMaxCutEnv
        env = GraphMaxCutEnv(num_nodes=8, edge_density=0.3, seed=42)
        obs = env.reset(seed=42)
        assert "edge_index" in obs
        obs, r, done, trunc, info = env.step(0)
        assert math.isfinite(r)

    def test_full_llm_rl_snippet(self):
        from tgraphx.rl import run_graph_rl, GraphMaxCutEnv

        env = GraphMaxCutEnv(num_nodes=8, edge_density=0.3, seed=42)

        result = run_graph_rl(
            algorithm="random",  # always safe; PPO is also supported but slower
            env=env,
            episodes=3,
            seed=42,
        )
        assert hasattr(result, "final_reward")
        assert math.isfinite(result.final_reward)

    def test_run_graph_rl_unknown_algorithm_helpful_error(self):
        from tgraphx.rl import run_graph_rl, GraphMaxCutEnv
        env = GraphMaxCutEnv(num_nodes=4, edge_density=0.2, seed=0)
        with pytest.raises(ValueError) as exc:
            run_graph_rl(algorithm="nonexistent_algo_xyz", env=env, episodes=1, seed=0)
        msg = str(exc.value)
        assert "Unknown algorithm" in msg
        assert "Choose from" in msg

    def test_max_cut_env_canonical_form_still_works(self):
        """The original MaxCutEnv(edge_index, num_nodes) API must keep working."""
        from tgraphx.rl import MaxCutEnv
        ei = torch.tensor([[0, 1, 2, 0, 1, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long)
        env = MaxCutEnv(edge_index=ei, num_nodes=3)
        env.reset(seed=0)
        env.step(0)


# ── Generation predictability ────────────────────────────────────────────────


class TestGenerationPredictability:

    def test_barabasi_albert_works(self):
        from tgraphx.generation import run_graph_generation
        result = run_graph_generation(method="barabasi_albert", num_nodes=20, m=2, seed=42)
        assert result is not None

    def test_vgae_raises_helpful_error(self):
        from tgraphx.generation import run_graph_generation
        with pytest.raises(ValueError) as exc:
            run_graph_generation(method="vgae", num_nodes=10)
        msg = str(exc.value)
        assert "not a classical graph generator" in msg
        assert "VGAEGraphGenerator" in msg

    def test_gae_raises_helpful_error(self):
        from tgraphx.generation import run_graph_generation
        with pytest.raises(ValueError) as exc:
            run_graph_generation(method="gae", num_nodes=10)
        msg = str(exc.value)
        assert "not a classical graph generator" in msg

    def test_autoregressive_raises_helpful_error(self):
        from tgraphx.generation import run_graph_generation
        with pytest.raises(ValueError) as exc:
            run_graph_generation(method="autoregressive", num_nodes=10)
        msg = str(exc.value)
        assert "AutoregressiveEdgeGenerator" in msg

    def test_transformer_raises_helpful_error(self):
        from tgraphx.generation import run_graph_generation
        with pytest.raises(ValueError) as exc:
            run_graph_generation(method="transformer", num_nodes=10)
        msg = str(exc.value)
        assert "GraphTransformerGenerator" in msg

    def test_unknown_method_lists_known_methods(self):
        from tgraphx.generation import run_graph_generation
        with pytest.raises(ValueError) as exc:
            run_graph_generation(method="totally_made_up", num_nodes=10)
        msg = str(exc.value)
        assert "Choose from" in msg
        assert "barabasi_albert" in msg

    def test_vgae_generator_available_for_direct_use(self):
        from tgraphx.generation import VGAEGraphGenerator
        assert VGAEGraphGenerator is not None


# ── Mining predictability ────────────────────────────────────────────────────


class TestMiningPredictability:

    def test_all_mining_imports(self):
        from tgraphx.mining import (
            graph_summary, motif_profile, degree_statistics, centrality_summary,
        )
        from tgraphx.mining.kernels import wl_subtree_kernel
        assert all(callable(f) for f in (
            graph_summary, motif_profile, degree_statistics,
            centrality_summary, wl_subtree_kernel,
        ))


# ── Evolutionary predictability ──────────────────────────────────────────────


class TestEvolutionaryPredictability:

    def _genome(self, seed=0):
        from tgraphx.evolutionary import GraphGenome
        torch.manual_seed(seed)
        return GraphGenome(edge_index=torch.randint(0, 8, (2, 10)), num_nodes=8)

    def test_nsga2_objective_list_works(self):
        from tgraphx.evolutionary import (
            NSGAIIOptimizer, EvolutionConfig,
            connectivity_fitness, sparsity_fitness,
        )
        cfg = EvolutionConfig(population_size=6, n_generations=2, seed=0)
        pop = [self._genome(i) for i in range(6)]
        r = NSGAIIOptimizer(cfg, [connectivity_fitness, sparsity_fitness]).optimize(pop)
        assert r.pareto_front is not None

    def test_composite_fitness_direct_raises_helpful(self):
        from tgraphx.evolutionary import (
            NSGAIIOptimizer, EvolutionConfig, composite_fitness,
        )
        with pytest.raises(TypeError) as exc:
            NSGAIIOptimizer(EvolutionConfig(population_size=4, n_generations=1, seed=0),
                            composite_fitness)
        assert "NSGAIIOptimizer expects" in str(exc.value)


# ── IO predictability ────────────────────────────────────────────────────────


class TestIOPredictability:

    def test_graphml_tensor_features_str_e(self, tmp_path):
        from tgraphx import Graph
        from tgraphx.io import write_graphml
        g = Graph(node_features=torch.randn(4, 3, 8, 8),
                  edge_index=torch.tensor([[0, 1], [1, 2]]))
        try:
            write_graphml(g, str(tmp_path / "x.graphml"), include_tensor_features=True)
            pytest.fail("Expected ValueError")
        except ValueError as e:
            msg = str(e)[:120]
            assert isinstance(msg, str) and len(msg) <= 120


# ── NeighborLoader predictability ────────────────────────────────────────────


class TestNeighborLoaderPredictability:

    def test_seed_logits_seed_y(self):
        from tgraphx import Graph
        from tgraphx.loaders import NeighborLoader

        N = 40
        x = torch.randn(N, 8)
        y = torch.randint(0, 3, (N,))
        ei = torch.randint(0, N, (2, 80))
        g = Graph(node_features=x, edge_index=ei, y=y)

        loader = NeighborLoader(graph=g, fanouts=[5, 3], batch_size=8,
                                shuffle=False, seed=0)
        batch = next(iter(loader))
        # seed-aware contract
        seed_y = batch.seed_y
        assert seed_y.shape[0] <= 8

        logits = torch.randn(batch.node_features.size(0), 3, requires_grad=True)
        seed_logits = batch.seed_logits(logits)
        loss = F.cross_entropy(seed_logits, seed_y)
        assert math.isfinite(loss.item())


# ── Reproducibility predictability ───────────────────────────────────────────


class TestReproducibilityPredictability:

    def test_deterministic_cpu(self):
        import tgraphx as tgx
        from tgraphx.reproducibility import set_seed

        def run():
            set_seed(7, deterministic=True)
            data = tgx.easy.synthetic_tensor_node_classification(
                num_nodes=32, node_shape=(4, 4, 4), num_classes=3,
                num_edges=120, seed=7,
            )
            return tgx.easy.train_node_classifier(
                data, epochs=1, batch_size=8, fanouts=[4, 2],
                verbose=False, seed=7, deterministic=True, device="cpu",
            )

        r1, r2 = run(), run()
        assert abs(r1.metrics["loss"] - r2.metrics["loss"]) < 1e-7

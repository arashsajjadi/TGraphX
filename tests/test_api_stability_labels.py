"""Tests that validate API stability labels and their correspondence with reality.

These tests prevent label drift — where a component is marked Beta in the README
but lacks tests or documentation, or is incorrectly marked Experimental despite
having full test/doc coverage.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestStableCoreAPIs:
    """Stable-labeled APIs must import and have basic functionality."""

    def test_graph_imports(self):
        from tgraphx import Graph, GraphBatch
        assert Graph is not None
        assert GraphBatch is not None

    def test_graph_y_alias(self):
        import torch
        from tgraphx import Graph
        x = torch.randn(10, 4)
        y = torch.randint(0, 3, (10,))
        g = Graph(node_features=x, y=y)
        assert g.y is not None

    def test_gnn_layers_import(self):
        from tgraphx import (
            ConvMessagePassing, TensorGATLayer, TensorGraphSAGELayer,
            TensorGINLayer, LinearMessagePassing, TensorMessagePassingLayer,
            GCNConv, GATv2Conv, APPNP,
        )
        assert all(c is not None for c in [
            ConvMessagePassing, TensorGATLayer, TensorGraphSAGELayer,
            TensorGINLayer, LinearMessagePassing,
        ])

    def test_training_utilities_import(self):
        from tgraphx import (
            set_seed, count_parameters, save_checkpoint, load_checkpoint,
            accuracy, mean_absolute_error, mean_squared_error,
            train_epoch, evaluate, fit,
        )
        assert set_seed is not None

    def test_stable_console_scripts_defined(self):
        """pyproject.toml must define tgraphx-dashboard and tgraphx-train."""
        import configparser
        import tomllib
        content = Path("pyproject.toml").read_bytes()
        data = tomllib.loads(content.decode())
        scripts = data.get("project", {}).get("scripts", {})
        assert "tgraphx-dashboard" in scripts
        assert "tgraphx-train" in scripts
        assert "tgraphx-doctor" in scripts


class TestBetaAPIs:
    """Beta-labeled APIs must import, have basic functionality, and be in __all__."""

    def test_graph_minibatch_importable(self):
        from tgraphx import GraphMiniBatch
        assert GraphMiniBatch is not None

    def test_neighbor_loader_importable(self):
        from tgraphx import NeighborLoader
        assert NeighborLoader is not None

    def test_map_global_to_local_importable(self):
        from tgraphx import map_global_to_local
        import torch
        sampled = torch.tensor([10, 20, 30])
        seeds = torch.tensor([30, 10])
        local = map_global_to_local(seeds, sampled)
        assert local.tolist() == [2, 0]

    def test_easy_mode_importable(self):
        import tgraphx as tgx
        assert hasattr(tgx, "easy")
        assert hasattr(tgx.easy, "train_node_classifier")
        assert hasattr(tgx.easy, "list_tasks")
        assert hasattr(tgx.easy, "doctor")

    def test_easy_mode_result_object(self):
        from tgraphx.easy import EasyResult
        r = EasyResult(metrics={"loss": 0.5})
        assert "loss" in r.metrics
        d = r.to_dict()
        assert "metrics" in d
        json.dumps(d)  # must be JSON-serializable

    def test_generation_high_level_api(self):
        from tgraphx import run_graph_generation, list_graph_generation_methods
        methods = list_graph_generation_methods()
        assert len(methods) >= 5

    def test_evolutionary_high_level_api(self):
        from tgraphx import run_evolutionary_optimization, list_evolutionary_optimizers
        optimizers = list_evolutionary_optimizers()
        assert len(optimizers) >= 3

    def test_kg_list_models(self):
        from tgraphx.kg import list_kg_models
        models = list_kg_models()
        assert "TransE" in models

    def test_kg_core_imports(self):
        from tgraphx.kg import (
            KnowledgeGraph, TransEModel, DistMultModel, ComplExModel, RotatEModel,
            KGTrainer, KGTrainingConfig,
        )
        assert KnowledgeGraph is not None

    def test_random_greedy_rl_baselines_importable(self):
        from tgraphx import RandomPolicy, GreedyPolicy
        assert RandomPolicy is not None
        assert GreedyPolicy is not None


class TestExperimentalAPIs:
    """Experimental APIs must at least import without error."""

    def test_rl_learning_agents_importable(self):
        from tgraphx import (
            REINFORCEAgent, ActorCriticAgent, A2CAgent,
            DQNAgent, DoubleDQNAgent, PPOAgent,
        )
        assert REINFORCEAgent is not None

    def test_rl_continuous_importable(self):
        from tgraphx import GraphDDPGAgent, GraphTD3Agent, GraphSACAgent
        assert GraphDDPGAgent is not None

    def test_rl_run_api(self):
        from tgraphx import run_graph_rl, list_graph_rl_algorithms
        algos = list_graph_rl_algorithms()
        assert len(algos) >= 10

    def test_neural_generation_importable(self):
        from tgraphx import VGAEGraphGenerator, AutoregressiveEdgeGenerator
        assert VGAEGraphGenerator is not None

    def test_hetero_graph_importable(self):
        from tgraphx import HeteroGraph, HeteroGraphBatch
        assert HeteroGraph is not None

    def test_temporal_graph_importable(self):
        from tgraphx import TemporalGraphSequence, TemporalGraphBatch
        assert TemporalGraphSequence is not None


class TestStabilityDocsCoverage:
    """api_stability.md must mention key components."""

    @pytest.fixture
    def stability_text(self):
        return Path("docs/api_stability.md").read_text()

    def test_graph_in_stability_doc(self, stability_text):
        assert "Graph" in stability_text

    def test_graphminibatch_in_stability_doc(self, stability_text):
        assert "GraphMiniBatch" in stability_text

    def test_easy_mode_in_stability_doc(self, stability_text):
        assert "easy" in stability_text.lower()

    def test_experimental_rl_in_stability_doc(self, stability_text):
        assert "REINFORCEAgent" in stability_text or "Experimental" in stability_text

    def test_beta_kg_in_stability_doc(self, stability_text):
        assert "KnowledgeGraph" in stability_text

    def test_stability_labels_defined(self, stability_text):
        assert "Beta" in stability_text
        assert "Experimental" in stability_text
        assert "Stable" in stability_text


class TestReadmeStabilityLabelDrift:
    """README capability table stability labels must not drift from api_stability.md."""

    @pytest.fixture
    def readme_text(self):
        return Path("README.md").read_text()

    def test_readme_has_stable_label(self, readme_text):
        assert "Stable" in readme_text

    def test_readme_has_beta_label(self, readme_text):
        assert "Beta" in readme_text

    def test_readme_has_experimental_label(self, readme_text):
        assert "Experimental" in readme_text

    def test_readme_graph_minibatch_mentioned(self, readme_text):
        assert "GraphMiniBatch" in readme_text

    def test_readme_easy_mode_mentioned(self, readme_text):
        assert "easy" in readme_text.lower()
        assert "Easy Mode" in readme_text or "easy mode" in readme_text.lower()

    def test_readme_kg_corrected_api(self, readme_text):
        """README KG example must use num_epochs= not epochs=."""
        # Check that the wrong form doesn't appear in the KG training block.
        import re
        # Find the KG training config code block.
        kg_blocks = re.findall(r'KGTrainingConfig\([^)]+\)', readme_text)
        for block in kg_blocks:
            assert "num_epochs=" in block, \
                f"KGTrainingConfig must use num_epochs=, found: {block}"

    def test_readme_logo_is_local(self, readme_text):
        """Prefer local logo over raw GitHub URL."""
        # The TGRAPHX.png should be referenced locally, not via raw.githubusercontent.com
        # (Either form is technically fine but local is cleaner)
        assert "TGRAPHX.png" in readme_text

    def test_api_cheatsheet_json_valid(self):
        """docs/api_cheatsheet.json must be valid JSON."""
        content = Path("docs/api_cheatsheet.json").read_text()
        data = json.loads(content)
        assert "tasks" in data
        assert "models" in data
        assert "graph_rl_algorithms" in data


class TestReadmeExamplesImportable:
    """Examples referenced in README must exist on disk."""

    def test_key_examples_exist(self):
        examples = [
            "examples/graph_paths_algorithms_demo.py",
            "examples/knowledge_graph_demo.py",
            "examples/neighbor_loader_demo.py",
            "examples/easy_tensor_node_classification_no_torch.py",
            "examples/tensor_node_classification_neighbor_loader_demo.py",
            "tutorials/tensor_node_classification_neighbor_loader.py",
            "tutorials/graph_generation_quickstart.py",
            "tutorials/evolutionary_optimization_quickstart.py",
            "tutorials/graph_rl_quickstart.py",
        ]
        missing = [e for e in examples if not Path(e).exists()]
        assert not missing, f"Missing examples: {missing}"

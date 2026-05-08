"""Backward-compatibility regression tests (v0.2 → v0.3 prep).

Covers the stable public surface listed in ``docs/deprecation_policy.md``
and ``docs/migration_v0_2_to_v0_3.md``.
"""
from __future__ import annotations

import inspect

import pytest
import torch


# ── Stable top-level exports ─────────────────────────────────────────────────

def test_stable_imports_work():
    from tgraphx import (
        Graph, GraphBatch, GraphDataLoader, GraphDataset,
        build_model, build_model_from_config, make_layer,
        fit, train_epoch, evaluate, set_seed,
        save_checkpoint, load_checkpoint,
        CSVLogger, TensorBoardLogger,
        env_report, recommended_device, estimate_message_memory,
        GraphClassifier, NodeClassifier, EdgePredictor,
        NodeRegressor, GraphRegressor,
        ConvMessagePassing, TensorGATLayer, TensorGraphSAGELayer,
        TensorGINLayer, LinearMessagePassing,
        AttentionMessagePassing,
        build_grid_graph, build_grid_graph_3d,
        build_fully_connected_graph, build_knn_graph, build_radius_graph,
        build_iou_graph, build_random_graph,
        image_to_patches, volume_to_patches,
        induced_subgraph, k_hop_subgraph, neighbor_sample,
        SubgraphDataLoader, NeighborSamplerLoader,
        write_graph_stats,
    )
    # If we reach here, all stable imports succeeded.
    assert True


def test_stable_graph_constructor_signature():
    """Old keyword args must still work."""
    import torch
    from tgraphx import Graph
    x = torch.randn(3, 4)
    ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    ew = torch.tensor([1.0, 0.5])
    ef = torch.randn(2, 3)
    nl = torch.tensor([0, 1, 2])
    el = torch.tensor([0, 1])
    gl = torch.tensor(7)
    g = Graph(
        node_features=x,
        edge_index=ei,
        edge_weight=ew,
        edge_features=ef,
        node_labels=nl,
        edge_labels=el,
        graph_label=gl,
        metadata={"k": "v"},
    )
    assert g.num_nodes == 3
    assert g.num_edges == 2


def test_stable_layer_constructors():
    """All four spatial layers + LinearMessagePassing keep working."""
    from tgraphx.layers import (
        ConvMessagePassing, TensorGATLayer, TensorGraphSAGELayer,
        TensorGINLayer, LinearMessagePassing,
    )
    cm = ConvMessagePassing((4, 4, 4), (4, 4, 4))
    gat = TensorGATLayer(4, 4, num_heads=2)
    sage = TensorGraphSAGELayer(4, 4)
    gin = TensorGINLayer(4, 4)
    lin = LinearMessagePassing((4,), (4,))
    for m in (cm, gat, sage, gin, lin):
        assert hasattr(m, "forward")


def test_stable_factory_signatures():
    from tgraphx import make_layer, build_model
    # Old-style factory calls.
    l = make_layer("gat", in_shape=(4, 4, 4), out_shape=(4, 4, 4), heads=2)
    assert l is not None
    m = build_model(
        task="graph_classification",
        layer="gat",
        in_shape=(4, 4, 4), hidden_shape=(8, 4, 4),
        num_layers=2, num_classes=3, heads=2, pooling="mean",
    )
    assert hasattr(m, "forward")


def test_stable_training_helpers():
    from tgraphx import set_seed, fit, train_epoch, evaluate
    assert callable(set_seed)
    assert callable(fit)
    assert callable(train_epoch)
    assert callable(evaluate)


def test_stable_logger_classes():
    from tgraphx import CSVLogger, TensorBoardLogger
    assert inspect.isclass(CSVLogger)
    assert inspect.isclass(TensorBoardLogger)


# ── Experimental imports — should still be available ─────────────────────────

def test_experimental_imports_available():
    from tgraphx import (
        HeteroGraph, HeteroGraphBatch,
        TemporalGraphSequence, TemporalGraphBatch,
        MLflowLogger,
    )
    assert all(inspect.isclass(c) for c in (
        HeteroGraph, HeteroGraphBatch,
        TemporalGraphSequence, TemporalGraphBatch,
        MLflowLogger,
    ))


def test_experimental_modules_importable():
    import tgraphx.interop  # noqa: F401
    import tgraphx.learned_graph  # noqa: F401
    import tgraphx.distributed  # noqa: F401
    import tgraphx.sampling  # noqa: F401
    import tgraphx.sampling_loaders  # noqa: F401
    from tgraphx.layers.hetero import HeteroConv  # noqa: F401
    from tgraphx.layers.graph_transformer import GraphTransformerLayer  # noqa: F401


# ── No optional heavy import on package import ───────────────────────────────

def test_no_eager_optional_imports():
    """Importing tgraphx must not pull in mlflow/torch_geometric/dgl/tensorboard."""
    import subprocess, sys
    code = (
        "import tgraphx, sys; "
        "for m in ('mlflow', 'torch_geometric', 'dgl', 'tensorboard'): "
        "    assert m not in sys.modules, f'eager import: {m}'"
    )
    # Multi-line in subprocess via -c is awkward with for; use exec-friendly.
    code = (
        "import tgraphx, sys\n"
        "for m in ('mlflow', 'torch_geometric', 'dgl', 'tensorboard'):\n"
        "    assert m not in sys.modules, f'eager import: {m}'\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

"""Optional adapter tests: PyG / DGL / OGB.

Each block skips cleanly if the upstream package is not installed.  A
mocked-class subblock asserts that calling the adapter without the
upstream package surfaces a clear :class:`OptionalDependencyError`.
"""
from __future__ import annotations

import builtins

import pytest
import torch

from tgraphx.datasets import (
    PyGDatasetAdapter,
    DGLDatasetAdapter,
    OGBDatasetAdapter,
    OGBEvaluatorWrapper,
    from_pyg_data,
    ogb_item_to_graph,
)
from tgraphx.datasets.errors import OptionalDependencyError


# ── PyG (lazy-import gating) ─────────────────────────────────────────────────


class TestPyGAdapter:
    def test_missing_torch_geometric_raises(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "torch_geometric" or name.startswith("torch_geometric."):
                raise ImportError("simulated missing torch_geometric")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(OptionalDependencyError, match="torch_geometric"):
            PyGDatasetAdapter(dataset_cls="Planetoid", root="/tmp/should-not-create",
                              dataset_kwargs={"name": "Cora"})

    def test_from_pyg_data_with_real_pyg_or_skip(self):
        torch_geometric = pytest.importorskip("torch_geometric",
                                              reason="torch_geometric not installed")
        from torch_geometric.data import Data

        data = Data(
            x=torch.randn(4, 3),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            edge_attr=torch.randn(3, 2),
            y=torch.tensor([0, 1, 0, 1]),
        )
        graph = from_pyg_data(data)
        assert graph.num_nodes == 4
        assert graph.num_edges == 3
        assert graph.has_edge_features


# ── DGL ──────────────────────────────────────────────────────────────────────


class TestDGLAdapter:
    def test_missing_dgl_raises(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "dgl" or name.startswith("dgl."):
                raise ImportError("simulated missing dgl")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(OptionalDependencyError, match="dgl"):
            DGLDatasetAdapter(dataset_cls="CoraGraphDataset")


# ── OGB ──────────────────────────────────────────────────────────────────────


class TestOGBAdapter:
    def test_missing_ogb_raises(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "ogb" or name.startswith("ogb."):
                raise ImportError("simulated missing ogb")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(OptionalDependencyError, match="ogb"):
            OGBDatasetAdapter(name="ogbn-arxiv")

    def test_evaluator_wrapper_with_fake_class(self):
        """OGBEvaluatorWrapper does not import OGB itself."""

        class _FakeEvaluator:
            def __init__(self, name):
                self.name = name

            def eval(self, input_dict):
                # Return a deterministic mock score.
                return {"acc": 0.5}

        wrapper = OGBEvaluatorWrapper(_FakeEvaluator, name="ogbn-test")
        out = wrapper.eval({"y_true": torch.zeros(3), "y_pred": torch.zeros(3)})
        assert out["acc"] == 0.5


# ── ogb_item_to_graph ────────────────────────────────────────────────────────


class TestOGBItemToGraph:
    def test_with_real_pyg_or_skip(self):
        pytest.importorskip("torch_geometric",
                            reason="torch_geometric not installed")
        from torch_geometric.data import Data

        data = Data(
            x=torch.randn(5, 4),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
            y=torch.tensor([0, 1, 0, 1, 0]),
        )
        graph = ogb_item_to_graph(data, task_type="node")
        assert graph.num_nodes == 5

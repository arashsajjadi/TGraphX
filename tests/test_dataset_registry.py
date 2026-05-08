"""Tests for tgraphx.datasets registry (v0.2.9)."""
from __future__ import annotations

import pytest

import tgraphx.datasets as dsmod


class TestListAndInfo:
    def test_list_datasets_includes_synthetic(self):
        names = dsmod.list_datasets()
        assert "synthetic:patch_graph" in names
        assert "synthetic:node_classification" in names
        assert "synthetic:hetero" in names

    def test_groups(self):
        groups = dsmod.available_dataset_groups()
        assert "synthetic" in groups
        # Optional adapters always have at least one entry registered.
        assert "torchvision" in groups
        assert "pyg" in groups
        assert "dgl" in groups
        assert "ogb" in groups

    def test_dataset_info_no_optional_imports(self):
        info = dsmod.dataset_info("synthetic:patch_graph")
        assert info["name"] == "synthetic:patch_graph"
        assert "synthetic" in info["tags"]

    def test_unknown_suggestions(self):
        from tgraphx.datasets.errors import DatasetNotFoundError
        with pytest.raises(DatasetNotFoundError, match="Did you mean"):
            dsmod.get_dataset("synthetic:patcch_graph", num_graphs=2)

    def test_filter_by_tag(self):
        names = dsmod.list_datasets(tags=["synthetic"])
        assert all(n.startswith("synthetic:") for n in names)

    def test_has_dataset(self):
        assert dsmod.has_dataset("synthetic:patch_graph") is True
        assert dsmod.has_dataset("not:a:dataset") is False


class TestGetDataset:
    def test_get_synthetic_patch(self):
        ds = dsmod.get_dataset("synthetic:patch_graph", num_graphs=3, seed=0)
        assert len(ds) == 3
        assert ds[0].graph_label is not None

    def test_pyg_missing_dependency(self, monkeypatch):
        # Force import to fail.
        import sys
        for mod in list(sys.modules):
            if mod.startswith("torch_geometric"):
                monkeypatch.delitem(sys.modules, mod, raising=False)

        # Block torch_geometric import for the duration of the test.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "torch_geometric" or name.startswith("torch_geometric."):
                raise ImportError("simulated missing torch_geometric")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        from tgraphx.datasets.errors import OptionalDependencyError
        with pytest.raises(OptionalDependencyError, match="torch_geometric"):
            dsmod.get_dataset("pyg:planetoid/cora", root="/tmp/should-not-create")

    def test_dgl_missing_dependency(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "dgl" or name.startswith("dgl."):
                raise ImportError("simulated missing dgl")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from tgraphx.datasets.errors import OptionalDependencyError
        with pytest.raises(OptionalDependencyError, match="dgl"):
            dsmod.get_dataset("dgl:cora")

    def test_ogb_missing_dependency(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "ogb" or name.startswith("ogb."):
                raise ImportError("simulated missing ogb")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from tgraphx.datasets.errors import OptionalDependencyError
        with pytest.raises(OptionalDependencyError, match="ogb"):
            dsmod.get_dataset("ogb:ogbn-arxiv")

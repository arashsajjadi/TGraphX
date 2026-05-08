"""Tests for tgraphx.datasets base classes (v0.2.9)."""
from __future__ import annotations

import pytest
import torch

from tgraphx import Graph
from tgraphx.datasets import (
    BaseGraphDataset,
    DatasetMetadata,
    DownloadableGraphDataset,
    InMemoryGraphDataset,
)


class _ToyInMemory(InMemoryGraphDataset):
    def _populate(self):
        self.graphs = [
            Graph(torch.randn(3, 2), torch.tensor([[0, 1], [1, 2]], dtype=torch.long)),
            Graph(torch.randn(4, 2), torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)),
        ]

    def _build_metadata(self):
        return DatasetMetadata(
            name="toy",
            task="graph_classification",
            graph_type="homogeneous",
            num_graphs=len(self.graphs),
        )


class TestInMemory:
    def test_len_and_get(self):
        ds = _ToyInMemory()
        assert len(ds) == 2
        g = ds[0]
        assert g.num_nodes == 3

    def test_indexing_neg(self):
        ds = _ToyInMemory()
        assert ds[-1].num_nodes == 4

    def test_index_out_of_range(self):
        ds = _ToyInMemory()
        with pytest.raises(IndexError):
            ds[10]

    def test_iteration(self):
        ds = _ToyInMemory()
        items = list(ds)
        assert len(items) == 2

    def test_metadata(self):
        ds = _ToyInMemory()
        m = ds.metadata
        assert m.name == "toy"
        assert m.num_graphs == 2

    def test_summary(self):
        ds = _ToyInMemory()
        s = ds.summary()
        assert s["len"] == 2
        assert s["sample_node_features_shape"] == [3, 2]

    def test_describe_no_crash(self):
        ds = _ToyInMemory()
        text = ds.describe()
        assert "toy" in text

    def test_transform_applied(self):
        recorded = {}

        def mark(g):
            recorded["called"] = True
            return g

        ds = _ToyInMemory()
        ds.transform = mark
        _ = ds[0]
        assert recorded.get("called") is True


class TestDatasetMetadata:
    def test_to_from_dict(self):
        m = DatasetMetadata(name="x", task="graph_classification", num_classes=4)
        d = m.to_dict()
        m2 = DatasetMetadata.from_dict(d)
        assert m2.name == "x"
        assert m2.num_classes == 4

    def test_save_load_json(self, tmp_path):
        m = DatasetMetadata(name="x", num_graphs=10)
        path = tmp_path / "meta.json"
        m.save_json(path)
        loaded = DatasetMetadata.load_json(path)
        assert loaded.num_graphs == 10

    def test_unknown_keys_ignored(self):
        m = DatasetMetadata.from_dict({"name": "y", "extra_unknown": 99})
        assert m.name == "y"


class TestDownloadableRefusesSilentDownload:
    """``DownloadableGraphDataset`` must not download silently."""

    def test_missing_files_raise(self, tmp_path):
        class _MissingDataset(DownloadableGraphDataset):
            raw_file_names = ("never_exists.bin",)

            def download(self):  # noqa: D401  - test stub
                raise AssertionError("download() must not be called")

            def process(self):  # pragma: no cover
                self.graphs = []

        from tgraphx.datasets.errors import DatasetFilesNotFoundError
        with pytest.raises(DatasetFilesNotFoundError):
            _MissingDataset(root=str(tmp_path), download=False)

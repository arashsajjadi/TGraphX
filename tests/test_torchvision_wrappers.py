"""Torchvision adapter tests using FakeData; no network (v0.2.9)."""
from __future__ import annotations

import pytest
import torch

from tgraphx.datasets import (
    FakeDataPatchGraphDataset,
    TorchvisionPatchGraphDataset,
)


class TestFakeData:
    def test_basic(self, tmp_path):
        ds = FakeDataPatchGraphDataset(
            root=str(tmp_path),
            upstream_kwargs={"size": 4, "image_size": (3, 16, 16),
                             "num_classes": 2},
            patch_size=4,
        )
        assert len(ds) == 4
        g = ds[0]
        assert g.node_features.shape[1:] == (3, 4, 4)
        assert g.graph_label.dtype == torch.long

    def test_metadata(self, tmp_path):
        ds = FakeDataPatchGraphDataset(
            root=str(tmp_path),
            upstream_kwargs={"size": 2, "image_size": (3, 16, 16)},
            patch_size=4,
        )
        m = ds.metadata
        assert m.upstream_library == "torchvision"
        assert "FakeData" in m.extra["upstream_class"]


class TestGenericTorchvisionAdapter:
    def test_unknown_class_raises(self):
        with pytest.raises(ValueError, match="torchvision.datasets has no attribute"):
            TorchvisionPatchGraphDataset(dataset_class_or_name="NoSuchClass")

    def test_does_not_download_by_default(self, tmp_path, monkeypatch):
        """``download=False`` must never appear as ``True`` to upstream loaders."""
        # Use MNIST as the spy target — it accepts ``download``.  We patch
        # __init__ so that calling MNIST does not actually download.
        from torchvision import datasets as tv_datasets

        called: dict = {}

        def spy_init(self, root, train=True, transform=None, target_transform=None, download=False):
            called["root"] = root
            called["download"] = download
            # Simulate a tiny in-memory MNIST with no on-disk artefacts.
            self.data = torch.zeros(0, 28, 28, dtype=torch.uint8)
            self.targets = torch.zeros(0, dtype=torch.long)
            self.train = train

        monkeypatch.setattr(tv_datasets.MNIST, "__init__", spy_init)
        # Bypass __len__ checks — but our spy already wrote zero-length data.

        TorchvisionPatchGraphDataset(
            dataset_class_or_name="MNIST",
            root=str(tmp_path),
            download=False,
            train=True,
            patch_size=7,
        )
        assert called["download"] is False

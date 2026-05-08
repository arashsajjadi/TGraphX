"""Folder dataset tests using tempfile only (v0.2.9). No external data."""
from __future__ import annotations

import numpy as np
import pytest
import torch

PIL = pytest.importorskip("PIL", reason="Pillow not installed")
from PIL import Image  # noqa: E402

from tgraphx.datasets import (
    ImageFolderPatchGraphDataset,
    VolumeFolderPatchGraphDataset,
)


class TestImageFolder:
    def _make_tree(self, root, n_per_class=2, image_size=16):
        for cls in ("a", "b"):
            (root / cls).mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                arr = (np.random.rand(image_size, image_size, 3) * 255).astype("uint8")
                Image.fromarray(arr).save(root / cls / f"img_{i}.png")

    def test_basic(self, tmp_path):
        self._make_tree(tmp_path, n_per_class=2, image_size=16)
        ds = ImageFolderPatchGraphDataset(
            root=tmp_path, patch_size=4, graph_builder="grid",
        )
        assert len(ds) == 4
        g = ds[0]
        assert g.node_features.shape[1:] == (3, 4, 4)
        assert g.graph_label.dtype == torch.long

    def test_class_to_idx(self, tmp_path):
        self._make_tree(tmp_path)
        ds = ImageFolderPatchGraphDataset(root=tmp_path, patch_size=4)
        assert ds.class_to_idx == {"a": 0, "b": 1}

    def test_padding_auto(self, tmp_path):
        # Image 18x18, patch 4 → must pad to 20x20.
        self._make_tree(tmp_path, n_per_class=1, image_size=18)
        ds = ImageFolderPatchGraphDataset(
            root=tmp_path, patch_size=4, padding="auto",
        )
        g = ds[0]
        assert g.metadata["grid_shape"] == (5, 5)

    def test_invalid_root(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ImageFolderPatchGraphDataset(root=tmp_path / "missing")


class TestVolumeFolder:
    def _make_tree(self, root, n_per_class=2, volume_size=8):
        for cls in ("x", "y"):
            (root / cls).mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                arr = np.random.rand(1, volume_size, volume_size, volume_size).astype("float32")
                np.save(root / cls / f"v_{i}.npy", arr)

    def test_basic(self, tmp_path):
        self._make_tree(tmp_path, n_per_class=2, volume_size=8)
        ds = VolumeFolderPatchGraphDataset(
            root=tmp_path, patch_size=4, graph_builder="grid",
        )
        assert len(ds) == 4
        g = ds[0]
        assert g.node_features.shape[1:] == (1, 4, 4, 4)

    def test_pt_format(self, tmp_path):
        cls = tmp_path / "z"
        cls.mkdir()
        torch.save(torch.randn(1, 8, 8, 8), cls / "v.pt")
        ds = VolumeFolderPatchGraphDataset(root=tmp_path, patch_size=4)
        assert len(ds) == 1

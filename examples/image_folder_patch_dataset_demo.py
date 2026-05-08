"""image_folder_patch_dataset_demo.py — convert a small temp image folder.

The script generates fake PNG images under a temporary directory, then
shows how :class:`ImageFolderPatchGraphDataset` walks them and converts
each image into a TGraphX patch :class:`Graph`.

No network access; PIL is required (skipped cleanly if missing).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:
    print(f"Skipping demo: {exc.__class__.__name__}: {exc}")
    sys.exit(0)

from tgraphx.datasets import ImageFolderPatchGraphDataset


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        raw_path = Path(raw)
        for cls in ("a", "b"):
            (raw_path / cls).mkdir()
            for i in range(2):
                arr = (np.random.rand(20, 20, 3) * 255).astype("uint8")
                Image.fromarray(arr).save(raw_path / cls / f"{cls}_{i}.png")

        ds = ImageFolderPatchGraphDataset(
            root=raw_path, patch_size=4, graph_builder="grid",
            padding="auto",
        )
        print(f"len={len(ds)}  classes={ds.class_to_idx}")
        g = ds[0]
        print(f"  node_features={tuple(g.node_features.shape)}  "
              f"label={int(g.graph_label)}  grid={g.metadata['grid_shape']}")


if __name__ == "__main__":
    main()

"""synthetic_datasets_demo.py — show every native synthetic dataset.

Run-time: a few seconds on CPU.  No network access.
"""
from __future__ import annotations

import torch

from tgraphx.datasets import (
    SyntheticEdgePredictionDataset,
    SyntheticGraphRegressionDataset,
    SyntheticHeteroGraphDataset,
    SyntheticNodeClassificationDataset,
    SyntheticPatchGraphDataset,
    SyntheticTemporalGraphDataset,
    SyntheticVolumeGraphDataset,
)


def main() -> None:
    torch.manual_seed(0)
    print("synthetic:patch_graph")
    ds = SyntheticPatchGraphDataset(num_graphs=4, image_size=16, patch_size=4, seed=0)
    print(f"  len={len(ds)}  node_features={tuple(ds[0].node_features.shape)}  "
          f"label dtype={ds[0].graph_label.dtype}")

    print("\nsynthetic:volume_graph")
    ds_v = SyntheticVolumeGraphDataset(num_graphs=2, volume_size=8, patch_size=4, seed=0)
    print(f"  len={len(ds_v)}  node_features={tuple(ds_v[0].node_features.shape)}")

    print("\nsynthetic:node_classification")
    ds_n = SyntheticNodeClassificationDataset(num_nodes=40, num_classes=3, seed=0)
    g = ds_n[0]
    masks = g.metadata["masks"]
    print(f"  num_nodes={g.num_nodes}  feature_dim={g.node_features.size(1)}  "
          f"train/val/test={int(masks['train_mask'].sum())}/"
          f"{int(masks['val_mask'].sum())}/{int(masks['test_mask'].sum())}")

    print("\nsynthetic:edge_prediction")
    ds_e = SyntheticEdgePredictionDataset(num_nodes=20, num_pos=8, num_neg=8, seed=0)
    print(f"  num_edges={ds_e[0].num_edges}  edge_labels unique="
          f"{ds_e[0].edge_labels.unique().tolist()}")

    print("\nsynthetic:graph_regression")
    ds_r = SyntheticGraphRegressionDataset(num_graphs=4, image_size=16, patch_size=4, seed=0)
    print(f"  graph labels (first 4): "
          f"{[float(x.graph_label) for x in ds_r]}")

    print("\nsynthetic:hetero")
    ds_h = SyntheticHeteroGraphDataset(num_papers=8, num_authors=5, num_venues=3, seed=0)
    hg = ds_h[0]
    print(f"  node types: {hg.node_types}")
    for et in hg.edge_types:
        print(f"    {et}: edges={hg.num_edges(et)}")

    print("\nsynthetic:temporal")
    ds_t = SyntheticTemporalGraphDataset(num_sequences=3, sequence_length=4, num_nodes=8, seed=0)
    seq = ds_t[0]
    print(f"  num_snapshots={seq.num_snapshots}  trend={seq.metadata['trend']}")


if __name__ == "__main__":
    main()

"""explainability_saliency_demo.py — saliency + IG + perturbation on a synthetic patch graph."""
from __future__ import annotations

import torch

from tgraphx import build_model
from tgraphx.datasets import SyntheticPatchGraphDataset
from tgraphx.explain import (
    edge_perturbation_attribution,
    integrated_gradients,
    node_feature_saliency,
    patch_saliency_to_image_grid,
)


def main() -> None:
    torch.manual_seed(0)
    ds = SyntheticPatchGraphDataset(num_graphs=1, image_size=16, patch_size=4, seed=0)
    g = ds[0]
    model = build_model(
        task="graph_classification", layer="conv",
        in_shape=(1, 4, 4), hidden_shape=(8, 4, 4),
        num_layers=2, num_classes=ds.metadata.num_classes, pooling="mean",
    )

    target = int(g.graph_label)
    sal = node_feature_saliency(model, g, target=target)
    ig = integrated_gradients(model, g, target=target, steps=8)
    edge_imp = edge_perturbation_attribution(model, g, target=target, max_edges=8)
    heatmap = patch_saliency_to_image_grid(sal, grid_shape=g.metadata["grid_shape"])

    print(f"saliency        : {tuple(sal.shape)}, max abs = {sal.abs().max().item():.4f}")
    print(f"integrated grad : {tuple(ig.shape)}, max abs = {ig.abs().max().item():.4f}")
    print(f"edge importance : {tuple(edge_imp.shape)}, top-3 = "
          f"{edge_imp.abs().sort(descending=True).values[:3].tolist()}")
    print(f"image heatmap   : {tuple(heatmap.shape)}")


if __name__ == "__main__":
    main()

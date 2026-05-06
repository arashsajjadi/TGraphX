"""03_volumetric_graph_classification.py

3-D volumetric patch graph classification with a factory-built SAGE model.

Pipeline:
  volumes [B, C, D, H, W]
  → volume_to_patches → [B, P, C, pd, ph, pw]
  → build_grid_graph_3d(n_d, n_h, n_w)  +  GraphBatch
  → build_model("graph_classification", "sage", ...)
  → logits [B, num_classes]
"""
import torch
from tgraphx import (
    Graph,
    GraphBatch,
    build_grid_graph_3d,
    build_model,
    volume_patch_grid_shape,
    volume_to_patches,
)


def main():
    B, C, D, H, W = 2, 2, 8, 8, 8
    patch_size = 4
    num_classes = 3

    volumes = torch.randn(B, C, D, H, W)
    patches = volume_to_patches(volumes, patch_size=patch_size)
    _, P, C_p, pd, ph, pw = patches.shape
    n_d, n_h, n_w = volume_patch_grid_shape(D, H, W, patch_size)

    print(f"Volumes       : {tuple(volumes.shape)}")
    print(f"Patches       : {tuple(patches.shape)}  (P={P}, each {C_p}×{pd}×{ph}×{pw})")
    print(f"Patch grid 3D : {n_d}×{n_h}×{n_w}  →  {P} nodes per graph")

    edge_index = build_grid_graph_3d(n_d, n_h, n_w, directed=False, self_loops=True)
    print(f"Edge index    : {tuple(edge_index.shape)}")

    # Pack into GraphBatch
    graphs = [Graph(patches[i], edge_index) for i in range(B)]
    gb = GraphBatch(graphs)
    print(f"GraphBatch    : {gb}")

    model = build_model(
        task="graph_classification",
        layer="sage",
        in_shape=(C_p, pd, ph, pw),
        hidden_shape=(4, pd, ph, pw),
        num_layers=2,
        num_classes=num_classes,
        pooling="mean",
    )
    print(f"Model         : {model.__class__.__name__}")

    out = model(gb.node_features, gb.edge_index, batch=gb.batch)
    print(f"Output logits : {tuple(out.shape)}  (expected [{B}, {num_classes}])")

    out.sum().backward()
    print("Backward      : OK")
    print("\nDone.")


if __name__ == "__main__":
    main()

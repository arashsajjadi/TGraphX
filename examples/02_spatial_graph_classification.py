"""02_spatial_graph_classification.py

2-D image patch graph classification with a factory-built GAT model.

Pipeline:
  images [B, C, H, W]
  → image_to_patches → [B, P, C, ph, pw]
  → build_grid_graph(n_h, n_w)  +  GraphBatch
  → build_model("graph_classification", "gat", ...)
  → logits [B, num_classes]
"""
import torch
from tgraphx import (
    Graph,
    GraphBatch,
    build_grid_graph,
    build_model,
    image_to_patches,
    patch_grid_shape,
)


def main():
    B, C, H, W = 3, 3, 8, 8
    patch_size = 4
    num_classes = 5

    images = torch.randn(B, C, H, W)
    patches = image_to_patches(images, patch_size=patch_size)
    _, P, C_p, ph, pw = patches.shape
    n_h, n_w = patch_grid_shape(H, W, patch_size)

    print(f"Images        : {tuple(images.shape)}")
    print(f"Patches       : {tuple(patches.shape)}  (P={P}, each {C_p}×{ph}×{pw})")
    print(f"Patch grid    : {n_h}×{n_w}  →  {P} nodes per graph")

    edge_index = build_grid_graph(n_h, n_w, directed=False, self_loops=True)
    print(f"Edge index    : {tuple(edge_index.shape)}")

    # Pack B separate graphs into a GraphBatch
    graphs = [Graph(patches[i], edge_index) for i in range(B)]
    gb = GraphBatch(graphs)
    print(f"GraphBatch    : {gb}")

    model = build_model(
        task="graph_classification",
        layer="gat",
        in_shape=(C_p, ph, pw),
        hidden_shape=(8, ph, pw),
        num_layers=2,
        num_classes=num_classes,
        heads=2,
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

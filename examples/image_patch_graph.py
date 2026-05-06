"""image_patch_graph.py — 2-D image → patch graph → GNN demo.

Synthetic [B, C, H, W] image → non-overlapping patches → build_grid_graph
→ run ConvMessagePassing and TensorGATLayer on the patch graph.
"""
import torch
from tgraphx.graph_builders import build_grid_graph, image_to_patches
from tgraphx.layers.conv_message import ConvMessagePassing
from tgraphx.layers.gat import TensorGATLayer


def main():
    # Synthetic image batch: B=2, C=3 channels, 8×8 pixels
    B, C, H, W = 2, 3, 8, 8
    images = torch.randn(B, C, H, W)
    print(f"Input images  : {tuple(images.shape)}")

    # Extract 4×4 non-overlapping patches → [B, P, C, ph, pw]
    patch_size = 4
    patches = image_to_patches(images, patch_size=patch_size)
    B, P, C_p, ph, pw = patches.shape
    print(f"Patches       : {tuple(patches.shape)}  (P={P} patches of {C_p}×{ph}×{pw})")

    # Build 2-D grid graph for one image: 2 rows × 2 cols of patches
    n_h, n_w = H // patch_size, W // patch_size
    edge_index = build_grid_graph(n_h, n_w, directed=False, self_loops=True)
    print(f"Grid graph    : {n_h}×{n_w} nodes, {edge_index.shape[1]} edges")

    # Use the first image's patches as node features: [P, C, ph, pw]
    x = patches[0]
    print(f"\nNode features : {tuple(x.shape)}")

    # ConvMessagePassing: in_shape=(C, ph, pw), out_shape=(8, ph, pw)
    in_shape = (C_p, ph, pw)
    out_shape = (8, ph, pw)
    conv = ConvMessagePassing(in_shape=in_shape, out_shape=out_shape, aggr="sum")
    out_conv = conv(x, edge_index)
    print(f"ConvMP output : {tuple(out_conv.shape)}")

    # TensorGATLayer: in_channels=C, out_channels=8, 2 heads
    gat = TensorGATLayer(
        in_channels=C_p, out_channels=8, num_heads=2,
        concat_heads=True, spatial_rank=2
    )
    out_gat = gat(x, edge_index)
    print(f"GAT output    : {tuple(out_gat.shape)}")

    print("\nDone.")


if __name__ == "__main__":
    main()

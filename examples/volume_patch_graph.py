"""volume_patch_graph.py — 3-D volume → patch graph → GNN demo.

Synthetic [B, C, D, H, W] volume → non-overlapping volumetric patches
→ build_grid_graph_3d → run ConvMessagePassing and TensorGATLayer.
"""
import torch
from tgraphx.graph_builders import build_grid_graph_3d, volume_to_patches
from tgraphx.layers.conv_message import ConvMessagePassing
from tgraphx.layers.gat import TensorGATLayer


def main():
    # Synthetic volume batch: B=2, C=2 channels, D=8 depth, 8×8 spatial
    B, C, D, H, W = 2, 2, 8, 8, 8
    volumes = torch.randn(B, C, D, H, W)
    print(f"Input volumes : {tuple(volumes.shape)}")

    # Extract 4×4×4 non-overlapping patches → [B, P, C, pd, ph, pw]
    patch_size = 4
    patches = volume_to_patches(volumes, patch_size=patch_size)
    B, P, C_p, pd, ph, pw = patches.shape
    print(f"Patches       : {tuple(patches.shape)}  (P={P} patches of {C_p}×{pd}×{ph}×{pw})")

    # Build 3-D grid graph: 2×2×2 patch grid
    n_d, n_h, n_w = D // patch_size, H // patch_size, W // patch_size
    edge_index = build_grid_graph_3d(n_d, n_h, n_w, directed=False, self_loops=True)
    print(f"Grid graph    : {n_d}×{n_h}×{n_w} nodes, {edge_index.shape[1]} edges")

    # Use the first volume's patches as node features: [P, C, pd, ph, pw]
    x = patches[0]
    print(f"\nNode features : {tuple(x.shape)}")

    # ConvMessagePassing (3-D): in_shape=(C, pd, ph, pw)
    in_shape  = (C_p, pd, ph, pw)
    out_shape = (4, pd, ph, pw)
    conv = ConvMessagePassing(in_shape=in_shape, out_shape=out_shape, aggr="sum")
    out_conv = conv(x, edge_index)
    print(f"ConvMP output : {tuple(out_conv.shape)}")

    # TensorGATLayer (3-D): in_channels=C, out_channels=4, 2 heads
    gat = TensorGATLayer(
        in_channels=C_p, out_channels=4, num_heads=2,
        concat_heads=True, spatial_rank=3
    )
    out_gat = gat(x, edge_index)
    print(f"GAT output    : {tuple(out_gat.shape)}")

    print("\nDone.")


if __name__ == "__main__":
    main()

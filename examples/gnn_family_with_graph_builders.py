"""gnn_family_with_graph_builders.py — all four GNN families on grid graphs.

Builds a small 2-D and 3-D grid graph with the graph builders and runs
ConvMessagePassing, TensorGATLayer, TensorGraphSAGELayer, TensorGINLayer
on both.  All shapes are printed for verification.
"""
import torch
from tgraphx.graph_builders import build_grid_graph, build_grid_graph_3d
from tgraphx.layers.conv_message import ConvMessagePassing
from tgraphx.layers.gat import TensorGATLayer
from tgraphx.layers.gin import TensorGINLayer
from tgraphx.layers.sage import TensorGraphSAGELayer


def run_all_layers(x: torch.Tensor, edge_index: torch.Tensor, label: str) -> None:
    """Run all four GNN families on x with edge_index and print shapes."""
    print(f"\n  [{label}]")
    print(f"    node_features : {tuple(x.shape)}")
    print(f"    edge_index    : {tuple(edge_index.shape)}")

    in_channels = x.shape[1]
    out_channels = 8
    spatial_rank = x.dim() - 2  # 2 or 3

    if spatial_rank == 2:
        in_shape  = tuple(x.shape[1:])       # (C, H, W)
        out_shape = (out_channels,) + tuple(x.shape[2:])
    else:
        in_shape  = tuple(x.shape[1:])       # (C, D, H, W)
        out_shape = (out_channels,) + tuple(x.shape[2:])

    # ConvMessagePassing
    conv = ConvMessagePassing(in_shape=in_shape, out_shape=out_shape, aggr="sum")
    out = conv(x, edge_index)
    print(f"    ConvMP  → {tuple(out.shape)}")

    # TensorGATLayer
    gat = TensorGATLayer(
        in_channels=in_channels, out_channels=out_channels,
        num_heads=2, concat_heads=True, spatial_rank=spatial_rank,
    )
    out = gat(x, edge_index)
    print(f"    GAT     → {tuple(out.shape)}")

    # TensorGraphSAGELayer
    sage = TensorGraphSAGELayer(
        in_channels=in_channels, out_channels=out_channels,
        spatial_rank=spatial_rank,
    )
    out = sage(x, edge_index)
    print(f"    SAGE    → {tuple(out.shape)}")

    # TensorGINLayer
    gin = TensorGINLayer(
        in_channels=in_channels, out_channels=out_channels,
        spatial_rank=spatial_rank,
    )
    out = gin(x, edge_index)
    print(f"    GIN     → {tuple(out.shape)}")


def main():
    print("=" * 60)
    print("  GNN family demo with graph builders")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 2-D grid graph: 3×3 nodes, each with [C=2, H=4, W=4] features
    # ------------------------------------------------------------------
    rows, cols = 3, 3
    C, H, W = 2, 4, 4
    ei_2d = build_grid_graph(rows, cols, directed=False, self_loops=True)
    x_2d = torch.randn(rows * cols, C, H, W)
    run_all_layers(x_2d, ei_2d, "2-D grid 3×3, node features [N,2,4,4]")

    # ------------------------------------------------------------------
    # 3-D grid graph: 2×2×2 nodes, each with [C=2, D=2, H=2, W=2] features
    # ------------------------------------------------------------------
    depth, rows3, cols3 = 2, 2, 2
    C3, D3, H3, W3 = 2, 2, 2, 2
    ei_3d = build_grid_graph_3d(depth, rows3, cols3, directed=False, self_loops=True)
    x_3d = torch.randn(depth * rows3 * cols3, C3, D3, H3, W3)
    run_all_layers(x_3d, ei_3d, "3-D grid 2×2×2, node features [N,2,2,2,2]")

    print("\nDone.")


if __name__ == "__main__":
    main()

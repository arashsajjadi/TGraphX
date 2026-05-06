# 2-D Spatial GNN Layers

TGraphX's main contribution: GNN message passing where **node features are
`[N, C, H, W]` spatial tensors** rather than flat vectors. 1×1 convolutions
replace linear projections, so the spatial layout is preserved through every
aggregation step.

## Supported spatial GNN families

| Layer | Class | Aggregation | Edge features |
|---|---|---|---|
| Conv-style | `ConvMessagePassing` | sum / mean / max | spatial `[E,C,H,W]` only |
| GAT | `TensorGATLayer` | softmax-weighted sum | vector `[E,D]` or spatial (mean-pooled) |
| GraphSAGE | `TensorGraphSAGELayer` | mean / max | vector or spatial |
| GIN | `TensorGINLayer` | sum | vector or spatial |

## ConvMessagePassing

```python
from tgraphx.layers.conv_message import ConvMessagePassing

layer = ConvMessagePassing(
    in_shape=(16, 8, 8),     # [C, H, W] per node
    out_shape=(32, 8, 8),
    aggr="sum",              # "sum" | "mean" | "max"
    use_edge_features=False,
    residual=False,
)
out = layer(node_features, edge_index)  # [N, 32, 8, 8]

# Optional chunked forward to reduce peak edge-buffer memory
out = layer(node_features, edge_index, chunk_size=512)
# chunk_size is supported for aggr="sum" and "mean"; falls back for "max"
```

## TensorGATLayer

True multi-head GAT with scalar attention per `(edge, head)`. Spatial
dimensions are mean-pooled to scalars for the scoring step; values keep
their full `[C_head, H, W]` layout for aggregation.

```python
from tgraphx.layers.gat import TensorGATLayer

layer = TensorGATLayer(
    in_channels=16,
    out_channels=32,         # must be divisible by num_heads if concat_heads=True
    num_heads=4,
    concat_heads=True,
    spatial_rank=2,          # 2 for [N,C,H,W]; 3 for [N,C,D,H,W]
    residual=False,
    attn_dropout=0.1,
)
out = layer(x, edge_index)  # [N, 32, H, W]
```

> **AMP note:** `TensorGATLayer` uses `index_add_` which requires matching
> dtypes. float16 autocast may raise a dtype mismatch. Use bfloat16 or
> full precision for stable AMP inference.
>
> **Chunking:** GAT chunking is deferred because destination-wise softmax
> requires all edge scores to be available simultaneously.

## TensorGraphSAGELayer

```python
from tgraphx.layers.sage import TensorGraphSAGELayer

layer = TensorGraphSAGELayer(
    in_channels=16,
    out_channels=32,
    aggr="mean",             # "mean" | "max"
    spatial_rank=2,
    use_edge_features=False,
    edge_features_kind="vector",  # "vector" | "spatial"
)
out = layer(x, edge_index)
```

## TensorGINLayer

```python
from tgraphx.layers.gin import TensorGINLayer

layer = TensorGINLayer(
    in_channels=16,
    out_channels=32,
    spatial_rank=2,
    train_eps=True,           # learnable ε
)
out = layer(x, edge_index)
```

## Example: 3×3 image patch graph

```python
import torch
from tgraphx import build_grid_graph, image_to_patches
from tgraphx.layers.gat import TensorGATLayer

images  = torch.randn(2, 3, 8, 8)                       # [B, C, H, W]
patches = image_to_patches(images, patch_size=4)         # [2, 4, 3, 4, 4]
x       = patches[0]                                     # [4, 3, 4, 4] — 4 nodes
ei      = build_grid_graph(2, 2, directed=False, self_loops=True)

layer   = TensorGATLayer(in_channels=3, out_channels=8, num_heads=2, spatial_rank=2)
out     = layer(x, ei)                                   # [4, 8, 4, 4]
```

## See also

- [Volumetric 3-D support](volumetric_3d.md)
- [Edge features](edge_features.md)
- [Graph builders](graph_builders.md)
- [Factories](factories.md)

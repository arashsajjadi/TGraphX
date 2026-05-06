# Edge Features

## Supported formats

| Format | Shape | Layers |
|---|---|---|
| Vector | `[E, D_e]` | GAT (attention bias), SAGE (vector bias), GIN (broadcast bias) |
| 2-D spatial | `[E, C_e, H, W]` | ConvMessagePassing, SAGE (concat), GIN (1×1 conv) |
| 3-D volumetric | `[E, C_e, D, H, W]` | ConvMessagePassing, SAGE (concat), GIN |

## ConvMessagePassing

```python
from tgraphx.layers.conv_message import ConvMessagePassing

layer = ConvMessagePassing(
    in_shape=(16, 8, 8),
    out_shape=(32, 8, 8),
    use_edge_features=True,
)
out = layer(x, edge_index, edge_features=edge_feats)
# edge_feats: [E, 16, 8, 8]  (channels must match node channels)
```

> ConvMessagePassing requires `edge_features.shape[1] == in_shape[0]` because
> source, destination, and edge tensors are concatenated along the channel axis.
> Use SAGE or GIN for an arbitrary `edge_dim`.

## TensorGATLayer

Edge features add an additive bias to the attention logit per `(edge, head)`.

```python
layer = TensorGATLayer(
    in_channels=16, out_channels=32, num_heads=4,
    use_edge_features=True, edge_dim=8,
    spatial_rank=2,
)
edge_feats = torch.randn(E, 8)             # vector [E, D_e]
out = layer(x, edge_index, edge_features=edge_feats)
```

Spatial edge tensors `[E, D_e, H_e, W_e]` are mean-pooled to vectors before
the bias projection; spatial dimensions need not match node spatial dims.

## TensorGraphSAGELayer

```python
# Vector edge features: added as per-edge channel bias
layer = TensorGraphSAGELayer(
    in_channels=16, out_channels=32, spatial_rank=2,
    use_edge_features=True, edge_dim=8,
    edge_features_kind="vector",   # "vector" | "spatial"
)
# Spatial edge features: concatenated to source before W_neigh
layer = TensorGraphSAGELayer(
    in_channels=16, out_channels=32, spatial_rank=2,
    use_edge_features=True, edge_dim=8,
    edge_features_kind="spatial",
)
```

## TensorGINLayer (GINEConv)

```python
layer = TensorGINLayer(
    in_channels=16, out_channels=32, spatial_rank=2,
    use_edge_features=True, edge_dim=8,
    edge_features_kind="vector",   # or "spatial"
)
# With vector: φ is nn.Linear(edge_dim, in_channels) + broadcast
# With spatial: φ is 1×1 Conv2d(edge_dim, in_channels)
out = layer(x, edge_index, edge_features=edge_feats)
```

## Using edge_features with Graph / GraphBatch

```python
from tgraphx import Graph

g = Graph(
    node_features,
    edge_index=edge_index,
    edge_features=edge_feats,   # [E, ...]
)
# Pass through a layer
out = layer(g.node_features, g.edge_index, edge_features=g.edge_features)
```

## See also

- [Weighted edges](weighted_edges.md)
- [Graph API](graph_basics.md)

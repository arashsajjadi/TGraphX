# Vector Node Features

For flat vector node features `[N, D]`, TGraphX provides two layers:

## LinearMessagePassing

A simple message-passing layer using linear projections.

```python
from tgraphx.layers.base import LinearMessagePassing

layer = LinearMessagePassing(
    in_shape=(32,),       # (D,)
    out_shape=(64,),
    aggr="sum",           # "sum" | "mean" | "max"
    use_edge_features=False,
    dropout_prob=0.0,
    residual=False,
)
out = layer(x, edge_index)   # [N, 64]
```

## AttentionMessagePassing (legacy)

Spatial-gating attention. Supports both vector and 2-D spatial inputs.
Retained for backward compatibility; use `TensorGATLayer` for true GAT.

```python
from tgraphx.layers.attention_message import AttentionMessagePassing

layer = AttentionMessagePassing(in_shape=(32,), out_shape=(64,))
out   = layer(x, edge_index)
```

> **Note:** `AttentionMessagePassing` does NOT implement the true GAT
> algorithm (Veličković et al. 2018). It applies per-edge sigmoid gating.

## NodeClassifier model

```python
from tgraphx.models import NodeClassifier

model = NodeClassifier(
    in_shape=(32,),
    hidden_shape=(64,),
    num_classes=5,
    num_layers=3,
    aggr="mean",
)
logits = model(x, edge_index)   # [N, 5]
```

## Factory API for vector features

```python
from tgraphx import build_model, make_layer

# Single layer
layer = make_layer("linear", in_shape=(32,), out_shape=(64,), aggr="mean")

# Full task model
model = build_model(
    task="node_classification",
    layer="linear",
    in_shape=(32,),
    hidden_shape=(64,),
    num_layers=3,
    num_classes=5,
)
out = model(x, edge_index)   # [N, 5]
```

## Limitations

- Only `"linear"` and `"legacy_attention"` support vector `(D,)` in_shape.
- `"conv"`, `"gat"`, `"sage"`, `"gin"` require spatial shapes `(C, H, W)`
  or `(C, D, H, W)`.
- For vector features with true GAT or SAGE/GIN semantics, use the standard
  PyG/DGL vector implementations directly.

## See also

- [Spatial 2-D GNN layers](spatial_tensor_gnn.md)
- [Factories](factories.md)
</content>

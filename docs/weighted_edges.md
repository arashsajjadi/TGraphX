# Weighted Edges

`edge_weight` is a 1-D float tensor of shape `[E]` that scales each message
**after** any edge-feature projection and **before** scatter aggregation.
It works across all four GNN layer families and the base message-passing class.

## Graph object

```python
from tgraphx import Graph

g = Graph(
    node_features,
    edge_index=edge_index,
    edge_weight=torch.rand(E),    # [E]
)
```

## Passing to a layer

```python
out = layer(g.node_features, g.edge_index, edge_weight=g.edge_weight)
```

## Per-layer semantics

| Layer | When weight is applied |
|---|---|
| `ConvMessagePassing` | After `self.conv(msg_input)`, before scatter |
| `TensorGATLayer` | After attention weighting, before destination sum |
| `TensorGraphSAGELayer` | After neighbour projection, before scatter |
| `TensorGINLayer` | Before scatter sum |

## Self-loop weights in TensorGATLayer

When `add_self_loops=True`, self-loop edge weights default to `1.0`.

## Normalisation example

```python
# Degree-normalised adjacency (simple GCN-style)
from tgraphx import build_grid_graph

ei = build_grid_graph(5, 5, directed=False, self_loops=True)
# Count in-degree for each destination
deg = torch.zeros(25).scatter_add_(0, ei[1], torch.ones(ei.size(1)))
ew  = 1.0 / deg[ei[1]].clamp(min=1)   # [E]

out = layer(x, ei, edge_weight=ew)
```

## GraphBatch with edge weights

All graphs in a `GraphBatch` must either all provide `edge_weight` or none.

```python
from tgraphx import Graph, GraphBatch

g1 = Graph(nf1, edge_index=ei1, edge_weight=ew1)
g2 = Graph(nf2, edge_index=ei2, edge_weight=ew2)
batch = GraphBatch([g1, g2])
out = layer(batch.node_features, batch.edge_index, edge_weight=batch.edge_weight)
```

## See also

- [Edge features](edge_features.md)
- [Graph API](graph_basics.md)

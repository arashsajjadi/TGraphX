# Batching

`GraphBatch` packs multiple `Graph` objects into a single super-graph so that
existing GNN layers can process an entire batch with one forward call.

## Creating a batch

```python
from tgraphx import Graph, GraphBatch

g1 = Graph(node_features_1, edge_index_1, edge_weight=ew1)
g2 = Graph(node_features_2, edge_index_2, edge_weight=ew2)
g3 = Graph(node_features_3, edge_index_3, edge_weight=ew3)

batch = GraphBatch([g1, g2, g3])
```

**Requirements:**
- All graphs must share the same per-node feature shape (`node_features.shape[1:]`).
- If any graph has `edge_weight` / `edge_features` / `edge_labels`, all graphs
  with edges must provide them (all-or-none per field).

## Batch attributes

```python
batch.node_features   # [N_total, ...]  — concatenated
batch.edge_index      # [2, E_total]    — per-graph node offsets applied
batch.edge_weight     # [E_total]       — or None
batch.edge_features   # [E_total, ...]  — or None
batch.batch           # [N_total]       — graph index per node: 0,0,...,1,1,...
batch.graph_labels    # [B, ...]        — stacked, or None
batch.metadata        # list[Any]       — one entry per graph
batch.num_graphs      # B
batch.num_nodes       # N_total
batch.num_edges       # E_total
```

## Graph-level pooling

The `batch` tensor maps nodes to graphs and enables scatter-based readout:

```python
import torch

x = layer(batch.node_features, batch.edge_index)   # [N_total, out_dim]

# Mean readout per graph
num_graphs = batch.num_graphs
pooled = torch.zeros(num_graphs, x.size(-1))
pooled.index_add_(0, batch.batch, x)               # [B, out_dim]
```

The `build_model` factory handles this automatically for
`task="graph_classification"` and `task="graph_regression"`.

## GraphDataLoader

```python
from tgraphx import GraphDataset, GraphDataLoader

dataset = GraphDataset(graph_list)
loader  = GraphDataLoader(dataset, batch_size=8, shuffle=True)

for batch in loader:
    out = model(batch.node_features, batch.edge_index, batch=batch.batch)
```

## See also

- [Graph API](graph_basics.md)
- [Factories](factories.md)

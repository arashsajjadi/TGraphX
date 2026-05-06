# Graph API

## Graph

`Graph` is the single-graph container. It validates all inputs eagerly.

```python
from tgraphx import Graph
import torch

g = Graph(
    node_features,          # torch.Tensor  [N, ...]              required
    edge_index=None,        # LongTensor    [2, E]                optional
    edge_weight=None,       # Tensor        [E]                   optional
    edge_features=None,     # Tensor        [E, ...]              optional
    node_labels=None,       # Tensor        [N, ...]              optional
    edge_labels=None,       # Tensor        [E, ...]              optional
    graph_label=None,       # Tensor        any shape             optional
    metadata=None,          # dict                                optional
)
```

### Node feature shapes

| Shape | Description |
|---|---|
| `[N, D]` | Vector features (one D-dim vector per node) |
| `[N, C, H, W]` | 2-D spatial (image-like per-node feature map) |
| `[N, C, D, H, W]` | 3-D volumetric (MRI/CT-like per-node feature map) |

At least 2 dimensions are required (`[N, ...]`). Arbitrary ranks beyond
vector/2-D/3-D are not supported by the GNN layers.

### Properties

```python
g.num_nodes        # int
g.num_edges        # int
g.feature_shape    # tuple: node_features.shape[1:]
g.has_edges        # bool
g.has_edge_weight  # bool
g.has_edge_features# bool
g.device           # torch.device
g.dtype            # torch.dtype
```

### Topology operations

```python
g.add_self_loops(fill_value=1.0)   # in-place; raises if edge_labels set
g.remove_self_loops()              # in-place
g.make_undirected(reduce="mean")   # symmetrize; raises if edge_labels set
g.coalesce(reduce="mean")          # sort + merge duplicate edges
g.is_undirected()                  # bool
```

### Device / dtype

```python
g.to(device="cuda", dtype=torch.float16)
g.cpu()
g.cuda()
g.clone()    # deep copy: tensors + metadata
```

## GraphBatch

Packs a list of `Graph` objects into one super-graph for batched inference.

```python
from tgraphx import GraphBatch

graphs = [g1, g2, g3]
batch = GraphBatch(graphs)

batch.num_graphs         # 3
batch.node_features      # [N_total, ...]
batch.edge_index         # [2, E_total]  (node offsets applied per graph)
batch.batch              # [N_total] LongTensor: graph index per node
batch.graph_labels       # [3, ...] stacked, or None
```

All graphs must share the same per-node feature shape.

## edge_index format

TGraphX uses the COO sparse format: `edge_index` is a `[2, E]` LongTensor
where `edge_index[0]` holds source node indices and `edge_index[1]` holds
destination indices (messages flow `src → dst`).

## Graph utility functions

```python
from tgraphx import add_self_loops, remove_self_loops
from tgraphx import make_undirected, coalesce_edges, is_undirected

new_ei, new_w, new_ef = add_self_loops(edge_index, edge_weight, edge_features,
                                        num_nodes=N, fill_value=1.0)
new_ei, new_w, new_ef = make_undirected(edge_index, edge_weight, edge_features)
```

## See also

- [Edge weights](weighted_edges.md)
- [Edge features](edge_features.md)
- [Graph builders](graph_builders.md)
- [Batching](batching.md)

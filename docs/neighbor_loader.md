# NeighborLoader — Ergonomic Batch API

**Status:** Beta (v0.5.0+). `GraphMiniBatch` added v1.0.1.

## What problem this solves

Training GNNs on large graphs requires mini-batch sampling. `NeighborLoader`
samples a subgraph centred on `batch_size` "seed" nodes by following
neighbours for `len(fanouts)` hops.

The key challenge: after sampling, you have more nodes in the subgraph than
seed nodes. You must compute your loss **only on seed nodes**.
`GraphMiniBatch` handles this mapping automatically.

---

## 30-second minimal example

```python
from tgraphx import Graph, NeighborLoader, ConvMessagePassing
import torch
import torch.nn.functional as F
import torch.nn as nn

# Build a graph with labels.
N, C, H, W = 1000, 8, 6, 6
x = torch.randn(N, C, H, W)
edge_index = torch.randint(0, N, (2, 5000))
y = torch.randint(0, 4, (N,))
g = Graph(node_features=x, edge_index=edge_index, y=y)

# Loader yields GraphMiniBatch objects.
loader = NeighborLoader(g, fanouts=[15, 10], batch_size=64, shuffle=True, seed=42)

# Model.
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvMessagePassing((C, H, W), (16, H, W))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(16, 4)
    def forward(self, x, ei):
        return self.head(self.pool(self.conv(x, ei).relu()).flatten(1))

model = Model()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in loader:
    logits = model(batch.node_features, batch.edge_index)
    # seed_logits extracts logits for the supervision nodes only.
    # seed_y returns labels for those same nodes.
    loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)
    opt.zero_grad(); loss.backward(); opt.step()
    break
```

---

## API contract

### `NeighborLoader`

```python
NeighborLoader(
    graph,          # tgraphx.Graph with y/node_labels set
    fanouts,        # list[int], per-hop neighbor count, e.g. [15, 10]
    mask=None,      # BoolTensor[N] — restrict seeds to these nodes
    batch_size=32,  # seed nodes per batch
    shuffle=True,
    num_workers=0,
    seed=None,      # for reproducibility
    direction="in", # "in" (GraphSAGE default) or "out"
    drop_last=False,
)
```

### `GraphMiniBatch` attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `node_features` / `x` | `[N_sub, ...]` | All subgraph node features |
| `edge_index` | `[2, E_sub]` | Subgraph edge index |
| `edge_features` / `edge_attr` | `[E_sub, ...]` or `None` | Edge features |
| `edge_weight` | `[E_sub]` or `None` | Edge weights |
| `y` / `labels` | `[N_sub]` or `None` | All subgraph labels (if source graph has labels) |
| `seed_y` / `seed_labels` | `[K]` | Labels for seed nodes only |
| `seed_node_ids` | `[K]` | Global IDs of seed nodes |
| `seed_local_indices` | `[K]` | Local positions of seed nodes in subgraph |
| `input_nodes` | `[N_sub]` | Global IDs of all subgraph nodes |
| `batch_size` | `int` | Number of seed nodes (K) |
| `num_nodes` | `int` | Total nodes in subgraph (N_sub) |
| `num_edges` | `int` | Total edges in subgraph (E_sub) |
| `metadata` | `dict` | Sampling metadata |

### `GraphMiniBatch` methods

| Method | Description |
|--------|-------------|
| `seed_logits(logits)` | Extract logits for seed nodes from `[N_sub, C]` tensor |
| `all_logits(logits)` | Return logits unchanged (all subgraph nodes) |
| `loss(logits, loss_fn=F.cross_entropy)` | Compute supervised loss over seed nodes |
| `to(device)` | Move all tensors to device in place |
| `as_tuple()` | Return `(subgraph, seed_node_ids)` for legacy code |

---

## Shape contract

- `batch.node_features`: `[N_sub, *node_shape]` — `N_sub ≥ batch_size`
- `batch.seed_logits(logits)`: `[K, *]` where `K == batch_size`
- `batch.seed_y`: `[K]`
- `N_sub` includes seed nodes **plus their sampled neighbours** (multi-hop)

---

## Common mistakes

### Wrong: slicing logits directly

```python
# WRONG — assumes seed nodes are the first batch_size nodes (not guaranteed)
loss = F.cross_entropy(logits[:batch_size], batch.seed_y)

# CORRECT
loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)
```

### Wrong: ignoring node labels

```python
# WRONG — no labels on graph, batch.seed_y will raise
g = Graph(node_features=x, edge_index=edge_index)  # missing y=y
loader = NeighborLoader(g, ...)

# CORRECT
g = Graph(node_features=x, edge_index=edge_index, y=y)
```

### Wrong: using full-graph features via batch directly

```python
# WRONG — batch.node_features is only the sampled SUBGRAPH nodes
loss = F.cross_entropy(model(g.node_features, g.edge_index), y)

# CORRECT — use batch attributes, not the source graph
for batch in loader:
    logits = model(batch.node_features, batch.edge_index)
    loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)
```

---

## Backward compatibility

Old code that unpacks the loader as a tuple still works:

```python
for subgraph, seed_ids in loader:
    # subgraph: Graph object (same as batch.graph)
    # seed_ids: LongTensor of global seed node IDs
    ...
```

`GraphMiniBatch` implements `__iter__` to yield `(subgraph, seed_ids)` on
iteration, so tuple unpacking continues to work without code changes.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: Batch labels are unavailable` | Source `Graph` has no `y/labels` | `Graph(..., y=labels)` or `g.y = labels` |
| `ValueError: seed_local_indices could not be computed` | Custom sampler without metadata | Use `NeighborLoader` (sets sampling metadata automatically) |
| `AttributeError: tuple object has no attribute node_features` | Still using old `for subgraph, seeds` but then calling `.node_features` on the batch | Use `for batch in loader: batch.node_features` |

---

## Links

- Tutorial: `tutorials/tensor_node_classification_neighbor_loader.py`
- Example: `examples/easy_tensor_node_classification_no_torch.py`
- Tests: `tests/test_user_friendly_llm_snippets.py::TestNeighborLoaderReturnsGraphMiniBatch`

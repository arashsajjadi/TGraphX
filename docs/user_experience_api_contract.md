# TGraphX User Experience API Contract

This document defines the stable UX contracts for TGraphX v1.0+.

---

## 1. Graph Object Fields

### Constructor

```python
from tgraphx import Graph

# Canonical form (recommended):
g = Graph(
    node_features=x,       # required: Tensor[N, ...]
    edge_index=edge_index, # optional: LongTensor[2, E]
    y=y,                   # optional: labels (alias: labels, node_labels)
    edge_attr=edge_attr,   # optional: Tensor[E, ...] (alias: edge_features)
    edge_weight=w,         # optional: Tensor[E]
    graph_features=gl,     # optional: graph-level label (alias: graph_label)
    train_mask=mask,       # optional: BoolTensor[N]
    metadata={},           # optional: dict
)

# Old form (still works):
g = Graph(node_features=x, edge_index=edge_index, node_labels=y)
```

### Properties

| Property | Description |
|----------|-------------|
| `g.x` | `node_features` (alias) |
| `g.y` | `node_labels` (alias) |
| `g.labels` | `node_labels` (alias) |
| `g.edge_attr` | `edge_features` (alias) |
| `g.num_nodes` | Count of nodes |
| `g.num_edges` | Count of edges |
| `g.feature_shape` | `node_features.shape[1:]` |
| `g.num_node_features` | Total scalar features per node |
| `g.num_classes` | Inferred from integer `y` |
| `g.train_mask` | Training mask from metadata |
| `g.val_mask` | Validation mask from metadata |
| `g.test_mask` | Test mask from metadata |
| `g.device` | Device of `node_features` |

### Methods

| Method | Description |
|--------|-------------|
| `g.has_labels()` | Returns `True` if `y/node_labels` is set |
| `g.get_labels()` | Returns `node_labels` or raises helpful error |
| `g.with_labels(y)` | Returns shallow copy with `y` set |
| `g.to(device)` | Move all tensors to device |
| `g.clone()` | Deep copy |
| `g.validate()` | Re-run validation |

---

## 2. Loader Batch Object

`NeighborLoader` yields `GraphMiniBatch` objects:

```python
for batch in loader:
    batch.node_features          # [N_sub, ...]
    batch.edge_index             # [2, E_sub]
    batch.seed_y                 # [K] labels for seed nodes
    batch.seed_logits(logits)    # [K, C] extract seed logits
    batch.seed_node_ids          # [K] global IDs
    batch.seed_local_indices     # [K] local positions
    batch.batch_size             # K (int)
    batch.to(device)             # move in-place
```

Legacy tuple unpacking still works: `for subgraph, seeds in loader: ...`

---

## 3. Labels

- **Node labels** are stored in `Graph.node_labels` (also accessible as `.y` and `.labels`).
- **Graph labels** are stored in `Graph.graph_label` (also accessible as `.graph_features`).
- **Edge labels** are stored in `Graph.edge_labels`.
- The `GraphMiniBatch.seed_y` property extracts labels for seed nodes.

---

## 4. Seed Node Mapping

NeighborLoader samples `N_sub ≥ K` nodes per batch. Seed nodes (the `K`
training targets) are identified by:

- `batch.seed_node_ids`: global IDs of the K seed nodes.
- `batch.seed_local_indices`: local positions of seeds within the `N_sub`-node subgraph.
- `batch.seed_logits(model_output)`: extracts the K-row slice from a `[N_sub, C]` tensor.

**Never slice** `logits[:batch_size]` — seed nodes are not guaranteed to be the first nodes.

---

## 5. High-Level Run APIs

| Function | Returns | Notes |
|----------|---------|-------|
| `run_graph_generation(method=..., ...)` | `GenerationResult` | `.graphs`, `.metrics` |
| `run_evolutionary_optimization(optimizer=..., ...)` | `OptimizationResult` | `.metrics`, `.best_genome` |
| `run_graph_rl(algorithm=..., env=..., ...)` | `RLResult` | `.metrics`, `.history` |
| `tgx.easy.train_node_classifier(...)` | `EasyResult` | `.metrics`, `.model`, `.graph`, `.config` |

All results have:
- `.metrics` — final metric dict
- `.to_dict()` — JSON-serialisable dict
- `.summary()` — print summary

---

## 6. Error Message Policy

All user-facing errors must:

1. Describe what happened (not what Python crashed on).
2. State the likely cause.
3. Show the fix (exact code or property name).
4. Link to docs if applicable.

### Examples

**Bad:**
```
AttributeError: tuple object has no attribute 'node_features'
```

**Good:**
```
ValueError: Batch labels are unavailable because the source Graph has no y/labels field.
Create the graph with Graph(..., y=labels) or assign graph.y = labels.
```

---

## 7. Backward Compatibility Policy

- All v1.0.x public APIs are preserved through v1.x.x.
- New keyword-only parameters are additive only.
- Old positional APIs keep working.
- Old tuple-style loader outputs keep working (via `GraphMiniBatch.__iter__`).
- No silent dtype or device coercion (raise `ValueError` with location).
- Deprecation warnings are added ≥2 minor versions before removal.

---

## 8. Naming Conventions

| Pattern | Examples |
|---------|---------|
| `list_*()` | `list_graph_rl_algorithms()`, `list_kg_models()`, `list_graph_generation_methods()` |
| `run_*()` | `run_graph_rl()`, `run_graph_generation()`, `run_evolutionary_optimization()` |
| `make_*()` | `make_graph_env()`, `make_layer()`, `make_tensor_node_classifier()` |
| `*Result` | `RLResult`, `EasyResult`, `GenerationResult`, `OptimizationResult` |
| `write_*_report()` | dashboard metadata writers |

# Graph algorithms

`tgraphx.algorithms` (Beta, v0.3.2) provides a focused set of
pure-PyTorch graph algorithms used by GNN training workflows.

**This is not a NetworkX replacement.**  Only algorithms that appear
naturally in GNN training — component analysis, BFS traversal, shortest
paths, and structural features — are included.  For general-purpose
graph analysis use NetworkX directly; no mandatory dependency on it is
introduced.

## Import

```python
from tgraphx.algorithms import (
    # Connectivity
    connected_components,
    weakly_connected_components,
    is_connected,
    number_connected_components,
    # Traversal
    bfs_layers,
    bfs_edges,
    shortest_path_length,
    # Structural
    degree,
    degree_features,
)
```

---

## Connectivity

### `connected_components`

```python
from tgraphx.algorithms import connected_components
labels = connected_components(edge_index, num_nodes=5)
# labels: LongTensor[5] in [0, K) where K is the number of components
```

Treats the graph as **undirected**: edge `(u, v)` connects `u` and `v`
regardless of direction.

| Argument | Default | Description |
|----------|---------|-------------|
| `edge_index` | — | `LongTensor[2, E]` |
| `num_nodes` | `None` | Inferred from `edge_index.max() + 1` when `None` |

**Returns:** `LongTensor[num_nodes]` of component labels in `[0, K)`.
Labels are deterministic: the component containing the smallest node id
is labelled 0.

**Algorithm:** Iterative `min`-label propagation — O(diameter × E)
time, O(N + E) memory.  Converges in at most `2 × num_nodes` iterations.

### `weakly_connected_components`

Same as `connected_components` but the name explicitly documents that
direction is ignored.  Use this when the input is a directed graph.

### `is_connected`

```python
assert is_connected(edge_index, num_nodes=5)
```

Returns `True` iff the undirected graph has exactly one component.
Empty graph (`num_nodes == 0`) returns `False`; single node returns `True`.

### `number_connected_components`

```python
K = number_connected_components(edge_index, num_nodes=5)
```

Returns the integer count of components.

---

## Traversal

### `bfs_layers`

```python
layers = bfs_layers(edge_index, source=0, num_nodes=6, max_hops=3)
# layers[0] = [0], layers[1] = neighbors of 0, ...
```

Returns a list of `LongTensor`s.  `layers[k]` contains all nodes whose
unweighted distance from `source` is exactly `k`.

| Argument | Default | Description |
|----------|---------|-------------|
| `edge_index` | — | `LongTensor[2, E]` (directed) |
| `source` | — | Source node id |
| `num_nodes` | `None` | Inferred when `None` |
| `max_hops` | `None` | Stop after `max_hops` layers |

**Direction note:** BFS follows directed edges.  To get undirected BFS,
apply `tgraphx.transforms.ToUndirected` first.

### `bfs_edges`

```python
bfs = bfs_edges(edge_index, source=0, num_nodes=6)
# bfs: LongTensor[2, M] — (predecessor, child) pairs in BFS order
```

Returns the BFS spanning-tree edges.  Each reachable non-source node
appears exactly once as a child.

### `shortest_path_length`

```python
dist = shortest_path_length(edge_index, source=0, num_nodes=6)
# dist[v] = shortest path length from source to v; -1 if unreachable
```

| Argument | Default | Description |
|----------|---------|-------------|
| `edge_index` | — | Directed `LongTensor[2, E]` |
| `source` | — | Source node id |
| `num_nodes` | `None` | Inferred when `None` |
| `max_hops` | `None` | Nodes beyond `max_hops` report `-1` |

**Returns:** `LongTensor[num_nodes]`.  Entry at `source` = 0; unreachable
nodes = -1.

---

## Structural

### `degree`

```python
from tgraphx.algorithms import degree

out_d = degree(edge_index, num_nodes=5, mode="out")
in_d  = degree(edge_index, num_nodes=5, mode="in")
total = degree(edge_index, num_nodes=5, mode="both")
```

| Argument | Default | Description |
|----------|---------|-------------|
| `edge_index` | — | `LongTensor[2, E]` |
| `num_nodes` | `None` | Inferred when `None` |
| `mode` | `"out"` | `"out"`, `"in"`, or `"both"` (in + out) |
| `dtype` | `torch.long` | Output dtype |

**Notes:**
- Self-loops contribute 1 to both out-degree and in-degree.
- Isolated nodes have degree 0.
- O(E) time, O(N) memory.

### `degree_features`

```python
from tgraphx.algorithms import degree_features

feats = degree_features(edge_index, num_nodes=5, log_scale=False)
# feats: LongTensor[N, 3] — columns: out_degree, in_degree, total_degree
# With log_scale=True: FloatTensor[N, 3] of log1p(degree)
```

Useful for prepending structural features to node feature matrices
before or between GNN layers.

---

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: edge_index must have shape [2, E]` | Wrong tensor shape | Pass `[2, E]` not `[E, 2]` |
| `ValueError: edge_index entries out of range` | Node id ≥ num_nodes | Check `num_nodes` or infer it |
| `ValueError: source=X out of range` | Source not in graph | Ensure `0 ≤ source < num_nodes` |
| `ValueError: mode must be 'out', 'in', or 'both'` | Bad mode string | Use one of the three |

## Limitations

- All algorithms are pure-PyTorch utilities for small-to-medium GNN
  workflows, not billion-edge production tools.
- BFS follows directed edges; no built-in undirected mode (use the
  `ToUndirected` transform first).
- `connected_components` treats the graph as undirected.  There is no
  strongly-connected-component algorithm.
- Complexity notes are accurate for sparse graphs; O(diameter × E)
  propagation converges quickly on real-world graphs.

## Related

- Tests: `tests/test_algorithms.py`, `tests/test_graph_utils.py`
- Examples: `examples/graph_algorithms_demo.py`
- Architecture: `docs/architecture.md`

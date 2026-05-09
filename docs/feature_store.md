# Feature store

`tgraphx.feature_store` provides two feature stores for large-scale
node-feature workflows:

- `InMemoryFeatureStore` — pure-PyTorch in-memory storage; fast
  random access.
- `MemmapFeatureStore` — disk-backed via `numpy.load(..., mmap_mode='r')`;
  supports features larger than RAM.

Both stores share the same API:

```python
store = InMemoryFeatureStore()
store.put("x", all_node_features)          # write all at once
feats = store.get("x", ids=sampled_ids)    # read only what you need
```

## Integration with NeighborLoader

Pass a feature store to `NeighborLoader` to fetch features for only the
sampled nodes (no full materialisation):

```python
from tgraphx import NeighborLoader, InMemoryFeatureStore

store = InMemoryFeatureStore()
store.put("x", torch.randn(num_nodes, 256))  # large feature matrix

loader = NeighborLoader(
    graph,                    # graph carries only structure
    fanouts=[15, 10],
    feature_store=store,      # features fetched per-batch
    feature_name="x",
    seed=0,
)
for sub, seeds in loader:
    # sub.node_features == store.get("x", ids=sampled_ids)
    pass
```

## MemmapFeatureStore

For features too large to fit in RAM:

```python
from tgraphx import MemmapFeatureStore

store = MemmapFeatureStore(root="/data/features")
store.put("x", large_tensor)           # writes .npy to disk
feats = store.get("x", ids=row_ids)    # memory-mapped read
```

Security: uses `numpy.load(..., allow_pickle=False)` — no unsafe pickle.

## Stability

Both stores are **Beta** in v0.5.0+. The `MemmapFeatureStore` requires
NumPy (`pip install numpy`).

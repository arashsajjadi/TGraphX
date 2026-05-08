# Negative sampling

`tgraphx.sampling_negative` (Beta, v0.3.2) provides link-prediction
building blocks that never require a heavy optional dependency.

## Import

```python
from tgraphx import (
    negative_sampling,
    structured_negative_sampling,
    batched_negative_sampling,
    hard_negative_sampling,
)
# or from the submodule:
from tgraphx.sampling_negative import negative_sampling
```

## `negative_sampling`

Uniformly sample negative edges absent from `edge_index`.

```python
import torch
from tgraphx import negative_sampling

edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
neg = negative_sampling(edge_index, num_nodes=5, num_neg_samples=4, seed=0)
# neg: LongTensor[2, 4] — no (u,v) appears in edge_index, no self-loops
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `edge_index` | `LongTensor[2, E]` | — | Positive edges |
| `num_nodes` | `int` | — | Total node count |
| `num_neg_samples` | `int \| None` | `None` | Target count; defaults to `E` |
| `method` | `"sparse" \| "dense"` | `"sparse"` | Rejection sampling or dense enumeration |
| `force_undirected` | `bool` | `False` | When `True`, excludes both `(u,v)` and `(v,u)` if either is positive |
| `seed` | `int \| None` | `None` | Optional RNG seed; no global RNG side effects |

**Returns:** `LongTensor[2, num_neg]` on the same device as `edge_index`.

**Method notes:**
- `"sparse"` (default) — rejection sampling up to 8 × target candidates.
  Safe for all graph sizes.  May return fewer than `num_neg_samples` for
  nearly-complete graphs (documented behaviour; the caller should check
  the returned size).
- `"dense"` — enumerates all absent edges; only safe for
  `num_nodes < 5_000` (raises `ValueError` otherwise).

**Complexity:** `"sparse"` O(E + target) time, O(E) memory.
`"dense"` O(N²) time and memory.

## `structured_negative_sampling`

For each positive edge `(i, j)`, sample a node `k` such that `(i, k)` is
not a positive edge.  Returns the aligned triplet `(i, j, k)`.

```python
i, j, k = structured_negative_sampling(edge_index, num_nodes=5, seed=0)
# i, j match edge_index endpoints; (i[t], k[t]) is never a positive edge
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `edge_index` | `LongTensor[2, E]` | — | Positive edges |
| `num_nodes` | `int` | — | Node count |
| `contains_neg_self_loops` | `bool` | `True` | Allow `k == i` (self-loop as valid negative) |
| `seed` | `int \| None` | `None` | RNG seed |

**Returns:** tuple of three `LongTensor[E]` on the same device as input.

**Raises** `RuntimeError` if a source node is connected to every other
node (no valid `k` exists).

## `batched_negative_sampling`

Negative sampling that respects `GraphBatch` boundaries.

```python
from tgraphx import batched_negative_sampling
neg = batched_negative_sampling(edge_index, batch, num_neg_samples=4, seed=0)
# No negative edge crosses a graph boundary in the batch
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `edge_index` | `LongTensor[2, E]` | — | Positive edges (global node ids) |
| `batch` | `LongTensor[N]` | — | Graph-batch assignment vector |
| `num_neg_samples` | `int \| None` | `None` | Target per graph; defaults to positives per graph |
| `method` | `str` | `"sparse"` | Same as `negative_sampling` |
| `force_undirected` | `bool` | `False` | Per-graph undirected constraint |
| `seed` | `int \| None` | `None` | Seed |

**Returns:** `LongTensor[2, total_neg]` with global node ids.

## `hard_negative_sampling`

Sample negatives that have high embedding similarity (harder for a
link-prediction model to reject).  Does **not** enumerate all O(N²) pairs.

```python
from tgraphx import hard_negative_sampling
neg = hard_negative_sampling(
    edge_index, node_embeddings, num_nodes=N,
    num_neg_samples=4, candidate_pool_size=512, seed=0,
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `edge_index` | `LongTensor[2, E]` | — | Positive edges |
| `node_embeddings` | `FloatTensor[N, D]` | — | Node representations |
| `num_nodes` | `int \| None` | `None` | Inferred when `None` |
| `num_neg_samples` | `int \| None` | `None` | Defaults to `E` |
| `candidate_pool_size` | `int` | `1024` | Random pairs to score; controls memory |
| `exclude_self_loops` | `bool` | `True` | Exclude self-loop candidates |
| `force_undirected` | `bool` | `False` | Block reverse positives |
| `seed` | `int \| None` | `None` | RNG seed |
| `method` | `"cosine" \| "dot"` | `"cosine"` | Similarity measure |

**Returns:** `LongTensor[2, num_neg]` — highest-scoring valid negative
candidates.  Issues a `UserWarning` if fewer than requested are found.

**Approximation note:** The function scores a random pool of
`candidate_pool_size` candidates, not all N² pairs.  Increase
`candidate_pool_size` to improve recall of the truly hardest pairs.

## Invariants (verified by tests)

- No positive edge ever appears in the output.
- No self-loop appears unless `contains_neg_self_loops=True` in
  `structured_negative_sampling`.
- No duplicate output edge (for `negative_sampling` and
  `hard_negative_sampling`).
- `batched_negative_sampling` never produces inter-graph negatives.
- All functions are deterministic for a given `seed`.
- No global RNG state is modified.
- Output is on the same device as the input `edge_index`.

## Limitations

- `negative_sampling` with `method="sparse"` may return fewer than
  `num_neg_samples` for dense graphs (no error is raised; check the
  returned tensor size).
- `hard_negative_sampling` approximates hard negatives — it is not a
  guaranteed top-K over all possible negatives.
- For large graphs, `method="sparse"` is always safer than
  `method="dense"`.

## Related

- Tests: `tests/test_negative_sampling.py`, `tests/test_hard_negative_sampling.py`
- Examples: `examples/negative_sampling_demo.py`
- Architecture: `docs/architecture.md`

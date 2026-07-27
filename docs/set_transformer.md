# SetTransformer: learned implicit relations (v1.5.0)

`SetTransformerModel` is the TGraphX family for the **learned implicit
relations** regime: pairwise interactions are inferred from node content
by global self-attention instead of being supplied as `edge_index`.

- **Relation-aware**: every node attends to every other node in its
  graph/set; this is not a pooling-only (DeepSets-style) model.
- **Explicit-input-topology-blind**: a supplied `edge_index` is never
  consumed.  `model.topology_source == "learned_implicit"`.
- Distinct from `TensorGATLayer` (attends only over supplied edges) and
  from `tgraphx.learned_graph` (constructs an explicit edge set first).

## Quick start

```python
import torch
from tgraphx import SetTransformerModel

model = SetTransformerModel(
    task="graph_classification",
    in_shape=(13, 32, 32),   # tensor-valued nodes; (D,) and (C,D,H,W) also work
    embed_dim=64,
    num_layers=2,
    num_heads=4,
    dropout=0.0,             # explicit; never silently nonzero
    num_classes=18,
)

x = torch.randn(9, 13, 32, 32)                 # 9 nodes across 2 graphs
batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
logits = model(x, None, batch=batch)           # [2, 18]
```

Or through the factory (`family=` is an alias for `layer=`):

```python
from tgraphx import build_model

model = build_model(
    task="graph_classification", family="set_transformer",
    in_shape=(13, 32, 32), hidden_shape=(64,),   # hidden_shape=(embed_dim,)
    num_layers=2, num_classes=18, heads=4, dropout=0.0,
)
```

## Architecture

1. **Shared node encoder** → `[N, embed_dim]` tokens.
   - vector `(D,)` input: 2-layer MLP;
   - 2-D spatial `(C, H, W)`: the package `CNNEncoder` with explicit
     defaults `{num_layers: 3, hidden_channels: 32, dropout_prob: 0.0,
     use_batchnorm: True, use_residual: False, pool_layers: 1}`
     (override via `encoder_config={...}`);
   - 3-D volumetric `(C, D, H, W)`: a small Conv3d encoder;
   - or pass any custom module via `encoder=` mapping
     `[N, *in_shape] → [N, embed_dim]`.
2. **`num_layers` pre-LN self-attention blocks** (`SetAttentionBlock`)
   with key-padding masks — permutation-*equivariant*; nodes only attend
   within their own graph.
3. **Permutation-invariant readout**: `pooling="attention"` (default;
   pooling by multi-head attention with `num_seeds` learned seed
   queries), or `"mean"` / `"sum"` / `"max"`.
4. Linear head sized by `num_classes` / `out_dim`.

Variable-size sets are handled natively: the flat TGraphX batch
convention (`[N, ...]` features + `batch` vector from `GraphBatch`) is
densified internally to `[B, M, embed_dim]` with a padding mask, so
`GraphDataLoader`, `fit`, `evaluate`, checkpoints, and the experiment
runner work unchanged. Tasks: `node_classification`, `node_regression`
(per-node outputs, no readout), `graph_classification`,
`graph_regression`.

## The `edge_index` contract

`forward(x, edge_index=None, edge_features=None, edge_weight=None,
batch=None)` accepts the standard pipeline arguments, but `edge_index`,
`edge_features`, and `edge_weight` are **never consumed**:

| `on_edge_index=` | Behaviour when `edge_index is not None` |
|---|---|
| `"warn"` (default) | emits `TopologyIgnoredWarning` once per instance |
| `"ignore"` | silent (the choice is recorded in `config()`) |
| `"error"` | raises `ValueError` |

## Configuration, repr, and serialization

All regularization is explicit: `dropout` and `attention_dropout`
default to `0.0` and appear in `repr(model)` and `model.config()`.

```python
cfg = model.config()                      # exact constructor config + family/topology metadata
clone = SetTransformerModel.from_config(cfg)
clone.load_state_dict(model.state_dict())  # exact reconstruction

from tgraphx import save_checkpoint, load_checkpoint
save_checkpoint(model, optimizer, epoch=10, path="ckpt.pt", config=model.config())
```

`encode_nodes(x, batch)` exposes the permutation-equivariant per-node
embeddings (`[N, embed_dim]`) before the readout.

## When to use it

Use `set_transformer` when entities vary in number and no trusted
`edge_index` exists. In the revised PASTIS-R validation experiments,
learned implicit relations were the strongest tested relation mode
(S2-only validation macro-F1 0.7023 vs 0.6326 for the corrected
explicit-topology TGraphX), while supplied topology still carried real
signal in matched comparisons. See
[tensor_relational_platform.md](tensor_relational_platform.md) for the
full regime map, provenance, and the honest interpretation of those
numbers (a set-attention win is a platform/family-level result, not a
message-passing result).

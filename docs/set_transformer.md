# TGraphXSetAttention: learned implicit relations

`TGraphXSetAttention` (canonical name since 1.5.1; `SetTransformerModel`
remains a permanent compatibility alias for the same class) is the
TGraphX family for the **learned implicit relations** regime: global
content-based relation learning over tensor-valued entities without
requiring a supplied edge graph. Pairwise interactions are inferred from
node content by global self-attention instead of being supplied as
`edge_index`.

- **Relation-aware**: every node attends to every other node in its
  graph/set; this is not a pooling-only (DeepSets-style) model.
- **Explicit-input-topology-blind**: a supplied `edge_index` is never
  consumed.  `model.topology_source == "learned_implicit"`.
- Distinct from `TensorGATLayer` (attends only over supplied edges) and
  from `tgraphx.learned_graph` (constructs an explicit edge set first).
- Paper/table label: **TGraphX-SetAttn**.  Stable machine name
  (`model_family`, factory, configs): `"set_transformer"`.

## Quick start

```python
import torch
from tgraphx import TGraphXSetAttention   # or: SetTransformerModel (alias)

model = TGraphXSetAttention(
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

Or through the factory (`family=` is an alias for `layer=`;
`"set_transformer"`, `"set_attention"`, and `"tgraphx_set_attention"`
all resolve to this same family):

```python
from tgraphx import build_model

model = build_model(
    task="graph_classification", family="tgraphx_set_attention",
    in_shape=(13, 32, 32), hidden_shape=(64,),   # hidden_shape=(embed_dim,)
    num_layers=2, num_classes=18, heads=4, dropout=0.0,
)
```

## Architecture

1. **Shared node encoder** → `[N, embed_dim]` tokens.
   - vector `(D,)` input: 2-layer MLP;
   - 2-D spatial `(C, H, W)`: selected by
     `encoder_config={"architecture": ...}` —
     `"cnn"` (default): the package `CNNEncoder` with explicit defaults
     `{num_layers: 3, hidden_channels: 32, dropout_prob: 0.0,
     use_batchnorm: True, use_residual: False, pool_layers: 1}`;
     `"strided"`: `StridedConvEncoder` — strided channel-growing 3×3
     convolutions (default schedule 32→64→128 via
     `hidden_channels`/`channel_multiplier`, or an explicit
     `channel_schedule` list), BatchNorm, no residual, adaptive average
     pool, linear projection;
   - 3-D volumetric `(C, D, H, W)`: a small Conv3d encoder;
   - or pass any custom module via `encoder=` mapping
     `[N, *in_shape] → [N, embed_dim]`.
2. **`num_layers` self-attention blocks** (`SetAttentionBlock`) with
   key-padding masks — permutation-*equivariant*; nodes only attend
   within their own graph.  `norm_order="pre"` (default) or `"post"`
   (the `torch.nn.TransformerEncoderLayer` convention);
   `activation="gelu"` (default) or `"relu"`.
3. **Permutation-invariant readout**: `pooling="attention"` (default;
   pooling by multi-head attention with `num_seeds` learned seed
   queries; its attention-weight dropout can be decoupled from the
   blocks via `pool_attention_dropout`), or `"mean"` / `"sum"` /
   `"max"`.
4. Head sized by `num_classes` / `out_dim`: a single linear layer, or
   `Linear → ReLU → Linear` when `head_hidden_dim` is set.

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

All regularization is explicit: `dropout`, `attention_dropout`, and
`pool_attention_dropout` default to inactive and appear in `repr(model)`
and `model.config()`.

```python
cfg = model.config()                      # exact constructor config + family/topology metadata
clone = TGraphXSetAttention.from_config(cfg)
clone.load_state_dict(model.state_dict())  # exact reconstruction

from tgraphx import save_checkpoint, load_checkpoint
save_checkpoint(model, optimizer, epoch=10, path="ckpt.pt", config=model.config())
```

Configs serialized by earlier package versions load unchanged — fields
introduced in 1.5.1 (`norm_order`, `activation`,
`pool_attention_dropout`, `head_hidden_dim`, encoder `architecture`)
take their defaults, which reproduce the earlier architecture exactly.

`encode_nodes(x, batch)` exposes the permutation-equivariant per-node
embeddings (`[N, embed_dim]`) before the readout.

## The evaluated reference configuration

The exact set-attention architecture evaluated in the TGraphX experiment
program is available as an explicit configuration — nothing about it is
a hidden default:

```python
cfg = TGraphXSetAttention.reference_config(in_shape=(13, 32, 32), num_classes=18)
model = TGraphXSetAttention(**cfg)
# StridedConvEncoder 32→64→128 · 2 post-LN ReLU blocks (dropout 0.1)
# · single-seed attention pooling (no attention-weight dropout) · linear head

# Checkpoints saved from the torch-primitives reference layout load
# via a documented, strict key mapping:
model.load_state_dict(
    TGraphXSetAttention.map_reference_state_dict(reference_state_dict),
    strict=True,
)
# or in one step:
model = TGraphXSetAttention.from_reference_state_dict(
    reference_state_dict, in_shape=(13, 32, 32), num_classes=18)
```

Parity of this configuration with the completed experiment (identical
parameter count, strict checkpoint load, identical predictions on the
full validation split for all five seeds) is documented in
[reports/SET_ATTENTION_REFERENCE_PARITY.md](reports/SET_ATTENTION_REFERENCE_PARITY.md).

## When to use it

Use this family when entities vary in number and no trusted
`edge_index` exists. In the revised PASTIS-R validation experiments,
learned implicit relations were the strongest tested relation mode
(S2-only validation macro-F1 0.7023 vs 0.6326 for the corrected
explicit-topology TGraphX), while supplied topology still carried real
signal in matched comparisons. See
[tensor_relational_platform.md](tensor_relational_platform.md) for the
full regime map, provenance, and the honest interpretation of those
numbers (a set-attention win is a platform/family-level result, not a
message-passing result).

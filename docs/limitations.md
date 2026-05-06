# Limitations

This document records known limitations honestly.
See [Performance](performance.md) for performance-specific constraints.

## GNN layer constraints

| Limitation | Status | Workaround |
|---|---|---|
| Arbitrary-rank tensors | Not supported | Use vector `(D,)`, 2-D `(C,H,W)`, or 3-D `(C,D,H,W)` only |
| Vector features in GAT/SAGE/GIN/Conv | Not supported | Use `"linear"` or `"legacy_attention"` for vectors |
| Graph Transformers (global attention + positional encodings) | Not implemented | — |
| Heterogeneous / temporal graphs | Not implemented | — |
| Per-channel / per-pixel attention in GAT | Not implemented (scalar only) | — |
| GAT chunked forward | Deferred | Softmax requires all edge scores |
| SAGE / GIN chunked forward | Deferred | See [performance.md](performance.md) |
| 3-D support in `AttentionMessagePassing` | Not supported | Use `TensorGATLayer(spatial_rank=3)` |

## Training framework

| Limitation | Status |
|---|---|
| `train_epoch` function | Not implemented; write your own loop |
| `evaluate` function | Not implemented |
| `fit` function | Not implemented |
| `TensorBoardLogger` | Not implemented; use `torch.utils.tensorboard` directly |
| `MLflowLogger` | Not implemented; use `mlflow` client directly |
| Neighbor sampling (GraphSAINT / ClusterGCN) | Not implemented |

## AMP / precision

| Limitation | Status |
|---|---|
| GAT float16 autocast | `index_add_` requires matching dtypes; may fail under float16 |
| Universal AMP support | Not claimed; device- and op-dependent |

Use `bfloat16` or full precision for stable inference across all layers.

## Dashboard

| Limitation | Status |
|---|---|
| Incremental / tail CSV reads for huge runs | Deferred; full file read on cache miss |
| TensorBoard log reading | Not implemented |
| MLflow run reading | Not implemented |
| Remote / cloud logdir support | Not implemented |

## Patch helpers

Patch helpers raise `ValueError` if the image/volume dimensions are not
exactly covered by `patch_size` and `stride`. No automatic padding is applied.

## PyG / DGL compatibility

TGraphX is not a drop-in replacement for PyG or DGL. The API and layer
semantics differ; there are no conversion utilities.

## Version

These limitations apply to TGraphX 0.1.1. Deferred items may be addressed
in future releases. See [CHANGELOG.md](../CHANGELOG.md).

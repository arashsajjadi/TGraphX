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

| Feature | Status |
|---|---|
| `train_epoch` function | **Implemented** — see [training_utilities.md](training_utilities.md) |
| `evaluate` function | **Implemented** — see [training_utilities.md](training_utilities.md) |
| `fit` function | **Implemented** — thin `train_epoch` + `evaluate` wrapper; see [training_utilities.md](training_utilities.md) |
| `TensorBoardLogger` | **Implemented** — optional; requires `pip install tensorboard` or `pip install "tgraphx[tracking]"`; see [training_utilities.md](training_utilities.md) |
| `MLflowLogger` | Not implemented; use `mlflow` client directly (`pip install mlflow`) |
| Neighbor sampling (GraphSAINT / ClusterGCN) | Not implemented |

## AMP / precision

v0.2.2 hardened dtype handling.  The remaining constraints are:

| Item | Status | Notes |
|---|---|---|
| CPU bfloat16 autocast | ✅ Best-effort | All four spatial layers tested |
| CUDA float16 autocast | ⚠️ Best-effort | Fixed in v0.2.2; requires PyTorch ≥ 1.13 scatter float16 support |
| CUDA bfloat16 autocast | ⚠️ Best-effort | Requires Ampere+ GPU |
| MPS AMP | ❌ Not tested | MPS operator coverage varies by PyTorch version |
| Universal float16 CPU | ❌ Not supported | CPU float16 is not a recommended AMP dtype |
| `edge_weight` under autocast | ✅ Fixed v0.2.2 | `broadcast_edge_weight` now casts to message dtype |
| GAT `index_add_` dtype mismatch | ✅ Fixed v0.2.2 | Attention weights now cast to activation dtype |
| Attention softmax precision | ✅ Fixed v0.2.2 | `edge_softmax` upcasts to fp32 for max-shift + exp computation |

> **Recommended usage:** for stable training use bfloat16 on CUDA (Ampere+)
> or bfloat16 on CPU; use a `GradScaler` only for float16 CUDA training.
> Always call `.float()` on the loss before `.backward()` when inputs are
> low-precision.

## Deterministic algorithms

`set_seed(seed, deterministic=True)` enables deterministic cuDNN operation
but intentionally does **not** call `torch.use_deterministic_algorithms(True)`.
Several scatter/index operations used by TGraphX layers (e.g. `scatter_reduce_`
with `reduce='amax'` in `TensorGATLayer` attention and graph max-pooling) do
not have deterministic CUDA kernels and would raise `RuntimeError` at runtime.

Enabling deterministic cuDNN (`cudnn.deterministic = True`,
`cudnn.benchmark = False`) typically reduces GPU throughput.

## Checkpoint loading

`load_checkpoint` defaults to `weights_only=True` (safe mode).
Checkpoints created by `save_checkpoint` are fully compatible with this mode.
Checkpoints created by older code or third-party tools that pickle custom Python
objects may fail with a `RuntimeError` explaining how to use `weights_only=False`
for trusted legacy files.

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

These limitations apply to the current TGraphX release. Deferred items may
be addressed in future releases. See [CHANGELOG.md](../CHANGELOG.md).

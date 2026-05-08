# Limitations

This document records known limitations honestly.
See [Performance](performance.md) for performance-specific constraints.

## GNN layer constraints

| Limitation | Status | Workaround |
|---|---|---|
| Arbitrary-rank tensors | Not supported | Use vector `(D,)`, 2-D `(C,H,W)`, or 3-D `(C,D,H,W)` only |
| Vector features in GAT/SAGE/GIN/Conv | Not supported | Use `"linear"` or `"legacy_attention"` for vectors |
| Graph Transformer (vector node features) | 🧪 Experimental | `tgraphx.layers.graph_transformer.GraphTransformerLayer`; tensor-aware variant ⏳ planned |
| Hetero graphs: container + batch + relation-dispatch + classifiers | 🧪 Experimental v0.2.5 | `HeteroGraph`, `HeteroGraphBatch`, `HeteroConv`, `HeteroGraphClassifier`, `HeteroNodeClassifier`; vector node features only |
| Hetero spatial/volumetric node features | Not yet | Possible by passing tensor-aware layers into `HeteroConv` per relation; not yet covered by canned classifiers |
| Temporal graphs: container + batch + readouts + classifier | 🧪 Experimental v0.2.5 | `TemporalGraphSequence`, `TemporalGraphBatch`, `temporal_readout`, `TemporalGraphClassifier`/`Regressor` |
| Temporal recurrent memory (TGN/TGAT) | Not implemented | Stateless snapshot-loop pattern only; planned for v0.2.6+ |
| GAT scalar attention | ✅ Stable | Default `attention_mode="scalar"` |
| GAT per-channel attention | 🧪 Experimental | `attention_mode="channel"` (v0.2.4) |
| GAT per-pixel / per-voxel attention | Not implemented | Score tensors would be O(E×K×H×W); deferred until memory analysis |
| GAT chunked forward | ✅ Stable | Two-pass log-sum-exp; pass `chunk_size=K` (v0.2.4) |
| SAGE chunked forward (`aggr="mean"` / `"max"`) | ✅ Stable | Pass `chunk_size=K` to `forward()` (v0.2.3) |
| GIN chunked forward (`aggr="sum"`) | ✅ Stable | Pass `chunk_size=K` to `forward()` (v0.2.3) |
| 3-D support in `AttentionMessagePassing` | Not supported | Use `TensorGATLayer(spatial_rank=3)` |

## Training framework

| Feature | Status |
|---|---|
| `train_epoch` function | **Implemented** — see [training_utilities.md](training_utilities.md) |
| `evaluate` function | **Implemented** — see [training_utilities.md](training_utilities.md) |
| `fit` function | **Implemented** — thin `train_epoch` + `evaluate` wrapper; see [training_utilities.md](training_utilities.md) |
| `TensorBoardLogger` | **Implemented** — optional; requires `pip install tensorboard` or `pip install "tgraphx[tracking]"`; see [training_utilities.md](training_utilities.md) |
| `MLflowLogger` | ✅ Implemented — optional; lazy `mlflow` import; `pip install "tgraphx[mlflow]"` |
| PyG / DGL data converters (homogeneous + hetero) | ✅ Implemented — `tgraphx.interop`; lazy imports; data-only, not API replacement |
| Learned graph helpers (soft adjacency, EdgeScorer) | ✅ Implemented — `tgraphx.learned_graph` |
| Subgraph / k-hop / neighbour sampling + loaders | ✅ Implemented — `tgraphx.sampling`, `SubgraphDataLoader`, `NeighborSamplerLoader` |
| Random-walk sampling | ✅ Implemented (v0.2.8) — `tgraphx.random_walk_sample` |
| Hetero sampling (induced + per-relation neighbour) | ✅ Implemented (v0.2.8) — `tgraphx.hetero_induced_subgraph`, `tgraphx.hetero_neighbor_sample` |
| Temporal window sampling (sequence + batch) | ✅ Implemented (v0.2.8) — `tgraphx.temporal_window_sample`, `tgraphx.temporal_window_sample_batch` |
| Distributed (DDP) helpers (rank-zero, world-size, barrier) | ✅ Implemented — `tgraphx.distributed`; never auto-initialises |
| GraphSAINT / ClusterGCN-style minibatch samplers | Not implemented — out of scope; the building-block samplers above can be composed by users |
| Full automatic multi-GPU training framework | Not implemented — DDP setup is the user's responsibility |
| Recurrent temporal memory module (TGN / TGAT) | Not implemented — temporal workflows use the stateless snapshot-loop pattern only |

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
| Byte-seek tail-read for growing metrics.csv | ✅ Implemented v0.2.3 — appends-only: reads only new bytes; full reparse on file rotation |
| Incremental client payload (`?since_row`) | ✅ Stable — browser receives only new rows; double-read bug fixed v0.2.3 |
| TensorBoard log reading | Not implemented |
| MLflow run reading | Not implemented |
| Remote / cloud logdir support | Not implemented |

## Patch helpers

Patch helpers raise `ValueError` if the image/volume dimensions are not
exactly covered by `patch_size` and `stride`. No automatic padding is applied.

## PyG / DGL compatibility

TGraphX is not a drop-in replacement for PyG or DGL. The API and layer
semantics differ. Data-format converters are available in
`tgraphx.interop` (`to_pyg_data`, `from_pyg_data`, `to_dgl_graph`,
`from_dgl_graph`, and their hetero counterparts); they convert graph
objects between formats but do not make the two APIs interchangeable.

## Version

These limitations apply to the current TGraphX release. Deferred items may
be addressed in future releases. See [CHANGELOG.md](../CHANGELOG.md).

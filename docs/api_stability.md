# API stability

## v0.3.0 stable surface

The following APIs are considered stable: breaking changes require a
deprecation cycle and a bump to at least v0.4.0.

### Core data objects
- `tgraphx.Graph`, `tgraphx.GraphBatch`
- `tgraphx.core.graph_utils` (add/remove self-loops, make_undirected, coalesce_edges)
- `tgraphx.GraphDataset`, `tgraphx.GraphDataLoader`

### Tensor-aware GNN layers
- `ConvMessagePassing`, `TensorGATLayer`, `TensorGraphSAGELayer`, `TensorGINLayer`
- `AttentionMessagePassing` (legacy sigmoid gating — preserved for backward compat)
- `LinearMessagePassing`, `TensorMessagePassingLayer`

### Vector model zoo (new in v0.3.0)
- `GCNConv`, `GATv2Conv`, `APPNP`
- `global_mean_pool`, `global_sum_pool`, `global_max_pool`

### Graph builders and patch helpers
- All functions in `tgraphx.graph_builders`

### Sampling
- All functions in `tgraphx.sampling`, `tgraphx.sampling_loaders`
- `tgraphx.hetero_sampling`, `tgraphx.temporal_sampling`

### Factories
- `make_layer`, `build_model`, `build_model_from_config`

### Model classes
- `GraphClassifier`, `NodeClassifier`, `EdgePredictor`, `NodeRegressor`, `GraphRegressor`

### Training utilities
- `train_epoch`, `evaluate`, `fit`, `set_seed`
- `save_checkpoint`, `load_checkpoint`, `count_parameters`
- `accuracy`, `mean_absolute_error`, `mean_squared_error`

### Logging / tracking
- `CSVLogger`, `TensorBoardLogger`, `MLflowLogger`
- `write_graph_stats` and all `write_*_metadata` helpers added in v0.3.0

### Datasets (stable)
- `get_dataset`, `list_datasets`, `dataset_info`
- `DatasetMetadata`, cache/download utilities
- All synthetic dataset classes
- `ImageFolderPatchGraphDataset`, `VolumeFolderPatchGraphDataset`

### Transforms (stable)
- All classes in `tgraphx.transforms`

### Metrics (stable)
- All functions in `tgraphx.metrics`

### Dashboard (stable)
- Dashboard CLI (`tgraphx-dashboard`)
- `launch_dashboard`, `launch_dashboard_background`
- `export_dashboard_html`

### Experiment manager (stable in v0.3.0)
- `load_config`, `ExperimentConfig`, `Runner`, `GridRunner`
- `EarlyStopping`, `ModelCheckpoint`, `CSVLoggerCallback`, `LearningRateLogger`
- `tgraphx-train`, `tgraphx-grid`, `tgraphx-report` CLI scripts

## Beta / evolving APIs

The following APIs are functional and tested, but may have minor surface
changes in a minor v0.3.x release:

- `tgraphx.explain` — saliency, integrated gradients, edge attribution,
  attention edge scores, patch heatmaps, export helpers.
- Dataset adapters for torchvision / PyG / DGL / OGB — the converter
  contract is stable; individual adapter kwargs may be refined.
- `HeteroGraph`, `HeteroGraphBatch`, `HeteroConv`, `HeteroGraphClassifier`,
  `HeteroNodeClassifier` — the hetero layer surface may expand.
- `TemporalGraphSequence`, `TemporalGraphBatch`, `temporal_readout`,
  `TemporalGraphClassifier`, `TemporalGraphRegressor`.
- `GraphTransformerLayer` and `tgraphx.layers.transformer_encodings`.

## Beta — added in v0.3.2 (in [Unreleased] until release)

These APIs ship with full test coverage but their signatures may evolve
before v0.4.0.  See [architecture.md](architecture.md) for the broader
plan.

- `tgraphx.negative_sampling`, `tgraphx.structured_negative_sampling`,
  `tgraphx.batched_negative_sampling`, `tgraphx.hard_negative_sampling`
  — link-prediction primitives (in `tgraphx.sampling_negative`).
- `tgraphx.algorithms.connected_components`,
  `weakly_connected_components`, `is_connected`,
  `number_connected_components`, `bfs_layers`, `bfs_edges`,
  `shortest_path_length`, `degree`, `degree_features`
  — pure-PyTorch graph algorithms.
- `tgraphx.temporal.sinusoidal_time_encoding` — deterministic time
  encoding.

## Experimental — added in v0.3.2 (in [Unreleased] until release)

- `tgraphx.temporal.LearnableTimeEncoding` — Time2Vec-style trainable
  time encoder.  Marked experimental until v0.3.4 evaluates it on a
  real temporal benchmark.

## Legacy / deprecated

- `AttentionMessagePassing` — kept for backward compatibility; prefer
  `TensorGATLayer` for true multi-head GAT.

## Not yet shipped

- Per-pixel / per-voxel GAT attention (memory-prohibitive naive form).
- Recurrent temporal memory modules (TGN, TGAT style).
- Full automatic multi-GPU training framework.
- Universal arbitrary-rank tensor support across all layers.

See [limitations.md](limitations.md) and [roadmap.md](roadmap.md).

# Changelog

All notable changes to TGraphX are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.0] — 2026-05

### Added — Core

- `Graph` and `GraphBatch` data structures with eager validation,
  `edge_weight`, `edge_features`, `node_labels`, `edge_labels`, `graph_label`,
  `metadata`, `.to(device)`, `.clone()`, topology operations.
- `GraphDataset` and `GraphDataLoader` wrapping `torch.utils.data`.
- Graph utility functions: `add_self_loops`, `remove_self_loops`,
  `make_undirected`, `coalesce_edges`, `is_undirected`.
- `graph_utils.py` pure-tensor helpers used by `Graph`.

### Added — GNN Layers

- `ConvMessagePassing` — 1×1 conv messages + `DeepCNNAggregator`;
  supports 2-D `[N,C,H,W]` and 3-D `[N,C,D,H,W]` node features;
  optional `chunk_size` for sum/mean aggregation to reduce peak memory.
- `TensorGATLayer` — true multi-head GAT with scalar attention per
  `(edge, head)`; spatial_rank 2 and 3; vector and spatial edge features.
- `TensorGraphSAGELayer` — separate W_self/W_neigh 1×1 convolutions;
  vector and spatial edge features.
- `TensorGINLayer` — GIN / GINEConv; learnable ε; optional batchnorm.
- `LinearMessagePassing` — vector features via linear projections.
- `AttentionMessagePassing` — legacy sigmoid gating; vector or 2-D spatial.
- `TensorMessagePassingLayer` — base class with sum/mean/max aggregation.

### Added — Graph Builders

- `build_grid_graph`, `build_grid_graph_3d` — 4/6-connected grids.
- `build_fully_connected_graph`, `build_knn_graph`, `build_radius_graph`,
  `build_iou_graph`, `build_random_graph`.
- Patch helpers: `image_to_patches`, `patch_grid_shape`,
  `volume_to_patches`, `volume_patch_grid_shape`.

### Added — Factories and Task Models

- `make_layer(name, in_shape, out_shape, **kwargs)` — layer factory.
- `build_model(task, layer, ...)` — task model factory.
- `build_model_from_config(path_or_dict)` — JSON / YAML / dict config.
- `EdgePredictor` — MLP edge scorer with spatial pooling.
- `NodeRegressor`, `GraphRegressor` — standalone regression models.
- `GraphClassifier` — ConvMessagePassing-based graph classifier.
- `NodeClassifier` — LinearMessagePassing-based node classifier.

### Added — Training Utilities

- `tgraphx.training`: `set_seed`, `count_parameters`, `save_checkpoint`,
  `load_checkpoint`, `accuracy`, `mean_absolute_error`, `mean_squared_error`.
- `tgraphx.tracking`: `CSVLogger` — append-mode, UTC timestamps,
  off by default, dashboard-compatible schema.

### Added — Dashboard

- Local HTTP dashboard (`tgraphx-dashboard` CLI and Python API).
- Responsive UI (desktop sidebar, mobile hamburger, TV fullscreen mode).
- API: `/api/status`, `/api/metrics`, `/api/hardware`, `/api/metadata`,
  `/api/graph`.
- Security: localhost bypass, LAN token enforcement, path traversal prevention,
  no external CDN assets.
- Graph visualization: SVG for ≤ 200 nodes / 1 000 edges; summary otherwise.
- `/api/metrics` uses mtime/size caching to avoid reparsing unchanged CSV.
- Optional hardware monitoring via `psutil` / `pynvml`.

### Added — Performance Utilities

- `tgraphx.performance`: `env_report`, `estimate_message_memory`,
  `recommended_device`.
- `benchmarks/benchmark_layers.py` — CUDA-event/perf_counter timing,
  AMP, torch.compile, JSON output.
- `benchmarks/benchmark_graph_builders.py` — builder timing, O(N²) warnings.

### Added — Examples

- Factory examples: 01–05 (node/graph classification, regression, edge prediction).
- Graph builder examples: directed vs undirected, image patch graph, volume patch graph.
- GNN family example with graph builders.
- Training with dashboard.
- Checkpoint save/load.
- torch.compile benchmark, mixed precision inference, memory report.
- Minimal layer examples, tiny overfit checks, gradient sanity stack.
- `training_minimal_fit.py`, `training_with_csvlogger.py`,
  `training_with_tensorboard.py` — training utility examples.
- `run_all_fast_examples.py` — runs all fast examples and reports results.

### Added — Training Utilities

- `train_epoch(model, loader, optimizer, loss_fn, ...)` — one supervised
  epoch; returns averaged loss + metrics dict.
- `evaluate(model, loader, loss_fn, ...)` — evaluation under `no_grad`;
  no file writes.
- `fit(model, train_loader, ...)` — thin loop wrapper over `train_epoch` /
  `evaluate`; returns per-epoch history list.
- Supported batch formats: `GraphBatch` (with `graph_labels` / `node_labels`)
  and `(Tensor, Tensor)` tuples.
- `[B, 1]` label tensors are squeezed to `[B]` for compatibility with
  `CrossEntropyLoss` and similar losses.

### Added — Tracking

- `TensorBoardLogger` — optional TensorBoard logger backed by
  `torch.utils.tensorboard.SummaryWriter`; lazy import; compatible
  `log(**kwargs)` interface matching `CSVLogger`.
  Requires: `pip install tensorboard` or `pip install "tgraphx[tracking]"`.

### Added — Dashboard

- Bounded metrics loading: `/api/metrics` returns at most `max_metric_rows`
  (default 5 000) most recent rows; response includes `truncated`,
  `total_row_count`, and `max_rows` fields.
- Metrics truncation notice displayed in the Metrics section of the dashboard
  UI when rows are omitted.
- `--max-metric-rows` CLI argument (default 5 000).
- `/api/metrics` mtime/size/max_rows caching to avoid reparsing unchanged CSV.

### Added — Performance

- `tgraphx.performance`: `env_report`, `estimate_message_memory`,
  `recommended_device`.
- `benchmarks/benchmark_layers.py` — layer throughput with CUDA events /
  `perf_counter`, AMP, `torch.compile`, JSON output, `--chunk-size`.
- `benchmarks/benchmark_graph_builders.py` — builder timing, O(N²) warnings.
- `ConvMessagePassing.forward(chunk_size=N)` — optional edge chunking for
  `aggr="sum"` and `aggr="mean"` to reduce peak message-buffer memory.

### Not Implemented (intentional)

- `MLflowLogger` — use the `mlflow` client directly: `pip install mlflow`.
- GAT / SAGE / GIN chunked forward — softmax constraint defers GAT;
  SAGE/GIN deferred for scope.
- Neighbor sampling, Graph Transformers, heterogeneous/temporal graphs.
- Per-channel/per-pixel attention in GAT.
- Incremental CSV tail-read by bytes (deferred; full file read on cache miss).
- GradScaler in `train_epoch` AMP — users who need stable float16 training
  should manage a `torch.cuda.amp.GradScaler` in their own loop.

---

[0.1.0]: https://github.com/arashsajjadi/TGraphX/releases/tag/v0.1.0

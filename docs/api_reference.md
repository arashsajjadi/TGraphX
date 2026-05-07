# API Reference

Quick reference for TGraphX's public API.
For full signatures see the source files; every public function has a docstring.

**Preferred imports** — all symbols below are available directly from the
top-level `tgraphx` namespace after `pip install tgraphx`.  Submodule paths
(e.g. `from tgraphx.training import fit`) continue to work and are listed at
the bottom of each section for advanced users.

```python
# Everything a typical user needs
from tgraphx import (
    Graph, GraphBatch, GraphDataLoader,
    build_model, make_layer,
    fit, evaluate, train_epoch,
    CSVLogger, TensorBoardLogger,
    env_report, recommended_device,
    GraphClassifier, NodeClassifier, EdgePredictor,
)
```

## Top-level (`tgraphx`)

### Data structures

| Symbol | Description |
|---|---|
| `Graph` | Single graph container — validated, device-aware |
| `GraphBatch` | Batched super-graph from a list of `Graph` objects |
| `GraphDataset` | `torch.utils.data.Dataset` wrapping a list of `Graph` |
| `GraphDataLoader` | Batched data loader using `GraphBatch` collation |

### Graph utilities

| Symbol | Description |
|---|---|
| `add_self_loops(edge_index, ...)` | Add one self-loop per node (pure tensor) |
| `remove_self_loops(edge_index, ...)` | Remove all self-loops |
| `make_undirected(edge_index, ...)` | Symmetrize + coalesce |
| `coalesce_edges(edge_index, ...)` | Sort + merge duplicate edges |
| `is_undirected(edge_index)` | bool |

### Graph builders

| Symbol | O() | Notes |
|---|---|---|
| `build_grid_graph(rows, cols, ...)` | O(E) | 4-connected 2-D grid |
| `build_grid_graph_3d(depth, rows, cols, ...)` | O(E) | 6-connected 3-D grid |
| `build_fully_connected_graph(N, ...)` | O(N²) | Complete graph |
| `build_knn_graph(coords, k, ...)` | O(N²) | torch.cdist |
| `build_radius_graph(coords, radius, ...)` | O(N²) | torch.cdist |
| `build_iou_graph(boxes, threshold, ...)` | O(N²) | (x1,y1,x2,y2) format |
| `build_random_graph(N, E, ...)` | O(E) | Deterministic with seed |

### Patch helpers

| Symbol | Input | Output |
|---|---|---|
| `image_to_patches(images, patch_size, stride)` | `[B,C,H,W]` | `[B,P,C,ph,pw]` |
| `patch_grid_shape(H, W, patch_size, stride)` | scalars | `(n_h, n_w)` |
| `volume_to_patches(volumes, patch_size, stride)` | `[B,C,D,H,W]` | `[B,P,C,pd,ph,pw]` |
| `volume_patch_grid_shape(D, H, W, patch_size, stride)` | scalars | `(n_d, n_h, n_w)` |

### Layer factory

| Symbol | Description |
|---|---|
| `make_layer(name, in_shape, out_shape, **kwargs)` | Create a GNN layer by name |

### Model factory

| Symbol | Description |
|---|---|
| `build_model(task, layer, in_shape, hidden_shape, num_layers, ...)` | Task model |
| `build_model_from_config(path_or_dict)` | Config-driven construction |

### Model classes

All importable directly from `tgraphx`:

| Symbol | Description |
|---|---|
| `GraphClassifier(...)` | ConvMessagePassing-based graph classifier |
| `NodeClassifier(...)` | LinearMessagePassing-based node classifier |
| `EdgePredictor(in_dim, hidden_dim, out_dim)` | MLP edge scorer |
| `NodeRegressor(in_shape, hidden_shape, out_dim, ...)` | Vector node regression |
| `GraphRegressor(in_shape, hidden_shape, out_dim, ...)` | Vector graph regression |

Also available from `tgraphx.models`: `CNNEncoder`, `CNN_GNN_Model`, `PreEncoder`.

## `tgraphx.layers`

| Symbol | Input | Description |
|---|---|---|
| `ConvMessagePassing(in_shape, out_shape, aggr, ...)` | `[N,C,H,W]` or `[N,C,D,H,W]` | Conv 1×1 message + DeepCNN aggregator |
| `TensorGATLayer(in_channels, out_channels, ...)` | `[N,C,H,W]` or `[N,C,D,H,W]` | Multi-head GAT |
| `TensorGraphSAGELayer(in_channels, out_channels, ...)` | spatial | GraphSAGE |
| `TensorGINLayer(in_channels, out_channels, ...)` | spatial | GIN / GINEConv |
| `LinearMessagePassing(in_shape, out_shape, ...)` | `[N,D]` | Linear projections |
| `AttentionMessagePassing(in_shape, out_shape, ...)` | vector or 2-D | Legacy sigmoid gating |
| `make_layer(name, in_shape, out_shape, **kwargs)` | — | Factory |

## `tgraphx.training` (also `from tgraphx import …`)

| Symbol | Description |
|---|---|
| `train_epoch(model, loader, optimizer, loss_fn, *, device, metrics, logger, log_level, epoch, amp, grad_clip)` | One supervised training epoch; returns `{"loss": …, …}`; `log_level=2` prints per-batch progress |
| `evaluate(model, loader, loss_fn, *, metrics, device)` | Evaluation under `torch.no_grad()`; no file writes |
| `fit(model, train_loader, val_loader, *, epochs, optimizer, loss_fn, device, metrics, logger, log_level, amp, grad_clip)` | Thin loop wrapper; `log_level=2` enables per-batch output |
| `set_seed(seed, deterministic=False)` | Seeds torch / numpy / random; `deterministic=True` also sets cuDNN flags |
| `count_parameters(model, trainable_only=True)` | Parameter count |
| `save_checkpoint(model, optimizer, epoch, path, **extra)` | Saves checkpoint dict via `torch.save` |
| `load_checkpoint(model, optimizer, path, map_location, weights_only=True)` | Loads checkpoint; defaults to safe deserialization; pass `weights_only=False` only for trusted legacy files |
| `accuracy(logits, labels)` | Multi-class argmax accuracy |
| `mean_absolute_error(predictions, targets)` | MAE |
| `mean_squared_error(predictions, targets)` | MSE |

## `tgraphx.tracking` (also `from tgraphx import …`)

| Symbol | Description |
|---|---|
| `CSVLogger(logdir, filename="metrics.csv")` | Append-mode CSV logger; dashboard-compatible schema |
| `logger.log(**metrics)` | Append one row; adds UTC timestamp automatically |
| `logger.close()` | Flush and close |
| `TensorBoardLogger(logdir, comment="")` | Optional TensorBoard logger; requires `pip install tensorboard`; **not imported at package load time** |
| `tb_logger.log(**metrics)` | Write scalars; `epoch=0` / `step=0` handled correctly |
| `tb_logger.log_scalar(tag, value, step)` | Write one scalar |
| `tb_logger.log_metrics(metrics_dict, step)` | Write multiple scalars |
| `tb_logger.close()` | Flush and close `SummaryWriter` |
| `write_graph_stats(graph_or_dict, path)` | Write `graph_stats.json` for dashboard display |

## `tgraphx.performance` (also `from tgraphx import …`)

| Symbol | Description |
|---|---|
| `env_report(include_hardware, include_sensors)` | Runtime environment dict |
| `estimate_message_memory(num_edges, out_shape, dtype)` | Peak buffer estimate |
| `recommended_device()` | Best `torch.device` (CUDA > MPS > CPU) |

## `tgraphx.dashboard`

| Symbol | Description |
|---|---|
| `launch_dashboard(logdir, host, port, token, refresh_interval_s)` | Blocking server launch |
| `launch_dashboard_background(logdir, ...)` | Background daemon thread; returns server handle |
| `export_dashboard_html(logdir, out_path)` | Standalone offline HTML snapshot (no server needed) |

### Dashboard API endpoints

| Endpoint | Returns |
|---|---|
| `GET /api/status` | Run name, status, epoch/step, timestamps |
| `GET /api/metrics` | Full metrics payload |
| `GET /api/metrics?since_row=N` | Incremental rows after index N |
| `GET /api/metrics?run=<name>` | Metrics for named run (multi-run mode) |
| `GET /api/runs` | `{mode, runs, capped}` — multi-run discovery |
| `GET /api/hardware` | Versions + CPU/RAM/GPU sensors |
| `GET /api/metadata` | `run_metadata.json` contents |
| `GET /api/graph` | Graph summary ± edge_index |
| `GET /api/graph_stats` | Precomputed stats from `graph_stats.json` |
| `GET /api/config` | Server config (`poll_ms`, `max_metric_rows`, …); never exposes token |

## `tgraphx.core.utils`

| Symbol | Description |
|---|---|
| `get_device(device_id)` | CUDA > MPS > CPU device selection |
| `load_config(path)` | Load JSON or YAML config file |

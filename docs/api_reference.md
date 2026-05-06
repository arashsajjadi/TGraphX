# API Reference

Quick reference for TGraphX's public API.
For full signatures see the source files; every public function has a docstring.

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

| Symbol | Description |
|---|---|
| `EdgePredictor(in_dim, hidden_dim, out_dim)` | MLP edge scorer |
| `NodeRegressor(in_shape, hidden_shape, out_dim, ...)` | Vector node regression |
| `GraphRegressor(in_shape, hidden_shape, out_dim, ...)` | Vector graph regression |

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

## `tgraphx.training`

| Symbol | Description |
|---|---|
| `set_seed(seed)` | Seeds torch / numpy / random |
| `count_parameters(model, trainable_only=True)` | Parameter count |
| `save_checkpoint(model, optimizer, epoch, path, **extra)` | torch.save wrapper |
| `load_checkpoint(model, optimizer, path, map_location)` | Returns saved epoch |
| `accuracy(logits, labels)` | Multi-class argmax accuracy |
| `mean_absolute_error(predictions, targets)` | MAE |
| `mean_squared_error(predictions, targets)` | MSE |

## `tgraphx.tracking`

| Symbol | Description |
|---|---|
| `CSVLogger(logdir, filename="metrics.csv")` | Append-mode CSV logger |
| `logger.log(**metrics)` | Append one row; adds UTC timestamp automatically |
| `logger.close()` | Flush and close |

## `tgraphx.performance`

| Symbol | Description |
|---|---|
| `env_report(include_hardware, include_sensors)` | Runtime environment dict |
| `estimate_message_memory(num_edges, out_shape, dtype)` | Peak buffer estimate |
| `recommended_device()` | Best `torch.device` (CUDA > MPS > CPU) |

## `tgraphx.dashboard`

| Symbol | Description |
|---|---|
| `launch_dashboard(logdir, host, port, token)` | Blocking server launch |
| `launch_dashboard_background(logdir, ...)` | Background thread; returns server |

## `tgraphx.core.utils`

| Symbol | Description |
|---|---|
| `get_device(device_id)` | CUDA > MPS > CPU device selection |
| `load_config(path)` | Load JSON or YAML config file |

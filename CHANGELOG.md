# Changelog


## [0.2.1] - 2026-05-07

### Fixed
- Corrected README/support-table contradictions around TensorGATLayer spatial and volumetric edge features.
- Corrected stale ConvMessagePassing `aggr="max"` documentation.
- Replaced overconfident Windows/macOS support wording with best-effort/no-CI wording.

### Added
- Added README support-status legend and backend/feature/scalability/attention support tables.
- Added runtime O(N²) warnings for large fully connected, kNN, radius, and IoU graph builders.
- Added documentation-claim regression tests.
- Added `docs/roadmap.md`.

### Changed
- Made README and docs more explicit about supported, best-effort, planned, and unsupported features.

All notable changes to TGraphX are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

---

## [0.2.2] — Unreleased draft

### Fixed

- **`broadcast_edge_weight` dtype cast** — caller-supplied `edge_weight` is now
  cast to the message tensor's dtype before broadcasting.  Previously, float32
  edge weights caused a dtype mismatch under `torch.autocast` when messages were
  float16 or bfloat16, manifesting as a multiplication error.

- **`TensorGATLayer` `index_add_` dtype mismatch** — the learned attention
  vectors `a_src` / `a_dst` are float32 `nn.Parameter` objects.  Under
  `torch.autocast` their element-wise product with low-precision activations
  (bf16 / fp16) is promoted to float32 by PyTorch's mixed-precision rules,
  making `attn_dropped` float32 even when the value tensor `h_src` is bf16.
  Added an explicit `.to(dtype=h_src.dtype)` cast for the attention weights
  before the value multiplication so that `out_per_head.index_add_` always
  sees matching dtypes.

- **`edge_softmax` numerical stability under AMP** — the max-shift + exp +
  scatter-sum computation is now performed in float32 when the input dtype is
  float16 or bfloat16, and the result is cast back to the original dtype.
  This prevents overflow/underflow in attention weights under low-precision
  autocast and matches the approach used by major GNN libraries.

### Added

- `tests/test_amp_compile.py` — 59 new tests covering:
  - CPU bfloat16 autocast for Conv, GAT, SAGE, GIN (forward + backward).
  - CUDA float16 / bfloat16 autocast for all four layers.
  - `edge_weight` dtype cast under autocast.
  - Vector and spatial edge features under autocast.
  - 3-D volumetric layers under autocast.
  - `torch.compile` correctness smoke for all four layers ± edge
    features ± edge weight.
  - `torch.compile + bfloat16 autocast` combined test.
  - `edge_softmax` dtype and numerical stability unit tests.
  - `broadcast_edge_weight` dtype cast unit tests.
  - No-side-effect import checks.

- `benchmarks/benchmark_layers.py` now reports:
  - `amp_dtype`: the AMP dtype actually used (`"float16"`, `"bfloat16"`, or
    `"none"`).
  - `finite_output`: whether the post-warmup output contains only finite values.
  Both fields appear in the terminal report and in `--output` JSON.

### Changed

- `examples/mixed_precision_inference.py` — fineness check added to the
  output line; the caught `RuntimeError` comment updated to reflect that dtype
  mismatches should no longer occur after v0.2.2 fixes.

### Documentation

- `docs/performance.md` — new **AMP policy** section with supported modes per
  backend, v0.2.2 fix summary, and recommended usage patterns.
- `docs/limitations.md` — AMP table updated; fixed items now ✅.
- `README.md` hardware/performance section — added AMP policy table and v0.2.2
  summary note.

### Not implemented (deferred)

- Universal float16 CPU support — CPU float16 kernels for `scatter_reduce_`
  with `reduce="amax"` are not consistently available across PyTorch versions;
  bfloat16 is the recommended CPU low-precision dtype.
- `GradScaler` integration in `train_epoch` — users needing stable float16
  CUDA training should manage `torch.cuda.amp.GradScaler` in their own loop.
- MPS AMP — MPS operator coverage varies by PyTorch version and is not
  tested in CI; deferred to v0.2.3+.

---

## [0.2.0] - 2026-05-07

### Security

- `load_checkpoint` now defaults to `weights_only=True` (safe deserialization)
  where supported by the installed PyTorch version.
- Unsafe/legacy checkpoint loading requires explicit opt-in via
  `weights_only=False` and emits a `UserWarning` on every call.  A clear
  `RuntimeError` explains how to opt in when safe loading fails.

### Fixed

- Dashboard static assets (`dashboard.css`, `dashboard.js`) now correctly
  included in wheel/sdist; PyPI-installed users no longer see a blank dashboard.
- `TensorBoardLogger.log()` now handles `epoch=0` / `step=0` correctly; falsy
  values no longer fall through to the internal auto-counter.
- Dashboard `/api/status` no longer reports `epoch=None` when the last CSV row
  contains `epoch=0`.
- `LinearMessagePassing` now honours `dropout_prob`, `residual`, and
  `use_batchnorm` flags; previously the `update()` override discarded them
  silently.
- `LinearMessagePassing` now rejects unsupported spatial/volumetric in-shapes
  with a clear `ValueError` at construction time.
- `TensorGATLayer(add_self_loops=True)` no longer duplicates self-loops that
  already exist in `edge_index`.
- Training utilities no longer hide internal `TypeError` exceptions inside
  model `forward()` calls; they propagate as `RuntimeError` with context.
- Failed metric functions emit a one-time `UserWarning` per metric name instead
  of silently disappearing from results.
- Float regression targets with shape `[B, 1]` are preserved; only integer
  `[B, 1]` tensors are squeezed for classification-loss compatibility.
- Stale `docs/limitations.md` rows for `train_epoch`, `evaluate`, `fit`, and
  `TensorBoardLogger` corrected; those utilities were fully implemented.

### Added

- Top-level convenience re-exports: `from tgraphx import fit, CSVLogger,
  env_report, write_graph_stats, ...` works without submodule paths.
- `make_layer("gin", ...)` now forwards `eps`, `train_eps`, `hidden_channels`,
  and `use_batchnorm` to `TensorGINLayer`.
- `make_layer("linear", ...)` now forwards `use_batchnorm`.
- `set_seed(seed, deterministic=False)`: optional `deterministic=True` sets
  `cudnn.deterministic = True` and `cudnn.benchmark = False`.
- Dashboard major upgrade:
  - Responsive professional layout; phone/tablet/desktop/TV breakpoints.
  - Okabe-Ito color-blind-safe palette toggle (persisted in localStorage).
  - Print stylesheet for save-as-PDF via the browser.
  - Focus-visible ring; skip-to-content link; ARIA labels; reduced-motion.
  - Pause/resume polling controls; stale-data warning banner.
  - Range/window selector for chart data (All / Last 100 / 500 / 1000).
  - Per-chart CSV and SVG export; metrics-table CSV export.
  - Print/save-as-PDF button.
  - Copy local and LAN URL tools page.
  - `/api/config` endpoint (exposes server config, never the token value).
  - `/api/metrics?since_row=N` incremental rows API.
  - `/api/runs` and `?run=<name>` multi-run selector.
  - `/api/graph_stats` endpoint + `write_graph_stats()` helper.
  - Offline standalone HTML snapshot export (`--export-html` CLI flag;
    `export_dashboard_html()` Python API).
  - GPU power draw and thermal status in hardware panel (requires `pynvml`).
  - Hover tooltip on charts (dependency-free, visual-only).
  - CLI flags: `--refresh-interval`, `--open-browser`, `--token auto`,
    `--export-html`, `--max-metric-rows`.
  - `no-referrer` policy; all user-controlled strings HTML-escaped.
- `docs/comparison.md`: when to use TGraphX vs PyG / DGL / NetworkX.
- CI hardening: wheel-install smoke, cross-platform (macOS/Windows), extras
  smoke, dashboard server and export smoke, risky-claims audit, README checks.

### Changed

- Dashboard `0.0.0.0` banner prints `Local → http://127.0.0.1:<port>` and a
  best-effort LAN URL with `?token=...` when applicable.
- `fit(log_level=2)` now produces per-batch progress lines via `train_epoch`.
- `load_checkpoint` wraps failed safe-mode loads in a `RuntimeError` that
  explains how to opt in to legacy mode.
- README/PyPI presentation: TGraphX logo added; PyPI badge added; stale
  "not yet published" text removed; installation section updated.
- `pyproject.toml` `Development Status` upgraded from Alpha (3) to Beta (4).
- Quickstart and API docs are vector-first and current-state focused.
- Dashboard documentation expanded: security model, export features, device
  support, accessibility, and troubleshooting guide.

### Documentation

- `docs/quickstart.md` opens with a vector-feature example.
- `docs/api_reference.md`, `docs/factories.md`, `docs/training_utilities.md`,
  `docs/performance.md` updated to match current API surface.
- README Limitations section corrected (graph builders and patch helpers are
  implemented; stale claims removed).
- `README.md` installation section updated: TGraphX is on PyPI; PyPI badge
  and logo added.
- `docs/comparison.md` — new page covering when to use TGraphX vs PyG / DGL /
  NetworkX / TensorBoard.

---

## [0.1.2] — 2026-05-07

### Added

- Added the official Colab tutorial link to `README.md` and
  `docs/quickstart.md` so users can open the interactive notebook directly
  from the documentation.

### Fixed

- `docs/limitations.md` incorrectly stated that `train_epoch`, `evaluate`,
  `fit`, and `TensorBoardLogger` were "not implemented". Those utilities are
  fully implemented in `tgraphx.training` and `tgraphx.tracking`; the
  limitation page now reflects reality and links to
  `docs/training_utilities.md`.
- `docs/api_reference.md` omitted `train_epoch`, `evaluate`, `fit`, and
  `TensorBoardLogger` from the `tgraphx.training` and `tgraphx.tracking`
  tables; all four are now documented there.
- `docs/installation.md` contained a stale version comment
  (`# e.g. "0.1.1"`) and incorrectly listed `mlflow` as an `[tracking]`
  extra (it was removed in 0.1.1); both are corrected.
- `pyproject.toml` was missing a `[tool.setuptools.package-data]` directive,
  which caused `tgraphx/dashboard/static/dashboard.css` and `dashboard.js`
  to be excluded from the wheel and sdist. Dashboard served a 404 on those
  assets for every PyPI-installed user. Static files are now correctly
  packaged.

---

## [0.1.1] — 2026-05-05

### Fixed

- Corrected PyPI-facing package metadata: `Arash Sajjadi` is now the sole
  listed package author and maintainer in `pyproject.toml`.
  Mark Eramian is Arash Sajjadi's PhD supervisor / academic advisor and
  co-author of the related preprint; he is acknowledged in that capacity in
  `CITATION.cff`, `README.md`, and the BibTeX citation block.
- Removed `mlflow` from the `tracking` optional extra (MLflowLogger is not
  implemented in TGraphX; users should install `mlflow` separately).
- Updated copyright year in `LICENSE` to 2025–2026.
- Updated `CITATION.cff` software-level `authors` to list Arash Sajjadi;
  paper co-authorship (Sajjadi & Eramian) preserved in `preferred-citation`.
- Added Python 3.13 classifier.
- Suppressed PyTorch-upstream `torch.jit.script_method` DeprecationWarning
  in pytest configuration.

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

[Unreleased]: https://github.com/arashsajjadi/TGraphX/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/arashsajjadi/TGraphX/releases/tag/v0.2.0
[0.1.2]: https://github.com/arashsajjadi/TGraphX/releases/tag/v0.1.2
[0.1.1]: https://github.com/arashsajjadi/TGraphX/releases/tag/v0.1.1
[0.1.0]: https://github.com/arashsajjadi/TGraphX/releases/tag/v0.1.0

# Changelog

All notable changes to TGraphX are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

---

## [0.2.7] — 2026-05-08 (release prep)

### Added

- **Graph Transformer maturity** (still 🧪 experimental, vector node
  features only):
  - `GraphTransformerLayer` accepts new optional arguments
    ``positional_encoding`` (``"degree"``, ``"laplacian"``), ``pe_dim``,
    ``edge_bias``.  ``forward`` now accepts ``positional`` and
    ``edge_bias_dense`` kwargs (both default-None — backward-compatible).
  - `tgraphx.layers.transformer_encodings` — pure-PyTorch helpers:
    `degree_encoding`, `laplacian_eigvec_encoding`,
    `build_adjacency_bias`.  No new dependencies.
  - **Factory integration**: `make_layer("graph_transformer", ...)` now
    works for vector node features.

- **Stability documentation** (new):
  - `docs/experimental_policy.md` — what 🧪 means, promotion criteria,
    promotion-target list per current experimental API.
  - `docs/deprecation_policy.md` — deprecation cycle for stable APIs,
    pre-1.0 stability summary.
  - `docs/migration_v0_2_to_v0_3.md` — migration recipe; reaffirms that
    v0.2 stable APIs are preserved into v0.3.

- **Tests**: `tests/test_graph_transformer_v027.py` (16 tests),
  `tests/test_backward_compatibility.py` (9 tests).

- **Example**: `examples/graph_transformer_demo.py`.

### Changed

- `pyproject.toml` version → 0.2.7.

### Deferred

- **Tensor-aware GraphTransformer** — open design question (token
  granularity).  Vector-only baseline stable at experimental level;
  full tensor-aware variant planned post v0.3.0 after a feasibility
  study.

---

## [0.2.6] — 2026-05-08 (release prep)

### Added

- **Sampling utilities** (`tgraphx.sampling`):
  - `induced_subgraph(graph, node_ids, relabel_nodes=True)`
  - `edge_subgraph(graph, edge_ids, relabel_nodes=True)`
  - `k_hop_subgraph(graph, seed_nodes, num_hops, direction="both"|"in"|"out")`
  - `sample_nodes(graph, num_nodes, seed)` — uniform without replacement
  - `sample_edges(graph, num_edges, seed)`
  - `neighbor_sample(graph, seed_nodes, fanouts, direction="in", seed)` —
    GraphSAGE-style multi-layer neighbour sampling.  Supports
    ``fanout=-1`` (keep all).
  - All samplers preserve features, edge_weight, edge_features,
    node_labels, edge_labels, graph_label, and metadata.  A
    ``metadata["sampling"]`` dict records ``original_node_ids`` /
    ``original_edge_ids`` and the sampler configuration.

- **Sampling-based loaders** (`tgraphx.sampling_loaders`):
  - `SubgraphDataLoader(graph, num_nodes, num_steps, seed)`
  - `NeighborSamplerLoader(graph, batch_size, fanouts, shuffle, seed,
    drop_last, input_nodes)`
  - Plain Python iterables; no hidden multiprocessing; deterministic
    with seed; integrate directly with existing TGraphX layers.

- **Distributed-helper module** (`tgraphx.distributed`):
  - `is_distributed_available_and_initialized()`
  - `get_rank(default=0)`, `get_world_size(default=1)`, `is_rank_zero()`
  - `rank_zero_print(...)`, `@rank_zero_only` decorator
  - `barrier()` — no-op when not initialized
  - **Never** calls ``init_process_group`` automatically; safe to import
    in any environment, including CPU-only single-process runs.
  - `examples/ddp_training_smoke.py` — single-process smoke + comments
    on how to launch real multi-process DDP via ``torchrun``.

- **Benchmark**: `benchmarks/benchmark_sampling.py` (CPU-safe ``--small``
  mode for CI; reports time per op and resulting subgraph sizes).

- **Tests**: 46 new tests in `test_sampling.py`, `test_sampling_loaders.py`,
  `test_distributed_compat.py`.

- **Examples**: `neighbor_sampling_demo.py`, `ddp_training_smoke.py`.

### Changed

- `__init__.py` exports all new sampling APIs at the top level.
- `pyproject.toml` version → 0.2.6.

### Deferred (with exact reason)

- **Hetero / temporal sampling** — uniform/k-hop/neighbor samplers for
  `HeteroGraph` and `TemporalGraphSequence` are planned but require
  per-relation / per-snapshot mask logic.  Planned for v0.2.7+.
- **Random-walk sampling** — useful for some self-supervised settings;
  defers because it needs careful handling of restart/teleport semantics.

---

## [0.2.5] — 2026-05-08 (release prep)

### Added

- **Hetero real functionality** (🧪 experimental):
  - `HeteroGraphBatch` — disjoint batching with per-type batch vectors,
    correct edge-index offsets per node type, and explicit errors for
    inconsistent stores (no silent dropping of edge weights / features /
    labels).
  - `HeteroConv` — relation-dispatch wrapper.  For each relation
    `(s, r, d)` it runs a user-supplied layer; per-destination-type
    aggregation across relations is `sum` / `mean` / `max`.  Source ≠
    destination type is handled by stacking `[x_src; x_dst]` and
    remapping edge indices into the destination subblock.
  - Hetero readouts: `hetero_mean_pool`, `hetero_sum_pool`,
    `hetero_max_pool`, `hetero_concat_pool`.  Optional `batch_dict` for
    graph-level pooling; stable type ordering for concat.
  - `HeteroGraphClassifier`, `HeteroNodeClassifier` — vector-feature
    composition with per-type input projections so that types not
    appearing as destinations of any relation still flow through with
    matching dim.

- **Temporal real functionality** (🧪 experimental):
  - `TemporalGraphBatch` — equal-length and variable-length sequence
    batching.  Per-snapshot iteration yields
    `(t, GraphBatch_active, mask[B])`.  Padded timestamps tensor.
  - `temporal_readout(seq_emb, mode, mask=None)` —
    `last` / `mean` / `max` over time with mask-aware reduction.
  - `TemporalGraphClassifier`, `TemporalGraphRegressor` — apply a
    stateless base graph encoder to each snapshot, then reduce.  No
    recurrent memory module (TGN/TGAT-style is deferred to v0.2.6+).

- **Hetero PyG/DGL converters** (🧪 experimental, optional, lazy):
  - `to_pyg_heterodata` / `from_pyg_heterodata`
  - `to_dgl_heterograph` / `from_dgl_heterograph`

- **Tests**: 52 new tests across `test_hetero_batch.py`,
  `test_hetero_layers.py`, `test_temporal_v025.py`.

- **Examples**: `hetero_graph_batch_demo.py`,
  `hetero_graph_classifier_demo.py`, `temporal_graph_batch_demo.py`,
  `temporal_graph_classifier_demo.py`.

### Changed

- `HeteroGraph` — added `node_label_stores`, `graph_label`,
  `edge_weight`/`edge_features` accessors, `device` property, `*_dict`
  property aliases.  All additions are optional and backward-compatible.
- `__init__.py` — exports `HeteroGraph`, `HeteroGraphBatch`,
  `TemporalGraphSequence`, `TemporalGraphBatch` at the top level.
- README, `docs/limitations.md`, `docs/roadmap.md` — updated to reflect
  the new functionality (no longer "container only").
- `pyproject.toml`: version bumped to 0.2.5.

### Deferred (with exact reason)

- **Tensor-aware spatial hetero classifiers** — `HeteroConv` already
  accepts tensor-aware layers per relation, but a canned spatial-feature
  classifier requires careful per-type spatial-rank validation.
  Planned v0.2.6.
- **Temporal recurrent memory (TGN/TGAT)** — requires a memory module
  with proper graph-level state management.  The current snapshot-loop
  classifier is sufficient for many tasks but is not a substitute.
  Planned v0.2.6+.

---

## [0.2.4] — 2026-05-08 (release prep)

### Added

- **`TensorGATLayer` two-pass chunked forward** — pass `chunk_size=K` to
  `forward()`.  Uses a numerically-stable log-sum-exp two-pass algorithm:
  Pass 1 accumulates per-destination/head max statistics over edge chunks;
  Pass 2 computes globally normalised exp-weighted values.  Memory use for
  intermediate edge tensors scales as O(K × K_heads × C_head × spatial)
  instead of O(E × …).  Supports `return_attention=True`, edge weight,
  vector/spatial edge features, 2-D/3-D spatial rank, and bfloat16 autocast.

- **`TensorGATLayer(attention_mode="channel")` — 🧪 Experimental** — one
  score per (edge, head, channel) instead of a single scalar per (edge, head).
  Attention is softmax-normalised per destination per head per channel.
  Supported by both unchunked and chunked paths.

- **`GraphTransformerLayer` — 🧪 Experimental** — global self-attention
  transformer layer for vector node features `[N, D]`.  Multi-head attention,
  feed-forward sublayer, residual, layer norm, dropout.  O(N²) with a warning
  for N > 1 000.  Tensor-aware (spatial/volumetric) input deferred.

- **`HeteroGraph` container — 🧪 Experimental** — lightweight typed-node /
  typed-edge data store.  Validation, `.to(device)`, repr.  No GNN layers.

- **`TemporalGraphSequence` container — 🧪 Experimental** — list of graph
  snapshots with optional timestamps.  Iteration, indexing, `.to(device)`.

- **`MLflowLogger`** — optional MLflow metric logger.  Lazy `mlflow` import
  (no mandatory dependency).  Context-manager API consistent with
  `CSVLogger`/`TensorBoardLogger`.  Added `mlflow` optional extra to
  `pyproject.toml`.

- **`tgraphx.interop`** — optional PyG/DGL data converters:
  `to_pyg_data`, `from_pyg_data`, `to_dgl_graph`, `from_dgl_graph`.
  All imports are lazy; no mandatory dependency.

- **`tgraphx.learned_graph`** — opt-in learned/soft graph construction:
  `soft_adjacency_from_embeddings` (differentiable), `top_k_edges_from_scores`
  (non-differentiable top-k), `build_knn_graph_from_embeddings`,
  `EdgeScorer` (learnable MLP edge scorer).

- **`image_to_patches` / `volume_to_patches` `padding="auto"`** — new
  optional `padding` argument.  Default `"none"` is unchanged (raises on
  non-divisible dims).  `"auto"` right-pads to make dimensions exactly
  divisible by `patch_size`.

- **README rewrite** — replaced the "What is NOT yet implemented" wall with a
  concise "Current scope and boundaries" section with a status table.
  Details moved to `docs/limitations.md` and `docs/roadmap.md`.

- **73 new tests** (`tests/test_gat_chunking.py`, `tests/test_v024_features.py`)
  covering all new features.

- **New examples**: `gat_chunking_demo.py`, `v024_new_features.py`.

### Changed

- `TensorGATLayer.__init__` accepts new `attention_mode` parameter (default
  `"scalar"` — fully backward-compatible).
- `TensorGATLayer.forward` accepts new `chunk_size` parameter (default
  `None` — fully backward-compatible).
- `tgraphx.tracking` module docstring updated.
- `pyproject.toml`: added `[mlflow]` optional extra; bumped version to 0.2.4.
- `__init__.py`: bumped `__version__` to `0.2.4`.

### CI / Release

- **Fixed Windows CI failure**: `pip install torch torchvision \\` used
  POSIX line-continuation that PowerShell rejects.  All `pip install`
  commands in `.github/workflows/tests.yml` now use single-line form.
- **Fixed Ubuntu Dashboard live server smoke flake**: replaced fragile
  fixed `time.sleep(1.5)` with a port-binding poll loop (up to 30s) that
  also captures and prints server stdout/stderr if the port never binds.
  This eliminates the "Connection refused" cascade that blocked the
  Ubuntu 3.10/3.11/3.12 matrix.
- Wheel install smoke and Optional extras smoke jobs run again
  automatically once `test` matrix succeeds (they `needs: test`).
- README and `tests/test_documentation_claims.py` updated to reflect
  v0.2.4 features (Graph Transformer, hetero/temporal containers,
  learned-graph helpers, PyG/DGL converters, MLflowLogger) instead of
  the prior "❌ Not supported" claims.

### Deferred (with exact reason)

- **GAT per-pixel / per-voxel attention** — score tensors would be
  `O(E × K × H × W)` per layer.  For E=10K, K=4, H=W=8: ~10M floats for
  scores alone.  Planned after memory-efficiency analysis.
- **Full hetero/temporal GNN layers** — container types added but message
  passing not implemented.
- **Tensor-aware GraphTransformerLayer** — spatial/volumetric [N,C,H,W]
  input requires redesigning the O(N²) attention to operate on spatial
  feature maps.  Planned for a future release.

---

## [0.2.3] — 2026-05-08

### Added

- **`TensorGraphSAGELayer` chunked forward** — pass `chunk_size=K` to
  `forward()` to process edges in chunks of size `K`, reducing the peak
  per-edge message buffer from O(E × spatial) to O(K × spatial).
  Supported for both `aggr="mean"` and `aggr="max"`.  Output matches
  unchunked within float32 precision; gradients flow correctly.

- **`TensorGINLayer` chunked forward** — same interface as SAGE chunking.
  The sum aggregation is exact (associativity); learnable epsilon and custom
  MLP paths both supported.

- **`build_knn_graph(chunk_size=K)`** — processes `K` rows of the pairwise
  distance matrix at a time, reducing peak memory from O(N²) to O(K×N).
  Output matches the full (unchunked) path exactly.  O(N²) time unchanged.

- **`build_radius_graph(chunk_size=K)`** — same benefit as kNN chunking.

- **`build_iou_graph(chunk_size=K)`** — processes `K` boxes at a time;
  O(K×N) peak memory.

- **`build_random_graph(algorithm="sample")`** — O(num_edges) memory sampling
  for directed graphs without self-loops.  Deterministic with `seed`.
  Default `algorithm="exact"` is unchanged (backward-compatible).

- **Dashboard byte-seek tail-read** — `DashboardServer` now tracks a byte
  offset for `metrics.csv`.  When the file only grows (same inode, larger
  size), only the new bytes are read and parsed; existing rows stay in the
  in-memory cache.  Full reparse triggered on inode change (log rotation) or
  file shrinkage (truncation).

- **Dashboard `?since_row` double-read fix** — the incremental path
  previously re-read the full file from disk even on a cache hit.  It now
  uses the in-memory full-row cache, eliminating the redundant disk read.

- **`tests/test_chunking.py`** — 46 new tests covering SAGE (mean/max), GIN,
  3-D volumetric variants, edge weights, vector/spatial edge features,
  isolated nodes, gradient flow, bfloat16 smoke, and graph builder chunking.

### Changed

- `benchmarks/benchmark_layers.py` — `--chunk-size` now also applies to
  SAGE and GIN layers (previously only ConvMessagePassing).

- O(N²) warning messages for `build_knn_graph` and `build_radius_graph`
  updated to mention `chunk_size` as a memory-reduction option.

### Not implemented (deferred)

- **`TensorGATLayer` chunked forward** — deferred to v0.2.4.  Correct
  implementation requires a two-pass algorithm (Pass 1: accumulate
  per-destination max/logsumexp statistics over chunked score batches;
  Pass 2: recompute normalised weights and aggregate values).  Single-pass
  normalisation inside chunks is mathematically incorrect and not shipped.

---

## [0.2.2] — 2026-05-08

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

## [0.2.1] — 2026-05-07

### Fixed
- Corrected README/support-table contradictions around TensorGATLayer
  spatial and volumetric edge features.
- Corrected stale ConvMessagePassing `aggr="max"` documentation.
- Replaced overconfident Windows/macOS support wording with
  best-effort/no-CI wording.

### Added
- README support-status legend and backend/feature/scalability/attention
  support tables.
- Runtime O(N²) warnings for large fully connected, kNN, radius, and IoU
  graph builders.
- Documentation-claim regression tests.
- `docs/roadmap.md`.

### Changed
- Made README and docs more explicit about supported, best-effort,
  planned, and unsupported features.

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

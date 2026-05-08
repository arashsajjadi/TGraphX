# TGraphX Roadmap

This document tracks planned work by release.  Items listed here are **not
yet implemented**.  "Planned" means the feature is on the roadmap; it is not
a committed delivery date.

See [Limitations](limitations.md) for the full list of known constraints and
[CHANGELOG.md](../CHANGELOG.md) for completed work.

---

## v0.2.1 (current prep)

Focus: documentation accuracy, contradiction fixes, runtime warnings.

- ✅ Fix `TensorGATLayer` edge-feature table contradiction (spatial edges
  are accepted via mean-pooling; the ✗ entry was wrong).
- ✅ Fix `ConvMessagePassing` docs: `aggr="max"` is supported (stale
  "NotImplementedError" claim removed).
- ✅ Fix hardware table: Windows/macOS "Fully supported" → "Best-effort (no CI)".
- ✅ Add `Support status` section with backend/feature/scalability/attention tables.
- ✅ Add runtime O(N²) `warnings.warn` to `build_knn_graph`, `build_radius_graph`,
  `build_fully_connected_graph`, `build_iou_graph`.
- ✅ Add `tests/test_documentation_claims.py`.
- ✅ Add this roadmap.

---

## v0.2.2 — AMP / dtype / backend robustness

- Investigate and document `TensorGATLayer` float16 autocast path (the
  `index_add_` dtype-mismatch issue).
- Add dtype-cast guard or clear runtime error for float16 GAT autocast.
- Smoke-test `torch.compile` across all four spatial GNN families.
- Document `torch.compile` known-good and known-broken op patterns.
- Add CI matrix entry for PyTorch 2.x (currently pinned to latest).

---

## v0.2.3 — Chunking and scalability ✅ (released)

- ✅ `TensorGraphSAGELayer` chunked forward (`aggr="mean"` and `"max"`).
- ✅ `TensorGINLayer` chunked forward (`aggr="sum"`).
- ✅ Dashboard byte-seek tail-read for `metrics.csv`.
- ✅ Dashboard `?since_row` incremental double-read fix.
- ✅ `build_knn_graph` / `build_radius_graph` / `build_iou_graph`: `chunk_size`
  parameter for O(K×N) peak memory.
- ✅ `build_random_graph(algorithm="sample")` for O(num_edges) memory sampling.

---

## v0.2.4 — GAT chunking, attention modes, ecosystem ✅ (release prep)

- ✅ `TensorGATLayer` two-pass chunked forward — log-sum-exp algorithm:
  Pass 1 accumulates per-destination max statistics, Pass 2 normalises and
  aggregates values.  Output matches unchunked within float32 tolerance.
- ✅ `TensorGATLayer(attention_mode="channel")` — 🧪 experimental per-channel
  attention with shape `[E, K, C_head]`.
- ✅ `image_to_patches(padding="auto")` and `volume_to_patches(padding="auto")`.
- ✅ `MLflowLogger` — optional, lazy `mlflow` import, `tgraphx[mlflow]` extra.
- ✅ `tgraphx.interop` — optional PyG/DGL data converters (lazy imports).
- ✅ `tgraphx.learned_graph` — soft adjacency, top-k edges, EdgeScorer,
  `build_knn_graph_from_embeddings`.
- ✅ `tgraphx.core.hetero_graph.HeteroGraph` — 🧪 experimental container.
- ✅ `tgraphx.core.temporal.TemporalGraphSequence` — 🧪 experimental container.
- ✅ `tgraphx.layers.graph_transformer.GraphTransformerLayer` — 🧪 experimental
  vector-only global self-attention with FFN, residual, LayerNorm.
- ✅ CI hardening: Windows pip line-continuation fix; Ubuntu dashboard live
  smoke uses port-binding poll instead of fixed sleep.

---

## v0.2.5 — Hetero / Temporal real functionality ✅ (release prep)

- ✅ `HeteroGraphBatch` — disjoint typed-node/typed-edge batching with
  per-type batch vectors and clean error reporting for inconsistent stores.
- ✅ `HeteroConv` — relation-dispatch wrapper with sum/mean/max
  cross-relation aggregation.  Vector node features fully supported;
  spatial features supported when relation modules accept them.
- ✅ Hetero readouts — `hetero_mean_pool`, `hetero_sum_pool`,
  `hetero_max_pool`, `hetero_concat_pool` with stable type ordering.
- ✅ `HeteroGraphClassifier`, `HeteroNodeClassifier` — vector-feature
  classifiers with per-type input projections.
- ✅ `TemporalGraphBatch` — equal-length and variable-length sequence
  batching with per-snapshot masks and padded timestamps.
- ✅ `temporal_readout` — `last`/`mean`/`max` with mask support.
- ✅ `TemporalGraphClassifier`, `TemporalGraphRegressor` — stateless
  snapshot-loop wrappers that delegate to a base graph encoder.
- ✅ Hetero PyG/DGL converters — `to_pyg_heterodata`, `from_pyg_heterodata`,
  `to_dgl_heterograph`, `from_dgl_heterograph` (lazy imports).
- ✅ 52 new tests + 4 new examples.

Deferred to v0.2.6+:
- ⏳ Hetero tensor-aware spatial classifiers (canned).
- ⏳ Temporal recurrent memory module (TGN/TGAT-style).
- ⏳ Temporal sampling utilities.

---

## v0.2.6 — Sampling, minibatching, distributed feasibility ✅ (release prep)

- ✅ `induced_subgraph`, `edge_subgraph`, `k_hop_subgraph`, `sample_nodes`,
  `sample_edges`, `neighbor_sample` in `tgraphx.sampling`.
- ✅ `SubgraphDataLoader`, `NeighborSamplerLoader` in
  `tgraphx.sampling_loaders` — plain iterables, deterministic with seed,
  no hidden multiprocessing.
- ✅ `tgraphx.distributed` helpers (`get_rank`, `get_world_size`,
  `is_rank_zero`, `rank_zero_print`, `@rank_zero_only`, `barrier`).
  Never auto-initialises DDP.
- ✅ `benchmarks/benchmark_sampling.py` (CI-safe `--small` mode).
- ✅ Examples: `neighbor_sampling_demo.py`, `ddp_training_smoke.py`.

Deferred to v0.2.7+:
- ⏳ Hetero / temporal sampling (per-relation, per-snapshot semantics).
- ⏳ Random-walk sampling.
- ⏳ Multi-GPU full DDP example (currently single-process smoke only).

---

## v0.2.7 — Graph Transformer maturity ✅ (release prep)

- ✅ `GraphTransformerLayer` (vector-only) hardened: ``positional_encoding``
  (degree / Laplacian), ``pe_dim``, ``edge_bias`` constructor args; new
  ``positional`` and ``edge_bias_dense`` forward kwargs (default-None,
  backward-compatible).
- ✅ `tgraphx.layers.transformer_encodings` — pure-PyTorch helpers:
  `degree_encoding`, `laplacian_eigvec_encoding`, `build_adjacency_bias`.
- ✅ Factory integration: `make_layer("graph_transformer", ...)`.
- ✅ Stability docs: `docs/experimental_policy.md`,
  `docs/deprecation_policy.md`, `docs/migration_v0_2_to_v0_3.md`.

Deferred to post-v0.3:
- ⏳ Tensor-aware Graph Transformer (token granularity is an open
  design question).
- ⏳ Memory-safe attention approximations (sparse / linear / chunked).

---

## v0.2.8 — Sampling completion + README honesty ✅ (release prep)

- ✅ `random_walk_sample` (homogeneous): direction `out` / `in`,
  configurable `num_walks_per_seed` and `restart_prob`, deterministic
  with `seed`, no global RNG side effects.
- ✅ Hetero sampling: `hetero_induced_subgraph` and
  `hetero_neighbor_sample` (per-relation fanout, multi-hop, BFS-style;
  preserves edge weights, edge features, node labels, graph label).
- ✅ Temporal window sampling: `temporal_window_sample` and
  `temporal_window_sample_batch` (variable-length aware).
- ✅ Top-level re-exports + new tests
  (`tests/test_random_walk_sample.py`, `tests/test_hetero_sampling.py`,
  `tests/test_temporal_sampling.py` — 49 new tests).
- ✅ Example: `examples/sampling_demo_v028.py` (CPU-safe).
- ✅ README/limitations/roadmap rewritten to reflect reality:
  - GAT chunked forward and `attention_mode="channel"` correctly
    labelled stable / experimental rather than "Planned v0.2.4".
  - Hetero / temporal classifiers, `HeteroConv`, sampling, distributed
    helpers correctly listed as shipped (instead of "out of scope").
  - Windows / macOS smoke CI properly credited (was "No CI").
  - Stale "v0.2.4" / "v0.2.5+ planned" wording removed.

Deferred:
- ⏳ Per-pixel / per-voxel GAT attention (only with a memory-safe
  variant — naive memory cost prohibitive).
- ⏳ Generic rank-agnostic message-passing layer (v0.3 design).
- ⏳ Recurrent temporal memory module (TGN / TGAT).
- ⏳ Full automatic multi-GPU training framework.

---

## v0.2.9 — Ecosystem expansion (planned)

- Robust PyG/DGL converter coverage (batches, hetero edge attributes,
  per-relation masks).
- Optional plugin architecture for external integrations.
- Optional curated public-dataset loaders (no required network calls).
- Optional GraphRAG-adjacent local utilities (purely local; no remote APIs).

---

## v0.3.0 — Stabilization

- Promote mature experimental APIs to stable.
- Backward-compatibility audit and deprecation policy.
- Migration guide.
- Final README/support-matrix cleanup.
- Full release audit.

---

## Honest positioning

- **"Planned"** items may be re-scoped, split, or moved to later versions.
- **"Feasibility study"** means we will evaluate before committing.
- Items in v0.3.x depend on v0.2.x work completing without major API
  changes that would break the v0.3.x design.
- No SOTA or superiority claims are made for any planned feature.

---

## See also

- [Limitations](limitations.md)
- [Performance](performance.md)
- [CHANGELOG.md](../CHANGELOG.md)

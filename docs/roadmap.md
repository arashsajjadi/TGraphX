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

## v0.2.3 — Chunking and scalability

- `TensorGraphSAGELayer` chunked forward (mean/max aggregation splits cleanly).
- `TensorGINLayer` chunked forward.
- Investigate `TensorGATLayer` two-pass chunked forward (first pass: collect
  all scores for destination-wise softmax; second pass: aggregate values).
- Dashboard incremental metrics: true byte-seek tail-read for large
  `metrics.csv` files (currently: mtime/size cache + full re-parse on miss).
- Graph builder scalability: document O(N²) alternatives (approximate-NN).

---

## v0.2.4 — Richer attention and edge features

- Per-channel attention in `TensorGATLayer` (attention vector per head per
  channel, not scalar).
- Per-pixel / per-voxel attention feasibility study and optional flag.
- Patch helper optional padding (pad-to-nearest-tile) with explicit warnings.
- Edge-feature consistency audit: ensure ConvMP, GAT, SAGE, GIN handle
  mismatched-rank edge tensors with clear errors everywhere.

---

## v0.2.5 — Ecosystem integrations and extended feature set

- Optional `MLflowLogger` (gated behind `pip install "tgraphx[mlflow]"`).
- Optional PyG/DGL `edge_index` converters (read-only; no full API compat).
- Heterogeneous graph containers (design doc; not full GNN implementation).
- Temporal graph containers (design doc; not full GNN implementation).
- Graph Transformer feasibility study:
  - Global self-attention with learnable positional encodings.
  - Decide on architecture and scope before committing to implementation.

---

## v0.3.x — Stable expanded feature set

Items below are deferred until the v0.2.x series stabilises.

- Stable Graph Transformer layer (if v0.2.5 feasibility is positive).
- Stable heterogeneous GNN layers (HeteroConv-style dispatch).
- Stable temporal GNN layers (sequence-aware message passing).
- Broader PyTorch version matrix in CI (1.13, 2.0, 2.x latest).
- Neighbor sampling (GraphSAINT / ClusterGCN style mini-batch training).
- Larger-scale benchmark suite (OGB or similar public datasets).

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

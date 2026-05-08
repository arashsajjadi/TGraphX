# TGraphX Roadmap

This document tracks **planned** future work toward TGraphX v0.4.0.  Items
listed here are not yet committed deliveries — they are the direction of
travel.

For completed work see [`CHANGELOG.md`](../CHANGELOG.md); for known
constraints see [`limitations.md`](limitations.md); for the architectural
plan that drives this roadmap see [`architecture.md`](architecture.md).

---

## Milestone summary

| Version | Theme | Status |
|---------|-------|--------|
| v0.1.x – v0.2.x | Core layers, datasets, transforms, metrics, dashboard | Released |
| v0.3.0 | Experiment manager, explainability, vector model zoo, dashboard hardening | Released |
| v0.3.1 | Audit hardening, doc fixes, CNNEncoder top-level export, Python 3.10+ | Released |
| **v0.3.2** | **Public benchmark campaign + foundations** | **In progress** |
| v0.3.3 | Mature samplers / loaders / link-prediction negative sampling | Planned |
| v0.3.4 | Hetero / temporal stabilization + deeper OGB/TGB integration | Planned |
| v0.3.5 | Optional acceleration + sparse backends + out-of-core foundation | Planned |
| v0.3.6 | Larger model zoo + graph algorithms foundation + community files | Planned |
| v0.4.0 | Stabilization release: stable API, real benchmarks, full docs nav | Planned |

---

## v0.3.2 — Public benchmark campaign and foundations (in progress)

**Theme:** convert TGraphX from "unit-tested framework" to "validated on real
public examples where feasible", with honest reporting.  Lay foundations for
later milestones via small primitives.

Planned:

- `benchmarks/public/` package — clean foundation for public-dataset
  benchmark scripts with a uniform CLI (`--root`, `--download`,
  `--max-samples`, `--max-nodes`, `--epochs`, `--device`, `--output-dir`,
  `--seed`, `--json`, `--strict`).
- Public benchmark scripts:
  - `mnist_patch_benchmark.py` (FakeData by default; real MNIST opt-in via
    `--download`),
  - `pyg_cora_benchmark.py` (skips cleanly without PyG),
  - additional scripts (FashionMNIST, EMNIST, OGB) to follow.
- `docs/benchmark_protocol.md` — how to run public benchmarks, exactly
  what JSON they emit, what they do *not* claim.
- `docs/public_benchmark_reports.md` — table of small-data engineering
  metrics from local runs (no SOTA, no superiority).
- Foundation primitives (small, isolated, well-tested):
  - `tgraphx.sampling.negative_sampling`,
    `structured_negative_sampling`, `batched_negative_sampling`.
  - `tgraphx.algorithms` package — `connected_components`,
    `weakly_connected_components`, `is_connected`, `bfs_edges`,
    `shortest_path_length`.
  - `tgraphx.temporal.time_encoding` — sinusoidal and Time2Vec-style
    learnable encodings.

What v0.3.2 is **not**:

- Not a NetworkX replacement.  Algorithms are GNN-oriented utilities, not
  full graph-analytics.
- Not a leaderboard.  Public benchmark reports are local engineering
  smoke runs, not benchmark wins.
- Not full TGN/TGAT.  Temporal additions are time-encoding primitives,
  not memory modules yet.

---

## v0.3.3 — Mature samplers, loaders, link prediction (planned)

- `NeighborSampler` / `NeighborLoader` — layer-wise fanouts, replace /
  no-replace, deterministic generators, original-ID preservation.
- `GraphSAINTNodeSampler` / `GraphSAINTEdgeSampler` /
  `GraphSAINTRandomWalkSampler` (label experimental until normalization
  is fully validated).
- Cluster-GCN-style partitioning: random and BFS partitioners (METIS
  optional, no mandatory dependency).
- `NodeLoader`, `LinkLoader`, `GraphLoader` minibatch interfaces.
- Hard-negative sampling using embedding scores.
- `tgraphx.pipeline` — GraphBolt-inspired lightweight pipeline
  (`ItemSet` → `sample_items` → `sample_neighbors` → `fetch_features`).
  Pure Python, no mandatory dependency.
- Sampling benchmarks under `benchmarks/sampling/`.

---

## v0.3.4 — Hetero / temporal stabilization, deeper OGB/TGB integration (planned)

- `tgraphx.hetero` package consolidation — `HeteroGraph`, `HeteroBatch`,
  relation-aware samplers, `RGCNConv`, `HGTConv`, `HANConv` (vector
  features stable first).
- `tgraphx.temporal` package consolidation — `TemporalGraphSequence`,
  `TemporalGraphBatch`, time encodings, optional memory module
  (TGN-inspired, experimental).
- `tgraphx.integrations.ogb` — `OGBDatasetAdapter` polish, evaluator
  wrapping, official split access.
- `tgraphx.integrations.tgb` — optional TGB adapter (skips cleanly if
  TGB is missing).
- Hetero / temporal benchmarks under `benchmarks/hetero/` and
  `benchmarks/temporal/`.

---

## v0.3.5 — Optional acceleration and storage foundation (planned)

- `tgraphx.backends` — backend registry (`pure_torch` default, optional
  `torch_scatter` / `pyg_lib` adapters) with strict numerical parity
  tests against the pure-PyTorch path.
- Sparse utilities: `edge_index_to_csr/csc` and back, `coalesce_edges`,
  `sort_edge_index`, `degree`, `segment_*` reductions, `sparse_softmax`,
  `edge_softmax`.
- `tgraphx.storage` — `InMemoryFeatureStore`, `MemmapFeatureStore`,
  `GraphStore`, partition metadata.  Out-of-core utilities, **not**
  billion-edge production training.

---

## v0.3.6 — Model zoo expansion, algorithms, community (planned)

- Vector model zoo additions: `GraphSAGEConv` (vector), `SGC`,
  `ChebConv`, `TAGConv`, `EdgeConv`, `GatedGraphConv`, `MLPBaseline`.
- Architectures: `GCNNet`, `GATNet`, `GraphSAGENet`, `GINNet`,
  `APPNPNet`, `HeteroRGCNNet`.
- Tensor-aware nets: `ConvMessagePassingNet`, `TensorGATNet`,
  `PatchGraphClassifier`, `VolumeGraphClassifier`.
- Algorithms expansion: `pagerank` (small graph), `clustering_coefficient`,
  `triangle_count`, structural feature helpers.
- Community: `CONTRIBUTING.md` overhaul, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, GitHub issue templates, contribution guide,
  example gallery with difficulty labels.

---

## v0.4.0 — Stabilization release (planned)

- Freeze the v0.3.x stable APIs.
- Move unfinished systems into a clearly labelled `tgraphx.experimental`
  namespace.
- Real benchmark report (small-graph, single-machine, honest).
- Docs site / tutorial gallery.
- Release-quality README, `api_stability.md`, `limitations.md`,
  `migration_v0_3_to_v0_4.md`.
- No half-implemented claims.

---

## Honest positioning

- "Planned" items may be re-scoped, split, or moved to later versions.
- "Feasibility study" means evaluation precedes commitment.
- TGraphX is **not** trying to replace PyTorch Geometric, DGL/GraphBolt,
  cuGraph, NetworkX, or OGB/TGB.  It is a tensor-aware graph learning
  library with a local-first dashboard, an explicit no-telemetry
  philosophy, and an honest documentation discipline.
- No SOTA or superiority claims are made for any planned feature.
- No CUDA CI claim, no full MPS support claim, no full automatic
  multi-GPU training claim.

---

## See also

- [Architecture plan](architecture.md)
- [Limitations](limitations.md)
- [API stability policy](api_stability.md)
- [Performance](performance.md)
- [CHANGELOG.md](../CHANGELOG.md)

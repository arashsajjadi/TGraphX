# TGraphX Architecture (v0.3.x → v0.4.0)

This document describes the architectural plan that drives the
[roadmap](roadmap.md) toward TGraphX v0.4.0.  It is the engineering
contract between the existing v0.3.x core and the new subsystems being
added incrementally.

The goal is to close the most useful gaps with major graph-learning
libraries (PyTorch Geometric, DGL/GraphBolt, OGB/TGB, NetworkX, cuGraph)
**while preserving TGraphX's identity**:

- tensor-aware image and volumetric graph learning,
- a local-first dashboard,
- explicit no-telemetry / no-hidden-downloads policy,
- clean PyTorch-native APIs,
- honest documentation discipline.

---

## Architectural goals

1. **Stability of v0.3.x stable APIs.**  Every new subsystem must be
   purely additive.  No silent breakage of `Graph`, `GraphBatch`,
   the four core layers, the dataset registry, transforms, metrics,
   experiment manager, explainability, or dashboard contracts.
2. **Optional dependencies stay optional.**  No new mandatory
   dependency.  Heavy libraries (`torch_scatter`, `pyg_lib`,
   `torch_geometric`, `dgl`, `ogb`, `tgb`, `networkx`, `cugraph`,
   `metis`) are imported lazily and gated by extras.
3. **Subsystem modules over monolith files.**  Each new system lives in
   its own package under `tgraphx/`, with a clear public surface and
   tests.
4. **Pure PyTorch fallbacks.**  Any optional acceleration must have a
   pure-PyTorch parity path so users without the optional dependency
   still get correct results.
5. **No hidden state.**  Pipelines, samplers, and stores accept explicit
   seeds, paths, and devices.  No global RNG mutation, no global cache,
   no auto-launched dashboards or processes.
6. **Honest documentation.**  Every new module documents its scope,
   its boundaries, what it is *not*, and its stability level (stable,
   beta, experimental, research).

---

## Module map (target v0.4.0)

The map below shows where each subsystem lives.  Entries marked
"existing" are already in v0.3.x.  Entries marked "planned" are part of
the v0.3.2 → v0.4.0 plan.

```
tgraphx/
├── __init__.py
│
├── core/                     # existing — Graph, GraphBatch, hetero, temporal containers
├── layers/                   # existing — tensor-aware + vector layers
├── models/                   # existing — Classifier/Regressor/EdgePredictor
├── graph_builders.py         # existing
│
├── datasets/                 # existing — registry + synthetic + folder + adapters
├── transforms/               # existing
├── metrics/                  # existing
│
├── training.py               # existing — train_epoch / evaluate / fit
├── tracking.py               # existing — CSV/TB/MLflow loggers + 14 metadata writers
├── performance.py            # existing
│
├── experiments/              # existing v0.3.0 — Runner / GridRunner / callbacks
├── explain/                  # existing v0.3.0 — saliency / IG / edge attribution
├── dashboard/                # existing
│
├── sampling.py               # existing — induced/k-hop/neighbor/random-walk
├── sampling_loaders.py       # existing — SubgraphDataLoader / NeighborSamplerLoader
├── hetero_sampling.py        # existing
├── temporal_sampling.py      # existing
├── distributed.py            # existing — rank helpers (no auto-init)
├── interop.py                # existing — PyG/DGL converters
├── learned_graph.py          # existing
│
├── algorithms/               # planned v0.3.2 — connectivity, traversal, paths, …
│   ├── connectivity.py       #   connected_components, is_connected
│   ├── traversal.py          #   bfs_edges, shortest_path_length
│   ├── centrality.py         #   degree centrality, pagerank (small graphs)
│   └── structural_features.py
│
├── temporal/                 # planned v0.3.2 → v0.3.4
│   ├── time_encoding.py      #   v0.3.2 — sinusoidal + Time2Vec
│   ├── memory.py             #   v0.3.4 — node memory module (TGN-inspired)
│   └── tgn.py / tgat.py      #   v0.3.4 — experimental
│
├── pipeline/                 # planned v0.3.3 — GraphBolt-inspired pipeline
│   ├── items.py              #   ItemSet / ItemSampler
│   ├── stages.py             #   sample_neighbors, fetch_features, …
│   └── minibatch.py          #   MiniBatch object
│
├── partition/                # planned v0.3.3 — Cluster-GCN partitioners
│   ├── random.py
│   └── bfs.py
│
├── hetero/                   # planned v0.3.4 — RGCN / HGT / HAN
│   ├── conv.py
│   ├── samplers.py
│   └── models.py
│
├── backends/                 # planned v0.3.5 — optional sparse acceleration
│   ├── registry.py
│   ├── scatter.py            #   pure_torch + optional torch_scatter
│   └── sparse.py             #   CSR/CSC utilities
│
├── storage/                  # planned v0.3.5 — feature store / out-of-core
│   ├── feature_store.py      #   InMemoryFeatureStore + MemmapFeatureStore
│   └── graph_store.py
│
└── integrations/             # planned v0.3.4
    ├── ogb.py                #   evaluator wrapping, official splits
    ├── tgb.py                #   optional TGB adapter
    └── cugraph.py            #   optional cuGraph interop hooks
```

`benchmarks/` mirrors this layout where it makes sense:

```
benchmarks/
├── benchmark_dataset_loading.py     # existing
├── benchmark_layers.py              # existing
├── benchmark_metrics.py             # existing
├── benchmark_sampling.py            # existing
├── benchmark_transforms.py          # existing
├── benchmark_training_synthetic.py  # existing
├── benchmark_tensor_vs_flatten.py   # existing
├── benchmark_graph_builders.py      # existing
│
├── public/                          # planned v0.3.2 — public-dataset benchmarks
│   ├── _common.py
│   ├── mnist_patch_benchmark.py
│   ├── pyg_cora_benchmark.py
│   └── …
│
├── sampling/                        # planned v0.3.3
├── hetero/                          # planned v0.3.4
├── temporal/                        # planned v0.3.4
├── backend/                         # planned v0.3.5
└── storage/                         # planned v0.3.5
```

---

## Subsystem boundaries

### Sampling and pipeline

- `tgraphx.sampling.*` — primitive samplers that take and return
  `Graph`/`HeteroGraph`/`TemporalGraphSequence` objects.  Pure functions
  with a `seed` argument.
- `tgraphx.sampling_loaders.*` — Python iterables wrapping samplers for
  training loops.  No multiprocessing by default.
- `tgraphx.pipeline.*` (planned) — composable pipeline stages built on
  top of the samplers.  Inspired by GraphBolt; not a GraphBolt clone.

### Algorithms

- `tgraphx.algorithms.*` — pure-PyTorch graph algorithms used by GNN
  workflows (connectivity, traversal, centrality, clustering).
- Optional NetworkX integration: `to_networkx_graph` /
  `from_networkx_graph` are kept in `tgraphx.interop`, **not** in the
  algorithms package, to keep the latter dependency-free.

### Heterogeneous and temporal

- `tgraphx.hetero.*` — relation-typed graph layers and samplers.
  Vector node features stable first, tensor-aware extension experimental.
- `tgraphx.temporal.*` — time-encoded GNN building blocks, snapshot
  utilities, memory modules.  Memory modules and TGN/TGAT-style
  layers are experimental until tested on real temporal benchmarks.

### Backends and storage

- `tgraphx.backends.*` — registry of optional acceleration paths.  The
  default is always `pure_torch`.  Each non-default backend ships a
  parity test against `pure_torch` on small graphs; mismatches fail CI.
- `tgraphx.storage.*` — feature/graph stores.  In-memory store is
  trivial; memmap store is the first out-of-core path.  Path safety is
  enforced (no traversal, no symlink escape).

### Integrations

- `tgraphx.integrations.*` — wrappers around external packages (PyG,
  DGL, OGB, TGB, cuGraph).  Each wrapper:
  - imports the external package lazily,
  - raises a clear `OptionalDependencyError` if the package is missing,
  - is gated by a `tgraphx[<extra>]` extra in `pyproject.toml`.

---

## Optional dependency policy

| Extra | Package(s) | Used by |
|-------|------------|---------|
| `tgraphx[dev]` | pytest, build, twine | development |
| `tgraphx[monitoring]` | psutil, pynvml | dashboard hardware panel |
| `tgraphx[tracking]` | tensorboard | `TensorBoardLogger` |
| `tgraphx[mlflow]` | mlflow | `MLflowLogger` |
| `tgraphx[pyg]` | torch-geometric | PyG dataset adapters + interop |
| `tgraphx[ogb]` | ogb | OGB adapters + evaluator |
| `tgraphx[pillow]` | Pillow | `ImageFolderPatchGraphDataset` |

DGL is intentionally not packaged as an extra (its wheels are
platform-sensitive); follow the upstream DGL install guide.

Planned future extras (v0.3.4–v0.3.5):

| Extra | Package(s) | Used by |
|-------|------------|---------|
| `tgraphx[tgb]` | tgb | TGB adapter + evaluator |
| `tgraphx[scatter]` | torch_scatter | optional sparse backend |
| `tgraphx[pyg_lib]` | pyg-lib | optional sparse backend |
| `tgraphx[cugraph]` | cugraph (cuGraph) | optional GPU graph analytics interop |

---

## Stability levels

Each public symbol is classified at one of four levels, recorded in
[`docs/api_stability.md`](api_stability.md):

| Level | Promise |
|-------|---------|
| **Stable** | API frozen across the v0.3.x series; deprecations follow the 2-minor-version policy in [`docs/deprecation_policy.md`](deprecation_policy.md). |
| **Beta** | Almost stable; minor signature changes possible during v0.3.x. |
| **Experimental** | API may change without deprecation across patch versions; documented as such in the symbol's docstring and in `docs/experimental_policy.md`. |
| **Research** | Not exposed at the top-level package; lives under `tgraphx.experimental.*` (added in v0.4.0). |

The current v0.3.x core (Graph, GraphBatch, the four spatial layers,
LinearMessagePassing, the dataset registry, transforms, metrics,
experiment manager, explainability, and the dashboard) is **stable**.
Hetero and temporal containers and helpers are **beta**.  Graph
Transformer, learned graph utilities, hetero PyG/DGL converters, and
per-channel attention are **experimental**.

New v0.3.2 additions follow the same discipline:

| Symbol | Level |
|--------|-------|
| `tgraphx.sampling.negative_sampling` | Beta |
| `tgraphx.sampling.structured_negative_sampling` | Beta |
| `tgraphx.sampling.batched_negative_sampling` | Beta |
| `tgraphx.algorithms.connected_components` | Beta |
| `tgraphx.algorithms.weakly_connected_components` | Beta |
| `tgraphx.algorithms.is_connected` | Beta |
| `tgraphx.algorithms.bfs_edges` | Beta |
| `tgraphx.algorithms.shortest_path_length` | Beta |
| `tgraphx.temporal.time_encoding.sinusoidal_time_encoding` | Beta |
| `tgraphx.temporal.time_encoding.LearnableTimeEncoding` | Experimental |

---

## Testing discipline

Every new subsystem must ship with:

1. **Unit tests** in `tests/` covering forward/backward,
   shape/dtype/device, edge cases (empty graph, isolated nodes,
   self-loops), and any documented invariants.
2. **Determinism tests** for anything that uses an RNG.
3. **Mathematical invariant tests** for graph algorithms (e.g.
   permutation equivariance, edge-order invariance, conservation laws).
4. **Optional-dependency tests** that verify the lazy-import contract
   and that missing dependencies produce actionable errors.
5. **Benchmark smoke** under `benchmarks/<area>/` with a `--small` mode
   safe for CI.

A new feature does not graduate from research to experimental until
unit tests exist, and does not graduate from experimental to beta until
benchmark and example coverage exists.

---

## Documentation discipline

Each new public module ships with:

- a `docs/<module>.md` describing purpose, install/dependency notes,
  import path, minimal example, an API table, key arguments, return
  types, performance notes, common errors, related examples, and
  related tests;
- an entry in `docs/index.md`;
- a one-line mention in `README.md` only if the module is stable and
  user-facing.

The README is a public contract — it stays concise.  Detailed
information lives in `docs/`.

---

## Dashboard integration contract

The local-first dashboard is a TGraphX differentiator.  Every new
subsystem that produces user-visible artifacts must:

- write its metadata as JSON to a path the user provides explicitly;
- never trigger an implicit download or background process;
- never embed code or scripts the dashboard could execute;
- match an entry in `docs/dashboard.md`'s artifact schema table;
- have a `tracking.write_<name>_metadata` helper if appropriate.

Existing artifacts are documented in `tracking.py` and
`docs/dashboard.md`.  Planned new artifacts include:

- `sampler_metadata.json` (existing)
- `loader_metadata.json` (planned v0.3.3)
- `pipeline_metadata.json` (planned v0.3.3)
- `partition_metadata.json` (planned v0.3.3)
- `feature_store_metadata.json` (planned v0.3.5)
- `backend_benchmark.json` (planned v0.3.5)

The dashboard server already tolerates missing or malformed JSON files;
new artifacts are best-effort renders, never crashes.

---

## Security and privacy invariants

The following invariants are enforced by tests in
`tests/test_documentation_claims.py`,
`tests/test_dataset_download_mocked.py`,
`tests/test_dashboard.py`, and
`tests/test_imports.py`:

1. Importing `tgraphx` makes **no** network calls.
2. Importing `tgraphx` does **not** import torch_geometric, dgl, ogb,
   tgb, networkx, cugraph, mlflow, or tensorboard.
3. Datasets never download unless the caller passes `download=True`
   explicitly (or `--download` to a CLI script).
4. Configs are loaded with `yaml.safe_load`; no `eval`, no `exec`.
5. The dashboard never executes user code, never loads checkpoints,
   never embeds external CDN assets, and never leaks tokens.
6. Archive extraction (zip / tar) blocks path-traversal attempts.
7. No background threads are started unless the user explicitly calls
   `launch_dashboard_background(...)`.

Every new subsystem must preserve these invariants.

---

## See also

- [Roadmap](roadmap.md)
- [API stability policy](api_stability.md)
- [Experimental policy](experimental_policy.md)
- [Deprecation policy](deprecation_policy.md)
- [Limitations](limitations.md)
- [CHANGELOG](../CHANGELOG.md)

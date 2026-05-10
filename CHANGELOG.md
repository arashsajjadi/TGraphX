# Changelog

All notable changes to TGraphX are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.0.3] — 2026-05-09

Final cleanup pass before the v1.1 development sprint.

### Fixed
- Benchmark smoke tests (`test_generation_rl_benchmarks_smoke.py`) no longer
  produce spurious TIMEOUT failures under concurrent CPU load; subprocess
  timeouts increased to 120 s / 90 s and the fragile wall-clock assertion
  removed.
- `examples/run_all_fast_examples.py` now has a per-example `TIMEOUT_OVERRIDES`
  dict; the TD3/SAC demo gets 120 s (was sharing the 60 s global default).

### Added
- `KnowledgeGraph.from_hrt(heads, relations, tails, ...)` — classmethod for
  users who have separate h/r/t tensors instead of a combined `[N_t, 3]` matrix.
  The existing `from_triples` classmethod accepts tuple lists and `[N_t, 3]`
  tensors.  The constructor error message now explicitly points to both helpers.
- `result.plot_loss()` and `result.plot_metrics()` now raise `ValueError` with
  an actionable message when no training history is available, and report
  available metric keys if `"loss"` is absent.
- `tgx.easy.list_models(group_by_task=True)` — nested dict grouped by task for
  better discoverability (default flat dict unchanged, backward-compatible).
- `.gitignore` explicit `!TGRAPHX.png` / `!TGRAPHX_LOGO.svg` exceptions so
  tracked logo assets are not accidentally re-ignored on fresh clones.

### Changed
- `docs/index.md` — added "Quick navigation" table and Easy Mode / LLM guide
  section; added NeighborLoader row in the Sampling table.
- `docs/neighbor_loader.md` — new section explaining dense vs sparse global-ID
  mapping paths in `map_global_to_local`.
- `docs/limitations.md` — added "Resolved in v1.0.1–v1.0.3" table, roadmap
  items, and explicitly-out-of-scope section.
- `docs/easy_mode.md` — added dashboard integration guidance, seed-cost note,
  and `tgraphx-info` alias clarification.
- `docs/api_stability.md` — version header updated to `v1.0.1+`; added
  `graph_features` field note and `KnowledgeGraph.from_triples` entry.
- `tgraphx/__main__.py` — docstring clarifies `tgraphx-info` is an alias for
  `tgraphx-doctor` (both registered console scripts; same behaviour).

---

## [1.0.2] — 2026-05-09

UX ergonomics cleanup.

### Fixed
- `float(loss)` in `easy/_workflows.py` replaced with `loss.detach().item()` to
  silence autograd scalar-conversion warnings.
- `benchmarks/ux/benchmark_easy_vs_manual.py` fairness: manual and easy branches
  now use the same two-layer architecture, same device (`--device cpu` default),
  same `set_seed` call, and median-of-3 timing.  Measured overhead: −0.46%.
- `map_global_to_local` (`tgraphx/loaders.py`): added `torch.searchsorted`
  fallback for global node IDs > 2 000 000 to avoid O(max_id) memory.
- `Graph(graph_features=...)`: now stores a **distinct** graph-level input
  feature in `self.graph_features` instead of aliasing to `graph_label`.

### Changed
- `tgraphx/easy/` split from a single 1 009-line file into a modular package.
  Public API unchanged.
- `pyproject.toml`: SPDX `license = "MIT"` + `license-files`; deprecated
  `License` classifier removed.  Build is now warning-free.
- Added `TestGraphFeaturesSemantics`, sparse-ID tests for
  `TestMapGlobalToLocal`, and deterministic `test_graph_num_classes`.

---

## [1.0.1] — 2026-05-09

First UX / API ergonomics hardening pass.

### Added
- `Graph(..., y=y)`, `Graph(..., labels=y)`, `Graph(..., edge_attr=ef)` and
  related PyG-style properties (`g.x`, `g.y`, `g.edge_attr`, `g.num_classes`,
  masks).
- `Graph.has_labels()`, `Graph.get_labels()`, `Graph.with_labels(y)`.
- `GraphMiniBatch` — ergonomic batch object from `NeighborLoader` with
  `batch.node_features`, `batch.seed_y`, `batch.seed_logits(logits)`, etc.
  Legacy tuple unpacking preserved.
- `map_global_to_local`, `seed_logits` helpers.
- `tgraphx.easy` package: `train_node_classifier`, `synthetic_*`, `list_tasks`,
  `list_models`, `list_samplers`, `doctor`, `EasyResult`, `EasyConfig`, and
  custom exceptions.
- `python -m tgraphx` CLI; `tgraphx-doctor` / `tgraphx-info` console scripts.
- Canonical tutorial `tutorials/tensor_node_classification_neighbor_loader.py`.
- `docs/easy_mode.md`, `docs/neighbor_loader.md`, `docs/llm_usage_guide.md`,
  `docs/api_cheatsheet.json`, `docs/user_experience_api_contract.md`.
- 96 new tests (`test_user_friendly_llm_snippets.py`,
  `test_api_stability_labels.py`).
- `benchmarks/ux/benchmark_easy_vs_manual.py`.
- README routing table, Easy Mode section, fixed KG quickstart API.

---

## [1.0.0] — 2026-05-09

First stable research release of TGraphX.

### Summary

- **Tensor-native GNN layers** preserving `[C,H,W]` and `[C,D,H,W]` node features through message passing (Beta)
- **Graph algorithms** — BFS/DFS, shortest paths, MST, max-flow, matching, coloring (Beta)
- **Graph mining** — motifs, centrality, spectral, WL features, Node2Vec/DeepWalk (Beta)
- **Sampling and loaders** — GraphSAINT, Cluster-GCN, NeighborLoader, LinkNeighborLoader, GraphLoader (Beta)
- **Feature store** — InMemoryFeatureStore, MemmapFeatureStore (Beta)
- **Sparse backend** — CSR/CSC, coalesce, segment ops (Beta)
- **Knowledge graphs** — TransE/DistMult/ComplEx/RotatE, filtered ranking, RGCN, multimodal entity features (Beta/Experimental)
- **Hypergraphs** — incidence matrix, clique/star expansion (Experimental)
- **GAE / VGAE** — dot-product and MLP edge decoders, link prediction (Experimental)
- **Heterogeneous graphs** — RGCN, HAN, HGT, typed neighbor sampling (Experimental)
- **Temporal graphs** — TGNMemory, TGATConv, time encodings, temporal splits (Experimental)
- **Graph generation** — 14 methods (classical + neural VGAE/autoregressive/transformer), quality metrics, one-liner API (Experimental)
- **Evolutionary graph optimization** — GA, SA, NSGA-II, hill climbing, random search, multi-objective Pareto, one-liner API (Experimental)
- **Graph RL** — 13 algorithms (random through SAC), 9 environments (navigation through continuous graph edit), one-liner API (Experimental)
- **Dashboard** — local HTTP server, offline HTML, generation/evolution/RL artifact writers, no telemetry (Beta)
- **Tutorials** — 3 CPU-runnable Colab-ready quickstarts (Stable)
- **Benchmarks** — 13 scripts with `--small --json` validation, all including `package_version`, `seed`, `device`, `status`, `limitations` (Stable)
- **Reproducibility** — `set_seed`, deterministic mode, `reproducibility_report.json` (Beta)
- **sklearn-like API** — estimators, GraphPipeline, EarlyStopping (Beta)
- **Experiment manager** — YAML/JSON configs, runners, CLI (`tgraphx-train`) (Beta)
- **Wheel** — pure Python wheel, no data files, lightweight import, optional features skip cleanly

### API changes from v0.6.0

- `run_evolutionary_optimization()` now accepts `n_generations` as an alias for `generations`
- `OptimizationResult` now has `.metrics` dict (best_fitness, n_generations, algorithm, objective)
- `list_graph_rl_algorithms()` returns a `dict` (not a list); use `.keys()` for names
- All 13 benchmark scripts now include `package_version`, `seed`, `device`, `status`, `limitations` in JSON output

### Tests and validation

- 217 generation/RL/evolutionary tests pass; full suite 2000+ tests pass
- All 13 benchmarks pass `--small --json`
- CPU + CUDA device validation: `all_passed: true`
- Wheel: `twine check PASSED`, wheel smoke PASSED
- All 3 tutorials: PASS

---

## [0.6.0] — 2026-05-08

### Added — Graph generation, evolutionary optimization, graph RL subsystems

- (Experimental): `tgraphx.generation` — classical generators (ER, BA, SBM, WS, grid, cycle, path, star, complete, random geometric, temporal, typed, anomaly/motif injected), neural generators (VGAE, autoregressive, transformer), action spaces, quality metrics (validity, uniqueness, novelty, diversity, MMD, WL hash), projectors for vector/image/volume features, high-level API `run_graph_generation`, dashboard report writers
- (Experimental): `tgraphx.evolutionary` — `GraphGenome`, mutation/crossover operators, selection (tournament, roulette, rank, elitism, diversity-preserving), fitness functions (connectivity, density, clustering, motif count, composite), NSGA-II multi-objective (non-dominated sort, crowding distance, hypervolume), algorithms (GA, SA, NSGA-II, hill climbing, random search), high-level API `run_evolutionary_optimization`, dashboard report writers
- (Experimental): `tgraphx.rl` — 9 graph environments (navigation, shortest path, coloring, max-cut, vertex cover, graph generation, KG reasoning, continuous navigation, continuous graph edit), 13 algorithms (random, greedy, REINFORCE, actor-critic, A2C, DQN, double DQN, dueling DQN, PPO, DDPG, delayed DDPG, TD3, SAC), exploration strategies (epsilon-greedy, Boltzmann, UCB, entropy), high-level API `run_graph_rl` / `list_graph_rl_algorithms`, auto-routing of continuous algorithms to continuous environments, dashboard report writers
- (Experimental): `tgraphx.generation.neural` — `VGAEGraphGenerator`, `AutoregressiveEdgeGenerator`, `GraphTransformerGenerator`
- Tutorials: `tutorials/graph_generation_quickstart.py`, `tutorials/evolutionary_optimization_quickstart.py`, `tutorials/graph_rl_quickstart.py` — CPU-runnable, deterministic, under 60 seconds each
- Examples: `examples/neural_graph_generation_demo.py`, `examples/generation_rl_high_level_api_demo.py`, `examples/graph_ppo_demo.py`, `examples/graph_td3_sac_demo.py`
- Benchmarks: `benchmarks/generation/`, `benchmarks/evolution/`, `benchmarks/rl/` — all support `--small --json` for CI-friendly runs
- Tests: 217 new tests for generation/evolution/RL subsystems; all pass on Python 3.10-3.13 (CPU and CUDA)
- Tensor-native: projectors (VectorNodeProjector, ImageNodeEncoder, VolumeNodeEncoder, EdgeFeatureProjector, GraphFeatureProjector) preserve `[N,C,H,W]` and `[N,C,D,H,W]` feature shapes; gradients reach projectors; no silent flattening
- Dashboard artifacts: `generation_*.json`, `evo_*.json`, `rl_*.json` written by high-level APIs; no raw tensors; malicious strings not eval'd

### Changed
- README: added generation/evolution/RL algorithm tables, tutorial section, all new example references
- CHANGELOG: formalized v0.6.0 entry

---

## [Unreleased] — v0.5.0 candidate

### Added — Large-scale loaders, feature store, sparse backend, VGAE, RGCN

-  (Beta): NeighborLoader, LinkNeighborLoader, GraphLoader, convenience factories
-  (Beta): InMemoryFeatureStore, MemmapFeatureStore
-  (Beta): coalesce, sort, remove/add self-loops, degree, segment ops, CSR/CSC, chunked top-k
-  (Experimental): GraphAutoencoder, VGAE, GCNEncoder, decoders, link prediction metrics
-  (Experimental): RGCNConv with basis decomposition
- Dashboard: 8 new API endpoints for KG, hypergraph, VGAE, loaders, feature store
- Report writers: write_kg_summary, write_hypergraph_summary, write_vgae_report, write_loader_summary, write_feature_store_summary
- Tests: 2045 pass (+53), 0 warnings; 60 examples pass

---

## [Unreleased] — v0.4.4 candidate

### Added — Matching, coloring, clique enumeration, and max-flow (`tgraphx/mining/matching_coloring.py`) — Beta/Experimental

- `greedy_maximal_matching` — greedy O(E) maximal matching
- `bipartite_greedy_matching` — greedy matching for bipartite graphs
- `greedy_coloring` / `welsh_powell_coloring` — valid graph coloring with largest-first ordering
- `greedy_maximal_independent_set` — seed-controlled O(N+E) MIS
- `enumerate_maximal_cliques` — Bron-Kerbosch with pivot (guarded at 50 nodes)
- `edmonds_karp_max_flow` — BFS-based Ford-Fulkerson max-flow for small graphs (guarded at 500 nodes)
- `min_cut_from_max_flow` — min-cut via max-flow min-cut theorem
- `wl_isomorphism_test` — WL graph isomorphism heuristic (necessary but not sufficient)
- `write_algorithm_report` — JSON dashboard artifact writer

Tests: `tests/test_matching_coloring_flow.py` (21 tests).

### Added — Node2Vec / DeepWalk unsupervised embeddings (`tgraphx/mining/node2vec.py`) — Experimental

- `node2vec_walks` — biased random walks with return parameter p and in-out parameter q
- `deepwalk_walks` — uniform walks (p=q=1 special case)
- `generate_skipgram_pairs` — (center, context, negative) pairs from walks
- `Node2VecEmbedding` — skip-gram embedding model with negative sampling
- `train_node2vec_step` — one training step
- `extract_node2vec_embeddings` — L2-normalised embedding extraction

Validation: loss decreases on synthetic 2-community SBM graph; intra-community similarity (0.936) >> inter-community (-0.679).

### Added — Knowledge graph foundation (`tgraphx/mining/knowledge_graph.py`) — Experimental

- `KnowledgeGraph` — triple container with entity/relation ID mapping, positive lookup, random split
- `negative_triple_sampling` — head/tail corruption with optional filtered negatives
- `filtered_ranking_metrics` — filtered MRR / Hits@1/3/10 evaluator
- `TransE` — margin-based scoring (Bordes et al., 2013)
- `DistMult` — trilinear scoring with L2 regularization (Yang et al., 2015)
- `train_kg_step` — one training step

Tests: `tests/test_node2vec_kg_hypergraph.py` (35 tests) — ID mapping, loss decreases, gradient health, filtered metrics.

### Added — Hypergraph foundation (`tgraphx/mining/hypergraph.py`) — Experimental

- `Hypergraph` — incidence-list representation with hyperdegree, edge degree, density, summary
- `incidence_to_bipartite_graph` — bipartite node-hyperedge graph conversion
- `clique_expansion` — connect all members of each hyperedge
- `star_expansion` — connect each member to a hyperedge node
- `hypergraph_density` — fraction of active (node, hyperedge) pairs

### Added — Graph IO (`tgraphx/mining/graph_io.py`) — Beta

- `read_edge_list_csv` / `write_edge_list_csv` — CSV roundtrip with optional edge weights
- `read_graph_json` / `write_graph_json` — atomic JSON graph format with schema versioning
- `save_graph_npz` / `load_graph_npz` — compressed NumPy format with node features support

All IO: atomic writes, path-safe, no unsafe pickle, clear errors for malformed input.

### Added — Examples

- `examples/graph_algorithms_advanced_demo.py` — max-flow, matching, coloring, hypergraph, IO
- `examples/knowledge_graph_demo.py` — TransE + DistMult training demo
- `examples/node2vec_demo.py` — Node2Vec walks + embedding training

All added to `run_all_fast_examples.py`.

**Total: 1992 tests pass (+56 new), 12 skipped, 0 warnings. 58 examples pass.**

---

## [Unreleased] — v0.4.3 candidate

### Added — Graph path algorithms (`tgraphx/mining/paths.py`) — Beta

Traversal, shortest paths, spanning trees, and cut metrics in pure Python/PyTorch:

- **Traversal**: `bfs_order`, `dfs_order`, `multi_source_bfs`, `reachable_nodes`
- **Shortest paths**: `dijkstra_shortest_path` (Dijkstra with non-negative weights),
  `batched_shortest_path_length` (multiple sources), `all_pairs_shortest_path_length`
  (size-guarded at 1 000 nodes), `reconstruct_path` (predecessor dict → node list)
- **Spanning trees**: `minimum_spanning_tree`, `maximum_spanning_tree`
  (Kruskal with deterministic tie-breaking; returns forest for disconnected graphs)
- **Cut metrics**: `cut_size`, `normalized_cut`, `conductance`, `volume`,
  `boundary_edges`, `write_path_summary` (dashboard JSON artifact writer)

Tests: `tests/test_mining_paths_algorithms.py` (33 tests) — hand-computed path
lengths, negative weight error, MST correctness, cut size, conductance, path reconstruction.

### Added — Graph learning utilities (`tgraphx/mining/graph_learning.py`) — Experimental

Self-supervised learning foundations: contrastive losses, augmentations, and objectives:

- **Losses**: `contrastive_loss` (NT-Xent), `supervised_contrastive_loss` (Khosla et al.),
  `triplet_loss`, `bpr_loss` (Bayesian Personalised Ranking), `reconstruction_loss`
- **Augmentations**: `drop_edges`, `drop_nodes`, `mask_node_features`,
  `add_random_edges`, `subgraph_sampling` — all deterministic with ``seed``
- **Self-supervised objectives**: `DGIObjective` (Deep Graph Infomax-style),
  `GraphCLObjective` (GraphCL NT-Xent)
- **Utilities**: `create_negative_pairs`, `create_positive_pairs_from_batch`

Tests: `tests/test_mining_graph_learning.py` (36 tests) — forward/backward, determinism,
no-false-negatives, augmentation shape correctness, DGI loss finite.

### Added — Structural/positional encodings (`tgraphx/mining/structural_encodings.py`) — Beta

Node-level structural encodings for use in GNNs and graph transformers:

- `degree_encoding` — out/in-degree, normalised
- `random_walk_structural_encoding` — random-walk landing probabilities (RRWP, guarded at 2 000 nodes)
- `shortest_path_anchor_encoding` — distances to random anchor nodes
- `centrality_encoding` — degree, PageRank, eigenvector, Katz centralities
- `community_encoding` — one-hot community assignment
- `StructuralEncodingModule` — learnable linear projection of any encoding
- `attach_structural_encodings` — concatenates to vector node features;
  for spatial/volumetric features raises a clear error with ``mode="side"`` guidance

### Added — Graph sequence models (`tgraphx/mining/sequence_models.py`) — Experimental

RNN/LSTM/GRU foundations for graph-as-sequence tasks:

- `bfs_sequence_encode` / `random_walk_sequence_encode` — encode graph as traversal sequences
- `pad_sequences` — batch variable-length sequences
- `GraphSequenceEncoder` — LSTM encoder with ``"last"``/``"mean"``/``"max"`` pooling
- `GraphSequenceClassifier` — LSTM encoder + classification head; passes tiny-overfit test
- `GraphRNNEdgeGenerator` — GraphRNN-inspired (You et al., 2018) edge-sequence graph generator
  for tiny graphs; autoregressive generation; backward-compatible training step

Tests: forward/backward, determinism, tiny-overfit, shape checks, symmetric adjacency generation.

### Added — Examples

- `examples/graph_paths_algorithms_demo.py` — BFS/DFS, weighted Dijkstra, MST, cuts
- `examples/graph_learning_demo.py` — NT-Xent, SupCon, BPR, DGI, augmentations, sequence model

Both added to `run_all_fast_examples.py`.

**Total: 1936 tests pass (+69 new), 12 skipped, 0 warnings.**

---

## [Unreleased] — v0.4.2 candidate

### Added — Traditional graph mining expansion (Beta)

**`tgraphx/mining/centrality.py`** — 12 centrality algorithms, pure PyTorch,
size-guarded, with hand-computed toy tests:
- `degree_centrality`, `in_degree_centrality`, `out_degree_centrality`
- `pagerank` (power-iteration, O(iter·E))
- `personalized_pagerank`
- `hits` (hubs + authorities)
- `katz_centrality`
- `closeness_centrality` (exact BFS; max 2000 nodes)
- `harmonic_centrality` (handles disconnected graphs)
- `betweenness_centrality` (Brandes'; max 500 nodes)
- `eigenvector_centrality` (power-iteration)
- `k_core_numbers` (iterative peeling)

**`tgraphx/mining/generators.py`** — 14 graph generators, all deterministic with seed:
- `erdos_renyi_graph`, `barabasi_albert_graph`,
  `stochastic_block_model_graph`, `watts_strogatz_graph`,
  `random_geometric_graph`, `planted_partition_graph`
- `grid_2d_graph`, `complete_graph`, `cycle_graph`, `path_graph`, `star_graph`
- `karate_club_graph` (Zachary's karate club, 34 nodes)
- `synthetic_anomaly_graph`, `motif_injected_graph`

**`tgraphx/mining/spectral.py`** — 9 spectral utilities:
- `graph_laplacian`, `normalized_laplacian`
- `laplacian_eigenvalues`, `fiedler_vector`, `algebraic_connectivity`
- `laplacian_eigvec_positional_encoding` (Dwivedi et al., for Graph Transformers)
- `spectral_clustering` (k-means on Laplacian eigenvectors)
- `spectral_distance`, `dirichlet_energy`

**`tgraphx/mining/label_prop.py`** — Semi-supervised learning:
- `label_propagation(edge_index, num_nodes, y, mask, num_classes, alpha, ...)`
- `LabelPropagationClassifier` — sklearn-style API

**`tgraphx/mining/embeddings.py`** — Embedding extraction:
- `extract_node_embeddings(model, edge_index, node_features, ...)`
- `extract_graph_embeddings(model, graphs, pooling="mean", ...)`
- `embedding_similarity_matrix`, `embedding_pairwise_distances`,
  `embedding_nearest_neighbors`

**`tgraphx/mining/api.py`** — High-level API:
- `analyze_graph(edge_index, num_nodes, ...)` — one-call structural analysis
- `graph_mining_report(...)` — full-featured report with optional JSON output
- `run_link_prediction_baseline(...)` — run all classical scorers at once

### Added — Dashboard Mining UI (v0.4.2+)

**`tgraphx/dashboard/static/dashboard.js`**:
- New "Mining" navigation section (⛏ icon).
- Mining panel renderer: fetches 8 artifact types in parallel from new API
  endpoints, handles missing files gracefully, shows empty state with
  help text.
- Panels: Graph Overview, Motifs/Structural, Anomaly Detection, Communities,
  Prototype Membership, Neural Mining, Reproducibility.
- Inline SVG mini bar chart for motif counts.
- KV table and list table helpers with row caps (max 20/50 rows).
- All user-provided strings are HTML-escaped (via `esc()`).
- Responsive two-column layout collapses to single column on mobile.

**`tgraphx/dashboard/app.py`**:
- New API endpoints: `/api/mining_summary`, `/api/motif_summary`,
  `/api/anomaly_summary`, `/api/community_summary`,
  `/api/prototype_membership`, `/api/neural_mining`,
  `/api/reproducibility`, `/api/mining_benchmark`,
  `/api/link_prediction_summary`.
- `_api_json_file_capped()` helper — reads JSON and caps top-level lists
  to prevent browser overload on large artifact files.

**`tgraphx/dashboard/static/dashboard.css`**:
- Mining panel styles: `.mining-panel`, `.kv-table`, `.kv-key/.kv-val`,
  `.mining-cols` (responsive grid), `.mining-chart-wrap`,
  `.empty-panel`, `.warn-note`.

### Tests added

- `tests/test_mining_centrality.py` (30 tests) — hand-computed toy graphs,
  bounds, size guards, determinism.
- `tests/test_mining_generators_spectral.py` (35 tests) — generator
  determinism/properties, spectral math, label propagation correctness,
  embedding shapes.

**Total: 1867 tests pass, 12 skipped, 0 warnings.**

---

## [Unreleased] — v0.4.1 candidate

### Added — Reproducibility module (`tgraphx.reproducibility`) — Beta

- `set_seed(seed, deterministic=False, benchmark=None, warn_only=True)`
  — returns a state dict; supports `warn_only` via
  `torch.use_deterministic_algorithms`.
- `make_generator(seed, device="cpu")` — seeded `torch.Generator`
  without global RNG side effects.
- `seed_worker(worker_id)` — DataLoader worker init function for
  reproducible multi-worker loading.
- `reproducibility_report()` — JSON-serialisable snapshot of current
  determinism state (cuDNN flags, `PYTHONHASHSEED`, etc.).
- `deterministic_mode(seed, warn_only=True)` — context manager that
  enables deterministic mode and restores previous state on exit.
- Tests: `tests/test_reproducibility.py` (21 tests) covering same-seed
  CPU output, cross-process WL determinism via subprocess, DataLoader
  `seed_worker`, context manager state restoration.

### Added/Fixed — WL kernel determinism

- `tgraphx/mining/kernels.py`: `weisfeiler_lehman_labels` now uses
  `repr(key).encode("ascii")` byte keys (via `_stable_compress`) for
  the internal label dictionary.  This makes WL label assignment stable
  across separate Python processes regardless of `PYTHONHASHSEED`.
  Previously the dict used raw tuple keys whose hash was affected by
  `PYTHONHASHSEED` for Python < 3.x (integers are unaffected, but
  documentation was unclear).  Added cross-process subprocess tests.

### Added — Neural mining benchmark (`benchmarks/mining/benchmark_neural_mining.py`) — Beta

- Benchmarks `PrototypeMembershipScorer`, `GraphPatternClassifier`, and
  `GraphAutoencoderAnomalyDetector` on synthetic tasks.
- Reports training time, initial/final loss, loss-decreased flag, and
  gradient health summary per task.
- Supports `--small`, `--json`, `--seed`, `--device`, `--epochs`,
  `--num-graphs` CLI flags.
- Fixed: `benchmark_graph_similarity.py` — `torch.allclose()` returns
  a Python `bool` (not a Tensor) in recent PyTorch; removed spurious
  `.item()` call.

### Added — Batched `PrototypeMembershipScorer.score_batch_fast` — Beta

- `score_batch_fast(candidates)` performs a **single GNN pass** over a
  disjoint batched graph of all candidates, then extracts per-graph
  embeddings.  Gradient-compatible.  Numerically equivalent to
  `score_batch` for the same model weights and inputs.
- Tests: `tests/test_neural_mining_batched.py` (13 tests) covering
  shape, gradient flow, no cross-graph leakage, CUDA optional smoke.

### Added — Documentation

- `docs/reproducibility.md` — set_seed, make_generator, seed_worker,
  deterministic_mode, WL determinism policy, hardware caveats.
- `docs/neural_graph_mining.md` — PrototypeMembershipScorer,
  GraphAutoencoderAnomalyDetector, GraphPatternClassifier, training
  helpers, backprop behavior, limitations.
- `docs/plotting.md` — Matplotlib dependency, layouts, graph plots,
  mining plots, save_figure, Colab usage, performance notes.
- `docs/index.md` — links to the three new docs pages.
- `docs/api_stability.md` — v0.4.0/v0.4.1 Beta/Experimental sections.

---

## [0.4.0] - 2026-05-08 — v0.4.0 candidate: neural mining + plotting + benchmarks

### Added — Code

- `tgraphx.sampling_negative` (Beta) — link-prediction primitives that do
  not pull in a heavy dependency:
  - `negative_sampling(edge_index, num_nodes, num_neg_samples=None,
    method="sparse"|"dense", force_undirected=False, seed=None)`
  - `structured_negative_sampling(edge_index, num_nodes,
    contains_neg_self_loops=True, seed=None)`
  - `batched_negative_sampling(edge_index, batch, num_neg_samples=None,
    method="sparse"|"dense", force_undirected=False, seed=None)`

  Invariants enforced by `tests/test_negative_sampling.py` (27 tests):
  no false negatives, no self-loops, no duplicates within the output,
  determinism with `seed`, no global RNG pollution, batched sampling
  never crosses graph boundaries.

- `tgraphx.algorithms` (Beta) — pure-PyTorch graph algorithms used by
  GNN workflows.  Not a NetworkX replacement.
  - `connectivity.py`: `connected_components`,
    `weakly_connected_components`, `is_connected`,
    `number_connected_components` — iterative `min`-label propagation,
    O(diameter) iterations, runs on CPU and GPU.
  - `traversal.py`: `bfs_layers`, `bfs_edges`, `shortest_path_length`
    (unweighted single-source).

  Tests: `tests/test_algorithms.py` (26 tests).

- `tgraphx.temporal` package — initial home for the temporal subsystem.
  Ships in v0.3.2:
  - `sinusoidal_time_encoding(timestamps, dim, base=10_000.0)` (Beta) —
    parameter-free Transformer-style positional encoding for timestamps.
  - `LearnableTimeEncoding(dim, init_scale=0.01)` (Experimental) —
    Time2Vec-style trainable encoder (Kazemi et al., 2019).

  Tests: `tests/test_time_encoding.py` (19 tests) covering shape,
  dtype, finiteness, determinism, no global RNG pollution, gradient
  health.

  The existing `tgraphx.temporal_sampling`,
  `tgraphx.core.temporal{,_batch}`, and snapshot-loop classifiers are
  unchanged; consolidation under `tgraphx.temporal` continues in
  v0.3.4.

### Added — Neural graph mining (`tgraphx.mining.neural`) — Experimental

New trainable neural mining models, fully differentiable and tested:

- `PrototypeMembershipScorer` — GNN encoder + Siamese similarity scorer
  for class-graph membership.  Accepts vector and spatial node features
  (via `flatten_spatial=True`).  Passes tiny-overfit test (loss 0.69 →
  0.0001 on 2-class synthetic task).

- `GraphAutoencoderAnomalyDetector` — MSE reconstruction auto-encoder.
  `node_anomaly_scores` and `graph_anomaly_score` with `@no_grad`.
  Passes injected-anomaly validation (known anomalous nodes receive
  higher reconstruction error after training on normal data).

- `GraphPatternClassifier` — GNN + mean-pool + MLP classifier for
  structural pattern families.  Achieves 100% test accuracy on the
  synthetic path/star/cycle/complete pattern dataset (well-separated
  embeddings + correct stratified split).

- `create_synthetic_pattern_dataset(num_graphs_per_class, num_nodes,
  in_dim, seed, noise_std)` — deterministic 4-class dataset of path /
  star / cycle / complete graphs with class-specific node features.

- Training helpers: `train_prototype_membership_step`,
  `train_anomaly_autoencoder_step`,
  `train_graph_pattern_classifier_step`.

Tests: `tests/test_neural_mining.py` (33 tests) covering forward shape,
backward works, gradients finite, gradients non-zero, optimizer updates
params, tiny overfit, injected-anomaly detection, train/eval mode,
CUDA optional smoke, edge cases.

### Added — Plotting infrastructure (`tgraphx.plotting`) — Beta

New Matplotlib-only visualization package.  No seaborn, no NetworkX.
Headless-safe (Agg backend).  Colorblind-friendly Okabe-Ito palette.

**Layouts** (`tgraphx.plotting.layouts`):
- `circular_layout`, `grid_layout`, `random_layout` — O(N).
- `spring_layout` — Fruchterman-Reingold, pure Python/NumPy, O(N²·iters).

**Graph plots** (`tgraphx.plotting.graph`):
- `plot_graph` — scatter/line graph plot with configurable layout,
  node values, size guard.
- `plot_degree_distribution` — histogram of node degrees.
- `plot_adjacency_matrix` — binary heatmap.
- `plot_connected_components` — nodes coloured by component.

**Mining plots** (`tgraphx.plotting.mining`):
- `plot_motif_summary`, `plot_graph_mining_summary`
- `plot_link_prediction_score_distribution`
- `plot_graph_similarity_heatmap` (with matrix size cap)
- `plot_anomaly_scores`, `plot_prototype_membership_scores`
- `plot_confusion_matrix` (annotated, normalized)
- `plot_training_curves`, `plot_community_assignments`

**Utilities**: `save_figure(fig, path, formats=("png","svg","pdf"))`.

Tests: `tests/test_plotting.py` (31 tests) covering layout math,
figure creation, size guards, save to PNG/SVG, headless, empty inputs.

### Added — Mining benchmarks (`benchmarks/mining/`) — Beta

Five new benchmark scripts with uniform `--small --json` CLI:
- `benchmark_motifs.py`
- `benchmark_prototype_membership.py`
- `benchmark_anomaly_detection.py`
- `benchmark_graph_similarity.py`
- `benchmark_link_prediction.py`

Each reports: task, version, num_nodes, num_edges, device, seed, timing, correctness.

### Added — Graph mining subsystem (`tgraphx.mining`)

First serious graph mining package for TGraphX.  Tensor-aware, pure
PyTorch, no mandatory heavy dependency, no hidden downloads.

**Level 1 — Beta:**

- `tgraphx.mining.structural` — `graph_density`, `degree_statistics`,
  `graph_summary`, `structural_features`, `add_structural_features`.
  `add_structural_features` is tensor-aware: spatial/volumetric node
  features are stored in metadata rather than silently flattened.

- `tgraphx.mining.link_prediction` — classical link prediction scores:
  `common_neighbors_score`, `jaccard_score`, `adamic_adar_score`,
  `resource_allocation_score`, `preferential_attachment_score`.  All
  return ``FloatTensor[P]`` for P candidate pairs; zero denominators
  return 0.

- `tgraphx.mining.motifs` — `triangle_count` (graph and node level),
  `wedge_count`, `local_clustering_coefficient`, `motif_counts`,
  `motif_features`.  O(N·d²) with a density guard for large graphs.

- `tgraphx.mining.kernels` — `weisfeiler_lehman_labels`,
  `wl_feature_histogram`, `wl_graph_features`, `wl_kernel_matrix`
  (normalised, symmetric), `degree_histogram_features`.

- `tgraphx.mining.similarity` — `degree_histogram_distance`,
  `wl_feature_similarity`, `graph_feature_cosine_similarity`,
  `pairwise_graph_similarity`.

- `tgraphx.mining.communities` — `label_propagation_communities`
  (synchronous, deterministic with seed, compact output labels),
  `modularity`, `community_summary`.

- `tgraphx.mining.random_walk` — `random_walks` (dead-ends stay in
  place; biased Node2Vec p/q supported on CPU),
  `generate_random_walks`.

**Level 2 — Experimental:**

- `tgraphx.mining.anomaly` — `DegreeAnomalyScorer` (robust MAD
  z-score), `EgoDensityAnomalyScorer`, `graph_level_anomaly_scores`.

- `tgraphx.mining.prototype` — **TGraphX-native class-graph membership
  paradigm**.  `ClassGraphBuilder` (cosine kNN support graphs per
  class, density cap, bridge edges for connectivity),
  `CandidateGraphBuilder` (adds a query node to a class graph),
  `GraphMembershipDataset`, `MembershipEvaluator` (accuracy, balanced
  accuracy, macro F1, confusion matrix, top confusion pairs),
  `cosine_graph_membership_baseline`.  Fully tensor-aware: spatial and
  volumetric node features are preserved unchanged.

- `tgraphx.mining.patterns` — `path_pattern_count`, `star_pattern_count`,
  `contains_triangle`, `small_pattern_counts`.

- `tgraphx.mining.frequent` — `frequent_node_labels`,
  `frequent_degree_bins`, `support_count`.

- `tgraphx.mining.temporal` — `temporal_degree`, `sliding_window_edges`,
  `temporal_chronological_split` (no future leakage), `burst_score`.

- `tgraphx.mining.hetero` — `typed_degree_features`,
  `relation_frequency_features`.

**Reports (Beta):**

- `tgraphx.mining.reports` — `write_graph_mining_summary`,
  `write_motif_summary`, `write_link_prediction_summary`,
  `write_anomaly_summary`, `write_prototype_membership_report`.
  All writers use atomic writes (temp-then-rename).

**Tests:** `tests/test_mining_structural.py` (20 tests),
`tests/test_mining_core.py` (69 tests).  Mathematical invariants
verified: K3 triangle = 1, K4 triangle = 4, WL identical-graph
similarity = 1.0, clustering coefficient of K3 nodes = 1.0,
common neighbours hand-computed, temporal split no future leakage.

**Examples (5):** `graph_mining_structural_demo.py`,
`graph_mining_link_prediction_demo.py`, `graph_mining_wl_kernel_demo.py`,
`graph_mining_anomaly_demo.py`, `prototype_graph_membership_demo.py`.
All added to `run_all_fast_examples.py`.

**Docs:** `docs/graph_mining.md`.

### Added — Top-level re-exports

- `tgraphx.negative_sampling`, `tgraphx.structured_negative_sampling`,
  `tgraphx.batched_negative_sampling` are now available from the
  top-level package.

### Added — Public benchmarks

- `benchmarks/public/` package with a uniform CLI
  (`--root`, `--download`, `--max-samples`, `--max-nodes`, `--epochs`,
  `--device`, `--output-dir`, `--seed`, `--json`, `--strict`):
  - `_common.py` — argparse helpers, device resolution, soft-skip
    handling, the four-file artefact writer.
  - `mnist_patch_benchmark.py` — FakeData by default (no network),
    real torchvision MNIST opt-in via `--download`.
  - `pyg_cora_benchmark.py` — Planetoid Cora; skips cleanly when
    `torch_geometric` is missing, instructs to pass `--download`.

  Tests: `tests/test_public_benchmarks.py` (8 tests) verify `--help`,
  optional-dependency skip, JSON schema of `benchmark_results.json`,
  no-network default for MNIST, `--strict` behaviour.

### Added — Documentation

- `docs/architecture.md` — master architecture plan for the v0.3.2 →
  v0.4.0 work, including the module map, optional-dependency policy,
  stability levels, testing/documentation discipline, dashboard
  artefact contract, and security/privacy invariants.
- `docs/roadmap.md` — rewritten around the new v0.3.2 → v0.4.0
  milestones (public benchmarks, samplers/loaders, hetero/temporal
  stabilisation, optional acceleration, model zoo expansion,
  stabilisation release).
- `docs/benchmark_protocol.md` — the protocol every public benchmark
  script in `benchmarks/public/` follows.
- `docs/public_benchmark_reports.md` — local engineering metrics
  recorded from public-dataset runs (no leaderboard claims).

### Safe Extras (v0.3.2 audit pass)

- `tgraphx.algorithms.structural` (Beta) — `degree(edge_index, num_nodes,
  mode="out"|"in"|"both")` and `degree_features(edge_index, num_nodes,
  log_scale)` for structural node features.  Tests: `tests/test_graph_utils.py`
  (17 tests).

- `tgraphx.sampling_negative.hard_negative_sampling` (Beta) — sample
  negatives with high embedding similarity without allocating an O(N²)
  matrix.  `candidate_pool_size` controls memory.  Deduplicates and
  excludes positives before ranking by cosine / dot similarity.
  Tests: `tests/test_hard_negative_sampling.py` (17 tests).

- `benchmarks/public/fashionmnist_patch_benchmark.py` — FakeData
  default (no network); real FashionMNIST opt-in via `--download`.  Same
  uniform CLI as `mnist_patch_benchmark.py`.

- Examples:
  - `examples/negative_sampling_demo.py`
  - `examples/graph_algorithms_demo.py`
  - `examples/time_encoding_demo.py`
  All three are added to `examples/run_all_fast_examples.py`.

- Docs:
  - `docs/negative_sampling.md` (full reference for all 4 sampling functions)
  - `docs/graph_algorithms.md` (connectivity, traversal, structural utilities)
  - `docs/temporal.md` (time encoding + planned v0.3.4 surface)

### Bug fixes (v0.3.2 audit pass)

- `tgraphx/sampling_negative.py`: `structured_negative_sampling` now
  returns output tensors on the same device as the input `edge_index`.
  Previously always returned CPU tensors for CUDA graphs.

- `tgraphx/algorithms/traversal.py`: `_check` now validates
  `edge_index` bounds when `num_nodes` is provided.  Previously a
  node-id ≥ num_nodes caused a cryptic PyTorch index error instead of a
  clear `ValueError`.

- `tgraphx/temporal/time_encoding.py`:
  - `sinusoidal_time_encoding`: removed redundant double-cast
    (`to(float32) if not fp else to(float32)`).
  - `sinusoidal_time_encoding`: `base ≤ 0` now raises a clear
    `ValueError` instead of the cryptic `math domain error` from
    `math.log`.

### Additional tests (v0.3.2 audit pass)

- `tests/test_negative_sampling.py`: +7 tests for complete-graph (0
  possible negatives), single-node, two-node, all-but-one-edge, and
  CUDA device preservation.

- `tests/test_algorithms.py`: +7 tests for self-loop/duplicate edge
  robustness, out-of-range node validation, large-sparse-ring smoke,
  optional NetworkX parity.

- `tests/test_time_encoding.py`: +6 tests for `base ≤ 0` validation,
  `dim=2` edge case, large-timestamp finiteness, and CUDA device
  preservation.

### Stability classifications

| Symbol | Level |
|--------|-------|
| `negative_sampling`, `structured_negative_sampling`, `batched_negative_sampling` | Beta |
| `connected_components`, `weakly_connected_components`, `is_connected`, `number_connected_components` | Beta |
| `bfs_layers`, `bfs_edges`, `shortest_path_length` | Beta |
| `sinusoidal_time_encoding` | Beta |
| `LearnableTimeEncoding` | Experimental |
| `benchmarks/public/*` scripts | Beta (CLI may evolve) |

### Honest scope

- v0.3.2 does **not** add a memory module (TGN/TGAT-style); only the
  time-encoding primitives.  Memory modules land in v0.3.4.
- The graph algorithms package is not a NetworkX replacement; only
  GNN-oriented utilities are included.
- Public benchmarks are smoke / engineering metrics, not leaderboard
  numbers.  No SOTA or superiority claim.

---

## [0.3.1] — 2026-05-08

Final v0.3.0 audit hardening released as a patch.

### Added — Documentation

- `docs/experiments.md` — full experiment-manager reference (config schema,
  Runner, GridRunner, callbacks, CLI, run artifacts).
- `docs/explainability.md` — explainability reference (saliency, integrated
  gradients, edge attribution, patch heatmaps, exports, dashboard compatibility).

### Added — Code

- `CNNEncoder` promoted to top-level `tgraphx` namespace (was only at
  `tgraphx.models`).  README stable core APIs table listed it alongside
  other top-level classes; the import path now matches that presentation.

### Fixed — Code

- `tgraphx/datasets/folder.py`: replaced `Image.getdata()` (deprecated in Pillow,
  removed in Pillow 14) with `numpy.array(im)` — equivalent result, no
  deprecation warning, no new dependency (numpy is always present with PyTorch).
- `tests/test_synthetic_datasets.py`: replaced `float(loss)` with
  `loss.detach().item()` — eliminates the "Converting a tensor with
  requires_grad=True to a scalar" UserWarning.

### Fixed — Packaging / metadata

- `pyproject.toml`: bumped `requires-python` from `>=3.9` to `>=3.10` and
  removed the `Python :: 3.9` classifier.  Python 3.9 reached end-of-life in
  October 2025 and CI only covers 3.10 / 3.11 / 3.12; the badge, README, and
  installation docs already said "Python 3.10+".
- `docs/installation.md`: updated requirement line from "Python ≥ 3.9" to
  "Python ≥ 3.10" to match `pyproject.toml` and README.

### Fixed — Documentation

- `docs/limitations.md` PyG / DGL section: corrected stale claim "there are no
  conversion utilities"; `tgraphx.interop` converters are noted.
- `docs/device_validation.md`: added dated manual Apple Silicon validation record
  for TGraphX 0.2.9 on macOS 15.5 arm64 (MPS available, smoke tests passed).
- `docs/datasets.md`: removed stale "(v0.2.9)" version pin from the opening
  sentence.
- `docs/performance.md`: corrected chunked-forward table row for `TensorGATLayer`
  — the two-pass log-sum-exp chunked forward shipped in v0.2.4 and is Stable;
  the row incorrectly said "Deferred v0.2.4".
- `docs/release_checklist.md`: added explicit pre-flight checks for stale dist/
  wheel rebuild and for stale tag pointing to wrong commit.

---

## [0.3.0] — 2026-05-07 (release prep)

First broad stabilisation release.  Adds an experiment manager, an
explainability foundation, a curated vector model zoo, dashboard
metadata writers for every major framework feature, and a
professionally rewritten README.

### Added — Experiment manager (`tgraphx.experiments`)

- `ExperimentConfig`, `DatasetConfig`, `ModelConfig`, `TrainingConfig`,
  `CallbackConfig`, `TransformConfig` dataclasses.
- `load_config(path_or_dict)` — YAML / JSON loader with safe parsing
  (no `eval`, no `exec`); rejects unknown top-level keys.
- `Runner` — config-driven training runner that writes
  `run_metadata.json`, `experiment_config.json`,
  `experiment_summary.json`, `metrics.csv`, and (when callbacks
  request them) `checkpoints/{best,latest}.pt` under an explicit
  `run_dir` only.
- `GridRunner` — multi-seed × cartesian-grid sweeps with a top-level
  `grid_summary.json`.
- Built-in callbacks: `EarlyStopping`, `ModelCheckpoint`,
  `CSVLoggerCallback`, `LearningRateLogger`.
- Console scripts: `tgraphx-train`, `tgraphx-grid`, `tgraphx-report`.
- `summarize_runs`, `write_markdown_report`, `write_summary_csv`.
- Three example configs under `examples/configs/` plus
  `examples/experiment_config_quickstart.py`.

### Added — Explainability (`tgraphx.explain`)

- `node_feature_saliency`, `integrated_gradients`,
  `edge_gradient_attribution`, `edge_perturbation_attribution`,
  `attention_to_edge_scores`, `patch_saliency_to_image_grid`,
  `patch_saliency_to_volume_projection`,
  `export_explanation_metadata`, `export_edge_scores_csv`,
  `export_patch_heatmap_json`.
- All methods CPU-safe, no autograd retention, no causal claims.
- Examples: `explainability_saliency_demo.py`,
  `explainability_attention_demo.py`.

### Added — Vector model zoo

- `tgraphx.layers.GCNConv` (Kipf & Welling, 2017).
- `tgraphx.layers.GATv2Conv` (Brody et al., 2022).
- `tgraphx.layers.APPNP` (Klicpera et al., 2019).
- `tgraphx.layers.global_sum_pool / global_mean_pool / global_max_pool`.
- `tgraphx.models.model_zoo.list_layers / make_zoo_layer`.
- Example: `model_zoo_demo.py`.

### Added — Dashboard metadata writers (`tgraphx.tracking`)

Eleven explicit, atomic JSON writers covering every dashboard-readable
file the v0.3.0 ecosystem can produce: `write_run_metadata`,
`write_dataset_metadata`, `write_transform_metadata`,
`write_metrics_summary`, `write_benchmark_results`,
`write_explanation_metadata`, `write_experiment_config`,
`write_hardware_report`, `write_sampling_metadata`,
`write_hetero_graph_metadata`, `write_temporal_metadata`.  Existing
artefacts (`metrics.csv`, `run_metadata.json`, `graph_metadata.json`,
`graph_stats.json`) are unchanged; the dashboard offline export and
live server stay fully backwards-compatible.

### Added — Tests (+88 vs. v0.2.9)

- `tests/test_experiments.py` — config validation, runner, early
  stopping, model checkpoint, run-dir-only writes, grid expansion,
  CLI invocation.
- `tests/test_explainability.py` — shape / finiteness / no-autograd-
  retention / export round-trip.
- `tests/test_model_zoo.py` — forward / backward / isolated nodes /
  registry / pooling.
- `tests/test_dashboard_metadata.py` — every new metadata writer +
  dashboard backwards compatibility + atomic-failure cleanup.
- `tests/test_math_invariants_v030.py` — permutation equivariance,
  edge-order invariance, GAT attention sums-to-one per destination
  per head, chunked-vs-unchunked GAT parity.
- `tests/test_tiny_overfit_v030.py` — synthetic-dataset trainability
  + gradient-health checks for 4-layer GCN stacks.
- `tests/test_documentation_claims.py` extended with two new tests
  (`test_readme_has_no_scary_symbols`,
  `test_readme_uses_calm_language`) that prevent regressions in the
  README's tone.

### Changed — README rewrite

The README is fully rewritten in calm, current-state, professional
prose:

- **Zero scary symbols.**  All `⚠️`/`❌`/`⛔`/`⏳`/`🧪`/`🚫`
  occurrences are removed.  Detailed limitations move to
  `docs/limitations.md`.
- New sections cover datasets/transforms/metrics/benchmarks, the
  experiment manager, explainability, the vector model zoo, the
  dashboard metadata writers, and an honest backend/platform table.
- Optional integrations are presented as a small table of adapters
  rather than a wall of warnings.
- A concise **Boundaries** section keeps the true technical limits
  visible without dramatising them.

`tests/test_documentation_claims.py::TestReadmeHonesty` prevents drift
back to the old style.

### Honest scope (kept, calmly worded)

- TGraphX is not a drop-in replacement for PyG or DGL.
- TGraphX provides DDP-aware helpers and a single-process smoke
  example, not an automatic multi-GPU training framework.
- Per-pixel / per-voxel GAT attention is not shipped; per-channel
  attention is shipped as `attention_mode="channel"`.
- Recurrent temporal memory modules (TGN, TGAT) are not shipped;
  temporal workflows use the stateless snapshot-loop pattern.
- Synthetic datasets are sanity / tutorial datasets, not benchmarks;
  benchmark scripts are reproducibility tools, not real-world
  performance comparisons.
- `kNN`, `radius`, `IoU`, and fully-connected graph builders are
  mathematically `O(N²)`; chunked variants reduce peak memory.
- Universal arbitrary-rank node-feature support across every layer is
  a future direction; the supported layouts today are vector,
  2-D spatial, and 3-D volumetric.

---

## [0.2.9] — 2026-05-07 (release prep)

### Added — Dataset ecosystem

- **`tgraphx.datasets`** — unified dataset registry, base classes,
  cache, safe download/extraction:
  - `BaseGraphDataset`, `InMemoryGraphDataset`,
    `DownloadableGraphDataset`, `ExternalDatasetAdapter`
  - `DatasetMetadata` (JSON-serialisable provenance record).
  - `register_dataset` / `get_dataset` / `list_datasets` /
    `dataset_info` / `available_dataset_groups`.
  - `cache_summary` / `clear_cache(dry_run=True)` / `resolve_dataset_root`.
  - `download_url`, `verify_checksum`, `extract_archive`,
    `maybe_download`, `safe_extract_zip`, `safe_extract_tar` —
    atomic downloads, SHA-256 verification, path-traversal-blocked
    extraction.

- **Native synthetic datasets** (deterministic, CPU-safe, no network):
  - `SyntheticPatchGraphDataset` (graph classification + regression).
  - `SyntheticVolumeGraphDataset` (3-D graph classification).
  - `SyntheticNodeClassificationDataset` (SBM with masks).
  - `SyntheticEdgePredictionDataset` (similarity-based pairs).
  - `SyntheticGraphRegressionDataset`.
  - `SyntheticHeteroGraphDataset` (paper / author / venue).
  - `SyntheticTemporalGraphDataset`.

- **Folder-backed datasets**: `ImageFolderPatchGraphDataset`,
  `VolumeFolderPatchGraphDataset` (`.npy` / `.npz` / `.pt`).

- **Optional torchvision wrappers** — generic
  `TorchvisionPatchGraphDataset` plus curated
  `MNIST`, `FashionMNIST`, `KMNIST`, `CIFAR10`, `CIFAR100`,
  `SVHN`, `STL10`, `FakeData` patch-graph subclasses.

- **Optional PyG wrappers** — `PyGDatasetAdapter`,
  `PyGPlanetoidDataset`, `PyGTUDatasetAdapter`.

- **Optional DGL wrappers** — `DGLDatasetAdapter`,
  `DGLCitationDatasetAdapter`.

- **Optional OGB wrappers** — `OGBDatasetAdapter`,
  `OGBNodePropertyDatasetAdapter`,
  `OGBLinkPropertyDatasetAdapter`,
  `OGBGraphPropertyDatasetAdapter`,
  `OGBEvaluatorWrapper`.

- **Converter utilities** — `ogb_item_to_graph`,
  `torchvision_image_to_patch_graph`, plus re-exports of the
  homogeneous + hetero PyG / DGL converters from
  `tgraphx.interop`.

### Added — Transforms (`tgraphx.transforms`)

- Composition: `Compose`, `LambdaTransform`, `RandomApply`.
- Structure: `AddSelfLoops`, `RemoveSelfLoops`, `ToUndirected`,
  `CoalesceEdges`, `DropEdges`.
- Features: `NormalizeFeatures`, `StandardizeFeatures`,
  `NormalizeEdgeFeatures`, `AddDegreeFeatures`,
  `AddConstantFeatures`, `FeatureNoise`, `NodeFeatureMask`.
- Splits: `RandomNodeSplit`, `RandomLinkSplit`, `RandomGraphSplit`,
  `FixedSplit`.
- Positional: `AddDegreeEncoding`, `AddLaplacianEigenvectors`
  (with O(N²) guard), `AddAdjacencyBias`.
- Patch: `PatchifyImage`, `PatchifyVolume`, `BuildGridGraph`,
  `BuildKNNGraph`, `BuildRadiusGraph`.

### Added — Metrics (`tgraphx.metrics`)

- Classification: `accuracy`, `top_k_accuracy`, `confusion_matrix`,
  `precision_recall_f1`, `classification_report`.
- Regression: `mae`, `mse`, `rmse`, `r2_score`, `regression_report`.
- Ranking / link prediction: `hits_at_k`, `mean_reciprocal_rank`,
  `ndcg_at_k`, `roc_auc`, `average_precision`,
  `link_prediction_report`.
- Reports: `graph_classification_report`,
  `node_classification_report`, `edge_classification_report`,
  `graph_regression_report`.
- OGB: `OGBEvaluatorWrapper` (lazy import).

### Added — Benchmarks (CI-safe `--small` mode + JSON output)

- `benchmarks/benchmark_dataset_loading.py`
- `benchmarks/benchmark_training_synthetic.py`
- `benchmarks/benchmark_tensor_vs_flatten.py`
- `benchmarks/benchmark_transforms.py`
- `benchmarks/benchmark_metrics.py`
- `benchmarks/make_benchmark_report.py` (Markdown report generator)

### Added — Tests, examples, docs

- New tests:
  - `tests/test_datasets_base.py`
  - `tests/test_dataset_registry.py`
  - `tests/test_dataset_cache.py`
  - `tests/test_dataset_download_mocked.py` (no network — monkey-patches
    `urlopen`; covers checksum / path-traversal blocking).
  - `tests/test_synthetic_datasets.py` (incl. tiny-overfit).
  - `tests/test_folder_datasets.py` (PIL via `pytest.importorskip`).
  - `tests/test_torchvision_wrappers.py` (uses `FakeData`, never
    downloads).
  - `tests/test_pyg_dgl_ogb_wrappers.py` (lazy-import-missing path
    asserted; real conversion test skipped when upstream missing).
  - `tests/test_transforms.py` (47 tests).
  - `tests/test_metrics.py` (35 tests with hand-computed values).
  - `tests/test_benchmark_smoke.py` (every benchmark `--small`).
  - `tests/test_dataset_docs_claims.py` (license/no-bundled/no-hidden-
    download wording, lazy-import contract).

- New examples:
  - `examples/datasets_quickstart.py`
  - `examples/synthetic_datasets_demo.py`
  - `examples/transforms_metrics_demo.py`
  - `examples/image_folder_patch_dataset_demo.py`
  - `examples/benchmark_quickstart.py`
  - `examples/pyg_dataset_adapter_demo.py`
  - `examples/dgl_dataset_adapter_demo.py`
  - `examples/ogb_dataset_adapter_demo.py`
  - `examples/mnist_patch_graph_demo.py` (FakeData by default; opt-in
    `--download` for actual MNIST).

- New docs:
  - `docs/datasets.md`
  - `docs/transforms.md`
  - `docs/metrics.md`
  - `docs/benchmarks.md`
  - `docs/dataset_license_policy.md`

### Added — Optional extras

- `pyg`, `ogb`, `pillow` extras in `pyproject.toml` (DGL is
  intentionally not packaged as an extra because its wheels are
  platform-sensitive).

### Honest scope (unchanged for v0.2.9)

- TGraphX **does not redistribute** third-party datasets.  Adapters
  call upstream loaders.
- Downloads happen only when `download=True` is explicitly passed.
- Synthetic datasets are tutorials/sanity, not benchmarks.
- TGraphX makes **no SOTA / leaderboard claims**.
- TGraphX is **not** a drop-in PyG / DGL replacement.

---

## [0.2.8] — 2026-05-07 (release prep)

### Added

- **Random-walk sampling** — `random_walk_sample(graph, seed_nodes,
  walk_length, num_walks_per_seed=1, direction="out", restart_prob=0.0,
  seed=None, relabel_nodes=True)` in `tgraphx.sampling`.
  Per-call `torch.Generator` (no global RNG side effects).

- **Hetero sampling** — new module `tgraphx.hetero_sampling`:
  - `hetero_induced_subgraph(hetero_graph, node_ids_dict, relabel_nodes=True)`
  - `hetero_neighbor_sample(hetero_graph, seed_nodes_dict, fanouts,
    seed=None, direction="in", relabel_nodes=True)` — multi-hop
    per-relation fanout neighbour sampling.

- **Temporal window sampling** — new module `tgraphx.temporal_sampling`:
  - `temporal_window_sample(seq, t_start, t_end)` for
    `TemporalGraphSequence`.
  - `temporal_window_sample_batch(batch, t_start, t_end)` for
    `TemporalGraphBatch` (variable-length sequences are clipped per
    sequence; sequences shorter than `t_start` raise).

- **Tests**:
  - `tests/test_random_walk_sample.py` — 16 tests.
  - `tests/test_hetero_sampling.py` — 20 tests.
  - `tests/test_temporal_sampling.py` — 13 tests.

### Changed

- **README rewritten for honesty** — every red flag was either
  implemented + tested or kept as a true scope boundary:
  - "Optional and experimental features (v0.2.4)" header now
    versionless.
  - "Full hetero/temporal GNN layers: containers, not GNN
    implementations" replaced with the truth: `HeteroConv`,
    `HeteroGraphClassifier`, `HeteroNodeClassifier`,
    `TemporalGraphClassifier`, `TemporalGraphRegressor` exist + are
    tested.
  - "Neighbor sampling, distributed training, multi-GPU: out of scope"
    split into accurate parts: sampling stable, distributed *helpers*
    stable, full multi-GPU framework intentionally out of scope.
  - Scalability table: `TensorGATLayer` chunked forward → ✅ Stable
    (was "⏳ Planned v0.2.4" despite being implemented since v0.2.4).
  - Attention table: per-channel attention → 🧪 Experimental (was
    "❌ Not supported").  Per-pixel/voxel kept as a true memory-driven
    scope boundary.
  - Hardware/Backend tables: Windows/macOS now correctly labelled
    "Smoke CI" (was "No CI").
  - Limitations text: `heterogeneous, temporal, graph transformers,
    learned graph` no longer listed as out of scope (they ship).
  - Project structure / tests / examples sections refreshed.

- **`docs/limitations.md`**: removed "Neighbor sampling
  (GraphSAINT/ClusterGCN): Not implemented" — replaced with positive
  note pointing at `tgraphx.sampling`.  Added rows for the new v0.2.8
  sampling helpers.

- **`docs/roadmap.md`**: v0.2.8 entry added with Hetero/temporal/random
  walk sampling marked complete.

### Honest scope boundaries (kept, with reasons)

- **Per-pixel / per-voxel GAT attention** — naive memory cost
  O(E·K·H·W) is prohibitive; deferred until a memory-safe variant
  (factorised / windowed / low-rank) is designed.
- **Universal arbitrary-rank node features across every layer** — only
  vector, 2-D, and 3-D shapes are supported.  Adding a generic
  rank-agnostic layer is a v0.3 design discussion.
- **Full TGN / TGAT recurrent memory module** — temporal workflows are
  stateless snapshot-loop only.
- **Full automatic multi-GPU training framework** — TGraphX provides
  rank-zero/world-size helpers; DDP setup remains the user's
  responsibility.
- **Graph builder mathematical O(N²) cost** for kNN/radius/IoU/fully
  connected — chunked variants exist; large graphs warn.

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

[0.4.0] - 2026-05-08: https://github.com/arashsajjadi/TGraphX/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/arashsajjadi/TGraphX/releases/tag/v0.2.0
[0.1.2]: https://github.com/arashsajjadi/TGraphX/releases/tag/v0.1.2
[0.1.1]: https://github.com/arashsajjadi/TGraphX/releases/tag/v0.1.1
[0.1.0]: https://github.com/arashsajjadi/TGraphX/releases/tag/v0.1.0

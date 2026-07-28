# Changelog

All notable changes to TGraphX are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.5.1] — 2026-07-27

Canonical set-attention naming and exact-reference-configuration
release. Backward-compatible PATCH release: every existing import,
config, checkpoint, and pickled model continues to work unchanged; all
defaults are unchanged.

### Added — canonical name and factory aliases

- `TGraphXSetAttention` is the canonical public class of the
  learned-implicit-relations family (paper/table label
  `TGraphX-SetAttn`). `SetTransformerModel` remains a permanent,
  stable compatibility alias for the **same class object** — imports,
  `isinstance` checks, `from_config`, state dicts, and pickled models
  are unaffected. The stable machine/family name in configs and the
  factory remains `"set_transformer"`.
- `build_model` family aliases: `"tgraphx_set_attention"` and
  `"set_attention"` resolve to the same family as `"set_transformer"`
  (identical architecture and metadata; `topology_source_of` accepts
  all three).
- Public-API registry: `TGraphXSetAttention` and `StridedConvEncoder`
  registered stable; alias table maps `SetTransformerModel` and the
  factory family names to `TGraphXSetAttention`.

### Added — explicit architecture axes (all defaults unchanged)

- `norm_order="pre"|"post"` on `TGraphXSetAttention` and
  `SetAttentionBlock`: post-LN follows the
  `torch.nn.TransformerEncoderLayer` (`norm_first=False`) convention.
- `activation="gelu"|"relu"` for the block FFNs.
- `pool_attention_dropout=None|float`: decouples the
  attention-pooling readout's attention-weight dropout from the block
  `attention_dropout` (default `None` follows it, as before).
- `head_hidden_dim=None|int`: optional `Linear → ReLU → Linear` head
  (default unchanged single linear head).
- `StridedConvEncoder` + `encoder_config={"architecture": "strided",
  ...}`: strided channel-growing 3×3 spatial encoder (default schedule
  32→64→128 via `hidden_channels`/`channel_multiplier`, explicit
  `channel_schedule` supported; BatchNorm, no residual, adaptive
  average pool, linear projection). `"cnn"` (the package `CNNEncoder`)
  remains the default architecture.
- All new fields appear in `config()`/`repr()`; configs serialized by
  earlier versions load unchanged (new fields take their defaults,
  reproducing the earlier architecture exactly).

### Added — evaluated reference configuration

- `TGraphXSetAttention.reference_config(in_shape, num_classes, task=...)`:
  the exact set-attention architecture evaluated in the TGraphX
  experiment program, as one explicit machine-readable configuration
  (strided encoder 32→64→128; two post-LN ReLU blocks with dropout
  0.1; single-seed attention pooling without attention-weight dropout;
  linear head). Data sizes are required arguments — nothing is a
  hidden dataset-specific default.
- `TGraphXSetAttention.map_reference_state_dict(sd)` /
  `from_reference_state_dict(sd, ...)`: documented, strict key mapping
  from the torch-primitives reference layout
  (`encoder.conv/proj`, `self_attn.layers.N`, `pma`, `head.net`) onto
  this class.
- Parity proof against the completed revised experiment: identical
  parameter count (189,650), strict checkpoint load (0 missing /
  0 unexpected; all 54 tensors bitwise equal), logits within 2.4e-06
  on a fixed real validation batch, and **identical predictions on the
  full validation split for all five seeds**, reproducing every
  recorded per-seed macro-F1 exactly (5-seed mean 0.7023 ± 0.0146).
  See `docs/reports/SET_ATTENTION_REFERENCE_PARITY.md` and
  `tools/verify_set_attention_reference_parity*.py`.

### Changed

- `repr()` of set-attention models now shows the canonical class name
  `TGraphXSetAttention` and the new `norm_order`/`activation` fields.
- Documentation (README, set_transformer.md,
  tensor_relational_platform.md, factories.md, api_stability.md,
  api_cheatsheet.json, examples) uses the canonical name and explains
  the alias.

### Testing

- New `tests/test_set_attention_v151.py` (36 tests): canonical/alias
  identity, factory aliases, new architecture axes, post-norm block ≡
  `torch.nn.TransformerEncoderLayer` numerical parity, self-contained
  torch-primitives reference replica mapping + fixed-batch parity,
  v1.5.0-style config compatibility, checkpoint round trip,
  permutation invariance/equivariance and padding-mask isolation for
  the post-norm path, CPU/CUDA parity, and (when the external evidence
  tree is present) strict mapping of the five completed experiment
  checkpoints. No existing test was modified or weakened.

---

## [1.5.0] — 2026-07-27

Tensor-relational platform release: a new learned-implicit-relations
model family (SetTransformer), an explicit topology-source vocabulary,
and the elimination of hidden dropout defaults. Backward-compatible
MINOR release; no existing scientific experiments were rerun for it —
documentation cites the frozen revised PASTIS-R artifacts.

### Added — SetTransformer / learned implicit relations

- `SetTransformerModel` (+ `SetAttentionBlock`, `AttentionPooling`):
  set attention over tensor-valued nodes — shared node encoder (MLP for
  vectors, the package `CNNEncoder` for `[C, H, W]`, a Conv3d encoder
  for `[C, D, H, W]`, or a custom module), pre-LN multi-head
  self-attention blocks with key-padding masks (permutation-
  equivariant), and a permutation-invariant readout (PMA `"attention"`
  pooling with learned seeds, or `"mean"`/`"sum"`/`"max"`).  Works with
  the flat `GraphBatch` convention (dense tokens + masks are derived
  internally), `fit`/`evaluate`, checkpoints, and the experiment runner.
  Deterministic `config()` / `from_config()` round trip.  CPU/CUDA.
  No new dependencies (PyTorch primitives only).
- `build_model` accepts `layer="set_transformer"` and a new `family=`
  alias for `layer=`; `hidden_shape=(embed_dim,)` sets the token width.
  `make_layer("set_transformer")` raises with a pointer to
  `build_model` (it is a model-level family, not a per-layer operator).
- Topology-source vocabulary: `TOPOLOGY_SOURCES = ("none", "fixed",
  "given", "learned_implicit", "learned_explicit", "hybrid")`,
  `topology_source_of(family)`, and `model.model_family` /
  `model.topology_source` attributes on every `build_model` output.
- Explicit ignored-topology contract: `SetTransformerModel` never
  consumes `edge_index`; by default it emits `TopologyIgnoredWarning`
  once per instance when one is supplied
  (`on_edge_index="warn"|"ignore"|"error"`).

### Changed — no more hidden dropout (migration: docs/migration_v1_4_to_v1_5.md)

- `CNNEncoder` and `DeepCNNAggregator` no longer default to a silent
  `dropout_prob=0.3`.  The documented default is now **0.0**; omitting
  the value emits `DropoutDefaultChangeWarning` (a `UserWarning`
  subclass) naming the construction site.  Passing any explicit value —
  including `0.0` — silences it.  Controlled PASTIS-R re-runs measured
  the hidden 0.3 at ≈ −0.04 to −0.06 validation macro-F1.
- `ConvMessagePassing` gained an explicit `dropout_prob` parameter
  (merged into `aggregator_params`; conflicting values raise
  `ValueError`); `GraphClassifier` gained `dropout_prob`;
  `CNN_GNN_Model` resolves a missing `cnn_params['dropout_prob']`
  loudly and no longer mutates the caller's `cnn_params` dict.
- **Bug fix:** `make_layer("conv", ..., dropout=X)` and
  `build_model(layer="conv", dropout=X)` previously **ignored** `X`
  (models silently carried `Dropout2d(p=0.3)` regardless of config).
  `dropout`, `use_batchnorm`, and `aggregator_params` are now forwarded
  to the conv aggregator.
- The effective dropout value is visible everywhere: `repr()`
  (`extra_repr` on `CNNEncoder`, `DeepCNNAggregator`,
  `TensorMessagePassingLayer` and subclasses), `.dropout_prob`
  attributes, and `.config()` dicts on `CNNEncoder` /
  `DeepCNNAggregator` / `SetTransformerModel`.
- Legacy behaviour is reconstructible intentionally (no warning):
  `CNNEncoder.legacy(...)`, `DeepCNNAggregator.legacy(...)`,
  `LEGACY_CNN_DROPOUT_PROB == 0.3`.
- Internal easy-mode/ux helper models (`tgx.easy` tensor classifiers,
  workflow demo models) now pass `dropout_prob=0.0` explicitly — in
  1.4.2 they silently carried the 0.3 aggregator dropout.
- `use_batchnorm` / `use_residual` defaults are **unchanged** (they
  affect checkpoint parameter layout) but are now surfaced in `repr()`
  and `config()`.  Aggregator BatchNorm is documented as
  graph-density-dependent (helps dense graphs; harmful with many
  zero-degree nodes), not universally good or bad.

### Compatibility

- Checkpoints: dropout modules hold no parameters, so `state_dict`
  layouts are identical across the default change; 1.4.2 checkpoints
  load unchanged, and eval-mode outputs never depended on dropout.
  Loaded legacy checkpoints are never silently altered.
- Training-time behaviour of code that *relied on the silent 0.3*
  changes (to no dropout) — loudly, via the warning.  Use
  `dropout_prob=0.3` or `.legacy(...)` to reproduce pre-1.5 training
  behaviour exactly.
- All v1.3.x/v1.4.x public APIs keep their signatures; new constructor
  parameters are keyword-optional additions.

### Documentation

- New: `docs/tensor_relational_platform.md` (operating-regime map +
  comparison table from the frozen revised PASTIS-R artifacts, with
  provenance and generalization caveats), `docs/set_transformer.md`,
  `docs/migration_v1_4_to_v1_5.md`,
  `docs/reports/TENSOR_RELATIONAL_PLATFORM_UPDATE.md` (internal
  engineering report), `examples/set_transformer_demo.py`.
- README gained a v1.5.0 platform overview; `docs/factories.md`,
  `docs/architecture.md`, `docs/api_stability.md`, `docs/index.md`,
  `docs/api_cheatsheet.json` updated; `CITATION.cff` version metadata
  fixed (was stale at 0.1.1).

### Testing

- New suites: `tests/test_explicit_dropout_v150.py` (22 tests) and
  `tests/test_set_transformer_v150.py` (37 tests, incl. permutation
  invariance/equivariance, padding-mask isolation, CPU/CUDA parity,
  config/checkpoint round trips, factory/registry integration, and two
  tiny synthetic sanity checks that are explicitly not benchmark
  results).
- Full suite at release: 3412 passed, 23 skipped (CUDA-dependent tests
  ran locally on GPU; the count includes them).

---

## [1.4.2] — 2026-05-23

Audit-fix and release-hardening patch. **No breaking API changes.** All
v1.3.x, v1.4.0, and v1.4.1 syntax is preserved.

### Fixed

- `tgx.make_graph(networkx_graph=G, x=..., labels=..., ...)` no longer
  silently discards external node features, labels, or graph kwargs.
  Topology now comes from the NetworkX graph, and supplied tensor fields
  are attached and re-validated; mismatched row counts raise a clear
  `ValueError` instead of producing a zero-filled placeholder graph.
  *(Codex/Composer TGX-AUDIT-002 / 001.)*
- `tgraphx.ux.serialization` now round-trips `Graph.edge_labels` and
  `Graph.graph_features` through `.tgx` bundles; older bundles without
  these keys still load unchanged. *(Codex TGX-AUDIT-009.)*
- `tgraphx.training._unpack_batch` raises a descriptive `ValueError` when
  `GraphBatch.edge_index` is `None`, instead of crashing on attribute
  access. *(Codex TGX-AUDIT-010.)*
- `Graph.num_classes` returns `0` for an empty integer label tensor
  instead of raising a `RuntimeError` from `Tensor.max()` on an empty
  tensor. *(Codex TGX-AUDIT-011.)*
- `Graph.from_adjacency` now rejects non-square SciPy-sparse adjacency
  with the same clear `ValueError` already used for dense adjacency.
  *(Codex TGX-AUDIT-012.)*
- `Graph.to(device)` also moves boolean/int mask tensors stored under
  `metadata['masks']` (used by `graph.train_mask` / `val_mask` /
  `test_mask`), so their device stays in sync with `node_features`.
  *(Codex TGX-AUDIT-013.)*
- `tgx.train_graph_rl(max_steps=...)` now actually forwards `max_steps`
  into the environment through a fresh `GraphEnvConfig`, and records the
  value on the returned RLResult's `.config` dict so callers can verify it
  took effect. *(Codex TGX-AUDIT-016.)*
- `tgraphx.audit_package_readiness()` reports `torch`, `torchvision`, and
  `pyyaml` under `required_dependencies` and limits `optional_dependencies`
  to genuinely optional packages, matching `pyproject.toml`.
  *(Composer TGX-AUDIT-015.)*
- `tgx.explain_error(...)` no longer suggests a non-existent
  `pip install tgraphx[vision]` extra; torchvision is a base dependency,
  so the message points users to plain `pip install torchvision` and to
  real extras (`tgraphx[pyg]`, ...) for genuinely optional pieces.
  *(Codex TGX-AUDIT-005.)*

### Added — public-API discoverability

- Top-level `tgraphx.__all__` now exposes the v1.4.1 helper aliases
  `generate`, `graph_generation_report`, `compare_generated_graphs`,
  `generation_metrics`, `graph_evolution`, `run_evolution`, `run_rl`, plus
  the importable top-level KG / generation / evolution / RL entry points
  `KnowledgeGraph`, `KGTrainer`, `KGTrainingConfig`,
  `run_graph_generation`, `run_evolutionary_optimization`, `run_graph_rl`.
  *(Codex TGX-AUDIT-003 / Composer TGX-AUDIT-004 / 010.)*
- `tgraphx.ux.public_api`'s `_STABILITY` registry and `_ALIASES` table now
  cover every v1.4.1 helper and its documented aliases, so
  `tgx.api_status("classify_nodes")`, `tgx.api_status("make_graph")`,
  `tgx.list_aliases("make_graph")`, etc., all resolve cleanly.
  *(Codex TGX-AUDIT-001 / Composer TGX-AUDIT-005.)*

### Packaging / Release metadata

- Added official-website URL: `https://tgraphx.com` is now the package
  `Homepage` in `pyproject.toml` `[project.urls]`, with the GitHub
  repository tracked under `Source` / `Repository`, and the website is
  cross-linked from the top of `README.md`.
- Made `torchvision` import lazy: `import tgraphx` no longer eagerly loads
  `torchvision.models`. The pretrained ResNet path in `PreEncoder` now
  imports `torchvision.models` only when the pretrained branch is taken.
  *(Composer TGX-AUDIT-003.)*
- `Graph.load` / `tgraphx.ux.serialization.load_tgraphx` accepts an opt-in
  `trust_source=False` kwarg that refuses pickle-backed loads, and the
  docstring now explicitly documents that `.tgx` bundles can execute
  pickle code (they include user metadata, so `weights_only=True` is not
  applicable). Existing bundles continue to load with the default
  `trust_source=True`. *(Codex TGX-AUDIT-014.)*

### Documentation

- README: replaced the misleading "no mandatory external dependencies"
  wording with an accurate description ("base package does not require
  PyG, DGL, OGB, PyKEEN, Stable-Baselines3, or RLlib; it depends only on
  the PyTorch stack and the lightweight runtime utilities declared in
  `pyproject.toml`"), and added the official website link near the top
  and in Quick links. README intentionally remains a *current-state*
  document, not a release-by-release changelog. *(Codex TGX-AUDIT-006 /
  Composer TGX-AUDIT-011.)*
- `docs/user_experience_api_contract.md`: corrected the `graph_features`
  / `graph_label` contract — `graph_label` is the graph-level target,
  `graph_features` is a *distinct* graph-level input feature tensor; the
  two are not aliases. *(Codex TGX-AUDIT-004.)*
- `docs/api_stability.md`: added a v1.4.0+ UX section and a v1.4.1+
  one-call helper section so the stability document is in sync with what
  is actually exported. *(Composer TGX-AUDIT-006.)*
- `docs/hetero_gnns.md`: replaced "parity with reference implementations"
  with a conservative description ("validated on small regression
  fixtures … no claim of numerical or training-throughput parity"). The
  module remains marked Experimental. *(Composer TGX-AUDIT-012.)*
- `CONTRIBUTING.md`: bumped the documented minimum Python version from
  3.9+ to 3.10+ to match `pyproject.toml`, and replaced the stale "do not
  claim `train_epoch`/`evaluate`/`fit`/`TensorBoardLogger`/`MLflowLogger`
  exist" rule with accurate guidance ("do not claim public APIs exist
  unless they are exported and tested; these APIs are currently exported
  and tested"). *(Codex TGX-AUDIT-015 / Composer TGX-AUDIT-002 / 008.)*
- `environment.yml`: bumped `python=3.9` → `python=3.10`. *(Composer
  TGX-AUDIT-008.)*
- `tgraphx/datasets/__init__.py` and `tgraphx/datasets/torchvision_wrappers.py`:
  install-hint text no longer references the non-existent
  `tgraphx[vision]` extra. *(Codex TGX-AUDIT-005.)*

### Tests

- New `tests/test_audit_fixes_v142.py` adds 17 targeted regression tests:
  `make_graph` × NetworkX feature/label preservation and shape-mismatch
  error; `api_status` / `list_aliases` for the v1.4.1 helpers; `__all__`
  top-level aliases; `.tgx` round-trip for `edge_labels` and
  `graph_features` plus backward-compat load of old payloads;
  `_unpack_batch` clear-error path; `num_classes` on empty labels;
  `from_adjacency` non-square sparse rejection; `Graph.to` mask move;
  `train_graph_rl` `max_steps` recorded on result; lazy-torchvision
  import; `python -m tgraphx list-methods` CLI smoke; readiness
  dependency classification; and explain_error not suggesting fake
  extras.
- `tests/test_advanced_notebook_workflows_v138.py::test_pyg_singleton_graph_label_normalized`
  no longer attempts a MUTAG download in CI: it skips automatically when
  the raw files are not already cached, and only opts in to a real
  download when `TGRAPHX_TESTS_ALLOW_NETWORK=1`. *(Codex TGX-AUDIT-008.)*

### Notes

- No breaking API changes; every v1.3.x / v1.4.0 / v1.4.1 public name and
  alias still resolves.
- No SOTA, parity, AMP, or universal `torch.compile` claims have been
  added.
- Verification commands and exact pass/skip counts are reported in the
  release report attached to the GitHub tag.

---

## [1.4.1] — 2026-05-11

Final usability hardening release. All v1.3.x and v1.4.0 syntax preserved.

### Added — Groups 101–115

- **GROUP 101** `tgx.classify_nodes(x, edge_index, labels, ...)` — one-call tensor node classification with leakage guard; aliases `node_classification`, `fit_node_classifier`, `train_node_classifier`.
- **GROUP 102** `tgx.kg_completion(triples, num_entities, num_relations, model=...)` — one-call KG link prediction; aliases `fit_kg`, `train_kg`.
- **GROUP 103** `tgx.make_graph(x, edges|edge_index|adjacency|networkx_graph, ...)` — one-call graph constructor from any format; aliases `build_graph`.
- **GROUP 104** `tgx.explain_error(e)` — maps common errors to actionable guidance including GraphML rank, NSGA-II, VGAE, mask overlap, CUDA, missing optional deps.
- **GROUP 105** `tgx.debug_batch(batch)` / `batch_summary` — NeighborLoader/GraphBatch content debugger.
- **GROUP 106** `tgx.dataset_card(dataset)`, `tgx.model_card(model)` — JSON-serializable metadata cards.
- **GROUP 107** `tgx.benchmark_card(result)` — no-SOTA-claim benchmark card from workflow result.
- **GROUP 108** `tgx.public_api()` / `api_status` / `list_aliases` — API stability registry (v1.4.0); consistency checks.
- **GROUP 109** CPU/GPU parity smoke via deterministic same-seed tests.
- **GROUP 110** `tgx.audit_package_readiness()` — full package readiness dict (optional deps, public API count, known limitations); `python -m tgraphx readiness` CLI; `list-datasets` and `list-methods` CLI subcommands.
- **GROUP 111** `tgx.generate_graph(method, ...)` — one-call graph generation with method aliases (`"ba"`, `"er"`, `"ws"`, ...), tensor node shapes, VGAE helpful error, artifact writing.
- **GROUP 112** `tgx.evaluate_generated_graphs(graphs, ...)` — structural evaluation report for generated graphs.
- **GROUP 113** `tgx.optimize_graph(objective, algorithm, ...)` — one-call evolutionary graph optimization; aliases `evolve_graph`, `graph_evolution`, `run_evolution`.
- **GROUP 114** `tgx.train_graph_rl(env, algorithm, ...)` — one-call graph RL; environment and algorithm aliases (`"maxcut"` → `"max_cut"`, etc.); aliases `graph_rl`, `run_rl`.
- **GROUP 115** Dashboard audit extensions: `audit_generation_run`, `audit_evolution_run`, `audit_rl_run`; `dashboard_audit(path, workflow="generation"|"evolution"|"graph_rl")`.

### Enhanced in v1.4.1

- `audit_run_dir` / `dashboard_audit` now return UX quality scores: `completeness_score`, `reproducibility_score`, `portability_score`, `scientific_reporting_score` (0–100 each) and support `return_markdown=True`.
- Readiness CLI: `python -m tgraphx readiness` + `list-datasets` + `list-methods` + `help`.

### Validation

- 70 new v1.4.1 group-101-to-115 tests pass.
- Full test suite: **3332 passed, 25 skipped, 0 failed.**
- Build + twine PASSED. Clean wheel smoke PASSED. CI PASSED. PyPI PASSED.

### Notes

- All v1.3.x and v1.4.0 syntax fully preserved.
- No SOTA claims. New APIs are Beta.
- TGraphX remains complementary to PyG, DGL, NetworkX, PyKEEN, SB3, RLlib.

---

## [1.4.0] — 2026-05-11

Major user-experience release: a new `tgraphx.ux` layer brings
LLM-predictable APIs, PyG / NetworkX / PyKEEN-style aliases, native tensor
graph save/load, reproducible run contexts, dataset registry aliases,
graph-construction helpers, leakage guards, and a dashboard audit utility —
**without** breaking any existing v1.3.x syntax.

### Added
- `tgraphx.ux` module (new) with:
  - `validate_graph`, `assert_tensor_native`, `check_graph_invariants`
  - `describe`, `summary` (object-aware)
  - `reproducible`, `seeded`, `reproducibility_state` (context manager)
  - `check_leakage`, `leakage_report`, `validate_split_policy`
  - `save`, `load`, `save_tgraphx`, `load_tgraphx` (native `.tgx` bundles —
    GraphML cannot store rank-4 tensors)
  - `knn_graph`, `build_class_prototypes`, `build_prototype_graph`,
    `image_to_patch_graph` (tensor-native, leakage-aware)
  - `audit_run_dir`, `dashboard_audit` (run-directory schema validation)
  - `workflow`, `run_workflow`, `list_workflow_tasks` (one-line task dispatcher)
  - `compare` (functionality, not throughput, comparison)
  - `public_api`, `api_status`, `list_aliases` (stability registry)
- `Graph.x=` PyG-style constructor alias (in addition to existing
  `node_features=`, `y=`, `labels=`, `edge_attr=`).
- `Graph.number_of_nodes()`, `Graph.number_of_edges()` NetworkX-style methods.
- `Graph.from_edges`, `Graph.from_adjacency`, `Graph.from_networkx`,
  `Graph.to_networkx`, `Graph.save`, `Graph.load` classmethods/methods.
- `Graph.summary()` instance method (alias for `tgraphx.describe(graph)`).
- `tgraphx.datasets.load_dataset(name, ...)` with friendly aliases
  (`mnist_graph`, `cifar10_patch`, `cora`, `mutag`, `proteins`, ...).
- `tgraphx.datasets.list_dataset_aliases()`.
- Top-level exports of all UX functions for LLM predictability
  (`tgx.validate_graph`, `tgx.workflow`, `tgx.save`, `tgx.load`,
  `tgx.knn_graph`, `tgx.reproducible`, `tgx.audit_run_dir`, ...).

### Fixed
- Improved helpful errors for unsupported shortcuts: unknown task names,
  unknown API names, unknown dataset names, bad metric, oversized k in
  `knn_graph`, non-square adjacency, unsupported save object types, corrupted
  `.tgx` bundles.
- All errors include closest-match suggestions via `difflib`.

### Documentation
- New `docs/user_friendly_syntax_audit.md` classifying groups 71–100 with
  status, canonical name, aliases, rejected forms, and tests.
- README updated with a concise "v1.4.0: user-friendly tensor-native workflows"
  section showing before/after examples (graph construction, dataset loading,
  reproducibility context, dashboard audit, save/load, migration aliases,
  public API registry).
- `docs/colab_gallery.md` now lists advanced notebooks 31–35 as Google Drive
  notebook files.

### Validation
- 90 new v1.4.0 group-71-to-100 tests pass.
- Full test suite: **3262 passed, 25 skipped, 0 failed.**
- All v1.3.x syntax still valid; advanced notebook tests (137 incl.
  consistency + execution + workflow) still green.
- Clean wheel smoke and PyPI smoke verified.

### Notes
- No SOTA / parity claims. TGraphX does not claim to replace PyG, DGL,
  NetworkX, PyKEEN, SB3, or RLlib.
- New aliases are **beta** in `docs/api_stability.md`; canonical APIs remain
  stable.
- Notebooks are stored on Google Drive (per `.gitignore` policy);
  `tools/build_advanced_notebooks.py` + `tools/execute_advanced_colab_drafts.py`
  are the source of truth.

---

## [1.3.8] — 2026-05-11

Executed-and-validated advanced Colab notebooks. The previous v1.3.7 release
claimed the advanced notebooks were "ready", but they had never been actually
executed and contained latent CUDA / device / API-signature bugs. This release
fixes all of them and runs the five notebooks end-to-end in FAST_MODE with
outputs preserved.

### Fixed
- **NB31 shape-trace CUDA index out-of-bounds:** Filter both source AND
  destination indices when slicing `edge_index` to a tiny subgraph, otherwise
  `node_features[src_idx]` can dereference prototype indices that exceed
  `tiny_x.size(0)`. Same bug fixed in NB33.
- **NB34 CUDA device mismatch:** `model = model.to(device)` must run before
  the manual `score_triples` call in the gradient sanity check; `torch.randint`
  for negative samples must also be on `device`.
- **NB34 KG report writer signatures:** `write_kg_training_report` and
  `write_kg_evaluation_report` take `(path, dict)`, not kwargs. Notebook updated
  to pass a dict.
- **NB35 PyG `MUTAG` singleton label normalization (package fix):**
  `tgraphx.interop.from_pyg_data` now normalizes `tensor([c])`-shaped graph
  labels to scalar `tensor(c)` so that `F.cross_entropy(logits, batch.graph_labels)`
  works after `GraphDataLoader` batching for PyG-sourced datasets.

### Added
- `tools/execute_advanced_colab_drafts.py` — executes notebooks 31–35 in place
  using `nbclient`, keeps minimal outputs, fails on any cell error.
- `tools/validate_executed_advanced_notebooks.py` — checks that shipped
  notebooks have non-empty `execution_count`, outputs, and a final completion
  message.
- `tests/test_advanced_notebook_execution_v138.py` (40 tests) — verifies the
  shipped notebooks have been executed (execution_count set, outputs present,
  no error outputs, completion message in outputs).
- `tests/test_advanced_notebook_report_consistency_v138.py` (70 tests) —
  strict consistency between source and report claims.
- `tests/test_advanced_notebook_workflows_v138.py` (6 tests) — runs each
  notebook's core workflow as a regression test.
- "Scientific and methodological notes" Markdown section in every notebook,
  with explicit learning setting, split policy, leakage policy, baseline
  meaning, metric interpretation, FAST_MODE disclaimer, and TGraphX-specific
  capability demonstrated.
- "Leakage policy" Markdown header in NB35 (previously implicit).
- Popularity baseline implemented in NB34 (previously only mentioned).
- Stronger test-isolation in `test_v024_features.py::TestInteropMissingDeps`
  and `test_dataset_docs_claims.py::TestLazyImports`: snapshot/restore
  optional `sys.modules` entries so earlier PyG imports do not break these tests.

### Validation
- All 5 advanced notebooks execute cleanly in FAST_MODE (`nbclient`, no
  network), with outputs preserved.
- 258 advanced-notebook tests pass (47 + 70 + 23 + 72 + 6 + 40).
- All 5 smoke scripts pass with `--fast --no-download`.
- Full test suite: **3172 passed, 25 skipped, 0 failed**.
- Build and `twine check` pass.

### Notes
- Notebook files are still NOT tracked in git (per .gitignore policy;
  they live in Google Drive / Colab links). The generator
  `tools/build_advanced_notebooks.py` is the source of truth.
- This release does NOT claim SOTA on MNIST, CIFAR-10, Cora, MovieLens, or
  MUTAG. Every notebook has an explicit FAST_MODE disclaimer.

---

## [1.3.7] — 2026-05-11

Advanced real-dataset notebook upgrades (31–35) and regression-test hardening.

### Fixed
- **MNIST class-graph notebook (31):** Added `edge_attr` tensor encoding
  `edge_type` (0=visual_similarity, 1=prototype_membership, 2=prototype_self_loop)
  on the TGraphX `Graph`; added explicit edge-type counts and leakage policy cell.
- **CIFAR-10 patch-graph notebook (32):** Added inductive task declaration and
  leakage policy in the opening markdown. The notebook correctly uses
  `CIFAR10PatchGraphDataset` (true patch graphs, not image-as-node graphs).
- **Cora notebook (33):** Renamed baseline class from `MLPBaseline` to
  `FlattenMLP` for consistency with the project report and acceptance tests.
- **MovieLens KG notebook (34):** Added explicit leakage policy section
  documenting the edge-wise split policy for link prediction.
- **MUTAG notebook (35):** Changed `edge_features=` to `edge_attr=` in
  `Graph` constructor to align with canonical parameter name.
- **KGTrainer CUDA-generator bug:** `torch.randperm(T, generator=cpu_gen,
  device='cuda')` raised `RuntimeError`; fixed by doing randperm on CPU then
  moving to device with `.to(dev)`.
- **`write_kg_summary` call in notebook 34:** Corrected argument from `kg`
  object to a plain dict (matches the function signature).

### Added
- `tests/test_advanced_notebook_report_consistency_v137.py` — 72 tests verifying
  that notebook source content matches report claims for all 5 notebooks.
- `tests/test_advanced_notebook_workflows_v137.py` — 23 workflow regression tests
  running the core code path of each notebook in FAST_MODE with no network access.

### Validation
- All 5 advanced notebook structural validations pass.
- 142 advanced notebook tests pass (47 + 72 + 23).
- All 5 smoke scripts pass with `--fast --no-download`.
- Full test suite: 3054 passed, 27 skipped, 1 warning.
- Build and `twine check` pass.

### Notes
- No SOTA claims are made in any notebook.
- No notebook requires private paths or repository-local files.
- Maintainer still needs to run full notebooks in Google Colab and upload them.

---

## [1.3.6] — 2026-05-10

Public Colab notebook regression fix + LLM-predictability sprint. No new
features; aliases and helpful errors only.

### Fixed
- **Colab draft notebooks regenerated.** `tools/generate_colab_drafts.py` and
  the corresponding `colab_drafts/*.ipynb` files for notebooks 14, 19, 22, and
  24 were stale; they now contain the corrected v1.3.5/v1.3.6 snippets:
  - Notebook 14 (evolutionary): uses `[connectivity_fitness, sparsity_fitness]`
    for NSGA-II; no longer passes `composite_fitness` directly.
  - Notebook 19 (GraphML): formats exceptions with `str(e)[:120]`.
  - Notebook 22 (structural roles): asserts on `max_total_degree` / `min_total_degree`
    (correct for a bidirectional edge_index) and explains the doubling.
  - Notebook 24 (benchmark suite): calls
    `from tgraphx.benchmarks import run_v13_benchmark_suite` instead of
    `subprocess.run` against a repo-local script path.

### Added — LLM-predictability aliases
- `from tgraphx import KnowledgeGraph, KGTrainer, KGTrainingConfig` — top-level
  re-exports of the canonical `tgraphx.kg.*` symbols, because that is the
  natural form LLM-generated code uses.
- `tgraphx.models.knowledge_graph` — compatibility shim that re-exports
  `TransEModel`, `DistMultModel`, `ComplExModel`, `RotatEModel`, `RESCALModel`,
  `SimplEModel`, `KnowledgeGraph`, `KGTrainer`.
- `KGTrainer` now accepts an LLM-friendly call form:
  `KGTrainer(model, kg_or_triples, lr=..., num_epochs=..., batch_size=..., ...)`
  in addition to the canonical `KGTrainer(model, config, train_triples)`.
- `KGTrainer.fit(epochs=..., batch_size=...)` is an alias for `train()` that
  can optionally override config fields.
- `KGTrainer.evaluate(triples=None)` returns the evaluator's metric dict, or
  a small training-summary dict when no evaluator is configured.
- `tgraphx.rl.GraphMaxCutEnv(num_nodes=..., edge_density=..., seed=...)` —
  LLM-friendly wrapper around `MaxCutEnv` that builds a random Erdos-Renyi
  graph under the hood. The canonical `MaxCutEnv(edge_index, num_nodes, ...)`
  form is unchanged.
- `RLResult.final_reward` and `RLResult.mean_return` — convenience properties
  that complement `RLResult.metrics`.

### Improved
- `run_graph_generation(method=...)` now raises a **helpful** `ValueError`
  for known neural-generator names (`"vgae"`, `"gae"`, `"autoregressive"`,
  `"transformer"`). The message lists the correct class
  (`VGAEGraphGenerator`, `AutoregressiveEdgeGenerator`,
  `GraphTransformerGenerator`) and explains that classical-vs-neural
  generators have different contracts.

### Documentation
- `docs/colab_gallery.md` — full visual redesign: themed sections, compact
  tables, reader-friendly display names (no raw filenames in the body),
  consistent “Google Drive notebook” labelling, mobile-readable formatting.
- `README.md` — “Notebook gallery” section rewritten with concise wording and
  a clean link to `docs/colab_gallery.md`. No long lists in the README.
- `docs/llm_usage_guide.md` — added KG (LLM-friendly form), RL
  (`GraphMaxCutEnv`), and graph-generation (classical vs neural) sections;
  expanded `ConvMessagePassing` shape contract with the v1.3.5+
  spatial-downsampling path.
- `docs/knowledge_graphs.md` — added LLM-friendly KG quickstart.
- `docs/graph_reinforcement_learning.md` — added high-level `run_graph_rl`
  + `GraphMaxCutEnv` quickstart.
- `docs/graph_generation.md` — added a classical-vs-neural explanation.
- `docs/api_cheatsheet.json` — added KG top-level alias, models compat path,
  benchmarks package import, and clarified generation/RL entries.

### Validation
- Added `tests/test_colab_notebook_regressions_v136.py` (27 tests) and
  `tests/test_llm_predictability_v136.py` (30 tests) — 57 new tests total.
- Full local suite: every targeted regression and predictability test passes.
- Built wheel passed clean-venv smoke tests outside the repository.
- Build and twine validation passed locally.

### Notes
- All v1.3.5 canonical APIs continue to work unchanged.
- This is a public Colab/notebook regression and LLM-predictability fix
  release. No unrelated features were added.

---

## [1.3.5] — 2026-05-10

Public Colab/API regression fix release. No new features.

### Fixed
- **NSGA-II notebook/API guidance**: `NSGAIIOptimizer` now raises a clear `TypeError`
  at construction time if `composite_fitness` (or any multi-argument callable) is passed
  directly. Error message explains the correct usage and points to `GeneticAlgorithmOptimizer`
  for scalar scalarization.
- **`sparsity_fitness`**: New fitness function (sparsity score in [0,1]; fewer edges = higher
  score) exported from `tgraphx.evolutionary`. Serves as a natural second objective for
  NSGA-II alongside `connectivity_fitness`.
- **`degree_statistics` aliases**: Added `min_degree`, `max_degree`, `mean_degree` as
  user-friendly aliases for total-degree statistics. Existing keys (`min_total_degree`,
  `max_total_degree`, etc.) preserved for backward compatibility.
- **`ConvMessagePassing` out_shape spatial downsampling**: `ConvMessagePassing` now
  honours `out_shape` spatial dimensions exactly when they differ from `in_shape`.
  Adds `nn.AdaptiveAvgPool2d/3d` after the aggregator when spatial dims change.
  Fixes `RuntimeError: mat1 and mat2 shapes cannot be multiplied` that occurred when
  building a classifier after `ConvMessagePassing(in_shape=(32,8,8), out_shape=(64,4,4))`.
- **Benchmark notebook**: `tools/generate_colab_drafts.py` and `tools/generate_notebooks.py`
  no longer reference `benchmarks/run_v13_benchmark_suite.py` as a local path. Updated to
  use `from tgraphx.benchmarks import run_v13_benchmark_suite` (package-public API).
- **Reproducibility**: Confirmed CPU deterministic mode is bitwise-exact; tests preserved.

### Documentation
- Removed stale "A Colab tutorial walks through every workflow" sentence and old single
  Colab badge from `README.md`.
- Added dedicated **Notebook Gallery** section to `README.md` (30 notebooks total).
- Updated `docs/colab_gallery.md` with all 30 Google Drive notebook links.
- Updated `docs/tutorials.md` to point to the gallery rather than a stale badge.
- Updated `docs/evolutionary_graph_optimization.md` to clarify that `composite_fitness`
  is for scalar scalarization and must not be passed directly to `NSGAIIOptimizer`.
- Updated `colab_drafts/14_graph_generation_evolutionary_optimization.ipynb` to use
  `[connectivity_fitness, sparsity_fitness]` as the NSGA-II objective list.

### Validation
- Added 51 Colab regression tests in `tests/test_colab_regressions_v135.py`.
- Full test suite (2800+ tests) passes locally.
- Built wheel passes clean-venv smoke tests.

### Notes
- This is a public Colab/API regression fix release.
- No unrelated features were added.

---

## [1.3.4] — 2026-05-10

Comprehensive Colab-facing regression fix: mining API, benchmark portability, notebook hygiene.

### Fixed
- Added `motif_profile` as a public alias for `motif_counts` in `tgraphx.mining`
  (notebook 20 ImportError).
- Added `wl_subtree_kernel(edge_index_a, num_nodes_a, edge_index_b, num_nodes_b, h, ...)`
  to both `tgraphx.mining` and `tgraphx.mining.kernels` (notebook 21 ImportError).
- Added `centrality_summary(edge_index, num_nodes, ...)` to `tgraphx.mining`
  (notebook 22 / structural-role ImportError).
- Fixed exception slicing in draft notebook generator: `e[:120]` → `str(e)[:120]`
  (TypeError when formatting ValueError).
- Fixed NSGA-II notebook to use `objectives=[fn1, fn2]` list instead of
  `composite_fitness` directly (composite_fitness requires an extra `components` arg).
- Made `run_v13_benchmark_suite` available as a package-level function
  (`from tgraphx.benchmarks import run_v13_benchmark_suite`) so it works
  from a pip-installed package without the repository source tree.
- Added `python -m tgraphx.benchmarks.run_v13_benchmark_suite` CLI entry point.
- Removed `notebooks/` from git tracking; added to `.gitignore`.
  Notebooks are now generated locally with `python tools/generate_notebooks.py`.
  Updated `tests/test_notebooks_v130.py` to skip gracefully when notebooks/ is absent.
- Updated `docs/colab_gallery.md` to remove references to tracked local notebooks
  and direct users to the Google Drive links instead.

### Added
- `tgraphx.mining.motif_profile` — alias for `motif_counts`.
- `tgraphx.mining.wl_subtree_kernel` — two-graph WL subtree kernel wrapper.
- `tgraphx.mining.centrality_summary` — degree-based centrality summary.
- `tgraphx.benchmarks.run_v13_benchmark_suite(small, device, seed, ...)` — package function.
- `tgraphx/benchmarks/run_v13_benchmark_suite.py` — `python -m` CLI entry.
- `tests/test_colab_regressions_v134.py` — 33 regression tests for all reported Colab bugs.

### Documentation
- Updated notebook gallery and README to reflect that notebooks/ is no longer tracked.
- Colab gallery consolidated as the single source for notebook links.

---

## [1.3.3] — 2026-05-10

Easy Mode reproducibility improvement and expanded notebook gallery.

### Fixed
- `train_node_classifier` now accepts a `deterministic=True` parameter that
  enables `cudnn.deterministic`, disables `cudnn.benchmark`, and requests
  `torch.use_deterministic_algorithms(True, warn_only=True)`.
  On CPU with `deterministic=True`, repeated runs with the same seed now
  produce exactly identical loss values (diff = 0.0).
- Reproducibility state is recorded in `result.config["reproducibility_state"]`
  including `seed`, `deterministic`, `torch_version`, `cuda_available`, and
  backend settings.

### Added
- `deterministic` parameter to `train_node_classifier` (default `False` —
  backward-compatible; no overhead unless explicitly enabled).
- 10 tests in `tests/test_reproducibility_easy_v133.py` covering: synthetic
  data seeding, NeighborLoader batch order, CPU deterministic exact match,
  config recording, CUDA smoke, and `set_seed` return value.

### Documentation
- Expanded notebook gallery (`docs/colab_gallery.md`) with 20+ available
  Google Drive notebook links organised into 8 themed sections (Easy Mode,
  tensor-native identity, sampling, KG, graph generation, RL, IO,
  reproducibility/workflows).

### Notes
- CUDA exact bitwise reproducibility is not guaranteed by PyTorch across
  all hardware/versions even with `deterministic=True`; the strict CPU test
  passes, the CUDA test asserts finite results only.
- The default `deterministic=False` is preserved for backward compatibility
  and speed; for demonstrations use `device="cpu", deterministic=True`.

---

## [1.3.2] — 2026-05-10

Evolutionary result API bugfix and expanded notebook gallery.

### Fixed
- `EvolutionResult` had no `history` attribute, raising `AttributeError` when users
  accessed `result.history` after `GeneticAlgorithmOptimizer.optimize()`.
  Added a `history` property returning a list of per-generation dicts
  `[{"generation": i, "best_fitness": ..., "diversity": ..., ...}]`.
- `NSGAIIOptimizer` did not populate `fitness_history` (so `result.history` was empty).
  Now tracks best-by-first-objective, diversity, and Pareto-front size per generation.
- `NSGAIIOptimizer.__init__` now accepts a single callable (wraps it in a list)
  in addition to the existing list-of-callables form.

### Added
- `EvolutionResult.summary()` — prints and returns a human-readable summary.
- `EvolutionResult.to_dict()` — returns a JSON-serialisable dict.
- 16 regression tests in `tests/test_evolutionary_history_v132.py`.

### Documentation
- Expanded notebook gallery in `docs/colab_gallery.md` with 13 available notebooks
  organised by category (Easy Mode, tensor-native identity, sampling, KG, graph generation).
- Notebooks are linked as Google Drive notebook files.
  Verified one-click Colab links will be added after maintainer-side Colab testing.
- README concise notebook gallery cross-link added.

---

## [1.3.1] — 2026-05-10

Emergency bugfix: feature-aware KG scoring crash in v1.3.0.

### Fixed
- `TransEModel._embed_entities` and `_embed_relations` called `_FeatureProjector`
  with only `feat` instead of `(emb, feat)`, raising
  `TypeError: _FeatureProjector.forward() missing 1 required positional argument: 'feat'`
  when `entity_feature_dim` or `relation_feature_dim` was set.
  Fixed both call sites to pass `(emb, feat.float())`.

### Tests
- Added `tests/test_kg_feature_aware_v131.py` (8 regression tests covering
  entity features, relation features, both together, gradient flow, and the
  exact Colab reproduction case).

### Notes
- Only `TransEModel` was affected; `DistMultModel` was already correct.
- No public API changes; the fix restores the documented behaviour.

---

## [1.3.0] — 2026-05-10

Strategic quality-upgrade release: new KG model, KG HPO, RL callback integration,
expanded notebook gallery, and v1.3 benchmark suite.

### Added
- `SimplEModel` — symmetric bilinear KG scoring `0.5*(⟨h_head, r_fwd, t_tail⟩ + ⟨t_head, r_inv, h_tail⟩)`.
  Captures asymmetric relations unlike DistMult. Beta: 12 tests including hand-computed D=1 values.
- `run_kg_hpo(kg, model_names, search_space, metric, strategy, max_trials, ...)` — lightweight KG
  hyperparameter search (grid/random). Returns `KGSearchResult` with `summary()`, `to_dict()`,
  and `write_dashboard_artifacts()`. Beta: 12 tests.
- `run_graph_rl(callbacks=...)` — callbacks parameter wired into `run_graph_rl` and `_run_discrete`.
  `EarlyStoppingCallback`, `CSVLoggerCallback`, and `CallbackList` now fire lifecycle hooks
  (on_train_start, on_episode_start/end, on_train_end). `result.stopped_early` reflects early stop.
- 7 educational notebooks in `notebooks/` (validated, CPU-runnable):
  - Easy Mode tensor node classification
  - Image-patch tensor graph (tensor-vs-flatten comparison, core TGraphX identity)
  - KG completion with RESCAL, TransE, SimplE, and HPO
  - Graph generation and evolutionary optimization
  - Graph RL with callbacks
  - GraphML IO round-trip
  - v1.3 benchmark suite and dashboard artifacts
- `tools/generate_notebooks.py` — idempotent notebook generator.
- `tools/validate_notebooks.py` — structural notebook validator.
- `tools/generate_colab_drafts.py` — generates 30 expanded Colab draft scenarios locally.
- `tools/validate_colab_drafts.py` — validates local draft notebooks.
- `benchmarks/run_v13_benchmark_suite.py` — 11-benchmark suite:
  7 inherited v1.2 + SimplE smoke, KG HPO smoke, RL callbacks smoke, notebook validation.
- `tests/test_kg_simple_v130.py` — 12 tests (hand-computed, asymmetry, CUDA, overfit, registry).
- `tests/test_kg_hpo_v130.py` — 12 tests (grid/random, selection, artifacts, errors).
- `tests/test_notebooks_v130.py` — 22 tests (existence, structure, content, validation tool).
- Updated `docs/api_stability.md`, `docs/api_cheatsheet.json`, `docs/index.md`,
  `docs/colab_gallery.md` for v1.3 additions.

### Notes
- Local Colab draft notebooks (`colab_drafts/`) are intentionally not included in this release.
  Verified Colab links will be added in v1.3.1 after maintainer upload/testing.
- `SimplE` added to `list_kg_models()` and `tutorials/kg_benchmark_quickstart.py`.
- This release does not claim parity with PyG/DGL, PyKEEN, NetworkX, SB3, or RLlib.

---

## [1.2.0] — 2026-05-10

Ecosystem-quality release with new KG model, graph IO, RL callbacks, tutorials, and benchmark suite.

### Added
- `RESCALModel` — bilinear KG scoring `f(h, r, t) = h^T M_r t` with hand-computed bilinear tests
  (zero-matrix → 0, identity → dot product, concrete [[1,2],[3,4]] case → 61, asymmetry capture).
  Listed in `list_kg_models()` and `docs/api_cheatsheet.json`.
- `tgraphx.io.write_graphml` / `read_graphml` — pure-stdlib GraphML round-trip for structure,
  `edge_weight`, node/edge labels, and 1-D tensor features. Multi-dim tensor features rejected
  with clear error. 14 tests covering round-trip, labels, weights, paths, and error paths.
- `tgraphx.rl.{Callback, CallbackList, EarlyStoppingCallback, CSVLoggerCallback}` — lightweight
  RL callback system with fan-out, patience-based early stopping, and lazy-file CSV logging.
  16 unit tests.
- `tutorials/real_dataset_cora_node_classification.py` — optional PyG / graceful synthetic fallback.
- `tutorials/image_patch_tensor_graph_demo.py` — image-patch tensor graph with tensor-vs-flatten comparison.
- `tutorials/kg_benchmark_quickstart.py` — TransE / DistMult / RESCAL + filtered MRR / Hits@K.
- `benchmarks/run_v12_benchmark_suite.py` — aggregates 7 representative benchmarks to a single
  JSON with stable schema (`name`, `status`, `runtime_s`, `device`, `seed`, `metrics`).
- `docs/benchmark_report.md` — formal benchmark structure with explicit smoke-vs-performance scope.
- `docs/colab_gallery.md` — script-runnable tutorial index.
- `docs/io.md` — GraphML usage, limitations, and roadmap.
- Updated `docs/api_stability.md`, `docs/api_cheatsheet.json`, `docs/index.md`, and
  `docs/limitations.md` for v1.2 additions.
- Hand-computed math tests for KG filtered ranking, TransE distance, DQN target, Double DQN,
  PPO clip (3 cases), GAE, and Polyak averaging (v1.1 additions, now part of stable test suite).
- Loader robustness tests: NeighborLoader determinism, label/feature preservation, GraphSAINT
  normalization, Cluster-GCN partition coverage, sparse backend fallback (v1.1 additions).

### Validation
- Full test suite passed locally: 2633 passed, 12 skipped, 0 failed.
- Focused v1.2 tests passed locally.
- Build and twine validation passed locally (zero warnings).
- CI passed before tag/release.

### Notes
- This release does not claim parity with PyG/DGL, NetworkX, PyKEEN, SB3, or RLlib.
- GEXF/Pajek IO, KG HPO, Gymnasium adapter, vectorized RL environments, and deeper graph mining
  remain roadmap items.

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

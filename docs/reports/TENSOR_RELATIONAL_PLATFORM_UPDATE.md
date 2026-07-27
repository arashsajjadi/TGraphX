# Engineering report: tensor-relational platform update (v1.5.0)

Internal record for the v1.5.0 release. Audience: maintainers.
Companion user-facing docs: `docs/tensor_relational_platform.md`,
`docs/set_transformer.md`, `docs/migration_v1_4_to_v1_5.md`.

## 1. Old package behaviour (≤ 1.4.2)

- `CNNEncoder.__init__` defaulted `dropout_prob=0.3` (also
  `use_batchnorm=True`, `use_residual=True`), invisible in `repr()`,
  configs, and checkpoints (`tgraphx/models/cnn_encoder.py:38` in
  1.4.2).
- `DeepCNNAggregator.__init__` defaulted `dropout_prob=0.3` (also
  `use_batchnorm=True`) and inserted the dropout module even for
  `p=0` (`tgraphx/layers/aggregator.py:42`).
- `ConvMessagePassing(aggregator_params=None)` inherited the silent 0.3;
  `GraphClassifier` had no dropout parameter at all and inherited it in
  every layer.
- `make_layer("conv", ..., dropout=X)` and `build_model(layer="conv",
  dropout=X)` **silently discarded `X`** and never forwarded
  `aggregator_params`, so config-driven models carried
  `Dropout2d(p=0.3)` per aggregator conv layer regardless of the
  config (verified empirically: `build_model(..., dropout=0.0)` → 8
  active `Dropout2d(p=0.3)` for a 2-layer model).
- No graph model family existed for the learned-implicit-relations
  regime (no SetTransformer/DeepSets-style component), and there was no
  machine-readable notion of a model's topology source.

## 2. Hidden-dropout root cause and measured impact

Root cause chain (from `TGraphX_revised/provenance/recon/`,
`LEAD_mechanism_verified.md` and `R2_library_internals.md`, auditing the
tgraphx 1.4.2 PyPI wheel): the old PASTIS-R wrapper built
`CNN_GNN_Model` with `cnn_params` lacking `dropout_prob`, so the node
CNN encoder ran `nn.Dropout2d(p=0.3)` in each of its 3 layers, while
every baseline's node encoder had **no dropout**.  The aggregator's own
0.3 default was neutralized by `gnn_dropout=0.0` via `setdefault`; the
encoder default was the one that bit.  Final-epoch (rather than
best-val) selection amplified the penalty.

Measured impact (revised workspace, PASTIS-R parcel classification,
5 paired seeds, macro-F1):

- CNN dropout 0.3 → 0.0 in isolation (C0 sub-sweep C, α-gate fusion):
  0.559 → 0.601 (**≈ −0.04 to −0.06 cost from the hidden 0.3**).
- Full correction (dropout 0 + α-gate fusion + aggregator BN +
  best-val selection): validation **0.5360 → 0.6326** (+0.0966,
  CI [+0.079, +0.114]).

## 3. Exact code changes (v1.5.0)

Dropout/config (commit "explicit dropout"):

- new `tgraphx/_compat.py`: `DropoutDefaultChangeWarning(UserWarning)`,
  `LEGACY_CNN_DROPOUT_PROB = 0.3`, `resolve_dropout_prob()` (None →
  0.0 + warning; validates `[0, 1)`).
- `CNNEncoder`: `dropout_prob=None` sentinel default; stores/exposes all
  constructor state; `extra_repr()`; `config()`; `legacy()` classmethod
  (0.3/BN/residual, no warning).
- `DeepCNNAggregator`: same treatment; dropout module now only inserted
  when `p > 0` (state_dict layout unaffected — dropout has no
  parameters).
- `ConvMessagePassing`: explicit `dropout_prob` parameter merged into
  `aggregator_params`; conflict raises `ValueError`; mirrors effective
  value onto `self.dropout_prob`.
- `TensorMessagePassingLayer`: `extra_repr()` exposing
  `dropout_prob`/`residual`/`use_batchnorm`.
- `GraphClassifier`: explicit `dropout_prob` parameter threaded into
  every layer.
- `CNN_GNN_Model`: resolves missing `cnn_params['dropout_prob']` loudly;
  no longer mutates the caller's `cnn_params` dict.
- `make_layer("conv")`: forwards `dropout` → `dropout_prob`,
  `use_batchnorm`, and `aggregator_params`.

SetTransformer + unified API (commit "set transformer"):

- new `tgraphx/models/set_transformer.py`: `SetTransformerModel`,
  `SetAttentionBlock`, `AttentionPooling` (details in §5).
- new `tgraphx/models/topology.py`: `TOPOLOGY_SOURCES = ("none",
  "fixed", "given", "learned_implicit", "learned_explicit", "hybrid")`,
  `topology_source_of()`, `TopologyIgnoredWarning`.
- `build_model`: accepts `family=` as alias for `layer=`; dispatches
  `set_transformer` at model level; tags every returned model with
  `model_family` and `topology_source`; `make_layer("set_transformer")`
  raises with a pointer to `build_model`.
- exports registered in `tgraphx/__init__.py`, `tgraphx/models/__init__.py`,
  `tgraphx/ux/public_api.py`, `docs/api_stability.md`.

## 4. Compatibility decision

Changing 0.3 → 0.0 alters *training-time* behaviour of code that relied
on the silent default.  We chose: new default 0.0 **plus a
per-construction-site `DropoutDefaultChangeWarning`** when the value is
unspecified, rather than (a) keeping 0.3 (would keep silently harming
users) or (b) changing silently (prohibited).  Rationale:

- dropout holds no parameters → `state_dict` layouts identical; old
  checkpoints load unchanged in both directions;
- `eval()` outputs never depended on dropout → inference on loaded
  checkpoints is bit-identical, so no loaded legacy checkpoint changes
  silently;
- the affected behaviour (hidden regularization) is a defect, and the
  warning marks exactly the construction sites whose behaviour changed;
- `.legacy(...)` constructors + `LEGACY_CNN_DROPOUT_PROB` make the old
  behaviour reproducible intentionally, without warnings;
- `use_batchnorm`/`use_residual` defaults are **unchanged** because they
  do affect checkpoint parameter layout; they are now visible in
  `repr()`/`config()` and documented as graph-density-dependent
  (BatchNorm helped dense temporal chains +0.017; harmful at 58.5%
  zero-degree rows — do not treat it as universally harmful).

## 5. SetTransformer architecture and conceptual position

`SetTransformerModel(task, in_shape, embed_dim=64, num_layers=2,
num_heads=4, ffn_dim=2*embed_dim, dropout=0.0, attention_dropout=0.0,
num_classes/out_dim, pooling="attention", num_seeds=1, layer_norm=True,
encoder=None, encoder_config=None, on_edge_index="warn")`:

- shared node encoder → `[N, embed_dim]` (vector: MLP; 2-D: the package
  `CNNEncoder` with explicit-zero dropout defaults; 3-D: small Conv3d
  encoder; or custom module);
- flat `[N, ...]` + `batch` → dense `[B, M, E]` + key-padding mask
  (stable argsort; order-robust; zero-node graphs rejected with a clear
  error);
- pre-LN MHSA blocks (`nn.MultiheadAttention`, batch_first) —
  permutation-equivariant, masked so nodes attend only within their
  graph;
- readout: PMA (learned seed queries, `nn.MultiheadAttention`) or
  mean/sum/max via the package pooling helpers — permutation-invariant;
- `config()`/`from_config()` deterministic round trip;
  `encode_nodes()` exposes pre-readout embeddings; CPU/CUDA parity
  verified to ~3e-8.

Why it is **relation-aware but explicit-topology-blind**: attention
weights are functions of node content, so pairwise interactions are
learned (dense implicit relations), but the computation never reads
`edge_index` — unlike `TensorGATLayer`, which softmax-attends only over
supplied edges, and unlike `tgraphx.learned_graph`, which constructs a
discrete/learned edge set that is then message-passed.  The
`on_edge_index` contract ("warn"/"ignore"/"error", warn-once default)
plus `topology_source="learned_implicit"` metadata makes the blindness
explicit rather than silent.

No new dependencies: PyTorch primitives only.

## 6. Tests performed

New suites (both green):

- `tests/test_explicit_dropout_v150.py` — 22 tests: unspecified→0.0+warn
  for all five construction surfaces; explicit==effective; zero → no
  active dropout path + deterministic train forward; nonzero → train
  stochastic/eval deterministic; repr/config parity; config round trip;
  conflict error; factory paths honour `dropout` (regression for the
  silent-0.3 bug); legacy constructors (no warning, 0.3); legacy↔new
  state_dict interchange; checkpoint save/load output parity; public
  importability/filterability of the warning.
- `tests/test_set_transformer_v150.py` — 37 tests: shapes (4 tasks,
  vector/2-D/3-D), variable node counts, batch=None, input validation,
  empty-graph rejection, permutation invariance (all four poolings),
  equivariance of `encode_nodes`, padding-mask isolation,
  batched==individual, edge_index warn-once/ignore/error + output
  independence from edges and edge order, topology metadata, gradient
  reach (encoder/blocks/pool/head), finite forward/backward, explicit
  dropout visibility, fixed-seed deterministic construction, config
  round trip, custom-encoder config refusal, checkpoint exactness,
  CPU/CUDA parity, factory/family-alias/config-file construction,
  GraphBatch+fit+evaluate integration, and two tiny synthetic sanity
  checks (memorize 6 sets; key-query relation task beats a pooling-only
  baseline 0.99 vs 0.85) — labelled non-scientific.

Full suite after all changes: see release notes (3351 baseline tests
before the change: all passing, 23 skipped; final counts recorded in
CHANGELOG/release notes).

## 7. Evidence reused; experiments deliberately NOT rerun

All numbers cited in docs come from the frozen artifacts in
`TGraphX_revised` (protocol: PASTIS-R parcel-level crop classification —
a constructed task, not the published PASTIS benchmark; 18-class
macro-F1; frozen tile/fold splits; 5 paired seeds; 10-epoch matched
budgets; best-val checkpoint selection; artifacts sha256-manifested in
`06_synthesis/REPRODUCIBILITY_MANIFEST.md`).  Reused:

- S2-only frozen-base validation (`01_frozen_base_revised`):
  set_transformer 0.7023, temporal_transformer 0.6914, pairset 0.6520,
  tgraphx_corrected 0.6326, fixed_imputed_cnn 0.6196, deepsets 0.6099,
  flatten_gnn 0.6012, tgraphx_original_explicit (bridge) 0.5360.
- Branch B multimodal validation (`04_branch_b_revised`):
  SetTransformer 0.6593, real-topology TGraphX 0.6306, S2-only 0.6232,
  matched-content blind 0.5813, shuffled topology 0.5718; topology
  contrasts +0.059 (vs shuffled) and +0.049 (vs matched blind), both
  significant.
- C0 dropout isolation (0.559 vs 0.601) and BN density findings.

**Frozen geographic test (not hidden):** on test tile `t30uxv`
(evaluated once per model/seed), all models drop ~0.25–0.30 macro-F1:
set_transformer 0.4272, temporal_transformer 0.4043, deepsets 0.3417,
tgraphx_original_explicit 0.3279, pairset 0.3245, **tgraphx_corrected
0.3243 (rank 6/8)**, flatten_gnn 0.3183, fixed_imputed_cnn 0.0950.
The corrected TGraphX's validation gain did **not** transfer
(corrected − original −0.0002, n.s.), and it **loses to DeepSets on
test** (−0.022, significant).  Documentation therefore claims topology
value on validation only and carries an explicit generalization caveat.

Deliberately not rerun: the entire PASTIS-R matrix (frozen-base 100
runs, C0 33, Branch B 25, Route E 30 — ≈ 82 GPU-hours).  No new
training beyond unit/sanity scope was executed for this release; the
only new compute is the test suite plus two tiny synthetic sanity
trainings (seconds each, CPU-scale).  No CNN baseline was rerun: the
matched `fixed_imputed_cnn` result already exists in the frozen-base
artifacts.

## 8. Remaining limitations

- No first-class **hybrid** (given + learned residual relations) family
  yet; the vocabulary and `GraphTransformerLayer(edge_bias=True)` leave
  a clean slot for it.
- SetTransformer attention is O(M²) per graph (no ISAB/inducing-point
  variant yet); fine at PASTIS-scale set sizes.
- `edge_features`/`edge_weight` are ignored (with the same contract as
  `edge_index`) rather than embedded into attention biases.
- The dropout transition warning fires per construction site for one
  minor release; removal is scheduled for the next major (documented in
  the migration guide).
- `GraphTransformerLayer` remains vector-only and experimental.
- Frozen-test generalization gap (above) is a property of the task
  distribution shift, not solved by this release.

## 9. Commits and release status

- explicit dropout configuration and migration: `6d67adb`
- SetTransformer + topology vocabulary + unified factory: `065b6ff`
- tests, docs, examples: the commit introducing this report
- version/changelog/release metadata: the commit carrying the `v1.5.0` tag

Release: version 1.5.0 (semantic-version MINOR: backward-compatible
family addition + loud configuration-semantics fix), tag `v1.5.0`,
pushed to `origin` (github.com/arashsajjadi/TGraphX); PyPI publication
via the repository's `publish.yml` workflow on GitHub-release creation.
Final statuses are recorded in `CHANGELOG.md` and
`docs/releases/v1.5.0.md`.

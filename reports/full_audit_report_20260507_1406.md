# TGraphX — Full Read-Only Audit Report

- **Generated:** 2026-05-07 14:06 (local)
- **Auditor mode:** read-only (no files modified, no git operations)
- **Repository:** `/home/arash/PycharmProjects/TGraphX`
- **Branch:** `main` (clean working tree at start and end)
- **Reviewed version:** TGraphX `0.1.2` (importable; matches `pyproject.toml` and `tgraphx/__init__.py`)
- **Python:** 3.13.12 · **PyTorch:** 2.11.0+cu130 · **CUDA available:** True · **MPS:** False

---

## 1. Executive verdict

TGraphX 0.1.2 is a **well-tested, honestly scoped** PyTorch package. The full pytest suite passes (`675 passed, 10 skipped`), all 28 fast examples run green, the wheel/sdist build is clean, and `twine check` passes. The package's distinctive value (tensor-aware message passing on `[N, C, H, W]` and `[N, C, D, H, W]` features) is genuinely implemented and well-covered by tests. Code is generally clear, with rank-aware helpers and validation messages that punch above the package's apparent age.

However, this audit found **one critical packaging bug that breaks the dashboard for every PyPI user**, **one high-severity logging bug** that silently corrupts TensorBoard step indices, **stale documentation that contradicts shipped features**, and **a cluster of UX/API ergonomics issues** that unnecessarily expose users to surprising defaults and incomplete factory pass-throughs. These are all fixable without breaking the public API, and most can be addressed before the next patch release.

**Release risk if no further changes are made:** Medium. The dashboard packaging bug alone justifies a `0.1.3` patch.

**Main strengths**
- Strict, descriptive input validation across `Graph`, layers, and builders.
- Genuinely rank-aware GAT / SAGE / GIN with consistent `spatial_rank` semantics.
- Tracking and dashboard are off by default and audited for path traversal.
- Healthy test count (~685 collected) including device skips and 3-D paths.

**Main weaknesses**
- Dashboard static assets (`dashboard.css`, `dashboard.js`) are missing from the wheel and sdist.
- `TensorBoardLogger.log` mishandles `epoch=0` / `step=0` (falsy `or`-chain).
- `LinearMessagePassing` silently drops `dropout_prob`, `residual`, and `use_batchnorm`.
- `docs/limitations.md` declares `train_epoch` / `evaluate` / `fit` / `TensorBoardLogger` "Not implemented" — they are implemented and documented elsewhere in the same `docs/` tree.
- `make_layer` does not pass through several layer parameters (e.g. GIN `eps`, `train_eps`, `hidden_channels`, `use_batchnorm`).
- `_call_model` and `_compute_metrics` swallow exceptions broadly enough to hide real bugs.

---

## 2. Table of contents

1. Executive verdict
2. Table of contents
3. Commands run and results
4. Working tree status and generated artifacts
5. Critical correctness findings
6. API / user-friendliness findings
7. Syntax simplification opportunities
8. Performance bottleneck findings
9. Layer / model correctness findings
10. Training / logging / dashboard findings
11. Graph builders / patch helpers findings
12. Docs / README / Colab findings
13. Test coverage findings
14. Security / privacy / side-effect findings
15. Release / CI findings
16. Backward compatibility risk table
17. Confirmed-issues table (Table 1)
18. Performance bottlenecks table (Table 2)
19. API/UX simplification table (Table 3)
20. Documentation/Colab issues table (Table 4)
21. Hypotheses requiring verification (Table 5)
22. Competitive / ecosystem audit
23. "No reason to leave" user-journey audit
24. Criticism-defense table
25. Prioritized roadmap (Batches A–H)
26. Final competitive roadmap (Phases 1–6)
27. Top 10 highest-value next actions
28. Top 10 things NOT to do yet
29. Questions requiring human decision
30. Final recommendation
31. Appendix — raw command outputs

---

## 3. Commands run and results

| # | Command | Result |
|---|---|---|
| 1 | `python -c "import tgraphx; print(tgraphx.__version__)"` | `0.1.2` ✅ |
| 2 | `cd /tmp && python -c "import tgraphx; ..."` | imports OK from outside repo ✅ |
| 3 | `python -c "import torch; print(torch.cuda.is_available())"` | `2.11.0+cu130` · CUDA True ✅ |
| 4 | `pytest -q` | `675 passed, 10 skipped in 28.58s` ✅ |
| 5 | `python -m build` | Built `tgraphx-0.1.2.tar.gz` and `tgraphx-0.1.2-py3-none-any.whl` ✅ |
| 6 | `twine check dist/tgraphx-0.1.2*` | Both `PASSED` ✅ |
| 7 | `python examples/run_all_fast_examples.py` | `OK 28 · FAIL 0 · TIMEOUT 0` ✅ |
| 8 | `python benchmarks/benchmark_layers.py --layer gat ...` | 0.184 ms ±0.037 ms ✅ |
| 9 | `unzip -l dist/tgraphx-0.1.2-py3-none-any.whl \| grep static` | **No static files in wheel** ❌ (see BUG-01) |
| 10 | `tar tzf dist/tgraphx-0.1.2.tar.gz \| grep static` | **No static files in sdist** ❌ (see BUG-01) |
| 11 | Manual repro of `TensorBoardLogger` `epoch=0` after `_step>0` | step recorded as `_step` instead of `0` ❌ (see TRACK-01) |
| 12 | Manual repro of `LinearMessagePassing(..., dropout_prob=0.5, residual=True, use_batchnorm=True)` | `train==eval` (dropout never applied), `bn` never called ❌ (see API-01) |
| 13 | `python -X importtime -c "import tgraphx"` | Only PyTorch's own conditional `pynvml` probe; no TGraphX-side heavy imports ✅ |

No commands failed. All commands returned with exit code 0.

---

## 4. Working tree status and generated artifacts

- **Working tree:** unchanged. No source files were created, modified, deleted, or renamed by this audit. `git status` is clean.
- **Generated artifacts** (created by `python -m build`, expected, no source changes):
  - `dist/tgraphx-0.1.2.tar.gz`
  - `dist/tgraphx-0.1.2-py3-none-any.whl`
  - Both files are produced by the official build system and would be re-created identically on any release.
- **No new files** outside `dist/` were created. `runs/`, `tgraphx.egg-info/`, and `.pytest_cache/` were already present before the audit.

---

## 5. Critical correctness findings

### BUG-01 (Critical) — Dashboard static assets missing from wheel and sdist

- **Location:** `pyproject.toml` (no `[tool.setuptools.package-data]`, no `include-package-data`, no `MANIFEST.in`); affected files served by `tgraphx/dashboard/app.py:STATIC_DIR` and `_serve_static`.
- **Current behavior:** The source tree contains `tgraphx/dashboard/static/dashboard.css` and `tgraphx/dashboard/static/dashboard.js`. The built wheel (`tgraphx-0.1.2-py3-none-any.whl`) and sdist (`tgraphx-0.1.2.tar.gz`) **do not** include those files. `unzip -l dist/tgraphx-0.1.2-py3-none-any.whl | grep static` returns nothing.
- **Why this is a problem:** Every PyPI user who runs `tgraphx-dashboard --logdir ...` and opens the URL will see the HTML shell load and immediately fail on `/static/dashboard.css` and `/static/dashboard.js` with the `Static file missing` 404 served by `_serve_static`. The dashboard UI is therefore non-functional after `pip install tgraphx`. Local editable installs (`pip install -e .`) hide the bug because Python resolves `STATIC_DIR` against the live source tree.
- **How to verify:** `python -m pip install --force-reinstall dist/tgraphx-0.1.2-py3-none-any.whl && tgraphx-dashboard --logdir runs/whatever` then open browser; both `/static/*` URLs return 404. Or run `unzip -l dist/tgraphx-0.1.2-py3-none-any.whl | grep static`.
- **Recommended fix direction:** In `pyproject.toml` add explicit package-data:

      [tool.setuptools.package-data]
      "tgraphx.dashboard" = ["static/*.css", "static/*.js"]

  Or add `include-package-data = true` and a `MANIFEST.in` line `recursive-include tgraphx/dashboard/static *`. Verify by re-running `python -m build` and inspecting the wheel.
- **Backward compatibility:** Safe — this only adds files into the distribution; no API changes.
- **Files likely affected:** `pyproject.toml` (or new `MANIFEST.in`).
- **Tests to add:** A new `tests/test_packaging.py` that:
  1. Builds the wheel via `python -m build --wheel` into a temp dir,
  2. Asserts that `tgraphx/dashboard/static/dashboard.css` and `dashboard.js` are present in the zip,
  3. Runs `tgraphx-dashboard --logdir <empty>` against the installed wheel in a venv if feasible (otherwise just static-file presence is enough).
- **Priority:** **Must fix before next release.**

### BUG-02 (High) — `LinearMessagePassing.update` silently ignores `dropout_prob`, `residual`, `use_batchnorm`

- **Location:** `tgraphx/layers/base.py:163-165` (`LinearMessagePassing.update`).
- **Current behavior:** `LinearMessagePassing.__init__` calls `super().__init__(... dropout_prob=..., residual=..., use_batchnorm=...)`, which constructs `self.bn`, `self.dropout`, and stores `self.residual`. Its overridden `update()` body is a one-line `return aggregated_message`, so none of those modules are ever applied.
- **Why this is a problem:** A user who constructs `LinearMessagePassing((D,), (D,), dropout_prob=0.5, residual=True, use_batchnorm=True)` is silently given a layer that does **none** of those things, and `train()` vs `eval()` produce identical outputs (verified by reproduction script). Worse, the `bn` parameters appear in `state_dict` but never receive gradients. `make_layer("linear", ..., dropout=0.3, residual=True)` silently propagates the same defect.
- **How to verify:** Reproducer:

      from tgraphx.layers.base import LinearMessagePassing
      layer = LinearMessagePassing((8,), (8,), dropout_prob=0.5, residual=True, use_batchnorm=True)
      x = torch.ones(4, 8); ei = torch.tensor([[0,1,2,3],[1,2,3,0]])
      a = layer(x, ei); layer.eval(); b = layer(x, ei)
      assert not torch.equal(a, b), "dropout did nothing"

  Currently fails (the assertion is incorrectly satisfied → bug confirmed).
- **Recommended fix direction:** Either delete the `update()` override in `LinearMessagePassing` (so the base class implementation runs and applies bn/dropout/residual), or copy the relevant logic explicitly. If the historical intent was to keep `LinearMessagePassing` minimal, then `__init__` should reject `dropout_prob > 0`, `residual=True`, `use_batchnorm=True` rather than accepting them silently. Because `make_layer` and `NodeRegressor`/`GraphRegressor` already pass these parameters in, the safe fix is to **make them effective** rather than to reject them.
- **Backward compatibility:** Medium-risk — users who rely on the no-op behavior (unlikely) would see different outputs. But the documented behavior matches "make these flags real," so this is a bug-fix not a feature change.
- **Files likely affected:** `tgraphx/layers/base.py`.
- **Tests to add:** in `tests/test_layers.py`, extend the existing layer table to verify `train()`/`eval()` divergence under `dropout_prob>0`, and that `residual=True` makes `out = x + msg`. Also assert `use_batchnorm=True` registers `running_mean` etc.
- **Priority:** **Should fix soon.**

### TRACK-01 (High) — `TensorBoardLogger.log` mis-resolves `epoch=0` / `step=0` because of `or`-chain on falsy zero

- **Location:** `tgraphx/tracking.py:224` — `step = kwargs.get("epoch") or kwargs.get("step") or self._step`.
- **Current behavior:** Python `or` treats `0` as falsy. If `kwargs["epoch"] == 0`, the expression skips it and falls through to `kwargs.get("step")` (which may be `None`) and then to the internal `self._step` counter. After any prior `logger.log(train_loss=...)` call without an `epoch` argument, `self._step > 0`, so the very first epoch (epoch 0) of training is recorded at the wrong global step.
- **Why this is a problem:** TensorBoard plots use `global_step` as the X-axis; a wrong step yields curves that visually skip epoch 0 or collide later epochs. This is a particularly nasty bug because nothing fails — the chart is simply wrong, and CSV-logging correlated runs will not match TB.
- **How to verify:** Verified live in this audit:

      logger.log(train_loss=0.9)         # _step -> 1
      logger.log(train_loss=0.8)         # _step -> 2
      logger.log(epoch=0, train_loss=0.5)  # observed step=2 (BUG)
      logger.log(epoch=2, train_loss=0.3)  # observed step=2

- **Recommended fix direction:** Replace the `or`-chain with explicit `is None` checks:

      if "epoch" in kwargs and kwargs["epoch"] is not None:
          step = int(kwargs["epoch"])
      elif "step" in kwargs and kwargs["step"] is not None:
          step = int(kwargs["step"])
      else:
          step = self._step
          self._step += 1

  Note `self._step` should only auto-increment when neither key is present, otherwise back-to-back calls with the same `epoch=N` collide on the same step (already the case today, intentional).
- **Backward compatibility:** Safe. Behavior changes only for the buggy paths.
- **Files likely affected:** `tgraphx/tracking.py`.
- **Tests to add:** `tests/test_tracking.py` — unit tests that fake the writer and assert `add_scalar` is called with the expected `global_step` for `epoch=0`, `step=0`, mixed sequences, and the default counter path.
- **Priority:** **Must fix before next release.** Trivially safe.

### BUG-03 (Medium) — `TensorGATLayer.add_self_loops=True` does not deduplicate against existing self-loops

- **Location:** `tgraphx/layers/gat.py:241-244` — `add_self_loops` branch.
- **Current behavior:** When `add_self_loops=True`, the layer always concatenates `arange(N)` self-loops to `src/dst` regardless of whether some nodes already have explicit self-loops in `edge_index`. `Graph.add_self_loops()` (the data-side helper in `tgraphx/core/graph_utils.py:148`) does deduplicate. The two paths therefore disagree.
- **Why this is a problem:** Nodes with pre-existing self-loops effectively get *two* self-attention edges — softmax over destinations now distributes attention across `existing_self + new_self + neighbours`, biasing toward self in a non-obvious way. Users mixing `Graph.add_self_loops()` and `TensorGATLayer(add_self_loops=True)` will see different behavior than expected.
- **How to verify:** Construct a graph with one self-loop on node 0, run `TensorGATLayer(add_self_loops=True, return_attention=True)` and inspect the attention vector for destination 0; it sums over more entries than expected. Add a test in `tests/test_layers.py`.
- **Recommended fix direction:** Mirror `core/graph_utils.add_self_loops` logic: detect existing self-loops and only append loops for nodes that lack them. Self-loop padding in `edge_weight` and `edge_features` (`pad = ones(N)` and `zeros(N, K)`) must be sized accordingly.
- **Backward compatibility:** Medium — outputs change for graphs that already had self-loops (likely few users in practice but the semantics drift).
- **Files likely affected:** `tgraphx/layers/gat.py`.
- **Tests to add:** assert that `out_with_loops_dedup ≈ out_without_loops_when_graph_already_had_them`.
- **Priority:** Should fix soon.

### BUG-04 (Low) — `_unpack_batch` auto-squeeze drops a real dimension for `[B, 1]` regression targets

- **Location:** `tgraphx/training.py:230-233`.
- **Current behavior:** When `targets.dim() == 2 and targets.size(-1) == 1`, the trailing singleton is squeezed. This is documented and convenient for `CrossEntropyLoss`. But for `MSELoss` with shape `[B, 1]` outputs and `[B, 1]` targets, the squeeze produces a `[B]` target while the model output is still `[B, 1]`, and `MSELoss` will broadcast (silently producing `[B, B]`). PyTorch nightly recently started warning about this; older PyTorch silently broadcasts.
- **Why this is a problem:** Surprising and silent for graph regression; users with `out_dim=1` regression won't see an error but get a wrong loss.
- **How to verify:** Run `examples/training_minimal_fit.py`-style code with `loss_fn=torch.nn.MSELoss()` and `out_dim=1`; check loss numerics versus an independent reference.
- **Recommended fix direction:** Make the squeeze classification-only: only squeeze when the loss function is known to expect 1-D class indices, or expose an explicit `squeeze_singleton_targets=True` parameter that defaults to the current behavior, with docs noting the regression-with-out_dim=1 case. The simpler conservative path is to detect target dtype `torch.long` and only squeeze in that case.
- **Backward compatibility:** Safe (current users with `[B,1]` long labels continue to work).
- **Files likely affected:** `tgraphx/training.py`.
- **Tests to add:** in `tests/test_training.py`, parametrize over `out_dim ∈ {1, 4}` and `loss_fn ∈ {CE, MSE}` and assert the produced loss matches a manual reference.
- **Priority:** Nice to have.

---

## 6. API / user-friendliness findings

### API-01 (High) — `make_layer` does not forward several useful layer parameters

- **Location:** `tgraphx/layers/factory.py:165-173` (gin), `:151-163` (sage).
- **Current behavior:** `make_layer("gin", ...)` accepts only `use_edge_features`, `edge_dim`, and `edge_features_kind`. The underlying `TensorGINLayer` exposes `eps`, `train_eps`, `hidden_channels`, `use_batchnorm`, and `mlp` — none reachable through the factory. Similarly, GAT's `attn_dropout` is mapped from `kwargs.get("dropout", 0.0)`, but `TensorGATLayer` also accepts `negative_slope` and exposes `concat=True` — both reachable but not documented in the same place.
- **Why this is a problem:** Users who build models with `build_model(..., layer="gin", ...)` cannot tune any GIN-specific knob without bypassing the factory. This makes the factory feel incomplete and forces users to read the layer source.
- **How to verify:** Read `tgraphx/layers/factory.py` and grep `make_layer` references in tests/examples — none use these knobs because they cannot.
- **Recommended fix direction:** Whitelist additional kwargs per layer:

      if name == "gin":
          return TensorGINLayer(..., eps=float(kwargs.get("eps", 0.0)),
                                train_eps=bool(kwargs.get("train_eps", False)),
                                hidden_channels=kwargs.get("hidden_channels"),
                                use_batchnorm=bool(kwargs.get("use_batchnorm", False)))

  And add a test that constructs each layer through the factory and verifies the parameter is honored.
- **Backward compatibility:** Safe — adding new kwargs cannot break existing callers.
- **Files likely affected:** `tgraphx/layers/factory.py`, `tests/test_factories.py`, `docs/factories.md`.
- **Priority:** Should fix soon.

### API-02 (High) — Public top-level `tgraphx` namespace omits training/tracking helpers

- **Location:** `tgraphx/__init__.py` (lines 11-86).
- **Current behavior:** Top-level imports expose `Graph`, `GraphBatch`, layers, builders, and `build_model`, but **not** `set_seed`, `count_parameters`, `train_epoch`, `evaluate`, `fit`, `CSVLogger`, `TensorBoardLogger`, `env_report`, `recommended_device`, `EdgePredictor` is exposed but `GraphClassifier` / `NodeClassifier` / `CNN_GNN_Model` / `CNNEncoder` / `PreEncoder` are not. README implies all of those are first-class symbols.
- **Why this is a problem:** New users follow the README's "what is implemented" table and try `from tgraphx import GraphClassifier` — which fails. They then have to learn the submodule layout (`tgraphx.models`, `tgraphx.training`, `tgraphx.tracking`) — strictly more boilerplate than PyG/DGL where the equivalent symbols are top-level.
- **Recommended fix direction:** Re-export safely with a separate `from .training import ...`/`from .tracking import ...` import block. Tracking imports `csv` and `datetime` only — cheap. Training imports `torch.nn` only — already loaded. No heavy deps. Add to `__all__` and update docs. Crucially: keep the existing submodule paths working too (no rename, just additional aliases).
- **Backward compatibility:** Safe — purely additive.
- **Files likely affected:** `tgraphx/__init__.py`, `docs/api_reference.md`.
- **Tests to add:** `tests/test_imports.py` add `from tgraphx import fit, CSVLogger, TensorBoardLogger, GraphClassifier, NodeClassifier`.
- **Priority:** Should fix soon.

### API-03 (Medium) — `_call_model` silently catches `TypeError` and retries with stripped kwargs

- **Location:** `tgraphx/training.py:252-264`.
- **Current behavior:** If `model(*args, **kwargs)` raises `TypeError`, the helper catches it and retries `model(*args)` (no kwargs). If the retry also fails, the original `TypeError` is suppressed and a generic `RuntimeError` is raised. Real bugs inside the user's model that produce a `TypeError` (e.g. invalid tensor reshapes that surface as `TypeError` in C++ code) get silently treated as "wrong signature."
- **Why this is a problem:** Confusing diagnostics, and the fall-through can succeed when it should not (e.g. a model that ignores `batch=` entirely will silently produce wrong-shape outputs without warning).
- **Recommended fix direction:** Inspect the model's `forward` signature once with `inspect.signature` and either pass only the kwargs it accepts, or re-raise the original TypeError immediately when the signature mismatch is *not* a known case. Add a one-time signature compatibility check at first batch.
- **Backward compatibility:** Medium — users who relied on silent fallback may now see explicit errors. Document migration.
- **Files likely affected:** `tgraphx/training.py`.
- **Tests to add:** `tests/test_training.py` — assert that a `TypeError` raised inside the model's forward is propagated rather than swallowed.
- **Priority:** Should fix soon.

### API-04 (Medium) — `_compute_metrics` swallows all metric exceptions silently

- **Location:** `tgraphx/training.py:267-281`.
- **Current behavior:** `try: result[name] = float(fn(...))` `except Exception: pass`. If a user's metric fails (e.g. shape mismatch), the metric is simply absent from results without any warning.
- **Why this is a problem:** Users debugging "why is my accuracy not appearing in the CSV" have no breadcrumb. Silent failures are harder to debug than loud ones.
- **Recommended fix direction:** Emit a single `warnings.warn(..., stacklevel=2)` per metric per epoch or per training run. Optionally an opt-in `strict_metrics=True` flag that re-raises.
- **Backward compatibility:** Safe (warnings only).
- **Files likely affected:** `tgraphx/training.py`.
- **Priority:** Nice to have.

### API-05 (Medium) — `LinearMessagePassing` shape ergonomics

- **Location:** `tgraphx/layers/base.py:148-162`.
- **Current behavior:** Uses `in_shape[0]` to pick the linear width, so spatial inputs would silently use only the channel count and produce wrong shapes. The factory rejects spatial shapes for `"linear"`, but a user constructing the class directly does not get that guard.
- **Recommended fix direction:** Add an `in_shape` rank check in `__init__` (`len(in_shape) != 1` raises) for parity with the factory.
- **Backward compatibility:** Safe — no working code depends on the broken path.
- **Priority:** Nice to have.

### API-06 (Medium) — `_graph_readout` is duplicated between `models/factory.py` and `models/regressors.py`

- **Location:** `tgraphx/models/factory.py:60-82` and `tgraphx/models/regressors.py:24-47`.
- **Current behavior:** Two byte-identical implementations of mean/sum/max scatter readout.
- **Why this is a problem:** Maintenance hazard (any fix in one place must be repeated). Also duplicates code path for tests.
- **Recommended fix direction:** Promote to a private helper in `tgraphx/layers/_scatter.py` or `tgraphx/models/_pool.py` and import from both call sites.
- **Backward compatibility:** Safe (private helper).
- **Priority:** Nice to have.

---

## 7. Syntax simplification opportunities (backward-compatible)

| ID | Current usage | Pain point | Proposed simpler usage | Backward-compatible? |
|---|---|---|---|---|
| UX-01 | `from tgraphx.training import fit; fit(model, loader, ...)` | Submodule path not obvious | `from tgraphx import fit` (re-export from `__init__`) | Yes (additive) |
| UX-02 | `model = build_model("graph_classification", "gat", in_shape=..., hidden_shape=..., num_layers=..., num_classes=...)` | Long positional + keyword mix | Add `tgraphx.classifier(layer="gat", in_shape=..., hidden=..., num_classes=...)` and `tgraphx.regressor(...)` thin wrappers around `build_model` | Yes (additive) |
| UX-03 | `from tgraphx import build_grid_graph; ei = build_grid_graph(rows, cols); g = Graph(node_features, ei)` | Two steps for a trivial case | Add `tgraphx.image_to_grid_graph(image, patch_size=...)` returning a `Graph` directly with patches as node features | Yes (additive helper) |
| UX-04 | `make_layer("gin", in_shape, out_shape)` ignores `eps`, `train_eps` | Layer features hidden | Forward extra kwargs (see API-01) | Yes |
| UX-05 | `targets.dim()==2 and targets.size(-1)==1: squeeze` (always) | Surprising for `out_dim=1` regression | Optional `squeeze_singleton_targets="auto"\|"never"` parameter | Yes (default keeps current behavior) |
| UX-06 | `tracking.CSVLogger(logdir).log(epoch=..., train_loss=...)` | Schema implicit; users don't know which keys are dashboard-recognized | Document and add `CSVLogger.log_epoch(epoch=..., train_loss=..., **rest)` typed wrapper | Yes (additive) |
| UX-07 | `from tgraphx.dashboard import launch_dashboard_background; server = launch_dashboard_background(...)` | Two-step pattern; users want training-loop wiring | Add `tgraphx.training.fit(..., dashboard=True, dashboard_port=...)` that starts the background server when explicitly requested | Yes (off by default) |

---

## 8. Performance bottleneck findings

### PERF-01 (Medium) — `validate_edge_index` triggers CUDA→CPU sync per `Graph` construction

- **Location:** `tgraphx/core/graph_utils.py:46-53`.
- **Current behavior:** `int(edge_index.min())` and `int(edge_index.max())` on a CUDA tensor force a synchronization. For data pipelines that build many `Graph` objects per second on GPU, this serializes execution.
- **Why this is a problem:** Hidden CUDA syncs in hot paths defeat torch async streams. For most users (graphs built CPU-side, moved later) this never fires; for those who keep `edge_index` on GPU, it can dominate small-batch latency.
- **How to verify:** `python -c "import torch; e=torch.randint(0,10,(2,1000),device='cuda'); ..."` and time `Graph(node_features=torch.zeros(10,4,device='cuda'), edge_index=e)` vs same on CPU.
- **Recommended fix direction:** Skip the range check when `edge_index.is_cuda` and emit a debug warning the first time, or perform the check on a preallocated 0-d tensor reused across calls. Better: replace `min/max` with `torch.where((edge_index<0) | (edge_index>=num_nodes))[0].numel()==0` and only `.item()` once if non-empty.
- **Backward compatibility:** Safe.
- **Priority:** Nice to have.

### PERF-02 (Medium) — `coalesce_edges` upcasts `edge_weight` to `float32` even when `float16`/`bfloat16` would be enough

- **Location:** `tgraphx/core/graph_utils.py:315-316`.
- **Current behavior:** Non-floating-point `edge_weight` is cast to `float32`; floating types are kept. That part is fine. But `scatter_add_` with `int` is unsupported, and the `to(float32)` cast allocates a new tensor.
- **Why this is a problem:** Minor — only affects the `coalesce`/`make_undirected` path. Memory blip is short-lived.
- **Recommended fix direction:** Use `to(weight.dtype if weight.is_floating_point() else torch.float32)` already does the right thing — could short-circuit when already floating.
- **Priority:** Nice to have.

### PERF-03 (Medium) — `build_random_graph` materializes `N*N` candidate pool

- **Location:** `tgraphx/graph_builders.py:493-514`.
- **Current behavior:** Builds `idx.repeat_interleave(N)` and `idx.repeat(N)` — both shape `[N*N]`. For `N=10000` that's `1e8` LongTensor entries (~800 MB each on CPU) before the `randperm` sample of `num_edges`.
- **Why this is a problem:** Out-of-memory on machines with <8 GB RAM for medium-N random graphs.
- **Recommended fix direction:** Sample with rejection: draw `2*num_edges` random pairs via `torch.randint(N, (2*num_edges,))`, drop duplicates and (optionally) self-loops, top up if undersampled, then optional reverse-edge append for `directed=False`. Document memory footprint as O(num_edges) not O(N²).
- **Backward compatibility:** Output edge order would differ; if `seed` is the same the old order is reproducible but the new one would not be bit-for-bit identical. Document and gate behind `algorithm="reservoir"` while keeping the old default.
- **Priority:** Should fix soon.

### PERF-04 (Medium) — `build_knn_graph` and `build_radius_graph` use `torch.cdist` (O(N²) memory)

- **Location:** `tgraphx/graph_builders.py:308`, `:364`.
- **Current behavior:** Documented as O(N²); no runtime warning.
- **Why this is a problem:** For N≈10k this still works; for N≈30k it OOMs without warning.
- **Recommended fix direction:** Add an opt-in chunked mode: split queries into chunks of `chunk_size` and compute partial distances. Add a runtime warning when `N > 5000` and no `chunk_size` is given. Keep current default for backward compatibility.
- **Backward compatibility:** Safe (warning only by default).
- **Priority:** Nice to have.

### PERF-05 (Low) — `_FactoryGNNModel` uses `int(batch.max())` per forward

- **Location:** `tgraphx/models/factory.py:65` and duplicated at `tgraphx/models/regressors.py:30`.
- **Current behavior:** `num_graphs = int(batch.max()) + 1` per forward call. CUDA→CPU sync.
- **Why this is a problem:** Each graph-classification forward pays one sync to learn `num_graphs`. Negligible for large batches; visible for tiny batches.
- **Recommended fix direction:** Allow callers to pass `num_graphs` explicitly; cache from `GraphBatch.num_graphs` when available (already exposed). The factory model could check `hasattr(batch, "num_graphs")`.
- **Backward compatibility:** Safe.
- **Priority:** Nice to have.

### PERF-06 (Hypothesis) — GAT memory at large E or large heads

- **Location:** `tgraphx/layers/gat.py:248-297`.
- **Current behavior:** `h = self.W(x).view(N, K, C_head, *spatial)` then `index_select` produces `[E, K, C_head, *spatial]`. For E=1e6, K=4, C_head=8, H=W=8, that is `1e6 * 4 * 8 * 64 * 4 bytes ≈ 8 GB`.
- **Why this is a problem:** Large image-patch graphs OOM under GAT before a chunked variant is implemented.
- **How to verify:** Add a benchmark that probes peak memory at a sweep of E.
- **Recommended fix direction:** Documented as a known limitation; future GAT chunked variant is hard because softmax is destination-global. Could implement a destination-grouped chunking (per-group softmax). Document explicitly in `docs/limitations.md`.
- **Priority:** Future major version.

---

## 9. Layer / model correctness findings

### LAYER-01 (Medium) — `AttentionMessagePassing` always uses `nn.Conv2d`, breaks on 3-D inputs

- **Location:** `tgraphx/layers/attention_message.py:19-24`.
- **Current behavior:** `if len(in_shape) > 1:` → `nn.Conv2d` regardless of whether `in_shape` is 3-D `[C,H,W]` or 4-D `[C,D,H,W]`. The factory already rejects 3-D for this layer (`tgraphx/layers/factory.py:116-121`), but a user constructing the class directly will get a confusing runtime error from inside Conv2d.
- **Why this is a problem:** Documented limitation in README, but the constructor itself doesn't enforce it. Inconsistent guard.
- **Recommended fix direction:** Add a `if len(in_shape) not in (1, 3):` raise in `__init__` matching the factory's wording.
- **Backward compatibility:** Safe (only fires for unsupported shapes that already break).
- **Priority:** Nice to have.

### LAYER-02 (Medium) — `aggregate(mean)` uses `index_add` while `scatter_mean` exists in `_scatter.py`

- **Location:** `tgraphx/layers/base.py:89-101` vs `tgraphx/layers/_scatter.py:136-150`.
- **Current behavior:** Two scatter-mean implementations: the base class hand-rolls one with `index_add`, while `scatter_mean` is the canonical one used by SAGE/GIN. They produce identical results, but the divergence is a maintenance liability.
- **Recommended fix direction:** Have `aggregate` route to the helpers in `_scatter.py` for `'sum'`, `'mean'`, and `'max'`. The factory- and shape-related logic can stay in the base class.
- **Backward compatibility:** Safe (functionally equivalent).
- **Priority:** Nice to have.

### LAYER-03 (Low) — `attn_dropout` only fires under `model.training`

- **Location:** `tgraphx/layers/gat.py:284-287`.
- **Behavior:** Standard PyTorch behavior — `F.dropout(..., training=True)` is gated. This is correct and matches user expectations, no action needed. Tests already cover this implicitly.
- **Priority:** Already correct, listed for completeness.

---

## 10. Training / logging / dashboard findings

(See BUG-04, TRACK-01, API-03, API-04 above; additional items below.)

### DASH-01 (Medium) — `_api_status` uses the same falsy `or` pattern as `TensorBoardLogger.log`

- **Location:** `tgraphx/dashboard/app.py:372`.
- **Current behavior:** `epoch = row_dict.get("epoch") or row_dict.get("step")`. If `epoch == 0` or `epoch == 0.0`, the dashboard reports "epoch: None" (or `step` if present). Consistent with TRACK-01.
- **Recommended fix direction:** Use explicit `is not None` checks. Same as TRACK-01.
- **Backward compatibility:** Safe.
- **Priority:** Should fix soon.

### DASH-02 (Low) — Dashboard `_print_banner` lies about display URL when binding `0.0.0.0`

- **Location:** `tgraphx/dashboard/__init__.py:101` and `tgraphx/dashboard/app.py:514`.
- **Current behavior:** When `host="0.0.0.0"`, the printed URL says `http://127.0.0.1:8765` — but the server is actually bound on all interfaces. Users on a LAN expecting the server to be accessible from another device will be confused.
- **Recommended fix direction:** Print `host` as-is; if the user binds `0.0.0.0`, print all detected interface IPs (best-effort via `socket.gethostbyname_ex`) or at least the LAN hostname.
- **Backward compatibility:** Safe (display only).
- **Priority:** Nice to have.

### DASH-03 (Low) — Hardware monitoring imports `pynvml` repeatedly

- **Location:** `tgraphx/dashboard/app.py:230`, `tgraphx/performance.py:138`.
- **Current behavior:** Each request to `/api/hardware` re-runs the `try: import pynvml` block. Cheap on hits, but `pynvml.nvmlInit()` on the dashboard side runs every time. The `psutil`/`pynvml` calls are cheap but the init/finalize cycle is not zero-cost.
- **Recommended fix direction:** Cache the imported module and the initialized handle on the server instance. Optionally add a 1-second cache for hardware probes.
- **Priority:** Nice to have.

### TRAIN-01 (Medium) — `fit()` does not propagate logger to `train_epoch`

- **Location:** `tgraphx/training.py:498-505`.
- **Current behavior:** `train_epoch` is called with `logger=None, log_level=0` because `fit` handles logging at the epoch level. That is intentional, but the per-batch log_level option is silently ignored in `fit`. Users who pass `log_level=2` to `fit` (expecting per-batch progress) get only per-epoch.
- **Recommended fix direction:** Forward the `log_level` argument to `train_epoch` and document that `fit(log_level=2)` produces per-batch progress.
- **Priority:** Nice to have.

### TRAIN-02 (Low) — `set_seed` does not set `torch.use_deterministic_algorithms` or `torch.backends.cudnn.deterministic`

- **Location:** `tgraphx/training.py:40-54`.
- **Current behavior:** Sets `random`, `torch`, `torch.cuda`, `numpy` seeds. Reproducibility under cuDNN can still vary.
- **Recommended fix direction:** Add an opt-in `deterministic=True` parameter that also sets `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`. Document the speed tradeoff.
- **Priority:** Nice to have.

---

## 11. Graph builders / patch helpers findings

(See PERF-03 / PERF-04 above; additional items below.)

### BLD-01 (Low) — `build_iou_graph(directed=False)` does not include self-loops in the candidate set when `self_loops=True`

- **Location:** `tgraphx/graph_builders.py:434-446`.
- **Current behavior:** `if not self_loops: conn.fill_diagonal_(False)` is correct. When `self_loops=True`, the diagonal stays True (self IoU is always 1.0). This is fine. But the documented behavior says "exactly one `i→i` per node" — which works, but is implicit, and the doc could clarify the IoU = 1.0 invariant.
- **Recommended fix direction:** Doc-only — clarify in the docstring that for `IoU(i,i)=1`, threshold ≤ 1 always yields a self-loop.
- **Priority:** Nice to have.

### BLD-02 (Low) — `image_to_patches` and `volume_to_patches` do not pad

- **Location:** `tgraphx/graph_builders.py:580-588`, `:676-682`.
- **Current behavior:** Documented and tested; raises with a clear message when dimensions don't tile exactly. This is the right design choice.
- **Recommended fix direction:** Add an `image_to_grid_graph` / `volume_to_grid_graph` convenience that returns a `Graph` whose `node_features` are the patches and `edge_index` is the matching grid graph (UX-03). Backward-compatible additive helper.
- **Priority:** Nice to have.

### BLD-03 (Low) — `_dedup` returns sorted edges; not invariant under input edge order

- **Location:** `tgraphx/graph_builders.py:64-70`.
- **Current behavior:** `torch.unique(keys, sorted=True)` enforces a canonical (src, dst) lexicographic order regardless of input. This is a feature, but tests should pin it explicitly.
- **Recommended fix direction:** Doc-only.
- **Priority:** Nice to have.

---

## 12. Docs / README / Colab findings

### DOC-01 (Critical) — `docs/limitations.md` declares `train_epoch`/`evaluate`/`fit`/`TensorBoardLogger` as "Not implemented" while they are shipped

- **Location:** `docs/limitations.md:23-26`.
- **Current behavior:** Lines 23–26 claim those four functions are "Not implemented; write your own loop." `tgraphx/training.py` defines all three; `tgraphx/tracking.py` defines `TensorBoardLogger`. `docs/training_utilities.md` documents them in detail.
- **Why this is a problem:** Users who read `limitations.md` first will think TGraphX is missing the entire training surface. The file also says "These limitations apply to TGraphX 0.1.1" while the package is 0.1.2.
- **Recommended fix direction:** Update each row to "Implemented (see [training_utilities.md](training_utilities.md))." Bump version line to `0.1.2` and consider a single source of truth for version (parse `tgraphx.__version__` in docs build).
- **Priority:** Must fix before next release.

### DOC-02 (High) — `docs/installation.md` references `0.1.1`

- **Location:** `docs/installation.md:48`.
- **Current behavior:** Code snippet says `print(tgraphx.__version__)  # e.g. "0.1.1"`. Stale comment.
- **Recommended fix direction:** Either update to `0.1.2` or drop the inline version (just say `"the installed version"`).
- **Priority:** Should fix soon.

### DOC-03 (High) — `docs/api_reference.md` omits `train_epoch`, `evaluate`, `fit`, `TensorBoardLogger`

- **Location:** `docs/api_reference.md:81-99`.
- **Current behavior:** The `tgraphx.training` table lists only utility helpers; it omits the actual loop functions. The `tgraphx.tracking` table omits `TensorBoardLogger`. Users referring to `api_reference.md` cannot find the most-used training APIs.
- **Recommended fix direction:** Add `train_epoch`, `evaluate`, `fit`, `TensorBoardLogger` rows.
- **Priority:** Must fix before next release.

### DOC-04 (Medium) — README's "What is currently implemented" table mentions `GraphClassifier`/`NodeClassifier` but those are not at top-level

- **Location:** `README.md:58-64` (table rows).
- **Current behavior:** Reader assumes `from tgraphx import GraphClassifier`. Actual path is `tgraphx.models.GraphClassifier`. Same for `NodeClassifier`, `CNN_GNN_Model`, `CNNEncoder`, `PreEncoder`.
- **Recommended fix direction:** Either re-export them at top-level (preferred — see API-02), or qualify the table entries with their import path.
- **Priority:** Should fix soon.

### DOC-05 (Medium) — `docs/quickstart.md` example uses spatial features but does not point to vector-feature path

- **Location:** `docs/quickstart.md:28-45`.
- **Current behavior:** First example uses `[N, 16, 8, 8]` features — which is the package's distinctive selling point but also the heaviest case. Beginners coming from PyG/DGL who only have `[N, D]` will not find a vector example until later.
- **Recommended fix direction:** Add a 5-line vector example *before* the spatial one and label it "If you only have vector features, start here." Keep the spatial example.
- **Priority:** Nice to have.

### DOC-06 (Low) — `CHANGELOG.md` does not have a `0.1.2` section

- **Location:** `CHANGELOG.md:8-28`.
- **Current behavior:** Latest entry is `[0.1.1] — 2026-05-05`. The package version is 0.1.2, the wheel was built today, and recent commits mention "v0.1.2" (`c49a1fd docs: add Colab tutorial link for v0.1.2`, `42ede54 Fix package authorship metadata for v0.1.1`).
- **Recommended fix direction:** Add a `[0.1.2] — 2026-05-07` section. Even a one-line "doc-only release" entry is enough.
- **Priority:** Should fix before next release.

### DOC-07 (Low) — `tracking.py` docstring says timestamps are ISO-8601 UTC but format-spec is `timespec="seconds"` only

- **Location:** `tgraphx/tracking.py:99`.
- **Current behavior:** `datetime.now(timezone.utc).isoformat(timespec="seconds")` produces `2025-01-01T12:00:00+00:00`. Doc example matches. No inconsistency, but downstream sub-second metric ordering is impossible.
- **Recommended fix direction:** Document this explicitly: rows logged in the same second can sort lexicographically equal. For high-frequency logging, users should also include `step`. Optional: add `time_resolution="seconds"|"ms"` parameter.
- **Priority:** Nice to have.

### DOC-08 (Low) — Several docs reference `tgraphx[monitoring]` and `tgraphx[tracking]` extras with no top-level mention in README

- **Location:** `README.md`, `pyproject.toml:[project.optional-dependencies]`.
- **Current behavior:** README and docs/dashboard.md mention `pip install tgraphx[monitoring]`. README does not advertise `tgraphx[tracking]`. `pyproject.toml` defines both.
- **Recommended fix direction:** Add a single "Installation" sub-section to the README that lists all optional extras: `dev`, `monitoring`, `tracking`. Keep verbatim names.
- **Priority:** Nice to have.

---

## 13. Test coverage findings

- **Strengths:** 685 tests collected, broad layer/families coverage, edge-feature variants, builder determinism, dashboard route tests, training-utility unit tests, gradient sanity, factory exhaustiveness, 3-D path coverage. CUDA/MPS-conditional skips are handled correctly.

- **Gaps:**

| ID | Missing test | Where to add | Priority |
|---|---|---|---|
| TEST-01 | Wheel/sdist contains dashboard static assets | new `tests/test_packaging.py` (skip if `build` not installed) | Must |
| TEST-02 | `LinearMessagePassing(dropout_prob>0)` actually applies dropout (train≠eval) | `tests/test_layers.py` | Must |
| TEST-03 | `TensorBoardLogger.log` step semantics for `epoch=0`, `epoch=0` after `_step>0`, mixed `epoch`/`step` | `tests/test_tracking.py` | Must |
| TEST-04 | `TensorGATLayer(add_self_loops=True)` deduplicates against pre-existing self-loops | `tests/test_layers.py` | Should |
| TEST-05 | `make_layer("gin", eps=...)` etc. forwarding | `tests/test_factories.py` | Should |
| TEST-06 | Auto-squeeze does not corrupt `[B, 1]` regression with `MSELoss` | `tests/test_training.py` | Should |
| TEST-07 | `_call_model` reraises non-signature `TypeError` | `tests/test_training.py` | Should |
| TEST-08 | Dashboard `_api_status` returns `epoch=0` when CSV has `epoch=0` | `tests/test_dashboard.py` | Should |
| TEST-09 | `build_random_graph` memory ceiling regression | `tests/test_graph_builders.py` (mem watcher) | Nice |
| TEST-10 | Vector-features Colab-style smoke through the full README first example | `tests/test_imports.py` | Nice |

---

## 14. Security / privacy / side-effect findings

- **No telemetry or analytics** detected in any module.
- **Imports at top-level `tgraphx`**: `pynvml` shows up only because PyTorch itself probes for it; not a TGraphX side effect (verified with `python -X importtime`). `tensorboard`, `psutil`, `mlflow` are *not* imported at base.
- **YAML safe-load**: `tgraphx/core/utils.py:23` and `tgraphx/models/factory.py:345` both use `yaml.safe_load`. ✅
- **No `eval`/`exec`** in source (verified via grep).
- **Dashboard token**: enforced for non-loopback hosts (`tgraphx/dashboard/app.py:465-470`). Path traversal is blocked by `_safe_path` (`:98-104`). Static file whitelist (`:329`) restricts to two files.
- **No file writes at import.** `CSVLogger` defers `os.makedirs` and the file open to the first `log()` call (`tgraphx/tracking.py:102-106`). ✅
- **`save_checkpoint` creates the parent dir** via `os.makedirs(..., exist_ok=True)`. The path is fully user-supplied; no traversal risk because nothing is read from the filesystem implicitly.
- **`load_checkpoint` uses `weights_only=False`**: `tgraphx/training.py:128`. This deserializes arbitrary pickle. If a user loads an untrusted checkpoint, code execution is possible.
  - **Recommended fix direction (SEC-01, Should):** Default to `weights_only=True` (PyTorch ≥ 2.4 supports a safe class allow-list), and expose `weights_only=False` as an opt-in for legacy checkpoints. Document the migration.
  - **Priority:** Should fix before next release for users on PyTorch ≥ 2.6 (which now warns by default).

- **Dashboard**: `CSVLogger.path` returns absolute path. No remote endpoints, no external CDN, no fonts. ✅

---

## 15. Release / CI findings

- **CI matrix:** Ubuntu, Python 3.10/3.11/3.12. Missing 3.9 and 3.13 (advertised in classifiers as supported).
  - **Recommended fix direction:** Add 3.9 and 3.13 entries to the matrix.
- **OS matrix:** Ubuntu only. macOS and Windows are not exercised; the package claims OS-independent.
  - **Recommended fix direction:** Add `macos-latest` and `windows-latest` rows; allow `fail-fast: false` to keep results visible.
- **Wheel install path is not exercised in CI.** The smoke check installs `pip install -e .[dev]` and tests against the source tree, masking BUG-01.
  - **Recommended fix direction:** Add a CI step that builds the wheel, installs it in a fresh venv, and runs `tgraphx-dashboard --logdir runs/empty` against a server health-check URL.
- **Trusted publishing workflow** (`.github/workflows/...`) uses a static API token (`TWINE_USERNAME=__token__`, `TWINE_PASSWORD=${{ secrets.PYPI_API_TOKEN }}`). Functional; PyPI's modern recommendation is OIDC trusted publishing (no static token).
  - **Recommended fix direction:** Migrate to OIDC trusted publishing once 0.1.3 is out.
- **No `wheel` smoke test** for static-asset presence.
- **No `tgraphx-dashboard --help`** smoke run.
- **`benchmark_layers.py` is exercised** in CI smoke. ✅
- **README/docs claims**: classifiers include 3.13, but CI does not run on 3.13. Consider tightening claims or expanding CI.

---

## 16. Backward compatibility risk table

| Change | Affects PyPI users | Affects README examples | Affects docs examples | Affects Colab tutorial | Affects existing tests | Backwards-compat |
|---|---|---|---|---|---|---|
| Add static files to wheel (BUG-01) | No (only adds files) | No | No | No | No | Safe |
| Fix TensorBoardLogger step (TRACK-01) | Yes (correct steps now) | No | No | No | New tests required | Safe (bug fix) |
| Make `LinearMessagePassing` flags effective (BUG-02) | Yes (output values change for users using these flags) | No | No | No | New tests required | Medium-risk (semantic correction) |
| Forward additional kwargs in `make_layer` (API-01) | No | No | No | No | New tests | Safe |
| Top-level re-export of training/tracking (API-02) | No | Yes (could simplify) | Yes (same) | Yes (same) | Add new import tests | Safe |
| Auto-squeeze guard (BUG-04) | No (default unchanged) | No | No | No | New tests | Safe |
| `weights_only=True` default (SEC-01) | Yes (existing checkpoints with custom classes may need allow-list) | No | No | No | Add test | Medium-risk |
| Update `docs/limitations.md` (DOC-01) | No (docs only) | No | Yes | No | No | Safe |
| Add `image_to_grid_graph` helper (UX-03) | No (additive) | No | No | No | New tests | Safe |
| Bump CI matrix to 3.9/3.13 + macOS/Windows | No | No | No | No | None | Safe |

---

## 17. Confirmed issues — Table 1

| ID | Severity | Category | Location | Current behavior | User impact | Recommended fix | Tests needed | Priority |
|---|---|---|---|---|---|---|---|---|
| BUG-01 | Critical | packaging | `pyproject.toml`; `tgraphx/dashboard/static/*` | Static CSS/JS missing from wheel and sdist | Dashboard UI 404s after `pip install` | Add `[tool.setuptools.package-data]` for `tgraphx.dashboard` | TEST-01 wheel asset check | Must |
| TRACK-01 | High | tracking | `tgraphx/tracking.py:224` | `or` chain treats `epoch=0`/`step=0` as falsy | TB step axis silently corrupted | Replace `or` chain with explicit `is None` checks | TEST-03 | Must |
| BUG-02 | High | correctness | `tgraphx/layers/base.py:163` | `LinearMessagePassing.update` ignores dropout/residual/bn | Silent feature loss; `train`==`eval` | Remove override or apply base-class behavior | TEST-02 | Must |
| DOC-01 | Critical | documentation | `docs/limitations.md:20-28,60` | Falsely declares `fit`/`evaluate`/`train_epoch`/`TensorBoardLogger` as not implemented | Users skip the package | Update table entries; bump version line | n/a | Must |
| DOC-03 | High | documentation | `docs/api_reference.md:81-99` | Omits `train_epoch`/`evaluate`/`fit`/`TensorBoardLogger` | Users can't find APIs | Add rows | n/a | Must |
| BUG-03 | Medium | correctness | `tgraphx/layers/gat.py:241` | `add_self_loops=True` doesn't dedup existing self-loops | Subtle attention bias | Mirror `core/graph_utils.add_self_loops` dedup | TEST-04 | Should |
| BUG-04 | Low | correctness | `tgraphx/training.py:230` | Auto-squeeze breaks `out_dim=1` regression | Wrong loss for `[B,1]` MSE targets | Squeeze only when target dtype is long | TEST-06 | Nice |
| API-01 | High | API/UX | `tgraphx/layers/factory.py:165` | `make_layer` drops several kwargs | Layer features unreachable via factory | Whitelist kwargs per layer | TEST-05 | Should |
| API-02 | High | API/UX | `tgraphx/__init__.py` | Training/tracking helpers not at top level | Long import paths | Re-export | TEST-10 | Should |
| API-03 | Medium | API/UX | `tgraphx/training.py:252` | Bare `except TypeError` masks user bugs | Confusing diagnostics | Use `inspect.signature` or re-raise | TEST-07 | Should |
| API-04 | Medium | API/UX | `tgraphx/training.py:267` | Metric exceptions silently swallowed | Silent metric loss | Emit warning | n/a | Nice |
| API-05 | Medium | API/UX | `tgraphx/layers/base.py:148` | No rank check in `LinearMessagePassing.__init__` | Confusing failure mode | Add rank-1 guard | n/a | Nice |
| API-06 | Medium | code-quality | `tgraphx/models/factory.py:60`, `tgraphx/models/regressors.py:24` | Duplicated `_graph_readout` | Maintenance hazard | Move to shared helper | n/a | Nice |
| LAYER-01 | Medium | correctness | `tgraphx/layers/attention_message.py:19` | No 3-D guard in constructor | Confusing error | Add explicit guard | n/a | Nice |
| LAYER-02 | Medium | code-quality | `tgraphx/layers/base.py:89` | Two scatter implementations | Maintenance hazard | Use `_scatter.py` helpers | n/a | Nice |
| DASH-01 | Medium | dashboard | `tgraphx/dashboard/app.py:372` | Falsy `or` chain (same pattern as TRACK-01) | Dashboard "epoch: None" for epoch=0 | Use explicit `is not None` | TEST-08 | Should |
| DASH-02 | Low | dashboard | `tgraphx/dashboard/__init__.py:101` | Banner prints `127.0.0.1` for `0.0.0.0` | Confusing for LAN users | Print real interface IPs | n/a | Nice |
| DASH-03 | Low | dashboard | `tgraphx/dashboard/app.py:230` | `pynvml.nvmlInit` per request | Small overhead | Cache initialization | n/a | Nice |
| TRAIN-01 | Medium | training | `tgraphx/training.py:498` | `fit` doesn't propagate `log_level` | Unexpected silence | Forward `log_level` | n/a | Nice |
| TRAIN-02 | Low | training | `tgraphx/training.py:40` | No determinism flag | Slightly worse reproducibility | Add `deterministic=True` opt-in | n/a | Nice |
| SEC-01 | Medium | security | `tgraphx/training.py:128` | `weights_only=False` | Untrusted-checkpoint code execution | Default to `True`; opt-in legacy | Add test | Should |
| PERF-01 | Medium | performance | `tgraphx/core/graph_utils.py:46` | CUDA sync on `Graph` validation | Hidden sync in hot path | Skip on CUDA or use mask | n/a | Nice |
| PERF-03 | Medium | performance | `tgraphx/graph_builders.py:493` | `N*N` candidate pool in `build_random_graph` | OOM at moderate N | Rejection sampling | TEST-09 | Should |
| PERF-04 | Medium | performance | `tgraphx/graph_builders.py:308,364` | `cdist` O(N²) without warning | OOM at large N | Optional chunked mode | n/a | Nice |
| PERF-05 | Low | performance | `tgraphx/models/factory.py:65` | `int(batch.max())` per forward | Tiny CUDA sync | Use `GraphBatch.num_graphs` if present | n/a | Nice |
| DOC-02 | High | documentation | `docs/installation.md:48` | Stale `0.1.1` version comment | Docs drift | Update or drop | n/a | Should |
| DOC-04 | Medium | documentation | `README.md:58` | `GraphClassifier` etc. claimed top-level | Failed `from tgraphx import` | Re-export or qualify | n/a | Should |
| DOC-05 | Medium | documentation | `docs/quickstart.md:28` | Spatial-first example | Beginners blocked | Add vector-first sub-section | n/a | Nice |
| DOC-06 | Low | release/CI | `CHANGELOG.md:8` | No 0.1.2 section | Release notes drift | Add entry | n/a | Should |
| CI-01 | Medium | release/CI | `.github/workflows/tests.yml` | Missing 3.9/3.13 and macOS/Windows | Claimed support unverified | Expand matrix | n/a | Should |
| CI-02 | Medium | release/CI | `.github/workflows/tests.yml` | Wheel install not exercised | Hides BUG-01 | Add wheel-install smoke job | TEST-01 | Should |
| BLD-01 | Low | docs | `tgraphx/graph_builders.py:434` | IoU self-loop semantics implicit | Minor confusion | Doc-only | n/a | Nice |

---

## 18. Performance bottlenecks — Table 2

| ID | Location | Complexity / bottleneck | Evidence | Affected workload | Suggested optimization | Risk | Benchmark to add |
|---|---|---|---|---|---|---|---|
| PERF-01 | `core/graph_utils.py:46-53` | CUDA sync per Graph validation | `int(edge_index.min/max())` always syncs | Pipelines that build many graphs on CUDA | Mask-based check; skip syncs | Low | `bench_graph_construction.py` with CUDA tensors |
| PERF-02 | `core/graph_utils.py:315` | `to(float32)` for non-float `edge_weight` | Source read | `coalesce_edges` / `make_undirected` | Short-circuit if already float | Low | included in `benchmark_graph_builders.py` |
| PERF-03 | `graph_builders.py:493` | O(N²) memory candidate pool | Source: `idx.repeat_interleave(N)` | Large random graph generation | Rejection sampling | Medium (output order changes) | `bench_random_graph.py` at N=10k, 50k |
| PERF-04 | `graph_builders.py:308,364` | O(N²) `torch.cdist` | Source/doc | k-NN / radius graph at N>5k | Chunked distances | Low (gated) | `bench_knn.py` |
| PERF-05 | `models/factory.py:65` | `int(batch.max())` per forward | Source | Graph-classification training step | Use `GraphBatch.num_graphs` | Low | per-epoch profile |
| PERF-06 | `layers/gat.py:248-297` | E·K·C·spatial memory under GAT | Source | Image-patch graphs with K>4 | Destination-grouped chunked GAT (future) | High | `bench_gat_memory.py` |

---

## 19. API/UX simplification opportunities — Table 3

| ID | Current syntax | Pain point | Proposed easier syntax | Backward-compatible? | Files affected | Tests/docs needed |
|---|---|---|---|---|---|---|
| UX-01 | `from tgraphx.training import fit` | Path discoverability | `from tgraphx import fit` | Yes | `tgraphx/__init__.py`, docs | new import test |
| UX-02 | `build_model("graph_classification", "gat", ...)` | Long string argument list | `tgraphx.classifier(layer="gat", in_shape=..., hidden=..., num_classes=...)` thin wrapper | Yes | `tgraphx/__init__.py`, new doc | factory tests |
| UX-03 | `image_to_patches(...) + build_grid_graph(...) + Graph(...)` | Three-step composition | `image_to_grid_graph(image, patch_size=..., self_loops=True)` returns `Graph` | Yes | `tgraphx/graph_builders.py` | new tests |
| UX-04 | `make_layer("gin", ...)` ignores `eps`, `train_eps`, `hidden_channels`, `use_batchnorm` | Layer features hidden | Forward extra kwargs | Yes | `tgraphx/layers/factory.py` | factory tests |
| UX-05 | Auto-squeeze of `[B,1]` targets | Silent surprise for regression | `squeeze_singleton_targets="auto"\|"never"` | Yes | `tgraphx/training.py` | new test |
| UX-06 | `CSVLogger.log(epoch=..., train_loss=...)` | Schema implicit | Add typed `log_epoch(...)` and document required keys | Yes | `tgraphx/tracking.py` | doc updates |
| UX-07 | `launch_dashboard_background(...)` + manual wiring | Two-step | `fit(..., dashboard=True)` opt-in | Yes | `tgraphx/training.py`, `tgraphx/dashboard/__init__.py` | new test |
| UX-08 | `tgraphx.GraphClassifier` (fails) | Top-level expectations from README | Re-export from `tgraphx` | Yes | `tgraphx/__init__.py` | import test |
| UX-09 | Custom error messages on shape mismatch | Already good but inconsistent across layers | Centralize "shape pretty-printer" helper | Yes | `tgraphx/layers/_dim.py` | n/a |
| UX-10 | Vector example is page-2 in quickstart | Beginners discouraged | Vector example first | Yes (doc) | `docs/quickstart.md` | n/a |

---

## 20. Documentation / Colab issues — Table 4

| ID | File / section / cell | Current wording or behavior | Problem | Exact correction direction | Priority |
|---|---|---|---|---|---|
| DOC-01 | `docs/limitations.md:20-28,60` | "train_epoch / evaluate / fit / TensorBoardLogger — Not implemented." Says limitations apply to 0.1.1 | False; everything is implemented | Replace rows with "Implemented (link to training_utilities.md)"; bump version | Must |
| DOC-02 | `docs/installation.md:48` | `print(tgraphx.__version__)  # e.g. "0.1.1"` | Stale | Update to `0.1.2` or drop | Should |
| DOC-03 | `docs/api_reference.md:81-99` | Omits `train_epoch`, `evaluate`, `fit`, `TensorBoardLogger` | Incomplete | Add rows | Must |
| DOC-04 | `README.md:58` | Lists `GraphClassifier` etc. as if top-level | Misleading | Re-export or qualify path | Should |
| DOC-05 | `docs/quickstart.md:28` | First example is spatial | Vector users blocked | Add vector example before spatial | Nice |
| DOC-06 | `CHANGELOG.md:8` | No `[0.1.2]` section | Release notes drift | Add entry summarizing the 0.1.2 docs/Colab changes | Should |
| DOC-07 | `tgraphx/tracking.py:99` (docstring) | Timestamps `timespec="seconds"` | Sub-second ordering impossible | Document explicitly | Nice |
| DOC-08 | README installation section | Does not advertise `tgraphx[tracking]` extras | Discoverability | Add a "Installation extras" sub-section | Nice |
| COLAB-01 | Colab tutorial (link in README) | Latest cell installs from PyPI | If BUG-01 is unfixed, dashboard cells will silently 404 in Colab | Must wait for BUG-01 fix; verify the tutorial after | Must |

---

## 21. Hypotheses requiring verification — Table 5

| ID | Hypothesis | Why it matters | How to verify | Expected fix if confirmed |
|---|---|---|---|---|
| H-01 | `TensorGATLayer` softmax may underflow under float16 autocast (`exp(score)` of large negative) | AMP correctness | Force `amp=True` and a deeply negative score; check finite output | Use `float32` accumulator inside `edge_softmax` |
| H-02 | `ConvMessagePassing._chunked_forward` divides by counts of zero for isolated nodes when `aggr='mean'` | Subtle if `clamp(min=1)` not applied; current code does clamp — confirm | Read `:175` — `clamp(min=1)` confirmed; hypothesis rejected | n/a |
| H-03 | `accuracy(logits, labels)` always triggers a `.item()` sync; per-batch metrics under CUDA may slow training measurably | Real if `metrics={"acc": accuracy}` is computed per batch | Bench training step on CUDA with and without metric | Lazy/no-sync metric computation; defer to epoch end |
| H-04 | `Graph.add_self_loops` returns a different `edge_index` order than `TensorGATLayer(add_self_loops=True)` (BUG-03) — hash-based attention shape change | Reproducibility of saved checkpoints across these two paths | Construct equivalent graph two ways; compare `edge_index` order | Document; or unify dedup |
| H-05 | `build_iou_graph` `iou` divisor `union.clamp(min=1e-8)` may overflow with `int` boxes | Edge case | Pass int box tensor; check for NaN | Always cast to float (already done) — hypothesis rejected via reading `:419` |
| H-06 | TB writer's `add_scalar` accepts `global_step` as `int` only; passing `float` may break | `int(step)` is already applied in TRACK-01 fix; safe | n/a | n/a |
| H-07 | `CSVLogger` re-opens the file with `"a"` mode but `os.path.getsize == 0` check at `:113` may race in multi-process training | Multi-worker DataLoader writing same file | Run two `CSVLogger` instances on the same file path | Document single-writer constraint, or add `fcntl` lock on POSIX |
| H-08 | Dashboard `_collect_hardware` may block for `psutil.cpu_percent(interval=0.1)` per request | Latency spikes for the dashboard at 10+ Hz polling | Stress-test dashboard with `wrk -t 4 -c 16` | Make `interval=0.0` and cache last reading |
| H-09 | `torch.compile` interaction with `index_add_` under bf16 autocast | Compile correctness | Compile a GAT layer; run forward+backward; compare to eager | Document caveat in `docs/limitations.md` |
| H-10 | `EdgePredictor.forward` does global average pool over `[N, C, *spatial]` even when caller already pre-pooled | Minor extra op | Pre-pool then call; check shape | Skip pool when `x.dim()==2`; already done at `_pool` |

---

## 22. Competitive / ecosystem audit (Section 25 of the prompt)

### 22.1 Frank positioning

TGraphX's distinctive value is unambiguous: **PyTorch-native, tensor-aware message passing on `[N, C, H, W]` and `[N, C, D, H, W]` node features, with a clean factory and lightweight training/dashboard/tracking surface.** It is **not** trying to be PyG/DGL, and it should not be benchmarked against them on raw vector-feature speed; it should be benchmarked on tasks where flattening would discard structure (image patches, voxel grids, ROI crops). For ordinary vector workflows, TGraphX is *adequate but not differentiated*: PyG/DGL have richer layer libraries, sampler ecosystems, and more documentation. TGraphX should make ordinary workflows pleasant enough that vector users do not feel the package is a worse PyG, while leaning on tensor-awareness as the reason to stay.

### 22.2 What TGraphX should learn from neighbors (and what to avoid)

| Library | Learn from | Avoid copying |
|---|---|---|
| **PyG** | Message-passing convention, broad layer zoo, `Data`/`Batch` patterns, mature samplers | Heavy install footprint, `torch_scatter`/`torch_sparse` extension dependence, vector-centric assumptions |
| **DGL** | Heterogeneous graph abstractions (future), efficient kernels | Multiple-backend complexity, separate runtime |
| **Spektral / TF-GNN** | Task-level convenience APIs and educational docs | TF lock-in |
| **NetworkX** | Beginner-friendly graph construction; algorithm utilities | CPU-only / non-tensor pipelines |
| **cuGraph** | Pure GPU graph algorithms | RAPIDS dependency footprint; CUDA-only |
| **DeepChem** | Domain-specific examples (molecules) | Domain lock-in |
| **TopoNetX** | Higher-order structure inspiration | Heavy abstractions for everyday users |
| **Neo4j GDS / GraphRAG** | Visualization, explainability, pipelines | Database/platform dependence |
| **PyReason / PyNeuraLogic / Stardog** | Explainability future | Symbolic reasoning is out of TGraphX's scope |
| **GraphNeuralNetworks.jl** | Clean abstraction style | Non-Python ecosystem |

### 22.3 Hardware / platform excellence audit (Section 25.3)

| Platform | TGraphX status | Findings |
|---|---|---|
| CPU-only laptop | Works (CI ubuntu) | ✅ Tests + examples pass on CPU |
| NVIDIA CUDA | Works (verified locally) | ✅ Layers and training run on CUDA |
| Apple Silicon MPS | Code paths exist; not tested in CI | ⚠ Hypothesis: AMP/`scatter_reduce_` differences may surface; needs CI runner |
| Colab CPU | Works (Colab tutorial) | ✅ |
| Colab GPU | Should work | Hypothesis: untested in CI, dashboard packaging bug breaks dashboard cell |
| Linux | ✅ |
| Windows / macOS | Untested in CI | ⚠ Should add to matrix |
| WSL2 | Likely works (Linux); untested |
| Phones/tablets/TV via dashboard | Dashboard is responsive; static-asset bug breaks PyPI users |

### 22.4 Best-of-ecosystem feature audit summary

- **PyG-style `Data` / `Batch`**: TGraphX's `Graph` / `GraphBatch` are simpler and tensor-shape-aware; they need no extension wheels. ✅
- **Layer zoo**: GAT/SAGE/GIN/Conv covers the staples. Missing: GCN (a literal Kipf-Welling layer), GraphConv with edge weight, and a typed-edge GINE. Adding GCN would be a 50-line file and would close a perception gap.
- **Sampler / dataset pipeline**: Not present. Out of scope unless the user base asks. The lightweight `GraphDataset`/`GraphDataLoader` is fine for synthetic and small datasets.
- **Visualization**: Dashboard fills part of this but is read-only. NetworkX-style `to_networkx()` for inspection (no dependency leak; lazy import) would help.
- **Examples / tutorials**: 28 examples is excellent. The README and Colab notebook are the discoverability magnet.

### 22.5 Ordinary graph workflows audit

| Task | TGraphX support today | Friction |
|---|---|---|
| Vector node classification | `LinearMessagePassing` + `build_model("node_classification","linear",...)` | LinearMessagePassing's BUG-02 — its quality flags are silent. Fix that and the workflow is solid. |
| Vector graph classification | `build_model("graph_classification","linear",...)` | Same caveat |
| Vector graph regression | `GraphRegressor` or `build_model("graph_regression","linear",...)` | OK |
| Vector edge prediction | `build_model("edge_prediction","linear",...)` | OK |
| Edge weights | First-class on every layer | OK |
| Edge features (vector) | Supported on SAGE/GIN/GAT (`edge_features_kind="vector"`) | OK; well-documented |
| Directed/undirected | `Graph.make_undirected()`, builder flags | OK |
| Self-loops | `Graph.add_self_loops()` — correct dedup; `TensorGATLayer(add_self_loops=True)` — does NOT dedup (BUG-03) | Inconsistent |
| Batching | `GraphBatch` with strict per-edge field policing | OK |
| Training | `fit`/`train_epoch`/`evaluate` | TRACK-01, BUG-04, API-03/04 |
| Logging | CSV + TB (with TRACK-01 fix) | OK after fix |
| Dashboard | After BUG-01 fix | OK |
| Comparison with raw PyTorch | TGraphX cuts boilerplate for graph wrangling but doesn't add helpers for "torchify your existing loop" patterns | Add `tgraphx.classifier(...)` UX-02 |

### 22.6 "Better than raw PyTorch" table (Section 25.7)

| Task | Raw PyTorch pain | Current TGraphX syntax | Remaining friction | Proposed easier API | BC? | Priority |
|---|---|---|---|---|---|---|
| Build a 3×3 grid + Conv-MP forward | 8 lines (manual edge_index, gather, scatter) | 4 lines (`build_grid_graph` + `Graph` + `ConvMessagePassing`) | Multi-step | `image_to_grid_graph(image, patch_size=...)` returns `Graph` ready to consume (UX-03) | Yes | Should |
| Train a graph classifier 5 epochs | ~30 lines (manual loop, batching) | 8 lines (`build_model` + `fit`) | Two imports needed | `from tgraphx import classifier, fit` (UX-01, UX-02) | Yes | Should |
| Log to CSV | Manual `csv` writer | `CSVLogger.log(...)` | Schema unclear | Type schema in docs (UX-06) | Yes | Nice |
| Inspect node embeddings | Manual extraction | Per-layer hooks | None added | Add `model.encode(...)` returning per-layer dict (future) | Yes | Future |
| Save+resume | Manual checkpoint dict | `save_checkpoint`/`load_checkpoint` | `weights_only=False` | Default `weights_only=True` (SEC-01) | Medium | Should |
| Multi-rank features | Lots of `.view`/`.permute` | Native — preserved | Limited rank set (1/3/4) | Already documented limitation | n/a | n/a |

---

## 23. "No reason to leave" user-journey audit

| Level | Current strengths | Current frictions | Recommended improvements |
|---|---|---|---|
| Beginner | Colab tutorial, quickstart, validated `Graph`, factory | DOC-01 lies; vector example buried; TB step bug; dashboard 404 | Fix DOC-01, BUG-01, TRACK-01; add UX-03 helper |
| Intermediate | Layer zoo, edge_features, edge_weight, factory, training utilities | API-01 (factory missing kwargs); BUG-02 silent flags; API-03 silent type fallback | Fix API-01 / BUG-02 / API-03 |
| Advanced | Custom `TensorMessagePassingLayer`, `make_layer`, dashboard tokens, `env_report`, `estimate_message_memory` | No `torch.compile` profile, no GAT chunking, no MPS CI, no WIN/macOS CI | Add CI matrix, document `torch.compile` caveats |

---

## 24. Criticism-defense table (Section 25.8)

| Possible criticism | Fair? | Current defense | Weakness remaining | Exact fix direction | Priority |
|---|---|---|---|---|---|
| "Dashboard doesn't work on PyPI" | Fair (BUG-01) | None | Static assets missing | Fix `pyproject.toml` package-data | Must |
| "TB charts skip epoch 0" | Fair (TRACK-01) | None | `or` chain | Replace with explicit checks | Must |
| "Docs lie about features" | Fair (DOC-01) | None | Stale page | Update `limitations.md` and `api_reference.md` | Must |
| "Linear flags do nothing" | Fair (BUG-02) | None | Silent override | Restore base-class behavior | Must |
| "make_layer feels half-baked" | Fair (API-01) | None | Kwargs dropped | Whitelist per layer | Should |
| "Imports are verbose" | Partly | Submodule paths are documented | Top-level missing some helpers | Re-export at top | Should |
| "Slow at large N" | Partly | Documented | No runtime warning | Add warnings + chunked modes | Nice |
| "No GCN layer" | Fair (perception) | LinearMessagePassing approximates | Beginners expect GCN | Add `TensorGCNLayer` thin wrapper | Future |
| "No PyG / DGL bridge" | Fair (by design) | Documented limitation | None | Future bridge module (utility-only) | Future |
| "No real datasets" | Fair (synthetic only) | Documented intentional | None | Optional `tgraphx-data` extras for tiny datasets | Future |
| "Heavy dashboard not needed" | Partly | Off by default | Some users still want zero deps | Already optional | n/a |
| "AMP support is sketchy" | Fair (limitations.md flag) | Documented | Real | Add `bfloat16` test matrix | Should |
| "Windows untested" | Fair (CI gap) | None | Real | Expand CI | Should |
| "weights_only=False default" | Fair (SEC-01) | None | Real | Default `True` | Should |
| "No graph transformers" | Fair (out of scope) | Documented | None | Future research only | Future |
| "No heterogeneous/temporal graphs" | Fair (out of scope) | Documented | None | Future research only | Future |

---

## 25. Prioritized roadmap (Batches A–H)

### Batch A — Critical bugs / correctness (release blocker)
- **Goal:** Restore the dashboard for PyPI users; fix silent metric/logging corruption and silently-no-op layer flags.
- **Issues:** BUG-01, TRACK-01, BUG-02, DOC-01, DOC-03.
- **Files:** `pyproject.toml`, `tgraphx/tracking.py`, `tgraphx/layers/base.py`, `docs/limitations.md`, `docs/api_reference.md`.
- **Implementation strategy:** (1) Add `[tool.setuptools.package-data]` clause; (2) replace `or`-chain with `is None` checks in `TensorBoardLogger.log`; (3) remove or fix the `LinearMessagePassing.update` override; (4) update both docs files to reflect shipped features.
- **Compatibility rules:** Public API surface unchanged. Keep all existing imports working. README examples and Colab tutorial must continue to run unchanged.
- **Tests to add:** TEST-01 (wheel asset check), TEST-02 (LinearMessagePassing flags effective), TEST-03 (TB step semantics), and a doc smoke test that the table rows in `limitations.md` match `tgraphx/training.py` symbols.
- **Verification commands:**

      python -m build && unzip -l dist/tgraphx-0.1.2-py3-none-any.whl | grep static
      pytest -q tests/test_packaging.py tests/test_layers.py::test_linear_flags tests/test_tracking.py
      python -m pytest -q
- **Risk:** Low.
- **Expected result:** Dashboard renders after `pip install`; TB charts show correct epochs; `LinearMessagePassing` flags work; docs match reality.

### Batch B — API usability and syntax simplification
- **Goal:** Reduce import friction and unify factory ergonomics.
- **Issues:** API-01, API-02, API-03, API-04, UX-01, UX-02, UX-03, UX-08.
- **Files:** `tgraphx/__init__.py`, `tgraphx/layers/factory.py`, `tgraphx/training.py`, `tgraphx/graph_builders.py`, docs.
- **Strategy:** (1) Forward additional kwargs in `make_layer` per layer; (2) re-export training/tracking/regressors and classifiers at top level; (3) replace bare `except TypeError` with signature inspection; (4) emit a warning on metric exceptions; (5) add `image_to_grid_graph` helper.
- **Compatibility rules:** All additions must be additive; existing imports must keep working; warnings count as new behavior — gate behind `warnings.warn`.
- **Tests to add:** TEST-05, TEST-07, TEST-10.
- **Risk:** Low–Medium (API-03 changes diagnostics).
- **Expected result:** A typical user can do `from tgraphx import fit, CSVLogger, GraphClassifier, image_to_grid_graph` and the factories accept all relevant knobs.

### Batch C — Documentation and Colab clarity
- **Goal:** Make every doc page reflect 0.1.2 reality and improve onboarding.
- **Issues:** DOC-01..08, COLAB-01.
- **Files:** `docs/*.md`, `README.md`, `CHANGELOG.md`, Colab notebook.
- **Strategy:** Single doc PR that:
  1. Fixes `limitations.md`, `installation.md`, `api_reference.md`.
  2. Adds `[0.1.2]` CHANGELOG entry.
  3. Adds a 5-line vector quickstart snippet before the spatial one.
  4. Adds `tgraphx[tracking]` advertising in README install section.
  5. Re-runs the Colab tutorial end-to-end after BUG-01 is fixed.
- **Compatibility rules:** No code changes; pure docs.
- **Risk:** Trivial.
- **Expected result:** New users can read any doc page in any order without contradiction.

### Batch D — Performance bottlenecks
- **Goal:** Eliminate hidden CUDA syncs and OOM risks at moderate scale.
- **Issues:** PERF-01, PERF-03, PERF-04, PERF-05.
- **Files:** `tgraphx/core/graph_utils.py`, `tgraphx/graph_builders.py`, `tgraphx/models/factory.py`.
- **Strategy:** (1) Mask-based range check on CUDA validation; (2) rejection-sampling random-graph builder; (3) optional `chunk_size` for `cdist`-backed builders with default warning at N>5k; (4) honor `GraphBatch.num_graphs` if set, falling back to `int(batch.max())`.
- **Tests to add:** TEST-09 plus a benchmark-only addition under `benchmarks/`.
- **Risk:** Medium for PERF-03 (output order may change; gate behind `algorithm=` parameter and keep old default).
- **Expected result:** Random-graph builder works at N=50k; dashboard stops syncing per request unintentionally.

### Batch E — Dashboard / logging / tracking polish
- **Goal:** Round out the monitoring story.
- **Issues:** DASH-01, DASH-02, DASH-03, TRAIN-01, TRAIN-02, SEC-01.
- **Files:** `tgraphx/dashboard/app.py`, `tgraphx/dashboard/__init__.py`, `tgraphx/training.py`.
- **Strategy:** (1) Use `is not None` checks consistently; (2) print actual interface IPs; (3) cache `pynvml` init; (4) forward `log_level` from `fit` to `train_epoch`; (5) `set_seed(deterministic=True)` opt-in; (6) `weights_only=True` default for `load_checkpoint`.
- **Tests to add:** TEST-08 plus `tests/test_dashboard.py` extension.
- **Risk:** Low; SEC-01 is medium (pickle compat).
- **Expected result:** Dashboard reports correct epochs from epoch 0; checkpoint loading is safer by default.

### Batch F — Test coverage hardening
- **Goal:** Add the 10 high-value tests above and shore up CI.
- **Issues:** TEST-01..10, CI-01, CI-02.
- **Files:** new `tests/test_packaging.py`; extend others; `.github/workflows/tests.yml`.
- **Strategy:** Implement each TEST-N row from §13 and CI-01/CI-02. Add Windows + macOS to the GitHub matrix. Add a wheel-install job.
- **Risk:** Low.
- **Expected result:** Stronger regression net; BUG-01 (and similar) cannot recur.

### Batch G — Release/CI automation
- **Goal:** Make `0.1.3` boring.
- **Issues:** CI-01, CI-02, DOC-06.
- **Files:** workflows, `CHANGELOG.md`, `pyproject.toml` (PyPI OIDC trusted publishing).
- **Strategy:** Migrate to OIDC; require successful wheel-smoke and full pytest before PyPI upload; auto-bump version source-of-truth.
- **Risk:** Low.
- **Expected result:** Release pipeline detects packaging regressions before publishing.

### Batch H — Future / research only (no work yet)
- Heterogeneous graphs, temporal graphs, graph transformers, learned adjacency, PyG/DGL bridge, GraphRAG integrations, symbolic reasoning. **Documented limitations stay documented.** Add only when there is concrete user demand.

---

## 26. Final competitive roadmap (Phases 1–6)

### Phase 1 — Make ordinary graph workflows effortless
- Issues: API-02, BUG-02, UX-01, UX-02, UX-03, DOC-01, DOC-04.
- Files: `tgraphx/__init__.py`, `tgraphx/layers/base.py`, `tgraphx/graph_builders.py`, docs.
- Tests: TEST-02, TEST-10.
- Compatibility: additive only.
- Expected benefit: a new vector user can train a node classifier in 6 lines without leaving the top-level namespace.
- Do NOT in this phase: rewrite layers, add GCN, change defaults.

### Phase 2 — Make tensor-aware workflows best-in-class
- Issues: BUG-03, API-01, BLD-02 (helper), DOC-05.
- Files: `tgraphx/layers/factory.py`, `tgraphx/layers/gat.py`, `tgraphx/graph_builders.py`, docs.
- Tests: TEST-04, TEST-05.
- Expected benefit: GAT semantics consistent with `Graph.add_self_loops`; full GIN/SAGE/GAT customization through the factory.
- Do NOT: add per-pixel/per-voxel attention.

### Phase 3 — Make training/logging/dashboard polished
- Issues: TRACK-01, BUG-04, API-03, API-04, DASH-01..03, TRAIN-01, TRAIN-02, SEC-01, BUG-01, DOC-01, DOC-03.
- Files: `tgraphx/tracking.py`, `tgraphx/training.py`, `tgraphx/dashboard/*`, `pyproject.toml`, docs.
- Tests: TEST-01, TEST-03, TEST-06, TEST-07, TEST-08.
- Expected benefit: dashboard works from pip-installed wheel; TB charts correct; safer checkpoint defaults.
- Do NOT: introduce a logger backend besides CSV/TB.

### Phase 4 — Make performance credible
- Issues: PERF-01..05.
- Files: `tgraphx/core/graph_utils.py`, `tgraphx/graph_builders.py`, `tgraphx/models/factory.py`, new `benchmarks/bench_*.py`.
- Tests: TEST-09 + bench scripts.
- Expected benefit: predictable behavior at moderate-to-large N; `torch.compile` reproducibility documented.
- Do NOT: chase chunked GAT yet.

### Phase 5 — Make ecosystem positioning clear
- Issues: DOC-04, DOC-05, DOC-08, README "Comparison" section (new).
- Files: README, new `docs/comparison.md`.
- Strategy: a one-page "When to use TGraphX vs PyG/DGL/NetworkX" with brutally honest scope. Reuse this audit's competitive positioning paragraph.
- Expected benefit: users self-select correctly; reduces "should be PyG clone" criticism.
- Do NOT: claim performance parity with PyG.

### Phase 6 — Future research only
- Graph transformers, heterogeneous/temporal graphs, learned adjacency, PyG/DGL bridges, GraphRAG, symbolic reasoning, higher-rank tensors.
- Each requires a separate design doc.
- Do NOT: bundle any of these into 0.1.x patches.

---

## 27. Top 10 highest-value next actions

1. **BUG-01** — Add `[tool.setuptools.package-data]` for dashboard static files; rebuild wheel; add wheel-content test.
2. **TRACK-01** — Replace `or` chain in `TensorBoardLogger.log` (and the same pattern in `_api_status` per DASH-01).
3. **BUG-02** — Make `LinearMessagePassing` honor its dropout/residual/batchnorm flags or reject them.
4. **DOC-01** — Update `docs/limitations.md` rows that lie about `train_epoch`/`evaluate`/`fit`/`TensorBoardLogger`; bump version line.
5. **DOC-03** — Add `train_epoch`/`evaluate`/`fit`/`TensorBoardLogger` rows to `docs/api_reference.md`.
6. **API-01** — Forward extra kwargs in `make_layer` for GIN (eps, train_eps, hidden_channels, use_batchnorm), SAGE (any missing), GAT (negative_slope).
7. **API-02 / UX-01 / UX-08** — Re-export training/tracking/classifier symbols at top-level; add the `tests/test_imports.py` row.
8. **CI-02** — Add a wheel-install smoke job that runs `tgraphx-dashboard --logdir runs/empty` and asserts `/static/dashboard.css` returns 200.
9. **SEC-01** — Default `load_checkpoint(..., weights_only=True)`; expose `weights_only=False` opt-in.
10. **PERF-03** — Replace `build_random_graph` candidate-pool construction with rejection sampling, gated behind `algorithm=` for backward compatibility.

---

## 28. Top 10 things NOT to do yet

1. Do not break the public API. Every change in this audit is additive or a bug fix.
2. Do not rename `LinearMessagePassing.update` interface even while restoring its behavior — keep the override delegation explicit so subclasses keep working.
3. Do not add `torch_scatter`/`torch_sparse` extension dependencies. Pure PyTorch is a feature.
4. Do not introduce heterogeneous/temporal graphs. Keep them as documented limitations.
5. Do not add a graph transformer layer. Future research only.
6. Do not chase chunked GAT in 0.1.x — the destination-wise softmax requires a careful design.
7. Do not pretend MPS support is fully tested if CI does not include a macOS runner.
8. Do not promise SOTA or benchmark wins; the package is honest about being for tensor-aware workflows.
9. Do not bundle a database, GraphRAG, or symbolic reasoner.
10. Do not silently change `TensorBoardLogger.log` step counter increment semantics — fix the falsy bug only; preserve auto-incrementing for the no-`epoch` path.

---

## 29. Questions requiring human decision

1. **Release cadence:** Should the dashboard packaging fix (BUG-01) ship as `0.1.3` immediately, or wait for the broader Batch A?
2. **`weights_only=True` default (SEC-01):** Are there third-party checkpoints in the wild from earlier TGraphX versions that would break under this default?
3. **`build_random_graph` ordering (PERF-03):** Is bit-for-bit reproducibility against earlier seeds a contract, or only "deterministic given seed" within a single version?
4. **CI matrix expansion:** Is there budget for macOS and Windows runners, or should the README/classifiers be tightened to Linux-only support claims?
5. **GCN layer:** Is "no GCN" a deliberate scope decision, or would a 50-line `TensorGCNLayer` be welcome?
6. **PyPI OIDC trusted publishing:** Migrate now (Batch G) or keep the API-token workflow for one more release?
7. **Colab tutorial regression test:** Should a CI job execute the Colab notebook (e.g. via `papermill`) so we catch packaging regressions like BUG-01 before users do?
8. **`set_seed(deterministic=True)`:** Should determinism be the default? It costs ~10–20% CUDA throughput on cuDNN paths.
9. **Top-level re-exports:** How aggressive — only training/tracking, or also `tgraphx.classifier` / `tgraphx.regressor` thin wrappers?
10. **CHANGELOG cadence:** Should we adopt "Keep a Changelog" with `[Unreleased]` section so 0.1.2 is properly documented even retrospectively?

---

## 30. Final recommendation

- **Implement fixes now?** Yes — Batch A only. The combination of BUG-01 (dashboard 404 for every PyPI user) + DOC-01 (docs lie about shipped features) + TRACK-01 (TB charts wrong from epoch 0) + BUG-02 (silent feature loss) is enough to justify a `0.1.3` patch release this week. The total diff is small (≤200 lines) and all changes are additive or strict bug fixes.
- **Which batch first?** Batch A. Then Batch C (docs catch-up), then Batch B (API ergonomics), then Batch F (test hardening), then Batch D / E in parallel.
- **Patch release after docs/API changes?** Yes — release `0.1.3` after Batch A + C, and `0.1.4` after Batch B + F. Defer Batch D / E to `0.2.0` if any breaking semantics emerge (none expected as scoped).
- **Major version planning:** Reserve `0.2.0` for any changes that are not strictly additive (e.g. SEC-01's `weights_only=True` default, PERF-03's sampling algorithm change behind a non-default flag becoming the default). Document the migration in CHANGELOG and `docs/limitations.md`.

---

## 31. Appendix — raw command outputs

### 31.1 `pytest -q` (truncated)

```
============================= test session starts ==============================
platform linux -- Python 3.13.12, pytest-9.0.2, pluggy-1.5.0
rootdir: /home/arash/PycharmProjects/TGraphX
configfile: pyproject.toml
testpaths: tests
plugins: cov-7.1.0, anyio-4.10.0
collected 685 items

tests/test_3d_support.py ...............................................
tests/test_dashboard.py ................................................
tests/test_devices.py ................sssss
tests/test_edge_features.py ..................
tests/test_edge_weight.py .............................................. .......
tests/test_factories.py ................................................ ........
tests/test_gnn_families.py ............................................. .......sss
tests/test_gradients.py ..................
tests/test_graph.py ........................
tests/test_graph_api.py ................................................ ...
tests/test_graph_builders.py ........................................... .........................................................
tests/test_imports.py .........
tests/test_layers.py ........................
tests/test_math.py ........................
tests/test_models.py ............................
tests/test_performance_smoke.py ........................................ ..
tests/test_tracking.py ...................ss
tests/test_training.py ........................................

======================= 675 passed, 10 skipped in 28.58s =======================
```

### 31.2 `python examples/run_all_fast_examples.py` (summary)

```
Script                                         Status      Time
----------------------------------------------------------------
  01_vector_node_classification.py             ok          1.4s
  02_spatial_graph_classification.py           ok          1.4s
  03_volumetric_graph_classification.py        ok          1.4s
  ...                                          ok          ...
  torch_compile_benchmark.py                   ok          5.0s
  tiny_overfit_tensor_gat.py                   ok          1.5s
  ...
----------------------------------------------------------------
  OK 28  |  FAIL 0  |  TIMEOUT 0  |  MISSING 0  |  SKIP 0
  All present examples passed.
```

### 31.3 `twine check dist/tgraphx-0.1.2*`

```
Checking dist/tgraphx-0.1.2-py3-none-any.whl: PASSED
Checking dist/tgraphx-0.1.2.tar.gz: PASSED
```

### 31.4 Wheel content (BUG-01 evidence)

```
Length      Date    Time    Name
---------  ---------- -----   ----
     2437  2026-05-06 07:03   tgraphx/__init__.py
     ...
    22815  2026-05-06 05:03   tgraphx/dashboard/app.py
       94  2026-05-06 03:59   tgraphx/dashboard/__main__.py
     3620  2026-05-06 05:03   tgraphx/dashboard/__init__.py
     ...
(no tgraphx/dashboard/static/ entries)
```

### 31.5 GAT benchmark (CPU, 16 nodes, 64 edges, 8x4x4)

```
  TGraphX Layer Benchmark
  Layer                  gat
  Device                 cpu
  Node shape             (8, 4, 4)   [16 nodes]
  Edges                  64
  Parameters             176
  Forward                0.184 ms  ±0.037 ms
  Output shape           (16, 16, 4, 4)
```

### 31.6 TensorBoardLogger reproducer (TRACK-01)

```
Calls: [('train_loss', 0.9, 0),
        ('train_loss', 0.8, 1),
        ('train_loss', 0.5, 2),     # epoch=0 logged at step 2 — BUG
        ('train_loss', 0.3, 2)]     # epoch=2 also at step 2 — collision
```

### 31.7 LinearMessagePassing reproducer (BUG-02)

```
attrs: 0.5 True True
has bn: True
train==eval (should differ if dropout active): True   # bug confirmed
```

### 31.8 Working tree status (end of audit)

```
$ git status
On branch main
nothing to commit, working tree clean
```

### 31.9 Generated artifacts (intentional, build only)

```
dist/tgraphx-0.1.0-py3-none-any.whl
dist/tgraphx-0.1.0.tar.gz
dist/tgraphx-0.1.1-py3-none-any.whl
dist/tgraphx-0.1.1.tar.gz
dist/tgraphx-0.1.2-py3-none-any.whl   # produced by `python -m build` during audit
dist/tgraphx-0.1.2.tar.gz             # produced by `python -m build` during audit
```

— *End of report.* —

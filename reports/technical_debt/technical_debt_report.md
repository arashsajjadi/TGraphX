# TGraphX Technical Debt Report

**Generated:** 2026-05-09T01:12:03.610590  
**Package version:** 0.6.0  
**Total Debt Score:** 12.5/100 — **EXCELLENT**

---

## 1. Executive Summary

- **Blockers:** 0
- **High debt:** 7
- **Medium debt:** 158
- **Low debt:** 580
- **Total findings:** 745

> **Verdict:** Release quality is acceptable. No blockers found.

---

## 2. Technical Debt Index

| Category | Score (0–100) | Weight |
|----------|:---:|:---:|
| Complexity Debt | 33.4 🟡 | 15% |
| Dead Code Debt | 0.0 🟢 | 12% |
| Test Coverage Debt | 0.0 🟢 | 15% |
| Docs Api Drift Debt | 24.0 🟡 | 15% |
| Type Debt | 8.8 🟢 | 10% |
| Architecture Import Debt | 27.6 🟡 | 10% |
| Performance Guard Debt | 2.6 🟢 | 10% |
| Security Debt | 0.0 🟢 | 5% |
| Packaging Release Debt | 0.0 🟢 | 3% |
| Ai Code Smell Debt | 0.6 🟢 | 5% |
| **Total (weighted)** | **12.5** | 100% |

---

## 3. Tools Available

- ruff: ✅
- mypy: ✅
- radon: ⬜ (skipped)
- vulture: ⬜ (skipped)
- bandit: ⬜ (skipped)

---

## 4. Blockers

_No blockers found._

---

## 5. Top 20 Highest-Risk Files

| File | Risk Score |
|------|:---:|
| `tgraphx/mining/__init__.py` | 97 |
| `tgraphx/__init__.py` | 89 |
| `tgraphx/dashboard/app.py` | 48 |
| `tgraphx/generation/high_level_api.py` | 34 |
| `tgraphx/rl/__init__.py` | 24 |
| `tgraphx/rl/high_level_api.py` | 19 |
| `tgraphx/evolutionary/high_level_api.py` | 12 |
| `tgraphx/evolutionary/operators.py` | 12 |
| `tgraphx/kg/__init__.py` | 12 |
| `tgraphx/graph_builders.py` | 10 |
| `tgraphx/hetero_sampling.py` | 10 |
| `tgraphx/layers/gat.py` | 10 |
| `tgraphx/sampling.py` | 10 |
| `tgraphx/rl/networks/__init__.py` | 10 |
| `tgraphx/kg/evaluation.py` | 9 |
| `tgraphx/rl/algorithms/continuous.py` | 9 |
| `tgraphx/tracking.py` | 9 |
| `tgraphx/generation/classical.py` | 8 |
| `tgraphx/layers/factory.py` | 8 |
| `tgraphx/mining/prototype.py` | 8 |

---

## 6. Top 20 Long Functions

| File | Function | Lines |
|------|----------|:---:|
| `tgraphx/hetero_sampling.py` | `hetero_neighbor_sample` | 172 |
| `tgraphx/layers/factory.py` | `make_layer` | 166 |
| `tgraphx/generation/high_level_api.py` | `run_graph_generation` | 158 |
| `tgraphx/layers/gat.py` | `forward` | 154 |
| `tgraphx/dashboard/app.py` | `_api_metrics` | 153 |
| `tgraphx/sampling_negative.py` | `hard_negative_sampling` | 152 |
| `tgraphx/evolutionary/high_level_api.py` | `run_evolutionary_optimization` | 148 |
| `tgraphx/dashboard/app.py` | `_collect_hardware` | 146 |
| `tgraphx/rl/high_level_api.py` | `_run_continuous` | 146 |
| `tgraphx/layers/gat.py` | `_chunked_forward` | 144 |
| `tgraphx/core/hetero_batch.py` | `__init__` | 140 |
| `tgraphx/core/graph.py` | `_batch_graphs` | 130 |
| `tgraphx/performance.py` | `env_report` | 129 |
| `tgraphx/hetero_sampling.py` | `hetero_induced_subgraph` | 122 |
| `tgraphx/sampling.py` | `random_walk_sample` | 121 |
| `tgraphx/core/hetero_graph.py` | `__init__` | 113 |
| `tgraphx/kg/data.py` | `__init__` | 112 |
| `tgraphx/layers/sage.py` | `forward` | 111 |
| `tgraphx/layers/graph_transformer.py` | `forward` | 110 |
| `tgraphx/graph_builders.py` | `build_iou_graph` | 108 |

---

## 7. Docs / API Drift

- **[medium]** `tgraphx/__init__.py` — Public export 'LinkNeighborLoader' has no reference in tests/examples/docs
- **[medium]** `tgraphx/kg/__init__.py` — Public export 'write_kg_model_report' has no reference in tests/examples/docs
- **[medium]** `tgraphx/kg/__init__.py` — Public export 'write_kg_gnn_report' has no reference in tests/examples/docs
- **[medium]** `tgraphx/kg/__init__.py` — Public export 'write_temporal_kg_report' has no reference in tests/examples/docs
- **[medium]** `tgraphx/kg/__init__.py` — Public export 'write_kg_reasoning_report' has no reference in tests/examples/docs
- **[medium]** `tgraphx/kg/__init__.py` — Public export 'write_kg_benchmark_report' has no reference in tests/examples/docs
- **[medium]** `tgraphx/kg/__init__.py` — Public export 'write_kg_multimodal_feature_report' has no reference in tests/examples/docs
- **[medium]** `tgraphx/rl/networks/__init__.py` — Public export 'StateFeatureProjector' has no reference in tests/examples/docs
- **[medium]** `tgraphx/rl/networks/__init__.py` — Public export 'ActionFeatureProjector' has no reference in tests/examples/docs
- **[medium]** `tgraphx/rl/networks/__init__.py` — Public export 'NodeActionPolicy' has no reference in tests/examples/docs
- **[medium]** `tgraphx/rl/networks/__init__.py` — Public export 'EdgeActionPolicy' has no reference in tests/examples/docs
- **[medium]** `tgraphx/rl/networks/__init__.py` — Public export 'GraphEditPolicy' has no reference in tests/examples/docs
- **[medium]** `tgraphx/rl/algorithms/__init__.py` — Public export 'BaseAgent' has no reference in tests/examples/docs
- **[medium]** `tgraphx/rl/algorithms/__init__.py` — Public export 'A2CAgent' has no reference in tests/examples/docs
- **[medium]** `tgraphx/rl/__init__.py` — Public export 'make_graph_policy' has no reference in tests/examples/docs
- **[medium]** `tgraphx/rl/exploration/__init__.py` — Public export 'LinearEpsilonDecay' has no reference in tests/examples/docs
- **[medium]** `tgraphx/rl/exploration/__init__.py` — Public export 'BoltzmannExploration' has no reference in tests/examples/docs
- **[medium]** `tgraphx/rl/exploration/__init__.py` — Public export 'UCBExploration' has no reference in tests/examples/docs
- **[medium]** `tgraphx/rl/exploration/__init__.py` — Public export 'EntropyRegularizer' has no reference in tests/examples/docs
- **[medium]** `tgraphx/rl/__init__.py` — Public export 'episode_length_mean' has no reference in tests/examples/docs

---

## 8. Public API Coverage

- Total public exports found: 633
- Exports without test/doc/example reference: 90

**Unreferenced exports (sample):**
- `Public export 'LinkNeighborLoader' has no reference in tests/examples/docs`
- `Public export 'write_kg_model_report' has no reference in tests/examples/docs`
- `Public export 'write_kg_gnn_report' has no reference in tests/examples/docs`
- `Public export 'write_temporal_kg_report' has no reference in tests/examples/docs`
- `Public export 'write_kg_reasoning_report' has no reference in tests/examples/docs`
- `Public export 'write_kg_benchmark_report' has no reference in tests/examples/docs`
- `Public export 'write_kg_multimodal_feature_report' has no reference in tests/examples/docs`
- `Public export 'StateFeatureProjector' has no reference in tests/examples/docs`
- `Public export 'ActionFeatureProjector' has no reference in tests/examples/docs`
- `Public export 'NodeActionPolicy' has no reference in tests/examples/docs`
- `Public export 'EdgeActionPolicy' has no reference in tests/examples/docs`
- `Public export 'GraphEditPolicy' has no reference in tests/examples/docs`
- `Public export 'BaseAgent' has no reference in tests/examples/docs`
- `Public export 'A2CAgent' has no reference in tests/examples/docs`
- `Public export 'make_graph_policy' has no reference in tests/examples/docs`

---

## 9. Tensor-Native Debt

_No tensor-native debt detected._

---

## 10. Performance Guard Debt

- **[medium]** `tgraphx/evolutionary/multi_objective.py:71` — Nested O(N²) loop: for i in range(n):
- **[medium]** `tgraphx/generation/actions.py:180` — Nested O(N²) loop: for src in range(n):
- **[medium]** `tgraphx/generation/classical.py:159` — Nested O(N²) loop: for i in range(n):
- **[medium]** `tgraphx/generation/high_level_api.py:207` — Nested O(N²) loop: for i in range(n):
- **[medium]** `tgraphx/mining/neural.py:595` — Nested O(N²) loop: src = [u for u in range(n) for v in range(n) if u != v]
- **[medium]** `tgraphx/mining/node2vec.py:88` — Nested O(N²) loop: adj: list = [[] for _ in range(num_nodes)]
- **[medium]** `tgraphx/plotting/graph.py:216` — Potential O(N²) pattern without guard: A = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
- **[medium]** `tgraphx/plotting/layouts.py:110` — Nested O(N²) loop: for i in range(num_nodes):

---

## 11. Dashboard Drift

_No dashboard drift detected._

---

## 12. Benchmark Drift

- **[medium]** `benchmarks/benchmark_metrics.py` — Benchmark lacks --json / machine-readable output
- **[low]** `benchmarks/benchmark_metrics.py` — Benchmark JSON output missing field: package_version
- **[low]** `benchmarks/benchmark_metrics.py` — Benchmark JSON output missing field: status
- **[low]** `benchmarks/benchmark_metrics.py` — Benchmark JSON output missing field: limitations
- **[medium]** `benchmarks/benchmark_tensor_vs_flatten.py` — Benchmark lacks --json / machine-readable output
- **[low]** `benchmarks/benchmark_tensor_vs_flatten.py` — Benchmark JSON output missing field: package_version
- **[low]** `benchmarks/benchmark_tensor_vs_flatten.py` — Benchmark JSON output missing field: status
- **[low]** `benchmarks/benchmark_tensor_vs_flatten.py` — Benchmark JSON output missing field: limitations
- **[medium]** `benchmarks/make_benchmark_report.py` — Benchmark lacks --small flag for fast CI runs
- **[medium]** `benchmarks/make_benchmark_report.py` — Benchmark lacks --json / machine-readable output
- **[low]** `benchmarks/make_benchmark_report.py` — Benchmark JSON output missing field: package_version
- **[low]** `benchmarks/make_benchmark_report.py` — Benchmark JSON output missing field: status
- **[low]** `benchmarks/make_benchmark_report.py` — Benchmark JSON output missing field: limitations
- **[medium]** `benchmarks/benchmark_utils.py` — Benchmark lacks --help / argparse
- **[medium]** `benchmarks/benchmark_utils.py` — Benchmark lacks --small flag for fast CI runs

---

## 13. Security Findings

_No security patterns detected._

---

## 14. Lint Summary (ruff)

Total lint findings: 374

**Top rule codes:**
- `F401`: 345
- `F841`: 14
- `E702`: 8
- `F821`: 3
- `E741`: 2
- `E731`: 1
- `F811`: 1

---

## 15. Type Debt (mypy)

Total mypy errors: 164

**By module:**
- `tgraphx`: 164

---

## 16. Suggested Cleanup Plan


### P1 — High Debt (fix in v1.x patch)
- File has 1426 lines (max 1200)
- Function '_api_metrics' has 153 lines
- Function 'run_graph_generation' has 158 lines
- Function 'hetero_neighbor_sample' has 172 lines
- Function 'make_layer' has 166 lines
- Function 'forward' has 154 lines
- Function 'hard_negative_sampling' has 152 lines

### P2 — Medium Debt (tech debt sprint)
- Function '_batch_graphs' has 130 lines
- Function '__init__' has 140 lines
- Function '__init__' has 113 lines
- Function '_collect_hardware' has 146 lines
- Function 'main' has 106 lines
- Function '_handle_api' has 106 lines
- File has 880 lines (warn at 800)
- Function 'run_evolutionary_optimization' has 148 lines
- Function '_validate' has 92 lines
- Function 'fit' has 81 lines

### P3 — Low (nice-to-have)
- [F401] `.generation.GeneratedGraph` imported but unused; consider removing, adding to `__all__`, or using a redundant alias
- [F401] `.generation.GraphEditState` imported but unused; consider removing, adding to `__all__`, or using a redundant alias
- [F401] `.generation.GraphGenerationTrajectory` imported but unused; consider removing, adding to `__all__`, or using a redundant alias
- [F401] `.generation.GraphGenerationBatch` imported but unused; consider removing, adding to `__all__`, or using a redundant alias
- [F401] `.generation.GraphActionType` imported but unused; consider removing, adding to `__all__`, or using a redundant alias

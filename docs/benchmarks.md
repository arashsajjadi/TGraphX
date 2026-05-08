# Benchmarks

The `benchmarks/` directory contains reproducibility / engineering
benchmarks that ship with TGraphX.  They are **not** real-world
performance claims and **not** SOTA comparisons.

| Script | Purpose |
|--------|---------|
| `benchmark_dataset_loading.py` | time the construction of every native synthetic + (optionally) folder/upstream-backed dataset |
| `benchmark_training_synthetic.py` | tiny training-loop sanity for graph-classification, node-classification, graph-regression |
| `benchmark_tensor_vs_flatten.py` | run an identical synthetic patch-graph task with a tensor-aware `ConvMessagePassing` model and a flatten-then-`LinearMessagePassing` baseline; report time and loss decrease |
| `benchmark_transforms.py` | per-transform throughput on a synthetic graph |
| `benchmark_metrics.py` | per-metric throughput |
| `benchmark_layers.py` | layer-level forward / backward timings (existing) |
| `benchmark_graph_builders.py` | grid / kNN / radius / IoU / fully-connected timing (existing) |
| `benchmark_sampling.py` | sampling helpers (existing) |
| `make_benchmark_report.py` | combine multiple benchmark JSONs into a markdown report |

## Running

Every benchmark accepts a `--small` flag for CI-safe defaults and
optionally an `--output PATH.json` argument:

```bash
python benchmarks/benchmark_dataset_loading.py --small
python benchmarks/benchmark_training_synthetic.py --small --output runs/bench/training.json
python benchmarks/benchmark_tensor_vs_flatten.py --small --output runs/bench/tvf.json
python benchmarks/benchmark_transforms.py --small --output runs/bench/transforms.json
python benchmarks/benchmark_metrics.py --small --output runs/bench/metrics.json

# Combine
python benchmarks/make_benchmark_report.py runs/bench/*.json --output runs/bench/report.md
```

## Honest interpretation

* **No download by default.**  No benchmark fetches data unless you
  pass an explicit `--download` flag (and even then, only on the very
  small set of benchmarks that opt into upstream adapters).
* **Synthetic data only by default.**  Patch graphs, volume graphs,
  SBM-style node graphs, etc. — use them to validate trainability
  and reproducibility, not to draw conclusions about real-world
  performance.
* **No SOTA claims.**  TGraphX is a tensor-aware GNN library; its
  benchmarks measure engineering properties (time, memory, loss
  decrease, gradient health), not leaderboard rankings.
* **Determinism.**  Pass `--seed` (where supported) to make output
  reproducible.

## Reading the JSON output

Every benchmark writes a flat top-level dict with at least:

| Key | Meaning |
|-----|---------|
| `version` | TGraphX version that produced the file |
| `small` | whether `--small` was set |
| `results` | list of per-row dicts (label, time, etc.) — for the dataset / metrics / transforms benchmarks |

`benchmark_tensor_vs_flatten.py` writes top-level `tensor` and
`flatten` dicts instead of a list.  `make_benchmark_report.py` knows
about both shapes.

## CI

`tests/test_benchmark_smoke.py` runs every benchmark in `--small`
mode and checks that:

* the script exits with status 0,
* the JSON output is valid,
* the report generator runs.

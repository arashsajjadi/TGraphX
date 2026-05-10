# TGraphX Benchmark Report (v1.2+)

This document describes the TGraphX benchmark structure and what each benchmark
reports.  **Before reading any number, read the scope statement below.**

---

## Scope statement (read first)

The benchmarks shipped in `benchmarks/` are **CI-safe smoke benchmarks**:

- They use **small synthetic data**.
- They run on a single device (CPU by default; CUDA optional).
- They use few repeats by default.
- They are **not** a competitive throughput claim against PyG, DGL, PyKEEN,
  Stable-Baselines3, RLlib, or NetworkX.

For adoption decisions, run the official upstream benchmarks of those
ecosystems, then run TGraphX with realistic-scale data of your own.

---

## How to run

### Single benchmark

Every benchmark script supports `--small`, `--json`, `--seed`, and
either `--device` or a sensible default:

```bash
python benchmarks/ux/benchmark_easy_vs_manual.py --small --json --epochs 2
python benchmarks/benchmark_layers.py --layer gat --nodes 16 --edges 64 \
    --shape 8,4,4 --device cpu --iters 3 --warmup 1
python benchmarks/benchmark_graph_builders.py --small
```

### Full suite

The v1.2 suite runner aggregates a curated subset into a single JSON report:

```bash
python benchmarks/run_v12_benchmark_suite.py --small \
    --out reports/benchmarks/v12_small.json
```

The output is a JSON document with one row per benchmark:

```json
{
  "suite": "tgraphx_v12_smoke",
  "package_version": "1.0.3",
  "device": "cpu",
  "seed": 42,
  "small": true,
  "benchmarks": [
    {
      "name": "easy_mode_train",
      "status": "ok",
      "runtime_s": 0.1576,
      "metrics": { "loss": 1.13, "accuracy": 0.41 }
    },
    ...
  ],
  "limitations": [...]
}
```

The schema is **stable for the suite runner**: `name`, `status`, `runtime_s`,
`device`, `seed`, `small`, `package_version`, `metrics`.  Failed rows include
an `error` field.

---

## Benchmark catalogue

| Benchmark | Type | What it measures | Expected runtime (--small, CPU) |
|-----------|------|------------------|---------------------------------|
| `benchmark_easy_vs_manual` | UX | Easy Mode wrapper overhead vs manual loop on matched config | < 1 s |
| `benchmark_layers` | Performance | Per-layer forward/backward time on tiny tensors | < 5 s |
| `benchmark_graph_builders` | Performance | Grid / kNN / radius graph construction | < 1 s |
| `benchmark_sampling` | Performance | NeighborLoader / GraphSAINT sampling time | < 5 s |
| `benchmark_dataset_loading` | I/O | Synthetic dataset load + first batch time | < 1 s |
| `benchmark_metrics` | Numeric | Pure-PyTorch metric throughput | < 1 s |
| `benchmark_transforms` | I/O | Transform pipeline throughput | < 1 s |
| `benchmark_tensor_vs_flatten` | Performance | Tensor-aware vs flatten-baseline forward time | < 5 s |
| `benchmark_training_synthetic` | Performance | End-to-end synthetic training | < 5 s |
| `benchmarks/generation/*` | Smoke | Graph generation methods + metrics | < 3 s each |
| `benchmarks/evolution/*` | Smoke | GA / SA / NSGA-II runtime | < 3 s each |
| `benchmarks/rl/*` | Smoke | RL algorithm episodes | < 5 s each |
| `benchmarks/kg/*` | Smoke | KG model training + filtered ranking | < 3 s each |
| `benchmarks/mining/*` | Smoke | Graph mining utilities | < 3 s each |
| `benchmarks/sampling/*` | Smoke | GraphSAINT / Cluster-GCN | < 3 s each |
| `run_v12_benchmark_suite.py` | Aggregate | Single-JSON summary of 7 representative benchmarks | < 1 s |

Every benchmark reports `package_version`, `seed`, `device`, `status`,
`limitations`, and a `metrics` block.  The dashboard reads the JSON output
directly through `tgraphx-dashboard --logdir <directory containing the
JSON>`.

---

## Honest limitations

- **Tiny graphs.** All `--small` benchmarks use < 5 000 nodes / < 50 000
  edges.  Performance characteristics at industrial scale (millions of edges,
  out-of-core sampling) are not exercised.
- **Single device, single process.** No distributed measurements.
- **No reference comparison.** TGraphX does not run PyG / DGL / PyKEEN
  benchmarks for direct comparison; doing so requires those packages, model
  ports, and careful normalization of training conditions.  **That is a
  separate roadmap item** ([roadmap.md](roadmap.md)).

If you need a comparison, the recommended approach is:

1. Pick one TGraphX benchmark that maps cleanly to a PyG / DGL example.
2. Match: dataset, model, optimizer, hyper-params, batch size, device, seed.
3. Report runtime AND final metric.  Don't report runtime alone.
4. State PyTorch / CUDA / driver versions.

---

## See also

- [Easy Mode dashboard integration](easy_mode.md#dashboard-integration)
- [API stability](api_stability.md)
- [Limitations](limitations.md)
- [Roadmap](roadmap.md)

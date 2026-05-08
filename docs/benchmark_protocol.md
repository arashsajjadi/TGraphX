# Public benchmark protocol

This document defines the protocol every TGraphX public-dataset
benchmark script follows.  It is a **smoke** protocol — it produces
small-data engineering metrics (training time, loss decrease, accuracy
on tiny subsets), not leaderboard numbers.

For the report tables produced by these scripts see
[`public_benchmark_reports.md`](public_benchmark_reports.md).

For the manual validation scripts (a separate, lighter set under
`examples/public_datasets/`) see
[`public_dataset_validation.md`](public_dataset_validation.md).

---

## Where the scripts live

```
benchmarks/public/
├── _common.py                    # shared CLI + artefact helpers
├── mnist_patch_benchmark.py      # FakeData by default; real MNIST opt-in
├── pyg_cora_benchmark.py         # skips cleanly without PyG
└── …                             # additional scripts in later milestones
```

The scripts import lazily from `tgraphx.datasets` and never run when
their optional dependency is missing (unless `--strict` is set).

---

## Uniform CLI

Every benchmark in `benchmarks/public/` accepts:

| Flag | Default | Meaning |
|------|---------|---------|
| `--root PATH` | `$TGRAPHX_DATA` or `~/.cache/tgraphx` | Cache directory for the upstream dataset |
| `--download` | off | Allow network downloads.  Off by default — TGraphX never bundles datasets and never downloads silently. |
| `--max-samples N` | `200` | Cap on graph-level samples |
| `--max-nodes N` | `10_000` | Cap on node-level slices for very large graphs |
| `--epochs N` | `5` | Number of training epochs |
| `--device {auto,cpu,cuda,mps}` | `auto` | Device override |
| `--output-dir PATH` | temp dir (cleaned on exit) | Where to write artefacts |
| `--seed N` | `0` | RNG seed |
| `--json` | off | Print a machine-readable JSON summary to stdout |
| `--strict` | off | Hard-fail (exit 2) instead of skipping when optional dependency missing |

`--help` is provided by `argparse`.

---

## Artefacts

Every script writes (at minimum) the following four JSON files under
`--output-dir`:

| File | Contents | Dashboard panel |
|------|----------|-----------------|
| `benchmark_results.json` | Full benchmark payload (see schema below) | Benchmark |
| `run_metadata.json` | Run name, status, device, seed, version, timestamps | Overview |
| `dataset_metadata.json` | Dataset name, task, num graphs/nodes/edges, license, upstream library | Dataset |
| `metrics_summary.json` | Final / best metrics extracted from the benchmark | Metrics |

Optional output (when implemented by the script):

- `metrics.csv` — per-epoch metrics for the dashboard's chart panel.
- `snapshot.html` — dashboard offline snapshot.

---

## `benchmark_results.json` schema

The required keys are:

```json
{
  "benchmark": "mnist_patch_benchmark",
  "data_source": "fake_data_synthetic | torchvision_mnist | …",
  "elapsed_s": 0.327,
  "epochs": 2,
  "num_graphs": 8,
  "num_nodes": 128,
  "num_edges": 384,
  "loss_start": 2.4149,
  "loss_end": 2.3805,
  "loss_decreased": true,
  "tgraphx_version": "0.3.1",
  "python": "3.11.5",
  "platform": "Linux-…",
  "torch_version": "2.…",
  "device": "cpu",
  "seed": 0,
  "started_at": "2026-05-08T12:34:56Z"
}
```

Task-specific keys are added by individual scripts:

| Script | Additional keys |
|--------|----------------|
| `mnist_patch_benchmark.py` | `node_feature_shape`, `model_param_count`, `final_accuracy` |
| `pyg_cora_benchmark.py` | `num_classes`, `train_accuracy`, `val_accuracy`, `test_accuracy` |

`benchmark_results.json` is a dashboard-readable artefact; the helper
that produces it follows the same schema as
`tgraphx.tracking.write_benchmark_results`.

---

## Honesty rules

These are enforced by code review and by
`tests/test_public_benchmarks.py`:

- Scripts must **not** download anything unless `--download` is passed
  explicitly.
- Scripts must skip cleanly (exit 0) when an optional dependency is
  missing.  `--strict` flips this to a hard failure (exit 2).
- Scripts must cap dataset size and report the actual sizes used.
- No SOTA wording.  No leaderboard claims.  No "TGraphX outperforms".
- Reported accuracies are tiny-overfit / tiny-train values, not
  best-of-published.
- `dataset_metadata.json` records `tgraphx_redistributes: false` for
  every public dataset.

---

## Running locally

CI-safe (no network):

```bash
python benchmarks/public/mnist_patch_benchmark.py --epochs 2 --max-samples 16
```

Real public dataset (manual; requires network):

```bash
python benchmarks/public/mnist_patch_benchmark.py \
    --download --epochs 5 --max-samples 200 \
    --output-dir runs/mnist_patch
python benchmarks/public/pyg_cora_benchmark.py \
    --download --epochs 30 \
    --output-dir runs/pyg_cora
```

Then start the dashboard:

```bash
tgraphx-dashboard --logdir runs/mnist_patch
```

The dashboard surfaces `benchmark_results.json`,
`dataset_metadata.json`, `metrics_summary.json`, and
`run_metadata.json` automatically.

---

## See also

- [Public dataset validation policy](public_dataset_validation.md)
- [Dataset license policy](dataset_license_policy.md)
- [Dashboard](dashboard.md)
- [Benchmarks (general)](benchmarks.md)
- [Roadmap](roadmap.md)

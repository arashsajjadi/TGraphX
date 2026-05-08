# Public benchmark reports

This page collects small-data engineering metrics from local runs of the
scripts under `benchmarks/public/`.  Numbers here are **not**
leaderboard claims; they are smoke-style sanity checks that prove the
TGraphX framework can train on real datasets when the user opts in.

For the protocol that produces these numbers see
[`benchmark_protocol.md`](benchmark_protocol.md).

For the full list of what is and isn't validated, see
[`public_dataset_validation.md`](public_dataset_validation.md) and
[`limitations.md`](limitations.md).

---

## How to read this page

Every row records a **single local run** of a `benchmarks/public/*.py`
script with the listed flags.  We deliberately do not aggregate across
seeds or hardware in this document — that is the job of a leaderboard,
and TGraphX is not a leaderboard runner.

If you want to reproduce a row, run the exact command, then paste your
own row into a personal copy of this file or attach the
`benchmark_results.json` from your local run to a GitHub release note.

---

## CI-safe FakeData smoke (no network, no download)

| Date | TGraphX | Device | Script | Flags | Loss start → end | Loss decreased? | Final acc |
|------|---------|--------|--------|-------|-----------------|-----------------|-----------|
| 2026-05-08 | 0.3.1 | cpu | `mnist_patch_benchmark.py` | `--epochs 2 --max-samples 8` | 2.4149 → 2.3805 | yes | 0.000 |

Notes:

- `final_accuracy = 0.000` is expected on FakeData with 2 epochs and 8
  graphs of random labels — there is no signal in synthetic noise.
  The point of the row is to confirm that the framework runs, the loss
  decreases, and the dashboard artefacts are written.

---

## Manual public-dataset runs (network required)

These rows are filled in by maintainers after running the scripts with
`--download`.  Add new rows by appending below — please include date,
TGraphX version, exact CLI flags, and the upstream library version.

### MNIST (`torchvision`)

```bash
python benchmarks/public/mnist_patch_benchmark.py \
    --download --epochs 5 --max-samples 200 \
    --output-dir runs/mnist_patch_v0_3_1
```

| Date | TGraphX | torchvision | Device | Loss start → end | Final train acc |
|------|---------|-------------|--------|-----------------|-----------------|
| _pending_ | 0.3.1 | _ | _ | _ | _ |

### Planetoid Cora (`torch_geometric`)

```bash
python benchmarks/public/pyg_cora_benchmark.py \
    --download --epochs 30 \
    --output-dir runs/pyg_cora_v0_3_1
```

| Date | TGraphX | PyG | Device | Train acc | Val acc | Test acc |
|------|---------|-----|--------|-----------|---------|----------|
| _pending_ | 0.3.1 | _ | _ | _ | _ | _ |

---

## What this page deliberately does not contain

- No comparison against PyG / DGL / OGB published numbers.  The MNIST
  patch-graph formulation is a TGraphX-specific data path; comparing
  it to standard MNIST CNN baselines or to vanilla GCN baselines on
  raw pixels is apples-to-oranges.
- No claim that TGraphX outperforms anything.  See
  [`limitations.md`](limitations.md) for the honest scope.
- No automatic publishing to GitHub Pages or external dashboards.
  These tables are local engineering notes.

---

## See also

- [Benchmark protocol](benchmark_protocol.md)
- [Public dataset validation policy](public_dataset_validation.md)
- [Dataset license policy](dataset_license_policy.md)
- [Roadmap](roadmap.md)

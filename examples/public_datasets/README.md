# Public dataset validation scripts (manual)

These scripts validate TGraphX's optional dataset adapters against the
**actual** upstream loaders.  They are intentionally **not** part of
the default test suite or CI: they can download data from the
internet and they target third-party datasets whose licenses TGraphX
does not own.

## Policy

* Each script exits without doing anything network-facing unless you
  pass `--download` (or `--train-tiny` for OGB).
* Each script caps dataset size by default
  (`--max-samples 100`, `--max-nodes 5000`).
* Each script skips cleanly with exit code 0 when the relevant
  optional package (`torchvision`, `torch_geometric`, `dgl`, `ogb`)
  is missing.  Pass `--strict` to turn the skip into a hard failure
  instead.
* Each script writes dashboard-compatible artefacts under
  `--output-run-dir` (or a temporary directory that is cleaned up on
  exit).

TGraphX **does not redistribute** any of these datasets.  Cite the
upstream dataset card (and TGraphX, if you wish) when you publish
results.

## Scripts

| Script | Network | Optional dep |
|--------|--------|--------------|
| `fake_torchvision_patch_smoke.py` | none — uses `torchvision.datasets.FakeData` | torchvision (already a TGraphX base dependency) |
| `mnist_patch_smoke.py` | only with `--download` | torchvision |
| `pyg_cora_smoke.py` | only with `--download` | `pip install "tgraphx[pyg]"` |
| `ogb_arxiv_smoke.py` | only with `--download` | `pip install "tgraphx[ogb]"` |
| `dgl_cora_smoke.py` | only with `--download` | DGL (install per upstream docs) |

## Suggested invocations

```bash
# CI-safe — no download
python examples/public_datasets/fake_torchvision_patch_smoke.py --epochs 2

# Small MNIST validation (real download)
python examples/public_datasets/mnist_patch_smoke.py \
    --download --max-samples 100 --epochs 3

# Cora via PyG
python examples/public_datasets/pyg_cora_smoke.py --download --epochs 3

# OGB ogbn-arxiv: load + split + evaluator only by default
python examples/public_datasets/ogb_arxiv_smoke.py --download

# OGB with optional capped training
python examples/public_datasets/ogb_arxiv_smoke.py \
    --download --train-tiny --max-nodes 5000 --epochs 3

# DGL Cora
python examples/public_datasets/dgl_cora_smoke.py --download --epochs 3
```

Each script prints a JSON summary at the end containing TGraphX
version, device, dataset stats, loss decrease indicator, and the run
directory containing `metrics.csv`, `run_metadata.json`,
`dataset_metadata.json`, `metrics_summary.json`, and (for MNIST) the
explanation artefacts the dashboard renders.

## What these scripts are not

* **Not benchmarks.**  They are reproducibility / sanity validations
  to verify that TGraphX's optional dataset adapters interoperate with
  upstream loaders, not real-world performance comparisons.
* **Not SOTA results.**  Sample caps make accuracy numbers
  meaningless beyond "did the loss decrease".
* **Not part of CI.**  These scripts must not be invoked from the
  default test suite — TGraphX tests never touch the network.

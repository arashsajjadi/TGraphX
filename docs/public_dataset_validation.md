# Public dataset validation

TGraphX includes manual scripts that validate the optional dataset
adapters against the **actual** upstream loaders.  These scripts live
under `examples/public_datasets/` and are intentionally **not** part
of the default test suite or CI.

## Policy

* Each script exits without doing anything network-facing unless you
  pass `--download` (or `--train-tiny` for OGB).
* Each script caps dataset size by default
  (`--max-samples 100`, `--max-nodes 5000`).
* Each script skips cleanly with exit code 0 when the relevant
  optional package is missing.  Pass `--strict` to turn the soft skip
  into a hard failure (exit code 2).
* Each script writes dashboard-compatible artefacts under
  `--output-run-dir` (or a temporary directory that is cleaned up on
  exit): `metrics.csv`, `run_metadata.json`, `dataset_metadata.json`,
  `metrics_summary.json`, and (where appropriate) explanation
  artefacts.

TGraphX **does not redistribute** any of these datasets.  Cite the
upstream dataset card (and TGraphX, if you wish) when you publish
results.

## Available scripts

| Script | Network | Optional dependency |
|--------|---------|---------------------|
| `fake_torchvision_patch_smoke.py` | none — uses `torchvision.datasets.FakeData` | torchvision (already a TGraphX base dependency) |
| `mnist_patch_smoke.py` | only with `--download` | torchvision |
| `pyg_cora_smoke.py` | only with `--download` | `pip install "tgraphx[pyg]"` |
| `ogb_arxiv_smoke.py` | only with `--download` | `pip install "tgraphx[ogb]"` |
| `dgl_cora_smoke.py` | only with `--download` | DGL (install per upstream docs) |

## Suggested invocations

```bash
# CI-safe — no download
python examples/public_datasets/fake_torchvision_patch_smoke.py --epochs 2

# Real download examples (manual only)
python examples/public_datasets/mnist_patch_smoke.py \
    --download --max-samples 100 --epochs 3
python examples/public_datasets/pyg_cora_smoke.py --download --epochs 3
python examples/public_datasets/ogb_arxiv_smoke.py --download
python examples/public_datasets/ogb_arxiv_smoke.py \
    --download --train-tiny --max-nodes 5000 --epochs 3
python examples/public_datasets/dgl_cora_smoke.py --download --epochs 3
```

## What these scripts are not

* **Not benchmarks.**  They are reproducibility / sanity validations
  to verify that TGraphX's optional dataset adapters interoperate with
  upstream loaders, not real-world performance comparisons.
* **Not SOTA results.**  Sample caps make accuracy numbers
  meaningless beyond "did the loss decrease".
* **Not part of CI.**  These scripts must not be invoked from the
  default test suite — TGraphX tests never touch the network.

## What is in CI

The `fake_torchvision_patch_smoke.py` script is exercised in
`tests/test_release_validation_v030.py::TestPublicDatasetScripts::test_fake_torchvision_runs`
because it relies only on `torchvision.datasets.FakeData`, which
synthesises images in memory.  All other scripts are tested at the
`--help` / "no-network" level only, which validates that the CLI
parsing and skip-on-missing-dependency paths work without ever
hitting an upstream loader.

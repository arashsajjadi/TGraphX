# Device validation

TGraphX ships a `examples/device_validation.py` script that runs a
small, deterministic smoke pass across the layer families it
advertises and emits a JSON report.  The script never claims a device
works that it has not actually tested — it always reports the
*requested* device, the *actual* device used, whether CUDA / MPS were
available, and per-layer forward/backward results.

## Usage

```bash
# CPU smoke (always available)
python examples/device_validation.py --device cpu --quick \
    --output-json /tmp/tgraphx_cpu_device_validation.json

# CUDA smoke (skips cleanly when no GPU is present)
python examples/device_validation.py --device cuda --amp \
    --output-json cuda_validation.json

# MPS smoke (skips cleanly on non-Apple-Silicon machines)
python examples/device_validation.py --device mps --quick

# Strict mode: exit 2 instead of skipping when the requested device
# is unavailable
python examples/device_validation.py --device cuda --strict
```

## What it covers

| Layer family | Vector | Spatial (2-D) |
|--------------|:-:|:-:|
| `LinearMessagePassing` | yes | — |
| `GCNConv` (v0.3.0 zoo) | yes | — |
| `GATv2Conv` (v0.3.0 zoo) | yes | — |
| `APPNP` | yes | — |
| `ConvMessagePassing` | — | yes |
| `TensorGATLayer` | — | yes |
| `TensorGraphSAGELayer` | — | yes |
| `TensorGINLayer` | — | yes |

For each layer the script:

* runs forward,
* checks the output is finite,
* runs a tiny backward,
* checks gradients are finite,
* records elapsed time.

A single dataset / metric smoke (`SyntheticPatchGraphDataset` +
`accuracy`) confirms that dataset and metric paths run on the chosen
device.

## AMP coverage

When `--amp` is set:

* on CUDA, the vector smoke runs inside
  `torch.autocast(device_type="cuda", dtype=torch.float16)`;
* on CPU, the vector smoke runs inside
  `torch.autocast(device_type="cpu", dtype=torch.bfloat16)`.

Output rows are tagged with the dtype so a CI diff between releases
can spot regressions.

## CI status

CI runs `examples/device_validation.py --device cpu --quick` on every
pull request via
`tests/test_release_validation_v030.py::TestDeviceValidation`.  CUDA
and MPS coverage is **local** — TGraphX has no GPU runners.  Run the
CUDA invocation on your own RTX system (or any CUDA-capable host) to
generate `cuda_validation.json`; pull-request reviewers can attach
the JSON when reporting CUDA test results.

## What this is not

* Not a benchmark.  Elapsed times are reported for diagnostic
  purposes; the script does not aggregate or compare across runs.
* Not a substitute for the dedicated math invariants tests in
  `tests/test_math_invariants_v030.py`, which assert permutation
  equivariance and other formal properties.

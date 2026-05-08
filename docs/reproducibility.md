# Reproducibility

`tgraphx.reproducibility` (Beta, v0.4.1+) provides utilities for
reproducible experiments: RNG seeding, deterministic operation, and
a per-run reproducibility report.

## Important caveats

- **Hardware-independent bit-exact identity is not guaranteed.**
  CPU and GPU use different kernel implementations.  The same seed
  gives the same sequence on the same device/backend, but GPU results
  may differ from CPU results.
- **`deterministic=True` reduces performance** by disabling
  cuDNN algorithm selection and enabling slower deterministic kernels.
  Use it only when reproducibility matters more than throughput.
- **CUBLAS workspace:** for full CUDA determinism you may also need to
  set the environment variable
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` **before** launching Python.
  This cannot be changed from Python after CUDA has been initialised.
- **Python hash randomisation** (`PYTHONHASHSEED`) affects string and
  bytes hashing, not integer or tuple-of-integer hashing.  TGraphX's
  WL kernel uses tuple-of-integer keys and is therefore deterministic
  across processes regardless of `PYTHONHASHSEED`.

## Import

```python
from tgraphx.reproducibility import (
    set_seed,
    make_generator,
    seed_worker,
    reproducibility_report,
    deterministic_mode,
)
```

## `set_seed`

```python
from tgraphx.reproducibility import set_seed

state = set_seed(42)
# Sets random, numpy (if installed), torch CPU, and torch.cuda (if available).
# Returns a state dict for logging.
print(state)
# {'seed': 42, 'deterministic': False, 'torch_version': '...', ...}
```

| Argument | Default | Description |
|----------|---------|-------------|
| `seed` | — | Integer seed |
| `deterministic` | `False` | Enable cuDNN deterministic mode and `use_deterministic_algorithms` |
| `benchmark` | `None` | Override `cudnn.benchmark`; when `None` set to `not deterministic` |
| `warn_only` | `True` | When `deterministic=True`, non-deterministic ops warn instead of raising |

**Returns:** dict with `seed`, `deterministic`, `torch_version`, `cuda_available`, `backend_settings`.

**Note:** `tgraphx.set_seed` in `tgraphx.training` is a simpler shim that
also works and is not changing.  `tgraphx.reproducibility.set_seed` adds
the return value and `warn_only` support.

## `make_generator`

Returns a `torch.Generator` seeded to a specific value.  Does **not**
affect the global RNG.

```python
from tgraphx.reproducibility import make_generator

g = make_generator(0)
samples = torch.randint(100, (5,), generator=g)
```

## `seed_worker`

For reproducible DataLoader workers:

```python
from tgraphx.reproducibility import seed_worker
import torch

g = torch.Generator()
g.manual_seed(0)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True,
    worker_init_fn=seed_worker, generator=g,
)
```

Each worker receives a deterministic seed derived from the initial
generator state.

## `deterministic_mode` context manager

```python
from tgraphx.reproducibility import deterministic_mode

with deterministic_mode(seed=42, warn_only=True):
    output = model(x, edge_index)
# Previous deterministic state is restored on exit.
```

## `reproducibility_report`

```python
from tgraphx.reproducibility import reproducibility_report

report = reproducibility_report()
# {'torch_version': '...', 'cuda_available': True, 'cudnn_deterministic': False, ...}
```

Write this to your run metadata to track reproducibility settings.

## WL kernel determinism

TGraphX's Weisfeiler-Lehman kernel (`tgraphx.mining.kernels`) uses
tuple-of-integer dictionary keys (not string keys), making it
independent of `PYTHONHASHSEED`.  The same graph with the same node
labels produces the same WL label sequence in every Python process,
regardless of how the process was started.

```python
# This is safe across processes with different PYTHONHASHSEED.
from tgraphx.mining import weisfeiler_lehman_labels
labels = weisfeiler_lehman_labels(edge_index, num_nodes, num_iterations=3)
```

## Limitations

- No bitwise cross-device guarantee.
- `deterministic=True` may raise `RuntimeError` for a few scatter ops
  that lack deterministic CUDA kernels.  Pass `warn_only=True` to
  convert errors to warnings.
- NumPy global seed is set when numpy is importable; no per-array
  generator support.

## Related

- Tests: `tests/test_reproducibility.py`
- Related: `docs/performance.md`, `docs/graph_kernels.md`

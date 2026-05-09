# Sparse backends

`tgraphx.sparse` ships pure-PyTorch implementations of every
scatter/segment operation.  Optional acceleration hooks pick up
`torch_scatter` and `pyg_lib` when they are installed; pure PyTorch is
always the safe fallback.

## API

```python
from tgraphx.sparse import backend_info, select_backend, segment_sum

info = backend_info()
# {'pure_torch': True, 'torch_scatter': False, 'pyg_lib': False, ...}

select_backend("auto")           # picks torch_scatter when available
select_backend("torch_scatter")  # explicit; falls back with a warning
```

The active backend is process-global and changes only when
`select_backend(...)` is called.  Optional dependencies are imported
lazily — `tgraphx.sparse` is safe to import even when `torch_scatter`
or `pyg_lib` is missing.

## Functions accelerated when `torch_scatter` is active

- `segment_sum`

The remaining `segment_mean` / `segment_max` / `segment_min` /
`segment_softmax` use the pure-PyTorch path; future releases may add
acceleration hooks where empirical speedups are clear.

## Stability

The backend selector is **Beta** in v0.5.0+.  Numerical results match
the pure-PyTorch path to within float32 tolerances on the regression
parity tests.

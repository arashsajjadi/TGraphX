# Performance

## Environment report

```python
from tgraphx.performance import env_report, estimate_message_memory, recommended_device

# Always-available fields: python, os, torch, tgraphx, cuda_available,
# cuda_device, mps_available, recommended_device
info = env_report()

# Optional hardware fields (requires psutil)
info = env_report(include_hardware=True)

# Optional sensor fields (requires pynvml)
info = env_report(include_sensors=True)

dev = recommended_device()   # CUDA > MPS > CPU as torch.device
```

`psutil` and `pynvml` are **never** imported at base `import tgraphx` time.
They are loaded lazily inside `env_report()` only when requested.

## Memory estimation

```python
m = estimate_message_memory(num_edges=1024, out_shape=(64, 8, 8))
print(f"~{m['total_mb']:.1f} MB  ({m['note']})")
```

This estimates the peak `[E, *out_shape]` message buffer before scatter aggregation.
Actual peak usage is typically 2–3× higher due to intermediate conv outputs.

## Benchmarks

### Layer benchmark

```bash
python benchmarks/benchmark_layers.py \
    --layer gat --nodes 64 --edges 256 --shape 8,4,4 --device cpu --iters 10

python benchmarks/benchmark_layers.py \
    --layer conv --nodes 256 --edges 2048 --shape 32,8,8 \
    --device cuda --amp 1 --backward 1

# Save JSON
python benchmarks/benchmark_layers.py --layer gin --shape 8,4,4 --output result.json
```

**Timing:** CUDA events on GPU, `time.perf_counter` on CPU/MPS. No CUDA sync
outside the timed region.

### Graph builder benchmark

```bash
python benchmarks/benchmark_graph_builders.py --small   # CI-safe
python benchmarks/benchmark_graph_builders.py           # full
```

O(N²) builders (`build_knn_graph`, `build_radius_graph`, `build_iou_graph`,
`build_fully_connected_graph`) are marked `[O(N²)]` in the output.

## torch.compile

```python
import torch
compiled = torch.compile(layer, mode="default")   # requires PyTorch ≥ 2.0
```

- Smoke-tested: eager and compiled outputs agree within tolerance (≤ 1e-4).
- **No universal speedup is guaranteed.** Compile overhead dominates for
  small graphs (< 64 nodes); potential gains at larger scales.
- Falls back gracefully when `torch.compile` is unavailable.

```bash
python examples/torch_compile_benchmark.py
```

## AMP / Mixed precision

```python
# CUDA — float16
with torch.autocast("cuda", dtype=torch.float16):
    out = layer(x, edge_index)

# CPU — bfloat16 (PyTorch ≥ 1.13)
with torch.autocast("cpu", dtype=torch.bfloat16):
    out = layer(x, edge_index)
```

**Known limitation:** `TensorGATLayer` uses `index_add_` which enforces
matching dtypes even under autocast. float16 autocast may raise a dtype
mismatch for GAT. Use bfloat16 or full precision as alternatives.

```bash
python examples/mixed_precision_inference.py
```

## Chunked edge processing (ConvMessagePassing)

```python
from tgraphx.layers.conv_message import ConvMessagePassing

layer = ConvMessagePassing(in_shape=(32, 8, 8), out_shape=(32, 8, 8), aggr="sum")
# Mathematically identical to unchunked; lower peak message buffer
out = layer(x, edge_index, chunk_size=256)
```

- Supported: `aggr="sum"` and `aggr="mean"`.
- `aggr="max"` falls back to unchunked with a warning.
- **GAT, SAGE, GIN chunking is deferred.** GAT requires all edge scores
  for destination-wise softmax; two-pass chunking provides no memory benefit.

## Profiling

No profiling or hardware polling is enabled by default.
Use PyTorch's built-in profiler for fine-grained analysis:

```python
with torch.profiler.profile() as p:
    out = layer(x, edge_index)
print(p.key_averages().table(sort_by="self_cpu_time_total"))
```

## Hardware compatibility

| Platform | Forward | Backward | AMP | torch.compile |
|---|:-:|:-:|:-:|:-:|
| CPU | ✅ | ✅ | ⚠️ bfloat16 | ✅ (may be slow) |
| CUDA | ✅ | ✅ | ⚠️ op-dependent | ✅ |
| MPS (Apple Silicon) | ✅ | ✅ | limited | ⚠️ partial |

⚠️ = supported but with known constraints documented above.

## Deferred / future work

- Incremental / tail-read for large `metrics.csv` files in the dashboard.
- GAT / SAGE / GIN chunked forward.
- Neighbor sampling (GraphSAINT / ClusterGCN style).
- Universal float16 AMP support (requires refactoring index-based scatter ops).

## See also

- [Limitations](limitations.md)
- [Dashboard API metrics caching](dashboard.md#performance)

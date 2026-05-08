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

**Correctness:** Eager and compiled outputs are smoke-tested to agree within
tolerance (≤ 1e-4 float32; ≤ 2e-2 bfloat16) for Conv, GAT, SAGE, and GIN
layers with and without edge weights / edge features.

**No universal speedup claim.** Compile overhead dominates for small graphs
(< 64 nodes).  Potential throughput gains exist at larger scales but are not
guaranteed and depend on the specific layer, shape, and device.

**Graceful degradation:** When `torch.compile` is unavailable (PyTorch < 2.0)
or fails for a specific op/backend combination, the layer falls back to eager
mode without error.

```bash
python examples/torch_compile_benchmark.py
```

## AMP policy

TGraphX v0.2.2 hardened dtype handling across all four spatial GNN layers.
The policy below describes the supported and best-effort AMP modes.

| Backend | Recommended AMP dtype | Status | Notes |
|---------|:---------------------:|:------:|-------|
| CPU | bfloat16 | ✅ Best-effort | PyTorch 1.13+; tested in CI |
| CUDA | float16 or bfloat16 | ⚠️ Best-effort | bfloat16 requires Ampere+ GPU |
| MPS | — | ❌ Not tested | MPS AMP support is PyTorch-version dependent |

**v0.2.2 fixes:**
- `broadcast_edge_weight` now casts `edge_weight` to the message dtype so
  float32 edge weights work correctly under any autocast context.
- `TensorGATLayer` now casts attention weights to the activation dtype before
  `index_add_`; the `a_src`/`a_dst` float32 parameters no longer pollute the
  aggregation dtype under autocast.
- `edge_softmax` upcasts to float32 for the numerically sensitive max-shift +
  exp + sum computation, then casts back to the original dtype.

**Output dtype behaviour:** Under `torch.autocast`, Conv2d/Conv3d/Linear
layers produce outputs in the autocast dtype (e.g. bfloat16). The final
output of each GNN layer will be in the activation dtype. Summing or using
the output in a loss function should `.float()` the result if full-precision
gradients are needed.

**Attention logits:** The `edge_softmax` computation always runs in float32
internally (upcast from float16/bfloat16) for numerical stability. The
returned attention weights are cast back to the input dtype.

```python
# CUDA — float16
with torch.autocast("cuda", dtype=torch.float16):
    out = layer(x, edge_index)

# CPU — bfloat16 (PyTorch ≥ 1.13)
with torch.autocast("cpu", dtype=torch.bfloat16):
    out = layer(x, edge_index)

# Backward: upcast loss to float32 for stable gradients
with torch.autocast("cpu", dtype=torch.bfloat16):
    out = layer(x, edge_index)
loss = out.float().sum()   # upcast before backward
loss.backward()
```

> ⚠️ **CUDA float16 best-effort:** `scatter_reduce_` with `reduce="amax"` for
> max aggregation (SAGE, base scatter) requires PyTorch ≥ 1.13 support for
> float16 CUDA scatter ops.  The non-max paths (sum, mean, GAT softmax) are
> robust.  If you see float16 errors on an older PyTorch version, switch to
> bfloat16 or full precision.

```bash
python examples/mixed_precision_inference.py
```

## Chunked edge processing

Reduce peak edge-buffer memory by processing edges in chunks.
All chunked paths are disabled by default (`chunk_size=None`).

| Layer | Supported aggr | Status | Notes |
|-------|:--------------:|:------:|-------|
| `ConvMessagePassing` | sum, mean | ✅ Stable | max falls back with warning |
| `TensorGraphSAGELayer` | mean, max | ✅ Stable v0.2.3 | |
| `TensorGINLayer` | sum | ✅ Stable v0.2.3 | |
| `TensorGATLayer` | — | ✅ Stable v0.2.4 | Two-pass log-sum-exp; pass `chunk_size=K` |

```python
from tgraphx.layers import ConvMessagePassing, TensorGraphSAGELayer, TensorGINLayer

# ConvMessagePassing
conv = ConvMessagePassing(in_shape=(32, 8, 8), out_shape=(32, 8, 8), aggr="sum")
out = conv(x, edge_index, chunk_size=256)  # same result; lower peak memory

# TensorGraphSAGELayer
sage = TensorGraphSAGELayer(32, 32, aggr="mean")
out = sage(x, edge_index, chunk_size=256)

# TensorGINLayer
gin = TensorGINLayer(32, 32)
out = gin(x, edge_index, chunk_size=256)
```

All chunked outputs match unchunked within float32 precision (exact for
mean/sum; exact for max).  Gradients flow correctly through all chunked paths.

> ⚠️ **GAT chunked forward deferred.** `TensorGATLayer` requires destination-wise
> softmax over **all** incoming edges before any attention weight can be
> finalised.  A correct two-pass implementation (Pass 1: collect per-destination
> max/logsumexp; Pass 2: recompute normalised weights and aggregate values) is
> planned for v0.2.4.  The single-pass unchunked path is unchanged.

## Dashboard performance overhead

When the dashboard is **disabled** (the default): zero overhead — no timers,
threads, or hardware polling.

When the dashboard is **enabled**:

- `/api/metrics` is mtime/size-cached; the CSV is re-parsed only when it
  changes.  The browser fetches incremental rows via `?since_row=N`, reducing
  payload size during long runs.  The server still parses the full CSV on a
  cache miss; true byte-seek tail-reading is not yet implemented.
- `/api/hardware` readings are cached for ~1.5 s; `pynvml.nvmlInit()` is
  called at most once per process.
- Polling frequency is set by `--refresh-interval` (default 2 s); the
  browser pauses polling when the tab is hidden.

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

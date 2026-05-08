# Temporal graph utilities

`tgraphx.temporal` is the growing home for temporal-graph building
blocks.  v0.3.2 ships only **time encoding** primitives.  A full
temporal GNN (TGN/TGAT-style memory module, chronological splits,
temporal negative sampling) is scheduled for v0.3.4.

**This page covers only the v0.3.2 surface.**  For the snapshot-loop
classifiers shipped in v0.2.5 see `docs/training_utilities.md` and the
examples in `examples/temporal_graph_classifier_demo.py`.

## Import

```python
from tgraphx.temporal import (
    sinusoidal_time_encoding,   # Beta
    LearnableTimeEncoding,      # Experimental
)
```

---

## `sinusoidal_time_encoding`

Deterministic Transformer-style positional encoding for timestamps.
No learnable parameters; output is fully determined by the input.

```python
import torch
from tgraphx.temporal import sinusoidal_time_encoding

t = torch.tensor([0.0, 1.0, 10.0, 100.0])
enc = sinusoidal_time_encoding(t, dim=16)
# enc: FloatTensor[4, 16]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `timestamps` | — | `Tensor[*]` — any shape; int or float; cast to float32 |
| `dim` | — | Encoding dimension; must be a **positive even** integer |
| `base` | `10_000.0` | Frequency base; must be positive |

**Returns:** `FloatTensor[*, dim]` — same leading shape as `timestamps`
with `dim` appended.

**Formula:**

```
even columns 2k:   sin(t / base^(2k / dim))
odd  columns 2k+1: cos(t / base^(2k / dim))
```

This matches the Transformer (Vaswani et al., 2017) positional encoding
formula.

**Properties:**
- Deterministic for any input; no global RNG.
- Finite for very large timestamps.
- Each sin/cos pair satisfies `sin²(t) + cos²(t) = 1`, so the encoding
  norm is constant: `‖enc[t]‖² = dim/2`.
- Device-preserving: output lives on the same device as `timestamps`.

| Device | Support |
|--------|---------|
| CPU | OK |
| CUDA | OK |
| MPS | Best-effort |

**Common errors:**

| Error | Cause |
|-------|-------|
| `ValueError: dim must be a positive even integer` | Pass an odd or ≤ 0 dim |
| `ValueError: base must be positive` | `base ≤ 0` |

---

## `LearnableTimeEncoding`

Time2Vec-style trainable time encoder (Kazemi et al., 2019).

- Channel 0: linear projection `w₀ · t + b₀`.
- Channels 1 … dim-1: sinusoidal `sin(wₖ · t + bₖ)` with learned
  frequencies and phases.

```python
from tgraphx.temporal import LearnableTimeEncoding

enc = LearnableTimeEncoding(dim=16)
t = torch.tensor([0.0, 1.0, 2.0])
out = enc(t)   # FloatTensor[3, 16]

# Train it like any nn.Module.
loss = out.sum()
loss.backward()
# enc.linear_w.grad, enc.periodic_w.grad are non-zero
```

| Constructor arg | Default | Description |
|-----------------|---------|-------------|
| `dim` | — | Output dimension; must be ≥ 2 |
| `init_scale` | `0.01` | Initial scale of frequency parameters |

**Forward:** `timestamps: Tensor[*]` → `Tensor[*, dim]` float32.

**Stability:** Experimental.  The API is expected to evolve in v0.3.4
once it is integrated with a TGAT-style layer and evaluated on a real
temporal benchmark.  The core Time2Vec semantics (channel 0 linear,
channels 1+ sinusoidal with learned params) will remain stable.

---

## Integration with existing temporal utilities

The `tgraphx.temporal` package is the future home of the temporal
subsystem.  The following **already-shipped** utilities live in the
older module layout and will be consolidated in v0.3.4:

| Symbol | Where today | Moving to |
|--------|-------------|-----------|
| `TemporalGraphSequence` | `tgraphx.core.temporal` | `tgraphx.temporal` (v0.3.4) |
| `TemporalGraphBatch` | `tgraphx.core.temporal_batch` | `tgraphx.temporal` (v0.3.4) |
| `temporal_window_sample` | `tgraphx.temporal_sampling` | `tgraphx.temporal` (v0.3.4) |
| `TemporalGraphClassifier` | `tgraphx.models.temporal_models` | `tgraphx.temporal` (v0.3.4) |

All existing import paths will continue to work; the move is additive.

---

## Planned for v0.3.4

- `tgraphx.temporal.memory` — node memory table with GRU update.
- `tgraphx.temporal.tgn` — TGN-inspired message-passing memory
  (experimental; will be labelled as such).
- `tgraphx.temporal.samplers` — chronological train/val/test split,
  temporal negative sampling.

These are **not** yet implemented.  See `docs/roadmap.md`.

## Related

- Tests: `tests/test_time_encoding.py`
- Examples: `examples/time_encoding_demo.py`
- Architecture: `docs/architecture.md`

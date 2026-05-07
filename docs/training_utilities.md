# Training Utilities

TGraphX provides **lightweight training utilities** in `tgraphx.training`
and metric logging helpers in `tgraphx.tracking`.

These are **thin helpers, not a full training framework**.  They handle the
common boilerplate (seeding, parameter counting, checkpointing, one-epoch
loops) while leaving training-loop design, callbacks, and scheduling to you.

> **Logging and file writes are off by default.**
> `logger=None` and `log_level=0` (both defaults in every function) produce
> no file writes, no stdout, and no dashboard activity.
>
> **MLflowLogger is not implemented.**  Use the `mlflow` client directly.

---

## Training loop helpers

### `train_epoch`

```python
from tgraphx.training import train_epoch

result = train_epoch(
    model,
    loader,            # GraphDataLoader or (x, y) DataLoader
    optimizer,
    loss_fn,           # fn(output, targets) -> scalar Tensor
    device="auto",     # auto selects CUDA > MPS > CPU
    metrics={"accuracy": acc_fn},   # optional extra metrics
    logger=None,       # CSVLogger / TensorBoardLogger / None
    log_level=0,       # 0=silent, 1=print epoch summary, 2=per-batch progress
    epoch=None,        # passed to logger if provided
    amp=False,         # CUDA autocast; no GradScaler — see note
    grad_clip=None,    # gradient norm clip value
)
# returns: {"loss": 0.42, "accuracy": 0.85, ...}
```

**Supported batch formats:**

| Format | Model call | Targets extracted from |
|---|---|---|
| `GraphBatch` | `model(nf, ei, batch=b.batch, ...)` | `batch.graph_labels` or `batch.node_labels` |
| `(Tensor, Tensor)` | `model(x)` | second element |
| Other | raises `ValueError` with a clear message | — |

**GraphBatch label squeezing:** if `graph_labels` has shape `[B, 1]` and
**integer dtype** (e.g. `torch.long` class indices), it is squeezed to `[B]`
for `CrossEntropyLoss` compatibility.  Float `[B, 1]` regression targets are
preserved as-is so `MSELoss` receives matching shapes.

**Model compatibility:** `train_epoch` inspects the model's `forward`
signature and passes only the kwargs it accepts.  Simple models (e.g.
`torch.nn.Sequential`) work without receiving graph-specific kwargs.
TGraphX factory models receive `batch=`, `edge_features=`, and `edge_weight=`
when available.  A `TypeError` raised *inside* the model propagates as a
`RuntimeError` with context — it is not swallowed.

**Metric failures** emit a one-time `UserWarning` per metric name rather than
silently disappearing from results.

**AMP note:** `amp=True` wraps the forward pass in `torch.autocast("cuda")`.
No `GradScaler` is used.  For stable float16 training, manage a
`torch.cuda.amp.GradScaler` in your own loop.

### `evaluate`

```python
from tgraphx.training import evaluate

result = evaluate(
    model, loader, loss_fn,
    metrics={"accuracy": acc_fn},
    device="auto",
)
# runs under torch.no_grad(); no file writes
# returns: {"loss": 0.38, "accuracy": 0.91}
```

### `fit`

```python
from tgraphx.training import fit
import torch
import torch.nn.functional as F

history = fit(
    model,
    train_loader,
    val_loader=val_loader,   # optional
    epochs=20,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),  # required
    loss_fn=F.cross_entropy,   # required
    device="auto",
    metrics={"accuracy": acc_fn},
    logger=None,       # no files written
    log_level=1,       # 0=silent, 1=per-epoch summary, 2=per-batch+per-epoch
)
# history: [{"epoch":0, "train_loss":0.9, "val_loss":0.85, ...}, ...]
```

`fit()` is a thin wrapper over `train_epoch` + `evaluate`.  It performs no
hidden checkpointing, no early stopping, and starts no dashboard.

---

## Reproducibility

```python
from tgraphx.training import set_seed

set_seed(42)   # sets torch, numpy (if installed), and random
```

### Deterministic cuDNN (optional)

```python
# Also sets torch.backends.cudnn.deterministic=True
# and torch.backends.cudnn.benchmark=False.
set_seed(42, deterministic=True)
```

Enabling deterministic mode reduces GPU throughput.  Use it only when
bit-exact reproducibility matters more than speed (e.g. debugging).

> **Note:** `torch.use_deterministic_algorithms(True)` is intentionally
> *not* set by `deterministic=True` because several scatter/index
> operations used by TGraphX layers do not have deterministic CUDA
> kernels and would raise `RuntimeError` at runtime.  If your workload
> supports fully deterministic algorithms, call
> `torch.use_deterministic_algorithms(True)` manually *after*
> `set_seed(seed, deterministic=True)`.

## Parameter count

```python
from tgraphx.training import count_parameters

n = count_parameters(model)                          # trainable only
n = count_parameters(model, trainable_only=False)    # all params
print(f"Parameters: {n:,}")
```

## Checkpointing

```python
from tgraphx.training import save_checkpoint, load_checkpoint

save_checkpoint(model, optimizer, epoch=10, path="runs/ep10.pt",
                loss=0.12, tag="best")

# Default: safe loading (weights_only=True)
epoch = load_checkpoint(model, optimizer, path="runs/ep10.pt")
epoch = load_checkpoint(model, None, path="runs/ep10.pt", map_location="cpu")
```

The checkpoint is a plain `torch.save` dict with keys `epoch`,
`model_state_dict`, `optimizer_state_dict`, and any extra kwargs.

### Checkpoint security

`load_checkpoint` defaults to `weights_only=True`, which restricts
deserialization to safe tensor/scalar types.  Standard TGraphX checkpoints
are fully compatible with this mode.

```python
# Default (safe, recommended for all TGraphX checkpoints)
epoch = load_checkpoint(model, optimizer, path="best.pt")

# Legacy/trusted checkpoints only — emits a UserWarning
epoch = load_checkpoint(model, optimizer, path="old.pt", weights_only=False)
```

> **Warning:** never load a checkpoint from an untrusted source with
> `weights_only=False`. Doing so allows arbitrary Python code to execute
> during deserialization.

## Metrics

```python
from tgraphx.training import accuracy, mean_absolute_error, mean_squared_error

acc = accuracy(logits, labels)              # [N,C] logits, [N] int labels → float
mae = mean_absolute_error(preds, targets)   # float
mse = mean_squared_error(preds, targets)    # float
```

---

## CSV metric logging

```python
from tgraphx.tracking import CSVLogger

with CSVLogger("runs/my_run") as logger:
    history = fit(model, train_loader, val_loader=val_loader,
                  epochs=20, optimizer=opt, loss_fn=F.cross_entropy,
                  logger=logger)
# writes runs/my_run/metrics.csv (dashboard-compatible)
```

UTC timestamps are added automatically.  To resume across sessions,
create a new `CSVLogger` pointing to the same directory.

## TensorBoard logging (optional)

```python
from tgraphx.tracking import TensorBoardLogger   # lazy import of tensorboard

# Requires: pip install tensorboard  or  pip install "tgraphx[tracking]"
with TensorBoardLogger("runs/tb_run") as tb:
    history = fit(model, train_loader, val_loader=val_loader,
                  epochs=20, optimizer=opt, loss_fn=F.cross_entropy,
                  logger=tb)
# view: tensorboard --logdir runs/tb_run
```

If TensorBoard is not installed, `TensorBoardLogger()` raises `ImportError`
with install instructions.  No TensorBoard files are written unless you
explicitly instantiate `TensorBoardLogger`.

---

## Complete example (with fit + CSVLogger)

```python
import torch
import torch.nn.functional as F
from tgraphx import build_model, build_grid_graph, Graph
from tgraphx.core.dataloader import GraphDataLoader, GraphDataset
from tgraphx.training import fit, set_seed, count_parameters
from tgraphx.tracking import CSVLogger

set_seed(0)
# ... build graphs, loaders, model ...
model = build_model("graph_classification", "gat",
                    in_shape=(8,4,4), hidden_shape=(16,4,4),
                    num_layers=2, num_classes=3, heads=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"Parameters: {count_parameters(model):,}")

with CSVLogger("runs/experiment1") as logger:
    history = fit(
        model, train_loader, val_loader=val_loader,
        epochs=50, optimizer=optimizer, loss_fn=F.cross_entropy,
        logger=logger, log_level=1,
    )
```

## See also

- [Dashboard](dashboard.md)
- [Factories](factories.md)
- [Examples: training_minimal_fit.py, training_with_csvlogger.py, training_with_tensorboard.py](../examples/)

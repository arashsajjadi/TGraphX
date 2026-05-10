# TGraphX Easy Mode

**tgraphx.easy** provides a high-level, beginner-friendly, LLM-friendly API that removes boilerplate while preserving full access to the underlying PyTorch objects.

---

## 1. Philosophy

- Easy mode removes boilerplate; core APIs remain PyTorch-native.
- No black box — every default is visible in `result.config`.
- Every result exposes the underlying PyTorch objects (model, optimizer, loader, graph).
- Advanced users can drop down to low-level code at any point.

---

## 2. Quick start (no direct `torch` import required)

```python
import tgraphx as tgx

# Create synthetic data.
data = tgx.easy.synthetic_tensor_node_classification(
    num_nodes=1000, node_shape=(16, 8, 8), num_classes=10, seed=42,
)

# Train a node classifier.
result = tgx.easy.train_node_classifier(
    data, model="tensor_gcn", sampler="neighbor", epochs=5, seed=42,
)

print(result.metrics)
result.summary()
```

TGraphX is built on PyTorch. Returned tensors are PyTorch tensors and the
model is an `nn.Module`. But you do not need to write `import torch` for the
beginner path.

---

## 3. Before / After

### Tensor node classification

**Before (manual):**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tgraphx import Graph, ConvMessagePassing, NeighborLoader
from tgraphx.reproducibility import set_seed

set_seed(42)
N, C, H, W, K = 1000, 16, 8, 8, 10
x = torch.randn(N, C, H, W)
edge_index = torch.randint(0, N, (2, 5000))
y = torch.randint(0, K, (N,))
g = Graph(node_features=x, edge_index=edge_index, y=y)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvMessagePassing((C, H, W), (16, H, W))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(16, K)
    def forward(self, x, ei):
        z = self.conv(x, ei).relu()
        return self.head(self.pool(z).flatten(1))

model = Model()
opt = Adam(model.parameters(), lr=1e-3)
loader = NeighborLoader(g, fanouts=[15, 10], batch_size=64, seed=42)

for epoch in range(5):
    for batch in loader:
        logits = model(batch.node_features, batch.edge_index)
        loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)
        opt.zero_grad(); loss.backward(); opt.step()
```

**After (easy mode):**
```python
import tgraphx as tgx

data = tgx.easy.synthetic_tensor_node_classification(
    num_nodes=1000, node_shape=(16, 8, 8), num_classes=10, seed=42,
)
result = tgx.easy.train_node_classifier(data, model="tensor_gcn", epochs=5, seed=42)
print(result.metrics)
```

---

## 4. Result object

Every `train_*` or `fit_*` function returns an `EasyResult`:

| Field | Type | Description |
|-------|------|-------------|
| `metrics` | `dict` | Final epoch metrics (`loss`, `accuracy`, etc.) |
| `history` | `list[dict]` | Per-epoch metrics |
| `model` | `nn.Module` | Trained PyTorch model |
| `graph` | `Graph` | Source graph |
| `config` | `dict` | Resolved config with all defaults |
| `artifacts` | `dict` | Generated artefact paths (if any) |
| `loader` | `DataLoader` | Last-used loader |
| `optimizer` | `Optimizer` | Last-used optimizer |
| `elapsed` | `float` | Wall-clock seconds |

```python
result.summary()          # print readable summary
result.to_dict()          # JSON-serialisable dict
result.to_markdown()      # Markdown metric table
result.plot_loss()        # requires matplotlib
result.save_report(path)  # save JSON report
```

---

## 5. When NOT to use easy mode

Use the low-level PyTorch API when you need:
- Custom research training loops.
- Non-standard loss functions.
- Unusual batching or multi-task setups.
- Advanced optimizer schedules.
- Distributed experiments.
- Per-layer gradient clipping.

---

## 6. Dropping down to low-level code

Easy mode never hides the underlying objects:

```python
result = tgx.easy.train_node_classifier(data, epochs=5)

# Drop to PyTorch at any time.
model = result.model          # nn.Module — train, export, wrap in DDP
graph = result.graph          # tgraphx.Graph — access tensors directly
loader = result.loader        # NeighborLoader — iterate manually
optimizer = result.optimizer  # Adam — add schedulers, custom steps

# Inspect the resolved config.
print(result.config["lr"])    # 0.001
print(result.config["device"]) # "cuda" or "cpu"
```

---

## 7. Discovery functions

```python
import tgraphx as tgx

tgx.easy.list_tasks()         # All task names
tgx.easy.list_models()        # All model names
tgx.easy.list_models("node_classification")  # Task-specific
tgx.easy.list_samplers()      # Sampling strategies
tgx.easy.list_workflows()     # High-level functions
tgx.easy.show_capabilities()  # Full capability overview
tgx.easy.doctor()             # Installation health check
```

---

## 8. Troubleshooting

| Problem | Fix |
|---------|-----|
| `TGraphXLabelError: Node labels are required` | `g = Graph(node_features=x, edge_index=ei, y=y)` or `g.y = y` |
| `TGraphXUnknownNameError: Unknown model 'xyz'` | Check `tgx.easy.list_models("node_classification")` |
| `TGraphXUnknownNameError: Unknown sampler 'xyz'` | Check `tgx.easy.list_samplers()` |
| `TGraphXConfigError: 'graph' must be a tgraphx.Graph` | Use `tgx.easy.synthetic_tensor_node_classification(...)` or `Graph(...)` |
| Device mismatch | Set `device="auto"` in `train_node_classifier` |

---

## 9. Dashboard integration *(v1.1+)*

Easy Mode now writes dashboard-compatible artifacts directly.

### Auto-write at training time

Pass `dashboard_dir=` to `train_node_classifier` and Easy Mode writes the
artifacts as soon as training finishes:

```python
import tgraphx as tgx

result = tgx.easy.train_node_classifier(
    data, model="tensor_gcn", epochs=5, seed=42,
    dashboard_dir="runs/easy_run",
)
```

Then open the dashboard:

```bash
tgraphx-dashboard --logdir runs/easy_run
```

### Manual artifact write

If you trained without `dashboard_dir=`, write the artifacts after the fact:

```python
result.write_dashboard_artifacts("runs/easy_run")
```

Both paths produce the same three files:

| File | Content |
|------|---------|
| `metrics.csv` | One row per epoch with all numeric metrics |
| `run_metadata.json` | Run name, status, total epochs, device, model, seed, tgraphx version, elapsed |
| `metrics_summary.json` | Final metrics + `best_loss` + `best_epoch` |

Calling `write_dashboard_artifacts()` with empty `result.history` raises
`ValueError` so misuse fails loudly instead of writing zero-row files.

---

## 10. Seed cost and reproducibility

When `seed` is not `None`, `train_node_classifier` calls `set_seed(seed)`,
which sets torch, numpy, random, and cuDNN determinism flags.  This incurs a
constant ~5-50 ms setup cost (higher when CUDA is available and cuDNN flags must
be flipped).

If you run many short training calls in a loop, pass `seed=None` for calls
after the first:

```python
tgx.easy.train_node_classifier(data, ..., seed=42)   # seeded
tgx.easy.train_node_classifier(data, ..., seed=None)  # no extra seed setup
```

The resolved seed is always visible in `result.config["seed"]`.

---

## 11. CLI / module access

```bash
# System health check:
python -m tgraphx doctor

# Alias (same as doctor):
python -m tgraphx info

# Show all capabilities:
python -m tgraphx capabilities

# List available tasks:
python -m tgraphx tasks
```

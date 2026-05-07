# Layer and Model Factories

Factories let you create layers and complete task models by name, without
writing boilerplate constructors or managing spatial ranks manually.

## Layer factory: `make_layer`

```python
from tgraphx import make_layer

# 2-D GAT layer
layer = make_layer("gat", in_shape=(8, 4, 4), out_shape=(16, 4, 4),
                   heads=2, residual=True, dropout=0.1)

# 3-D SAGE layer
layer = make_layer("sage", in_shape=(4, 4, 4, 4), out_shape=(8, 4, 4, 4))

# Vector linear layer
layer = make_layer("linear", in_shape=(32,), out_shape=(64,), aggr="mean")
```

### Supported layer names

| Name | Class | Shape constraint |
|---|---|---|
| `"conv"` | `ConvMessagePassing` | 2-D or 3-D spatial only |
| `"gat"` | `TensorGATLayer` | 2-D or 3-D spatial only |
| `"sage"` | `TensorGraphSAGELayer` | 2-D or 3-D spatial only |
| `"gin"` | `TensorGINLayer` | 2-D or 3-D spatial only |
| `"linear"` | `LinearMessagePassing` | vector `(D,)` only |
| `"legacy_attention"` | `AttentionMessagePassing` | vector or 2-D spatial |

### Key kwargs forwarded per layer

| kwarg | Applies to | Notes |
|---|---|---|
| `aggr` | conv, sage, linear, legacy_attention | `"sum"`/`"mean"`/`"max"` |
| `heads` | gat | number of attention heads |
| `concat` | gat | concat heads (`True`) or average (`False`) |
| `residual` | all | skip connection when shapes match |
| `dropout` | gat (→`attn_dropout`), linear, legacy_attention (→`dropout_prob`) | |
| `use_edge_features`, `edge_dim` | all | `edge_dim` required when `use_edge_features=True` |
| `edge_features_kind` | sage, gin | `"spatial"` (default) or `"vector"` |
| `add_self_loops` | gat | add self-loops inside forward |
| `negative_slope` | gat | LeakyReLU slope (default `0.2`) |
| `normalize` | sage | L2-normalise output |
| `bias` | gat, sage | learnable bias |
| `use_batchnorm` | gin, linear | BatchNorm after aggregation |
| `eps` | gin | GIN ε initial value (default `0.0`) |
| `train_eps` | gin | make ε a learnable `nn.Parameter` |
| `hidden_channels` | gin | MLP hidden dim (defaults to `out_channels`) |

**Not forwarded** (set directly on the constructor if needed):
- `ConvMessagePassing`: `aggregator_params` dict (for deep CNN aggregator internals)
- `TensorGraphSAGELayer`: no `dropout` or `use_batchnorm` at this time
- `TensorGATLayer`: `edge_features_kind` is not applicable (handled by tensor rank detection)

**Unknown kwargs** are silently ignored by the factory rather than causing an error; only the listed kwargs are forwarded.  Passing unrecognised kwargs does not crash, but has no effect.

## Model factory: `build_model`

```python
from tgraphx import build_model

model = build_model(
    task="graph_classification",   # see supported tasks below
    layer="gat",                   # any make_layer name
    in_shape=(8, 4, 4),
    hidden_shape=(16, 4, 4),
    num_layers=2,
    num_classes=5,                 # required for classification
    # out_dim=1,                   # required for regression/edge_prediction
    heads=2,                       # forwarded to make_layer
    pooling="mean",                # graph readout: "mean"/"sum"/"max"
)
out = model(x, edge_index, batch=batch)
```

### Supported tasks

| Task | Output shape | `batch` required? |
|---|---|---|
| `"node_classification"` | `[N, num_classes]` | No |
| `"node_regression"` | `[N, out_dim]` | No |
| `"graph_classification"` | `[G, num_classes]` | Yes |
| `"graph_regression"` | `[G, out_dim]` | Yes |
| `"edge_prediction"` | `[E, out_dim]` | No |
| `"link_prediction"` | — | Not implemented; use `"edge_prediction"` |

### Spatial → vector pooling

For spatial `[N, C, *spatial]` node features, the factory applies global
spatial average-pooling → `[N, C]` before the linear head and graph readout.

## Config-based construction

```python
from tgraphx import build_model_from_config

# From a Python dict (no eval, no exec)
model = build_model_from_config({
    "model": {
        "task": "graph_classification",
        "layer": "gat",
        "in_shape": [8, 4, 4],
        "hidden_shape": [16, 4, 4],
        "num_layers": 2,
        "num_classes": 5,
        "heads": 2,
    }
})

# From a JSON file
model = build_model_from_config("config.json")

# From a YAML file (requires PyYAML)
model = build_model_from_config("config.yaml")
```

No `eval`, no `exec`. YAML uses `safe_load` only.

## See also

- [Training utilities](training_utilities.md)
- [Graph builders](graph_builders.md)

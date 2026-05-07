# Quickstart

## Colab tutorial

For an interactive, browser-based introduction that requires no local setup,
open the official TGraphX Colab notebook:

**[Open the TGraphX Colab Tutorial](https://colab.research.google.com/drive/1agls1xtqE5WxbWthcG0HEa3Gbk3fvoCD?usp=sharing)**

The notebook installs TGraphX from PyPI and walks through vector, 2-D spatial,
and 3-D volumetric node classification, regression, edge prediction, and the
full layer zoo on synthetic sanity-check tasks.

> These are controlled synthetic tasks intended to verify installation and API
> behaviour.  They are not benchmark results or real-world performance claims.

---

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # CPU only
pip install -e .   # from source (editable)
```

## Your first graph — vector features

If you only have flat feature vectors (no spatial structure), start here.

```python
import torch
from tgraphx import Graph, LinearMessagePassing

# Eight nodes with 32-dimensional feature vectors
node_features = torch.randn(8, 32)           # [N, D]

# A simple edge list (undirected cycle)
src = torch.arange(8)
edge_index = torch.stack([src, (src + 1) % 8])  # [2, 8]

g = Graph(node_features, edge_index)
print(g)  # Graph(num_nodes=8, num_edges=8, feature_shape=(32,))

layer = LinearMessagePassing(in_shape=(32,), out_shape=(64,))
out = layer(g.node_features, g.edge_index)   # [8, 64]
out.sum().backward()                         # differentiable
```

## Spatial node features — image patches

TGraphX's distinctive feature: keeping `[C, H, W]` spatial structure intact
through message passing.

```python
import torch
from tgraphx import Graph, ConvMessagePassing

# Six nodes, each with a [16-channel, 8×8] spatial feature map
node_features = torch.randn(6, 16, 8, 8)

# Directed cycle: 0→1→2→3→4→5→0
src = torch.arange(6)
edge_index = torch.stack([src, (src + 1) % 6])    # [2, 6]

g = Graph(node_features, edge_index)               # validated eagerly
print(g)   # Graph(num_nodes=6, num_edges=6, feature_shape=(16, 8, 8))

layer = ConvMessagePassing(in_shape=(16, 8, 8), out_shape=(32, 8, 8))
out = layer(g.node_features, g.edge_index)         # [6, 32, 8, 8]
out.sum().backward()                               # fully differentiable
```

## Build a graph from a builder

```python
from tgraphx import build_grid_graph

edge_index = build_grid_graph(3, 3, directed=False, self_loops=True)
# [2, 33]: 24 neighbour edges + 9 self-loops on a 3×3 grid
```

## Use the factory API

```python
from tgraphx import build_model

model = build_model(
    task="graph_classification",
    layer="gat",
    in_shape=(8, 4, 4),
    hidden_shape=(16, 4, 4),
    num_layers=2,
    num_classes=5,
    heads=2,
)
out = model(x, edge_index, batch=batch)   # [num_graphs, 5]
```

## Config-based construction

```python
from tgraphx import build_model_from_config

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
```

## Save a checkpoint

```python
from tgraphx import save_checkpoint, load_checkpoint

save_checkpoint(model, optimizer, epoch=10, path="run/epoch10.pt")
epoch = load_checkpoint(model, optimizer, path="run/epoch10.pt")
# Default: safe deserialization (weights_only=True)
```

## Log metrics

```python
from tgraphx.tracking import CSVLogger

with CSVLogger("runs/my_run") as logger:
    for epoch in range(20):
        # ... training ...
        logger.log(epoch=epoch, train_loss=loss, accuracy=acc)
```

## See also

- [Installation](installation.md)
- [Graph API](graph_basics.md)
- [Graph Builders](graph_builders.md)
- [Factories](factories.md)
- [Training Utilities](training_utilities.md)
- [Dashboard](dashboard.md)
- [Limitations](limitations.md)

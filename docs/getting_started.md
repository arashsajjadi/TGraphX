# Getting started with TGraphX

## Install

```bash
pip install tgraphx
```

Optional extras:

```bash
pip install "tgraphx[tracking]"     # TensorBoard
pip install "tgraphx[mlflow]"       # MLflow
pip install "tgraphx[monitoring]"   # psutil + pynvml (dashboard hardware panel)
pip install "tgraphx[pyg]"          # PyG dataset adapter
pip install "tgraphx[ogb]"          # OGB dataset adapter
pip install "tgraphx[pillow]"       # image-folder dataset
pip install "tgraphx[dev]"          # pytest + build + twine
```

## Quickstart: vector node features

```python
import torch
from tgraphx import Graph, LinearMessagePassing

x = torch.randn(8, 32)
edge_index = torch.stack([torch.arange(8), (torch.arange(8) + 1) % 8])
g = Graph(x, edge_index)

layer = LinearMessagePassing(in_shape=(32,), out_shape=(64,))
out = layer(g.node_features, g.edge_index)   # [8, 64]
out.sum().backward()
```

## Quickstart: spatial `[C, H, W]` node features

```python
import torch
from tgraphx import Graph, ConvMessagePassing

N, C, H, W = 6, 16, 8, 8
g = Graph(torch.randn(N, C, H, W),
          torch.stack([torch.arange(N), (torch.arange(N) + 1) % N]))

layer = ConvMessagePassing(in_shape=(C, H, W), out_shape=(32, H, W))
out = layer(g.node_features, g.edge_index)   # [6, 32, 8, 8]
out.sum().backward()
```

## Dataset registry

```python
from tgraphx.datasets import list_datasets, get_dataset

print(list_datasets())
ds = get_dataset("synthetic:patch_graph", num_graphs=32, seed=0)
g = ds[0]   # Graph(node_features=[P,C,ph,pw], edge_index=[2,E], graph_label=tensor)
```

## Experiment manager

```python
from tgraphx.experiments import Runner, load_config

cfg = load_config("examples/configs/synthetic_patch_graph.yaml")
runner = Runner(cfg)
history = runner.fit()
```

Or from the command line:

```bash
tgraphx-train examples/configs/synthetic_patch_graph.yaml
tgraphx-report runs/
```

## Next steps

* [tutorials.md](tutorials.md) — 10-minute tutorial path
* [datasets.md](datasets.md) — dataset registry, synthetic datasets, adapters
* [experiments.md](experiments.md) — full experiment-manager reference
* [explainability.md](explainability.md) — saliency, IG, edge attribution
* [dashboard.md](dashboard.md) — local training dashboard

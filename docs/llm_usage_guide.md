# TGraphX LLM / Code Assistant Usage Guide

This guide helps LLMs and code assistants generate correct TGraphX code on the
first attempt.

---

## Canonical imports

```python
# Minimum imports for node classification:
from tgraphx import Graph, ConvMessagePassing, NeighborLoader, GraphMiniBatch

# Training utilities:
from tgraphx import set_seed
from tgraphx.reproducibility import set_seed

# Easy mode (no-boilerplate):
import tgraphx as tgx
tgx.easy.train_node_classifier(...)

# KG:
from tgraphx.kg import KnowledgeGraph, TransEModel, KGTrainer, KGTrainingConfig

# Generation:
from tgraphx import run_graph_generation, list_graph_generation_methods

# RL:
from tgraphx import run_graph_rl, list_graph_rl_algorithms, make_graph_env

# Mining:
from tgraphx.mining import analyze_graph

# Discovery:
import tgraphx as tgx
tgx.easy.list_tasks()
tgx.easy.list_models("node_classification")
tgx.easy.list_samplers()
```

---

## Graph constructor

```python
# Correct (labels via y=):
g = Graph(node_features=x, edge_index=edge_index, y=y)

# Correct (labels via labels=):
g = Graph(node_features=x, edge_index=edge_index, labels=y)

# Correct (labels via node_labels=):
g = Graph(node_features=x, edge_index=edge_index, node_labels=y)

# Correct (edge features via edge_attr=):
g = Graph(node_features=x, edge_index=edge_index, edge_attr=ef)

# Old positional form (still valid):
g = Graph(x, edge_index)

# Access:
g.x              # → node_features
g.y              # → node_labels (None if not set)
g.edge_attr      # → edge_features (None if not set)
g.num_classes    # → int (inferred from integer labels)
g.num_nodes      # → int
g.num_edges      # → int
g.device         # → torch.device
g.has_labels()   # → bool
g.get_labels()   # → Tensor or raises helpful error
g.with_labels(y) # → new Graph with y set
```

---

## NeighborLoader batch contract

```python
loader = NeighborLoader(g, fanouts=[15, 10], batch_size=64, seed=42)

for batch in loader:
    # Subgraph attributes:
    batch.node_features      # [N_sub, *] subgraph features
    batch.x                  # alias
    batch.edge_index         # [2, E_sub]
    batch.edge_attr          # [E_sub, *] or None
    batch.num_nodes          # int
    batch.num_edges          # int

    # Seed node access:
    batch.seed_y             # [K] labels for seed nodes
    batch.seed_labels        # alias
    batch.seed_node_ids      # [K] global node IDs
    batch.seed_local_indices # [K] local positions in subgraph
    batch.batch_size         # K (int)

    # Critical: extract logits for seed nodes only:
    logits = model(batch.node_features, batch.edge_index)  # [N_sub, C]
    seed_logits = batch.seed_logits(logits)                 # [K, C]
    loss = F.cross_entropy(seed_logits, batch.seed_y)

    # Or use batch.loss() shortcut:
    loss = batch.loss(logits)

    # Device:
    batch.to("cuda")

# Legacy: tuple unpacking still works (backward compat):
for subgraph, seed_ids in loader:
    subgraph.node_features  # Graph object
    seed_ids                # LongTensor[K]
```

---

## ConvMessagePassing shape contract

```python
# Input: [N, C_in, H, W] → Output: [N, C_out, H, W]
# Spatial dimensions H, W are PRESERVED. Only channels change.
conv = ConvMessagePassing(in_shape=(C_in, H, W), out_shape=(C_out, H, W))

# To change spatial dims, use explicit pooling:
pool = nn.AdaptiveAvgPool2d((1, 1))
head = nn.Linear(C_out, num_classes)

# Typical two-layer classifier:
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvMessagePassing((C, H, W), (16, H, W))
        self.conv2 = ConvMessagePassing((16, H, W), (16, H, W))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(16, num_classes)

    def forward(self, x, edge_index):
        z = self.conv1(x, edge_index).relu()
        z = self.conv2(z, edge_index).relu()
        return self.head(self.pool(z).flatten(1))
```

---

## Easy mode (recommended for beginners)

```python
import tgraphx as tgx

# Synthetic data:
data = tgx.easy.synthetic_tensor_node_classification(
    num_nodes=1000, node_shape=(16, 8, 8), num_classes=10, seed=42,
)

# Train:
result = tgx.easy.train_node_classifier(
    data, model="tensor_gcn", sampler="neighbor",
    fanouts=[15, 10], batch_size=64, epochs=5, seed=42,
)

# Results:
result.metrics         # {'loss': ..., 'accuracy': ...}
result.history         # list of per-epoch dicts
result.model           # nn.Module
result.graph           # Graph
result.config          # resolved config with all defaults
result.summary()       # print summary
result.to_dict()       # JSON-serialisable

# Discovery:
tgx.easy.list_tasks()
tgx.easy.list_models("node_classification")
tgx.easy.list_samplers()
tgx.easy.doctor()
```

---

## Common mistakes to avoid

```python
# WRONG: slicing logits by batch_size (seed nodes may not be first)
loss = F.cross_entropy(logits[:batch_size], batch.seed_y)

# CORRECT:
loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)

# WRONG: out_shape with different spatial dims (spatial preserved in ConvMP)
conv = ConvMessagePassing(in_shape=(C, 8, 8), out_shape=(16, 4, 4))  # raises

# CORRECT (spatial dims must match):
conv = ConvMessagePassing(in_shape=(C, 8, 8), out_shape=(16, 8, 8))
# Then pool to reduce spatial:
pool = nn.AdaptiveAvgPool2d((1, 1))

# WRONG: graph without y then accessing batch.seed_y
g = Graph(node_features=x, edge_index=ei)  # no labels
# batch.seed_y → raises ValueError with fix instructions

# CORRECT:
g = Graph(node_features=x, edge_index=ei, y=y)
```

---

## API cheatsheet index

See also: `docs/api_cheatsheet.json` for a machine-readable schema.

---

## Task → Module mapping

| Task | Module | Key function |
|------|--------|--------------|
| Node classification | `tgraphx` | `Graph`, `NeighborLoader`, `ConvMessagePassing` |
| Graph classification | `tgraphx` | `GraphLoader`, `GraphBatch`, `global_mean_pool` |
| Link prediction | `tgraphx` | `LinkNeighborLoader` |
| KG completion | `tgraphx.kg` | `KnowledgeGraph`, `TransEModel`, `KGTrainer` |
| Graph generation | `tgraphx.generation` | `run_graph_generation` |
| Evolutionary opt. | `tgraphx.evolutionary` | `run_evolutionary_optimization` |
| Graph RL | `tgraphx.rl` | `run_graph_rl` |
| Graph mining | `tgraphx.mining` | `analyze_graph` |
| Explainability | `tgraphx.explain` | `node_feature_saliency`, `integrated_gradients` |
| Experiment tracking | `tgraphx.experiments` | `ExperimentConfig`, `Runner` |
| Dashboard | `tgraphx.dashboard` | `tgraphx-dashboard` CLI |

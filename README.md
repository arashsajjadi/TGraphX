<p align="center">
  <img src="TGRAPHX.png" alt="TGraphX logo" width="220">
</p>

<h1 align="center">TGraphX</h1>

<p align="center">
  <strong>Tensor-native graph learning, mining, knowledge graphs, generation, evolutionary optimization, reinforcement learning, and dashboard-ready research workflows — in PyTorch.</strong>
</p>

<p align="center">
  <a href="https://github.com/arashsajjadi/TGraphX/actions/workflows/tests.yml"><img src="https://github.com/arashsajjadi/TGraphX/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://pypi.org/project/tgraphx/"><img src="https://img.shields.io/pypi/v/tgraphx.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-1.13%2B-orange.svg" alt="PyTorch 1.13+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

<p align="center">
  Tensor-aware GNNs &nbsp;·&nbsp; Graph Mining &nbsp;·&nbsp; Knowledge Graphs &nbsp;·&nbsp; Graph Generation &nbsp;·&nbsp;
  Evolutionary Optimization &nbsp;·&nbsp; Graph RL &nbsp;·&nbsp; Dashboard &nbsp;·&nbsp; Easy Mode &nbsp;·&nbsp; Benchmarks &nbsp;·&nbsp; Tutorials
</p>

---

**Preprint:** [TGraphX: Tensor-Aware Graph Neural Network for Multi-Dimensional Feature Learning](https://arxiv.org/abs/2504.03953) · *Sajjadi & Eramian, arXiv 2025*

TGraphX is a **tensor-native graph intelligence framework** for research workflows that combine graph learning, graph mining, knowledge graphs, graph generation, evolutionary optimization, graph reinforcement learning, reproducibility, and dashboard-ready reporting — all in PyTorch, with no mandatory external dependencies.

It preserves **multi-dimensional node/edge features** (`[C, H, W]`, `[C, D, H, W]`, `[D]`) through every message-passing step, supports scalable mini-batch samplers (GraphSAINT, Cluster-GCN), multimodal tensor-aware knowledge graphs, 13 graph RL algorithms, classical and neural graph generation, multi-objective evolutionary optimization, a local dashboard with offline HTML export, sklearn-like estimators, and a full benchmark + tutorial suite.

**Quick links:** [Graph algorithms](docs/graph_algorithms.md) · [Graph mining](docs/graph_mining.md) · [Tensor GNNs](docs/vector_gnn.md) · [Sampling](docs/graphsaint.md) · [FeatureStore](docs/feature_store.md) · [KG](docs/knowledge_graphs.md) · [Dashboard](docs/dashboard.md) · [Easy Mode](docs/easy_mode.md) · [LLM guide](docs/llm_usage_guide.md)

---

## Install

```bash
pip install tgraphx
```

Optional extras (all lazy-imported, none required for the base package):

```bash
pip install "tgraphx[tracking]"     # TensorBoard
pip install "tgraphx[mlflow]"       # MLflow
pip install "tgraphx[monitoring]"   # psutil + pynvml (dashboard hardware panel)
pip install "tgraphx[pyg]"          # PyTorch Geometric dataset adapter
pip install "tgraphx[ogb]"          # OGB dataset adapter
pip install "tgraphx[pillow]"       # ImageFolder dataset
pip install "tgraphx[dev]"          # pytest + build + twine
```

DGL has platform-sensitive wheels and is not packaged as a TGraphX extra. Follow the [DGL install guide](https://www.dgl.ai/pages/start.html); the DGL adapter is available either way once DGL is on the import path.

For a custom PyTorch build (CPU-only or specific CUDA), install PyTorch first, then TGraphX:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install tgraphx
```

---

## Easy mode — start without boilerplate

For beginners, Colab users, and LLM-generated code, the `tgraphx.easy` namespace
provides zero-boilerplate workflows:

```python
import tgraphx as tgx

data = tgx.easy.synthetic_tensor_node_classification(
    num_nodes=1000, node_shape=(16, 8, 8), num_classes=10, seed=42,
)
result = tgx.easy.train_node_classifier(
    data, model="tensor_gcn", sampler="neighbor", epochs=5, seed=42,
)
print(result.metrics)
result.summary()
```

Advanced users can always drop down to PyTorch: `result.model`, `result.graph`,
`result.loader`, `result.optimizer` are all standard PyTorch objects.

→ [docs/easy_mode.md](docs/easy_mode.md) · [docs/llm_usage_guide.md](docs/llm_usage_guide.md) · [examples/easy_tensor_node_classification_no_torch.py](examples/easy_tensor_node_classification_no_torch.py)

---

## v1.4.1: final usability hardening

v1.4.1 adds one-call entry points for common workflows. All v1.4.0 and
v1.3.x syntax is preserved.

**Tensor node classification in one call.**
```python
result = tgx.classify_nodes(
    x=images, edge_index=edge_index, labels=y,
    model="tensor_gcn", seed=42, device="auto",
)
print(result.metrics["val_accuracy"])
```

**KG link prediction in one call.**
```python
result = tgx.kg_completion(
    triples=triples, num_entities=N, num_relations=R,
    model="transe", seed=42,
)
```

**Graph construction from any format.**
```python
g = tgx.make_graph(x=x, edges=[(0,1),(1,2)], labels=y)
g = tgx.make_graph(x=x, adjacency=adj)
g = tgx.make_graph(networkx_graph=G)
```

**Graph generation, optimization, and RL in one call.**
```python
g = tgx.generate_graph("ba", num_nodes=100, m=3, node_shape=(8,), seed=42)

result = tgx.optimize_graph(objective="connectivity", algorithm="ga", num_nodes=30, seed=42)

rl = tgx.train_graph_rl(env="maxcut", algorithm="random", episodes=5, seed=42)
```

**Troubleshooting.**
```python
print(tgx.explain_error(error))  # returns actionable guidance
```

**Package readiness.**
```bash
python -m tgraphx readiness
python -m tgraphx list-datasets
```

---

## v1.4.0: user-friendly tensor-native workflows

v1.4.0 keeps every canonical API stable while adding **mathematically-safe**
aliases for PyG, NetworkX, dataset, graph-construction, reproducibility,
and dashboard patterns. Unsupported shortcuts fail with actionable errors
rather than silent surprises.

**Graph construction.**
```python
# Before: many helper imports, careful shape work
# After: one line, tensor-native preserved
g = tgx.Graph(x=images, edge_index=edge_index, labels=y).to("cuda")
tgx.validate_graph(g, strict=True)
```

**Datasets.**
```python
# Before: torchvision/PyG-specific incantations
# After: friendly aliases route to TGraphX bridges
dataset = tgx.load_dataset("cifar10_patch", download=True, patch_size=8)
```

**Edge / kNN / patch construction.**
```python
edge_index = tgx.knn_graph(x, k=10, metric="cosine", make_symmetric=True)
patches, ei = tgx.image_to_patch_graph(image, patch_size=8)
proto, proto_edges, all_x = tgx.build_prototype_graph(x, y, train_mask, num_classes=10)
```

**Reproducible workflow.**
```python
# Before: random/numpy/torch seeds + cudnn flags + run_metadata bookkeeping
# After: one context manager
with tgx.reproducible(seed=42, deterministic=True):
    result = tgx.workflow(task="node_classification", fast_mode=True)
```

**Native tensor graph save/load (GraphML cannot store rank-4 tensors).**
```python
g.save("my_tensor_graph.tgx")
g = tgx.Graph.load("my_tensor_graph.tgx")
```

**Dashboard audit.**
```python
tgx.dashboard_audit("runs/advanced_notebooks/31_mnist")
# → {"ok": True, "files_present": [...], "issues": []}
```

**Functionality (not throughput) comparison.**
```python
out = tgx.compare(
    workflows=[{"name": "tensor", "task": "node_classification"},
               {"name": "mining", "task": "graph_mining"}],
    fast_mode=True, seed=42,
)
```

**Migration aliases from PyG / NetworkX / PyKEEN.**
```python
# PyG-style
g.x  # node_features
g.y  # node_labels
g.edge_attr
g.num_node_features

# NetworkX-style
g.number_of_nodes()
g.number_of_edges()
Graph.from_networkx(G)
g.to_networkx()

# PyKEEN-style
kg.triples
kg.num_entities
kg.num_relations
KnowledgeGraph.from_hrt(heads, rels, tails)
```

**Public API stability registry.**
```python
tgx.api_status("Graph")        # → "stable"
tgx.api_status("workflow")     # → "beta"
tgx.api_status("Graff")        # → KeyError with closest-match suggestion
tgx.public_api()               # → {"stable": [...], "beta": [...], "experimental": [...]}
```

All v1.4.0 aliases preserve tensor shape, dtype, device, autograd, masks, and
labels. Unsupported shortcuts (e.g. unknown task names) raise `ValueError` /
`KeyError` with closest-match suggestions. The full audit lives in
[docs/user_friendly_syntax_audit.md](docs/user_friendly_syntax_audit.md).

---

## If you want to...

| Goal | Use this | Minimal example | Tutorial |
|------|----------|-----------------|---------|
| Train on image-like node tensors | `Graph + ConvMessagePassing + NeighborLoader` | [below](#train-a-gnn) | [tensor_node_classification_neighbor_loader.py](tutorials/tensor_node_classification_neighbor_loader.py) |
| Train on vector node features | `Graph + GCNConv` | [below](#train-a-gnn) | [01_vector_node_classification.py](examples/01_vector_node_classification.py) |
| Zero-boilerplate training | `tgx.easy.train_node_classifier(...)` | [easy mode](#easy-mode--start-without-boilerplate) | [easy_mode.md](docs/easy_mode.md) |
| Sample large graphs | `NeighborLoader / GraphSAINT / Cluster-GCN` | [below](#scalable-mini-batch-training) | [docs/graphsaint.md](docs/graphsaint.md) |
| Mine graph structure | `analyze_graph / graph_summary` | [below](#analyse-a-graph) | [docs/graph_mining.md](docs/graph_mining.md) |
| Train a KG model | `KGTrainer + TransEModel` | [below](#knowledge-graph-learning) | [docs/knowledge_graphs.md](docs/knowledge_graphs.md) |
| Build multimodal KG | `KnowledgeGraph(entity_features=...)` | [example](examples/kg_multimodal_tensor_features_demo.py) | [docs/kg_multimodal_tensor_features.md](docs/kg_multimodal_tensor_features.md) |
| Generate graphs | `run_graph_generation(method=...)` | [below](#one-liner-graph-generation) | [tutorials/graph_generation_quickstart.py](tutorials/graph_generation_quickstart.py) |
| Optimize graphs | `run_evolutionary_optimization(...)` | [below](#graph-generation-and-evolutionary-optimization) | [tutorials/evolutionary_optimization_quickstart.py](tutorials/evolutionary_optimization_quickstart.py) |
| Run graph RL | `run_graph_rl(algorithm=..., env=...)` | [below](#one-liner-graph-rl) | [tutorials/graph_rl_quickstart.py](tutorials/graph_rl_quickstart.py) |
| Open dashboard | `tgraphx-dashboard` | [docs/dashboard.md](docs/dashboard.md) | — |
| Check installation | `python -m tgraphx doctor` | — | — |
| Discover capabilities | `tgx.easy.list_tasks()` | — | [docs/easy_mode.md](docs/easy_mode.md) |
| LLM code generation | Follow the contract in the guide | — | [docs/llm_usage_guide.md](docs/llm_usage_guide.md) |

---

## 60-second quickstart

**Vector node features:**

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

**Spatial `[C, H, W]` node features — preserved through message passing:**

```python
import torch
from tgraphx import Graph, ConvMessagePassing

N, C, H, W = 6, 16, 8, 8
g = Graph(torch.randn(N, C, H, W), torch.stack(
    [torch.arange(N), (torch.arange(N) + 1) % N]))

layer = ConvMessagePassing(in_shape=(C, H, W), out_shape=(32, H, W))
out = layer(g.node_features, g.edge_index)   # [6, 32, 8, 8]
out.sum().backward()
```

## Notebook gallery

A curated set of 30 CPU-runnable notebooks covers Easy Mode, tensor-native
message passing, sampling, knowledge graphs, classical graph generation,
evolutionary optimization, graph reinforcement learning, graph IO, graph
mining, benchmarking, and reproducibility.

Browse the full index → **[docs/colab_gallery.md](docs/colab_gallery.md)**

Each notebook focuses on one concept, uses synthetic or toy data, and runs in
under two minutes on CPU. Notebooks are hosted on Google Drive; open the file
in Drive and choose **Open with → Google Colaboratory**, or download and run
locally in Jupyter.

---

## Capability map

| Area | Capabilities | Stability | Start here |
|------|-------------|-----------|------------|
| **Tensor-aware graphs** | vector / image / volume node features, edge features, graph metadata, PyG-compatible `.x`/`.y`/`.edge_attr` aliases | Beta | [docs/graph_basics.md](docs/graph_basics.md) |
| **Graph algorithms** | BFS/DFS, shortest paths, MST, max-flow, matching, coloring | Beta | [examples/graph_paths_algorithms_demo.py](examples/graph_paths_algorithms_demo.py) |
| **Graph mining** | motifs, centrality, spectral analysis, WL features, similarity | Beta | [docs/graph_mining.md](docs/graph_mining.md) |
| **GNN layers** | GCN/SAGE/GAT/GIN/GATv2/APPNP vector layers; tensor GAT/SAGE/GIN/ConvMP | Beta | [docs/vector_gnn.md](docs/vector_gnn.md) |
| **Sampling & loaders** | `NeighborLoader` → `GraphMiniBatch` (seed-node aware), `LinkNeighborLoader`, `GraphLoader`, GraphSAINT, Cluster-GCN | Beta | [docs/neighbor_loader.md](docs/neighbor_loader.md) |
| **Feature store** | In-memory & memmap tensor feature storage; NeighborLoader integration | Beta | [docs/feature_store.md](docs/feature_store.md) |
| **Sparse backend** | CSR/CSC, coalesce, segment ops, optional torch_scatter acceleration | Beta | [docs/backends.md](docs/backends.md) |
| **Knowledge graphs** | triples, TransE/DistMult/ComplEx/RotatE, filtered ranking, KG+RGCN, multimodal entity features, temporal KG, reasoning | Beta | [docs/knowledge_graphs.md](docs/knowledge_graphs.md) |
| **Classical graph generation** | ER/BA/SBM/temporal/typed generators, generation metrics (validity/uniqueness/novelty/diversity/MMD) | Beta | [docs/graph_generation.md](docs/graph_generation.md) |
| **Neural graph generation** | VGAE generation, autoregressive generation, transformer generation | Experimental | [docs/neural_graph_generation.md](docs/neural_graph_generation.md) |
| **Evolutionary optimization** | genetic algorithm, simulated annealing, NSGA-II multi-objective, mutation/crossover/selection | Beta | [docs/evolutionary_graph_optimization.md](docs/evolutionary_graph_optimization.md) |
| **Graph reinforcement learning** | RL environments, policy/Q/value networks, 13 algorithms (REINFORCE/A2C/DQN/PPO/TD3/SAC/…) | Experimental | [docs/graph_reinforcement_learning.md](docs/graph_reinforcement_learning.md) |
| **RL baselines** | Random, Greedy — environment-agnostic no-learn baselines | Beta | [docs/graph_rl_algorithms.md](docs/graph_rl_algorithms.md) |
| **Easy Mode** | `tgraphx.easy` — zero-boilerplate workflows, discovery, diagnostics, EasyResult | Beta | [docs/easy_mode.md](docs/easy_mode.md) |
| **Hypergraphs** | incidence matrix, clique/star expansion | Experimental | [examples/graph_algorithms_advanced_demo.py](examples/graph_algorithms_advanced_demo.py) |
| **Heterogeneous graphs** | RGCN, HAN, HGT; typed neighbor sampling | Experimental | [docs/hetero_gnns.md](docs/hetero_gnns.md) |
| **Temporal graphs** | TGNMemory, TGATConv, time encoding, temporal splits | Experimental | [docs/temporal_graph_learning.md](docs/temporal_graph_learning.md) |
| **Graph autoencoders** | GAE / VGAE; dot-product & MLP edge decoders | Experimental | [examples/vgae_link_prediction_demo.py](examples/vgae_link_prediction_demo.py) |
| **Representation learning** | Node2Vec, DeepWalk, graph embeddings | Beta | [examples/node2vec_demo.py](examples/node2vec_demo.py) |
| **Semi-supervised** | label propagation, masks, graph splits | Beta | [docs/graph_mining.md](docs/graph_mining.md) |
| **Experiment manager** | YAML/JSON configs, runners, callbacks, CLI (`tgraphx-train`) | Beta | [docs/experiments.md](docs/experiments.md) |
| **Explainability** | saliency, integrated gradients, edge attribution | Beta | [docs/explainability.md](docs/explainability.md) |
| **sklearn-like API** | estimators, GraphPipeline, splits, EarlyStopping | Beta | [docs/sklearn_api.md](docs/sklearn_api.md) |
| **Calibration** | ECE, temperature scaling, reliability diagram data | Beta | `tgraphx.calibration` |
| **Dashboard** | local HTTP server, offline HTML, run artifacts, benchmark panels | Beta | [docs/dashboard.md](docs/dashboard.md) |
| **Reproducibility** | `set_seed`, deterministic mode, `reproducibility_report.json` | Beta | [docs/reproducibility.md](docs/reproducibility.md) |
| **Distributed helpers** | rank-zero utilities, DDP wrapping, shard helpers | Experimental | [docs/distributed_training.md](docs/distributed_training.md) |
| **OGB / TGB wrappers** | optional evaluators with no hidden downloads | Beta (optional) | [docs/ogb_tgb_integration.md](docs/ogb_tgb_integration.md) |
| **Tutorials** | CPU-runnable Colab-ready quickstarts for generation, evolutionary optimization, and graph RL | Stable | [tutorials/](tutorials/) |
| **Benchmarks** | 13 benchmark scripts with `--small --json` for CI-friendly validation | Stable | [benchmarks/](benchmarks/) |

---

## Choose your workflow

### Analyse a graph

```python
from tgraphx.mining import graph_summary, degree_statistics

summary = graph_summary(graph.edge_index, num_nodes=graph.num_nodes)
# {'num_nodes': ..., 'num_edges': ..., 'density': ..., 'is_directed': ...}
```

→ [docs/graph_mining.md](docs/graph_mining.md)

### Train a GNN

```python
import torch.nn.functional as F
from tgraphx import Graph, NeighborLoader
from tgraphx.reproducibility import set_seed

set_seed(42)
g = Graph(node_features=x, edge_index=edge_index, y=y)  # y= for labels
loader = NeighborLoader(g, fanouts=[15, 10], batch_size=64, seed=42)

for batch in loader:
    logits = model(batch.node_features, batch.edge_index)
    # batch.seed_logits() extracts logits for supervision nodes only.
    # batch.seed_y returns labels for those same nodes.
    loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)
    loss.backward()
```

→ [docs/neighbor_loader.md](docs/neighbor_loader.md) · [docs/vector_gnn.md](docs/vector_gnn.md)

### Scalable mini-batch training

```python
from tgraphx.graphsaint import GraphSAINTNodeSampler, GraphSAINTLoader

sampler = GraphSAINTNodeSampler(graph, budget=512, num_steps=100, seed=0)
for sub in GraphSAINTLoader(sampler, attach_norm=True):
    out = model(sub.node_features, sub.edge_index)
```

→ [docs/graphsaint.md](docs/graphsaint.md) · [docs/cluster_gcn.md](docs/cluster_gcn.md)

### Knowledge graph learning

```python
import torch
from tgraphx.kg import KnowledgeGraph, TransEModel, KGTrainer, KGTrainingConfig

triples = torch.tensor([[0, 0, 1], [1, 0, 2], [2, 1, 0]], dtype=torch.long)
kg = KnowledgeGraph(triples, num_entities=3, num_relations=2)
model = TransEModel(num_entities=kg.num_entities, num_relations=kg.num_relations, embedding_dim=32)
config = KGTrainingConfig(num_epochs=10, lr=1e-3, seed=42)
trainer = KGTrainer(model, config, kg.triples)
trainer.train()
```

→ [examples/knowledge_graph_demo.py](examples/knowledge_graph_demo.py)

### Multimodal tensor-aware knowledge graphs

TGraphX KG can represent different entity types such as image nodes, user nodes, text/document nodes, item nodes, paper nodes, method nodes, and dataset nodes. Each entity type can carry its own tensor features, while relations and triples can also carry features such as weights, timestamps, confidence, or provenance.

```
image_001 --viewedBy--> user_123
user_123  --wrote-->    text_doc_045
text_doc_045 --describes--> image_001
```

- **Image entities** can carry image tensors or precomputed image embeddings through a lightweight modality-specific projector.
- **User entities** can carry profile vectors or learned user embeddings.
- **Text entities** should currently be provided as precomputed embeddings (e.g. from a sentence encoder); raw text tokenization is not built in.
- **Modality masks** handle missing modalities gracefully, so heterogeneous graphs with partially observed features are supported.
- **Modality-specific projectors** are differentiable: gradients flow through them and the model learns from these features, not only stores them.
- Tested behaviour: image/user/text feature sensitivity is verified, gradients flow through all projectors, and a toy multimodal KG training loop shows loss decrease.

This is not a full vision-language foundation model and makes no claim to SOTA. The image projector is intentionally lightweight. For an end-to-end demo and the API reference:

→ [docs/kg_multimodal_tensor_features.md](docs/kg_multimodal_tensor_features.md) · [examples/kg_multimodal_tensor_features_demo.py](examples/kg_multimodal_tensor_features_demo.py)

### Temporal / heterogeneous graphs

```python
from tgraphx.temporal import TGNMemory, TGATConv
from tgraphx.layers.hgt import HGTConv

# TGN memory for temporal link prediction
mem = TGNMemory(num_nodes=N, memory_dim=64, message_dim=64)
mem.update(node_ids, messages, timestamps)  # raises on future-data leakage
```

→ [docs/temporal_graph_learning.md](docs/temporal_graph_learning.md) · [docs/hetero_gnns.md](docs/hetero_gnns.md)

### Graph generation and evolutionary optimization

```python
from tgraphx.generation import FeatureAwareERGraph, uniqueness_score
from tgraphx.evolutionary import GraphGenome, GeneticAlgorithmOptimizer, GeneticAlgorithmConfig, connectivity_fitness
import torch

# Generate graphs with tensor features
graphs = [FeatureAwareERGraph(n=20, p=0.3, node_feature_dim=8, seed=i) for i in range(10)]
print("Uniqueness:", uniqueness_score(graphs))

# Evolve a graph to maximize connectivity
def make_genome(n=6, seed=0):
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    return GraphGenome(edge_index=ei, num_nodes=n)

config = GeneticAlgorithmConfig(population_size=10, n_generations=20, seed=42)
result = GeneticAlgorithmOptimizer(config, connectivity_fitness).optimize([make_genome(seed=i) for i in range(10)])
print(f"Best connectivity: {result.best_fitness:.4f}")
```

→ [docs/graph_generation.md](docs/graph_generation.md) · [docs/evolutionary_graph_optimization.md](docs/evolutionary_graph_optimization.md) · [examples/classical_graph_generation_demo.py](examples/classical_graph_generation_demo.py) · [examples/evolutionary_graph_optimization_demo.py](examples/evolutionary_graph_optimization_demo.py)

### One-liner graph generation

```python
from tgraphx.generation import run_graph_generation

graphs = run_graph_generation(
    method="barabasi_albert",
    num_graphs=16,
    num_nodes=50,
    m=2,
    node_feature_dim=8,
    seed=42,
)
print(f"Generated {len(graphs.graphs)} graphs. Validity: {graphs.metrics['validity']:.2f}")
```

→ [docs/graph_generation.md](docs/graph_generation.md) · [docs/evolutionary_graph_optimization.md](docs/evolutionary_graph_optimization.md)

### Graph reinforcement learning

```python
from tgraphx.rl import GraphNavigationEnv, GraphEnvConfig, GraphPolicyNetwork, REINFORCEAgent
import torch

ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
env = GraphNavigationEnv(ei, 5, node_features=torch.randn(5, 8), target_node=4,
                         config=GraphEnvConfig(max_steps=20))

policy = GraphPolicyNetwork(node_in_dim=8, hidden_dim=32, num_actions=4)
agent = REINFORCEAgent(policy, torch.optim.Adam(policy.parameters(), lr=1e-3))

for ep in range(50):
    traj = agent.collect_episode(env, max_steps=20)
    agent.update(traj)
```

→ [docs/graph_reinforcement_learning.md](docs/graph_reinforcement_learning.md) · [docs/graph_rl_algorithms.md](docs/graph_rl_algorithms.md) · [examples/graph_reinforce_demo.py](examples/graph_reinforce_demo.py) · [examples/graph_dqn_demo.py](examples/graph_dqn_demo.py)

### One-liner graph RL

```python
from tgraphx.rl import run_graph_rl

result = run_graph_rl(
    env="graph_navigation",
    algorithm="dqn",
    episodes=50,
    seed=42,
)
print(f"Mean return: {result.metrics['mean_return']:.2f}")
```

→ [docs/graph_reinforcement_learning.md](docs/graph_reinforcement_learning.md)

### Graph RL algorithms

| Algorithm | Action type | Core idea | Stability | One-liner |
|-----------|-------------|-----------|-----------|-----------|
| Random | discrete | Uniform random sampling of valid actions | Beta | `run_graph_rl(..., algorithm="random")` |
| Greedy | discrete | Highest Q-value action, no learning | Beta | `run_graph_rl(..., algorithm="greedy")` |
| REINFORCE | discrete | Monte Carlo policy gradient + entropy | Experimental | `run_graph_rl(..., algorithm="reinforce")` |
| Actor-Critic | discrete | Synchronous actor-critic with GAE | Experimental | `run_graph_rl(..., algorithm="actor_critic")` |
| A2C | discrete | Advantage Actor-Critic with GAE | Experimental | `run_graph_rl(..., algorithm="a2c")` |
| DQN | discrete | Deep Q-Network + ε-greedy + replay | Experimental | `run_graph_rl(..., algorithm="dqn")` |
| Double DQN | discrete | Decoupled action selection/evaluation | Experimental | `run_graph_rl(..., algorithm="double_dqn")` |
| Dueling DQN | discrete | V(s) + A(s,a) − mean(A) decomposition | Experimental | `run_graph_rl(..., algorithm="dueling_dqn")` |
| PPO | discrete | Clipped surrogate objective | Experimental | `run_graph_rl(..., algorithm="ppo")` |
| DDPG | continuous | Deterministic policy + critic, soft update | Experimental | `run_graph_rl(..., algorithm="ddpg")` |
| Delayed DDPG | continuous | DDPG with delayed actor updates | Experimental | `run_graph_rl(..., algorithm="delayed_ddpg")` |
| TD3 | continuous | Twin critics + clipped target noise + delayed update | Experimental | `run_graph_rl(..., algorithm="td3")` |
| SAC | continuous | Entropy-regularized twin-critic, stochastic actor | Experimental | `run_graph_rl(..., algorithm="sac")` |

### Dashboard-ready experiments

```python
from tgraphx.mining.reports import write_graph_mining_summary
write_graph_mining_summary("logs/graph_mining_summary.json", summary)
# → python -m tgraphx.dashboard logs/
```

→ [docs/dashboard.md](docs/dashboard.md)

---

## Stability labels

| Label | Meaning |
|-------|---------|
| **Stable** | Public API will not change in v1.x; backward compatibility guaranteed |
| **Beta** | Tested and documented; API stable within the v1.x series; may expand |
| **Experimental** | Correct foundations; API or semantics may evolve in future minor releases |
| **Optional** | Requires an optional dependency or explicit `--download` |

See [docs/api_stability.md](docs/api_stability.md) for the detailed per-component stability contract.

---

## Why tensor-aware GNNs

Standard GNN frameworks expect flat vector node features. Flattening a `[C, H, W]` feature map into a `[C·H·W]` vector discards the spatial structure that makes CNNs effective. TGraphX keeps each node's representation as a tensor and applies 1×1 convolutions during message passing, so every neighbourhood aggregation step acts like a miniature CNN across neighbouring feature maps.

```
M_{i→j} = φ( X_i, X_j, E_{i→j} )
A_j     = AGG_{i ∈ N(j)} M_{i→j}
X'_j    = ψ( X_j, A_j )
```

Spatial dimensions `H` and `W` are preserved through every learned transform. Aggregations are permutation-invariant; the layers are permutation-equivariant under node relabelling (verified by `tests/test_math_invariants_v030.py`).

---

## Stable core APIs

| Component | Class | Input node shape | Notes |
|-----------|-------|-----------------|-------|
| Graph data structure | `Graph`, `GraphBatch` | any | Validated; supports `.to(device)` |
| Tensor-aware GCN-style message passing | `ConvMessagePassing` | `[N, C, H, W]` | `aggr="sum" / "mean" / "max"`; chunked forward |
| Tensor-aware multi-head GAT | `TensorGATLayer` | `[N, C, H, W]` | Softmax over incoming edges per destination per head; two-pass log-sum-exp chunked forward |
| Tensor-aware GraphSAGE | `TensorGraphSAGELayer` | `[N, C, H, W]` | Mean / max aggregation; chunked forward |
| Tensor-aware GIN / GINEConv | `TensorGINLayer` | `[N, C, H, W]` | `(1+ε)·h_j + Σ h_i`; chunked forward |
| Vector message passing | `LinearMessagePassing` | `[N, D]` | Base layer with linear projections |
| Custom-layer base class | `TensorMessagePassingLayer` | any | Override `message` / `update` |
| Vector model-zoo | `GCNConv`, `GATv2Conv`, `APPNP` | `[N, D]` | New in v0.3.0; permutation-equivariance and tiny-overfit tested |
| Pooling helpers | `global_mean_pool`, `global_sum_pool`, `global_max_pool` | `[N, *]` | Per-graph reductions |
| CNN patch encoder | `CNNEncoder` | `[N, C_in, pH, pW]` | Outputs spatial feature maps |
| Graph classification | `GraphClassifier` | `[N, C, H, W]` | Mean / sum / max readout |
| Node classification | `NodeClassifier` | `[N, D]` | Vector features |

3-D volumetric node features `[N, C, D, H, W]` are supported by `ConvMessagePassing`, `TensorGATLayer`, `TensorGraphSAGELayer`, and `TensorGINLayer` (pass `spatial_rank=3`).

### Factories

```python
from tgraphx import make_layer, build_model

layer = make_layer("gat", in_shape=(8, 4, 4), out_shape=(16, 4, 4), heads=2, residual=True)
model = build_model(
    task="graph_classification", layer="conv",
    in_shape=(3, 4, 4), hidden_shape=(8, 4, 4),
    num_layers=2, num_classes=10, pooling="mean",
)
```

| Task | `[N,D]` | `[N,C,H,W]` | `[N,C,D,H,W]` |
|------|:-:|:-:|:-:|
| `node_classification` | yes | yes | yes |
| `node_regression` | yes | yes | yes |
| `graph_classification` | yes | yes | yes |
| `graph_regression` | yes | yes | yes |
| `edge_prediction` | yes | yes (spatial pool) | yes (spatial pool) |

---

## Datasets, transforms, metrics, benchmarks

TGraphX ships a unified dataset registry, native synthetic datasets, folder-backed datasets, and optional adapters for torchvision, PyG, DGL, and OGB. **TGraphX does not redistribute third-party datasets**; adapters delegate download and parsing to the upstream library, and any download requires the user to pass `download=True` explicitly.

```python
from tgraphx.datasets import list_datasets, dataset_info, get_dataset

print(list_datasets())                        # 34 registered datasets
ds = get_dataset("synthetic:patch_graph", num_graphs=32, seed=0)
g = ds[0]
```

Cache layout: `<root>` (or `$TGRAPHX_DATA`, or `~/.cache/tgraphx/datasets`). Inspect with `tgraphx.datasets.cache_summary()`; clear with `clear_cache(...)`.

| Layer | Highlights |
|-------|-----------|
| `tgraphx.datasets` | Native synthetic + folder + torchvision/PyG/DGL/OGB adapters; safe atomic downloads with SHA-256 checksums and path-traversal-blocked archive extraction |
| `tgraphx.transforms` | `Compose`, `AddSelfLoops`, `RemoveSelfLoops`, `ToUndirected`, `NormalizeFeatures`, `StandardizeFeatures`, `AddDegreeFeatures`, `RandomNodeSplit`, `RandomLinkSplit`, `AddDegreeEncoding`, `AddLaplacianEigenvectors`, `PatchifyImage`, `BuildGridGraph`, … |
| `tgraphx.metrics` | Pure-PyTorch `accuracy`, `top_k_accuracy`, `precision_recall_f1`, `classification_report`, `mae`, `mse`, `rmse`, `r2_score`, `hits_at_k`, `mean_reciprocal_rank`, `ndcg_at_k`, `roc_auc`, `average_precision` |
| `benchmarks/` | `benchmark_dataset_loading.py`, `benchmark_training_synthetic.py`, `benchmark_tensor_vs_flatten.py`, `benchmark_transforms.py`, `benchmark_metrics.py`, `benchmark_layers.py`, `benchmark_graph_builders.py`, `benchmark_sampling.py`. Every benchmark accepts `--small` and `--output JSON`. |

Detailed write-ups: [docs/datasets.md](docs/datasets.md), [docs/transforms.md](docs/transforms.md), [docs/metrics.md](docs/metrics.md), [docs/benchmarks.md](docs/benchmarks.md), [docs/dataset_license_policy.md](docs/dataset_license_policy.md).

Importing `tgraphx`, `tgraphx.datasets`, `tgraphx.transforms`, `tgraphx.metrics`, `tgraphx.experiments`, or `tgraphx.explain` does **not** import torch_geometric / dgl / ogb.

---

## Experiment manager

`tgraphx.experiments` (new in v0.3.0) is a lightweight, dashboard-compatible experiment manager. YAML / JSON configs are validated against an explicit schema (no `eval`, no `exec`). Every run writes its artefacts under `run_dir` only:

```
runs/<run_name>/<timestamp>/
├── run_metadata.json           # run name, status, device, seed, version
├── experiment_config.json      # exact config copy
├── experiment_summary.json     # epochs, best metric, final loss
├── metrics.csv                 # dashboard-compatible
└── checkpoints/{best,latest}.pt
```

```python
from tgraphx.experiments import Runner, load_config

cfg = load_config("examples/configs/synthetic_patch_graph.yaml")
runner = Runner(cfg)
history = runner.fit()
```

Or via the CLI (registered as console scripts):

```bash
tgraphx-train  examples/configs/synthetic_patch_graph.yaml
tgraphx-grid   examples/configs/grid_sweep.yaml
tgraphx-report runs/                  # → Markdown summary
```

Built-in callbacks: `EarlyStopping`, `ModelCheckpoint`, `CSVLoggerCallback`, `LearningRateLogger`. Multi-seed and grid sweeps via `GridRunner`. The dashboard reads the same files when you run `tgraphx-dashboard --logdir runs/<run_name>`.

See [docs/experiments.md](docs/experiments.md) for the full schema.

---

## Explainability

`tgraphx.explain` (new in v0.3.0) provides diagnostic explainability tools. They run on CPU by default, do not retain autograd graphs, and never make causal claims about a model's predictions.

```python
from tgraphx.explain import (
    node_feature_saliency, integrated_gradients,
    edge_perturbation_attribution, attention_to_edge_scores,
    patch_saliency_to_image_grid,
    export_explanation_metadata, export_edge_scores_csv,
)

sal      = node_feature_saliency(model, graph, target=label)
ig       = integrated_gradients(model, graph, target=label, steps=16)
edge_imp = edge_perturbation_attribution(model, graph, target=label, max_edges=64)

# For TensorGATLayer outputs.
out, attn = layer(x, edge_index, return_attention=True)
edge_scores = attention_to_edge_scores(attn, edge_index, head_reduce="mean")

# Patch-level heatmap from grid_shape metadata.
heatmap = patch_saliency_to_image_grid(sal, grid_shape=graph.metadata["grid_shape"])

# Export artefacts the dashboard can render (explicit paths only).
export_explanation_metadata("runs/demo/explanation_metadata.json",
                            method="saliency", target=int(label))
export_edge_scores_csv("runs/demo/explanation_edges.csv",
                       graph.edge_index, edge_scores, top_k=20)
```

See [docs/explainability.md](docs/explainability.md) for examples and limits.

---

## Optional integrations

| Adapter | Install | Notes |
|---------|---------|-------|
| Native synthetic / folder datasets | (none) | Deterministic, learnable, no network |
| Torchvision-backed datasets | already a TGraphX base dependency | `MNIST`, `CIFAR-10/100`, `SVHN`, `STL-10`, `FakeData`, … converted to patch graphs |
| PyTorch Geometric | `pip install "tgraphx[pyg]"` | `Planetoid`, `TUDataset`, generic adapter; data-format converters only |
| DGL | follow upstream install | citation graphs, generic adapter |
| OGB | `pip install "tgraphx[ogb]"` | node / link / graph property prediction wrappers + `OGBEvaluatorWrapper` |
| MLflow | `pip install "tgraphx[mlflow]"` | `MLflowLogger`, lazy import |
| TensorBoard | `pip install "tgraphx[tracking]"` | `TensorBoardLogger`, lazy import |
| Hardware monitoring | `pip install "tgraphx[monitoring]"` | `psutil` + `pynvml` for the dashboard's hardware panel |

TGraphX is interoperable with the PyG / DGL / OGB ecosystems through small data-format converters; it is not a drop-in replacement for either framework.

---

## Backend and platform support

| Platform | Forward | torch.compile | AMP | CI coverage |
|----------|---------|--------------|-----|-------------|
| Linux + CPU | yes | yes | bfloat16 (recommended) | Full Ubuntu CI on Python 3.10 / 3.11 / 3.12 |
| NVIDIA CUDA | yes | yes | float16 / bfloat16 | Local tests; no GPU runners in CI |
| Apple Silicon MPS | yes | partial | partial | macOS smoke CI (import + build) |
| Windows + CPU | yes | yes | yes | Windows smoke CI (Python 3.11) |
| macOS + CPU | yes | yes | yes | macOS smoke CI (Python 3.11) |
| Multi-GPU (DDP) | helpers | user-managed | user-managed | `tgraphx.distributed` rank-zero / barrier helpers; full DDP setup is the user's responsibility |

For the AMP recommendations, dtype-cast policy, and per-platform caveats, see [docs/performance.md](docs/performance.md). For the precise API stability classification, see [docs/api_stability.md](docs/api_stability.md).

---

## Dashboard, local-first, privacy

TGraphX ships a local-first training dashboard that reads run artefacts (`metrics.csv`, `run_metadata.json`, `experiment_config.json`, `experiment_summary.json`, `dataset_metadata.json`, `transform_metadata.json`, `metrics_summary.json`, `benchmark_results.json`, `explanation_metadata.json`, `explanation_edges.csv`, `explanation_patch_heatmap.json`, `hetero_graph_metadata.json`, `temporal_metadata.json`, `sampling_metadata.json`, `hardware_report.json`).

```bash
tgraphx-dashboard --logdir runs/demo
# → http://127.0.0.1:8765 (localhost only, no token needed)

tgraphx-dashboard --logdir runs/demo --host 0.0.0.0 --token MY_SECRET_TOKEN
tgraphx-dashboard --logdir runs/demo --export-html snapshot.html
```

Properties:

* Off by default. The dashboard is opt-in; nothing about importing TGraphX runs a server.
* Local-first. Reads files from your `--logdir`; never executes code, never loads checkpoints, never runs models.
* No telemetry, no analytics, no external CDN; the offline HTML export is fully self-contained.
* LAN access requires `--token`; localhost mode does not.
* Path-traversal protected: every file read is validated against the resolved logdir.
* Background threads are launched only when you call `launch_dashboard_background(...)` explicitly.

Helpers for writing metadata files the dashboard understands:

```python
from tgraphx import (
    write_run_metadata, write_dataset_metadata, write_transform_metadata,
    write_metrics_summary, write_benchmark_results,
    write_explanation_metadata, write_experiment_config,
    write_hardware_report, write_sampling_metadata,
    write_hetero_graph_metadata, write_temporal_metadata,
)
```

`write_graph_stats` from earlier releases is preserved.

See [docs/dashboard.md](docs/dashboard.md) for the full UI, security model, and offline export.

---

## Privacy summary

| Behaviour | Default |
|---|---|
| Telemetry / analytics | None — never |
| Remote calls at import | None |
| Dashboard | Off — launch explicitly |
| CSV / TensorBoard / MLflow logging | Off — create a logger explicitly |
| Hardware monitoring | Off — pass `include_hardware=True` to `env_report` |
| Checkpoints | Off — call `save_checkpoint` or attach `ModelCheckpoint` explicitly |
| Dataset downloads | Off — adapters require `download=True` |
| Background threads | None unless `launch_dashboard_background` is called |
| File writes | Only to paths the user provides |

No `~/.tgraphx` directory or user-level config is created.

---

## Examples

### Graph algorithms and mining
```bash
python examples/graph_paths_algorithms_demo.py      # shortest paths, MST, BFS/DFS
python examples/graph_algorithms_advanced_demo.py   # max-flow, matching, coloring
python examples/graph_mining_structural_demo.py     # motifs, centrality, WL
python examples/knowledge_graph_demo.py             # KG triples, TransE
python examples/node2vec_demo.py                    # Node2Vec embeddings
```

### Sampling and scalable training
```bash
python examples/neighbor_loader_demo.py             # NeighborLoader / LinkNeighborLoader
python examples/graphsaint_sampler_demo.py          # GraphSAINT node/edge/RW samplers
python examples/cluster_loader_demo.py              # Cluster-GCN partitioners
```

### Neural graph learning
```bash
python examples/graph_learning_demo.py              # GCN/SAGE/GAT training
python examples/vgae_link_prediction_demo.py        # GAE / VGAE link prediction
python examples/gat_chunking_demo.py                # chunked GAT forward parity
```

### Graph generation, evolutionary optimization, and RL (Experimental)
```bash
python examples/classical_graph_generation_demo.py       # ER/BA/temporal/typed generators
python examples/neural_graph_generation_demo.py          # VGAE/autoregressive/transformer
python examples/evolutionary_graph_optimization_demo.py  # GA, SA, NSGA-II
python examples/graph_rl_environments_demo.py            # all RL environments demo
python examples/graph_reinforce_demo.py                  # REINFORCE on navigation
python examples/graph_dqn_demo.py                        # DQN on graph coloring
python examples/graph_ppo_demo.py                        # PPO on navigation
python examples/graph_td3_sac_demo.py                    # TD3/SAC continuous RL
python examples/generation_rl_high_level_api_demo.py     # one-line API demo
python examples/graph_lstm_sequence_demo.py              # GraphRNN sequence model
```

### Quickstart tutorials (CPU runnable, deterministic)
```bash
python tutorials/graph_generation_quickstart.py       # ER/BA/SBM + metrics + dashboard
python tutorials/evolutionary_optimization_quickstart.py # GA/SA/NSGA-II + Pareto front
python tutorials/graph_rl_quickstart.py               # random/DQN/PPO/TD3/SAC comparison
```

### Dashboard, experiments, sklearn API
```bash
python examples/training_with_dashboard.py          # fit() + CSVLogger → dashboard
python examples/sklearn_style_graph_pipeline_demo.py # estimator/pipeline API
python examples/distributed_smoke.py --world-size 2 --subprocess-pair  # DDP smoke
```

### All fast examples
```bash
python examples/run_all_fast_examples.py            # runs 75+ demos in sequence
```

### Validation scripts

A small set of stand-alone validation scripts proves that the surfaces
TGraphX advertises actually work end-to-end:

| Script | Purpose |
|--------|---------|
| `examples/device_validation.py` | Runs vector + spatial layer smokes on CPU / CUDA / MPS, optionally under autocast (`--amp`). Emits a JSON report. |
| `examples/dashboard_artifact_validation.py` | Writes every supported dashboard metadata file, exports an offline HTML snapshot, and checks for CDN / token / `eval` leaks. |
| `examples/experiment_end_to_end_validation.py` | Trains a tiny synthetic experiment, asserts dashboard files exist, resumes from a checkpoint. |
| `examples/explainability_end_to_end_validation.py` | Trains, runs saliency / integrated gradients / edge perturbation, exports dashboard-readable explanation artefacts. |
| `examples/public_datasets/*.py` | Manual, opt-in scripts that exercise the torchvision / PyG / DGL / OGB adapters against real upstream loaders. They require `--download`, cap dataset size, skip cleanly if the optional package is missing, and never run in CI. |

See [docs/public_dataset_validation.md](docs/public_dataset_validation.md)
and [docs/device_validation.md](docs/device_validation.md) for the
exact invocations and policy.

---

## Maturity and scope

TGraphX v1.0 is a validated research framework with tested foundations across tensor-aware GNNs, graph algorithms, graph mining, scalable sampling, sparse utilities, feature stores, dashboard reporting, knowledge graphs, hypergraphs, temporal graphs, heterogeneous GNNs, sklearn-style workflows, calibration, benchmarks, and reproducibility tooling. Components are labeled Beta or Experimental based on evidence — see [docs/api_stability.md](docs/api_stability.md) for the full contract.

Current maturity boundaries:

- **Distributed utilities** — DDP-aware helpers and a validated two-process CPU/gloo smoke path; broader multi-node training remains a roadmap item.
- **GraphSAINT / Cluster-GCN** — available as Beta foundations with benchmark scripts; production-scale benchmarks should continue to expand.
- **HAN / HGT / TGN / TGAT** — Experimental foundations with unit, toy-overfit, and no-leakage validation; broader reference-parity comparisons remain future work.
- **OGB / TGB integrations** — optional wrappers around official evaluators; require explicit user setup.
- **Dense graph operations** — some builders (`kNN`, `radius`, `IoU`, fully-connected) are O(N²) and emit warnings on large N; the spectral partitioner is O(N³) and is restricted to ≤ 4096 nodes.
- **Per-pixel / per-voxel GAT scores** — not shipped; `[E, K, H, W]` score tensors are memory-prohibitive.

TGraphX is built to complement and interoperate with mature graph ecosystems. PyG and DGL provide large-scale GNN infrastructures; NetworkX provides extensive classical graph algorithms; PyKEEN focuses on knowledge graph embeddings. TGraphX focuses on a different integration point: tensor-aware graph learning, graph mining, reproducible experiments, and dashboard-ready research workflows in one package.

Detailed limitations: [docs/limitations.md](docs/limitations.md) · Roadmap: [docs/roadmap.md](docs/roadmap.md)

---

## Citation

```bibtex
@misc{sajjadi2025tgraphxtensorawaregraphneural,
    title={TGraphX: Tensor-Aware Graph Neural Network for Multi-Dimensional Feature Learning},
    author={Arash Sajjadi and Mark Eramian},
    year={2025},
    eprint={2504.03953},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2504.03953},
}
```

---

## Authorship

**Software author and maintainer:** Arash Sajjadi, PhD Candidate in Computer Science, University of Saskatchewan ([arash.sajjadi@usask.ca](mailto:arash.sajjadi@usask.ca)).

**Academic supervision:** Mark Eramian, PhD supervisor / academic advisor, University of Saskatchewan.

Developed as part of doctoral research in the Department of Computer Science at the University of Saskatchewan. TGraphX is an independent research software project; it does not represent an official product or endorsement of the University of Saskatchewan.

---

## License

TGraphX is released under the [MIT License](LICENSE).

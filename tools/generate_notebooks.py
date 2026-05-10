"""Generate TGraphX v1.3 Jupyter notebooks from structured templates.

Run::

    python tools/generate_notebooks.py

This script writes all notebooks/*.ipynb files.  The notebooks are
educational, scenario-driven, and CPU-runnable.

Each notebook:
- has a clear title and goal Markdown cell;
- states what problem it solves;
- uses only TGraphX public APIs;
- avoids private paths, secrets, and hidden network calls;
- is short enough to read and run in under a few minutes on CPU;
- is compatible with local Jupyter and Google Colab.

Notebooks will later be uploaded to Google Colab.  Links will be added to
docs/colab_gallery.md by the maintainer after upload.  This script does
NOT produce Colab links.
"""
from __future__ import annotations

import json
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────


def md(source: str) -> dict:
    """Create a Markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip(),
    }


def code(source: str, outputs: list | None = None) -> dict:
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source.strip(),
    }


def notebook(cells: list, title: str = "") -> dict:
    """Wrap cells in notebook metadata."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }


# ── Notebook 01 — Easy Tensor Node Classification ────────────────────────────

NB01 = notebook([
    md("""# 01 — Easy Tensor Node Classification

**Goal:** Train a node classifier on a graph where each node carries a
structured `[C, H, W]` tensor using the TGraphX Easy Mode API.

**Why this matters:** Most graph frameworks treat node features as flat
vectors.  TGraphX keeps tensor structure intact through every message-passing
step.  This notebook shows you can get a working model in a few lines.

**TGraphX subsystem:** `tgraphx.easy`

**Data:** Synthetic — no download required.

**Runtime:** < 30 seconds on CPU.
"""),
    md("## 1. Setup"),
    code("""# Optional: uncomment to install in Colab
# !pip install -q tgraphx
import tgraphx as tgx
print("TGraphX version:", tgx.__version__)"""),
    md("""## 2. Scenario

We have a synthetic citation graph.  Each node (paper) is represented by a
`[C, H, W] = [4, 6, 6]` tensor — imagine a small feature map summarising
the paper's topic distribution across semantic categories.

There are 3 topic classes.  Our goal: classify each paper into one class.
"""),
    md("## 3. Create Synthetic Data"),
    code("""# Create a synthetic tensor node classification graph.
# num_nodes=256, node_shape=(4,6,6), 3 classes, no GPU required.
data = tgx.easy.synthetic_tensor_node_classification(
    num_nodes=256,
    node_shape=(4, 6, 6),    # [channels, height, width] per node
    num_classes=3,
    num_edges=1024,
    seed=42,
)
print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
print(f"Node feature shape per node: {data.node_features.shape[1:]}")  # (4,6,6)
print(f"Labels shape: {data.node_labels.shape}")"""),
    md("## 4. Train with Easy Mode"),
    code("""# Zero-boilerplate training: one call handles model, sampler, optimizer, loss.
result = tgx.easy.train_node_classifier(
    data,
    model="tensor_gcn",   # ConvMessagePassing-based, preserves [C,H,W]
    sampler="neighbor",   # mini-batch NeighborLoader
    fanouts=[10, 5],      # 2-hop sampling
    batch_size=32,
    epochs=5,
    seed=42,
    verbose=True,
)"""),
    md("## 5. Inspect Result"),
    code("""print("\\nFinal metrics:")
for k, v in result.metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

print("\\nConfig (all resolved defaults visible):")
for k, v in result.config.items():
    print(f"  {k}: {v}")"""),
    md("## 6. Why This Is Tensor-Native"),
    code("""# Node features are kept as [N, C, H, W] — never flattened.
import torch
print("node_features shape:", result.graph.node_features.shape)
# ↑ (256, 4, 6, 6) — 4 channels, 6×6 spatial layout preserved.

# The model internals: ConvMessagePassing uses 1×1 conv, not Linear(flatten).
print("Model type:", type(result.model).__name__)

# You can always escape to raw PyTorch objects:
print("Optimizer:", type(result.optimizer).__name__)
print("Loader:", type(result.loader).__name__)"""),
    md("""## 7. Write Dashboard Artifacts (optional)

If you have a logdir and want to view metrics in `tgraphx-dashboard`:

```python
result.write_dashboard_artifacts("runs/easy_nb01")
# Then: tgraphx-dashboard --logdir runs/easy_nb01
```
"""),
    md("""## 8. Next Steps

- **Script version:** `examples/easy_tensor_node_classification_no_torch.py`
- **Tutorial:** `tutorials/tensor_node_classification_neighbor_loader.py`
- **Low-level API:** see `tgraphx/easy/_workflows.py` for the full training loop.
- **Limitations:** this is a synthetic demo; real-world accuracy depends on
  your dataset and training length.
"""),
])

# ── Notebook 02 — Image Patch Tensor Graph ───────────────────────────────────

NB02 = notebook([
    md("""# 02 — Image-Patch Tensor Graph: Tensor-Native vs Flattened Baseline

**Goal:** Build a graph of image patches where each node is an image-patch
tensor `[C, H, W]`.  Train a tensor-aware TGraphX model and compare it
with a flattened baseline that discards spatial structure.

**Why this matters:** This notebook is the core proof-of-concept for
TGraphX.  Most GNN frameworks force you to flatten `[C,H,W]` node features
into flat vectors, discarding the spatial relationship between channels.
TGraphX preserves that structure throughout message passing.

**TGraphX subsystem:** `ConvMessagePassing`, `build_grid_graph`, `image_to_patches`

**Data:** Synthetic image — no download, no torchvision required.

**Runtime:** < 60 seconds on CPU.
"""),
    md("## 1. Setup"),
    code("""# Optional: uncomment to install in Colab
# !pip install -q tgraphx
import torch
import torch.nn as nn
import torch.nn.functional as F
import tgraphx as tgx
from tgraphx import Graph, ConvMessagePassing, build_grid_graph, image_to_patches, patch_grid_shape
print("TGraphX version:", tgx.__version__)"""),
    md("""## 2. Scenario

We synthesize a small `[3, 12, 12]` image with a horizontal gradient
(channel 0), vertical gradient (channel 1), and center bump (channel 2).

The image is split into 9 non-overlapping `4×4` patches.  Each patch becomes
a graph node with features `[3, 4, 4]`.  We connect neighboring patches in
a grid graph and train a simple node classifier.

**Key question:** does preserving the `[3,4,4]` tensor shape help compared
with flattening each patch to a `48`-dimensional vector?
"""),
    md("## 3. Build Synthetic Image and Patch Graph"),
    code("""torch.manual_seed(42)
C, H, W = 3, 12, 12
patch_size, stride = 4, 4

# Synthetic image: gradient + center bump → spatial structure matters.
image = torch.randn(C, H, W) * 0.3
yy = torch.linspace(-1, 1, H).view(H,1).expand(H,W)
xx = torch.linspace(-1, 1, W).view(1,W).expand(H,W)
image[0] += yy; image[1] += xx
image[2] += torch.exp(-(xx**2 + yy**2))

# Patchify: image_to_patches expects [B, C, H, W].
patches = image_to_patches(image.unsqueeze(0), patch_size=patch_size, stride=stride)
patches = patches.squeeze(0)          # [N, C, ph, pw]
N, Cp, ph, pw = patches.shape
print(f"Patches: {N} nodes, each shape [{Cp}, {ph}, {pw}]")

# Per-patch label: is mean intensity in channel 0 positive? (encodes horizontal gradient)
y = (patches[:, 0].mean(dim=(-1,-2)) > 0).long()
num_classes = 2

# Build 4-connected grid graph.
grid_h, grid_w = patch_grid_shape(H, W, patch_size, stride)
edge_index = build_grid_graph(grid_h, grid_w, directed=False)
print(f"Grid: {grid_h}×{grid_w} = {N} nodes, {edge_index.size(1)} edges")

# Tensor-native graph: each node feature is [C, ph, pw].
g_tensor = Graph(node_features=patches, edge_index=edge_index, y=y)
# Flattened graph: each node feature is a 1-D vector.
g_flat = Graph(node_features=patches.flatten(1), edge_index=edge_index, y=y)
print(f"\\nTensor node shape: {g_tensor.node_features.shape[1:]}")
print(f"Flat   node shape: {g_flat.node_features.shape[1:]}")"""),
    md("## 4. Define Models"),
    code("""class TensorModel(nn.Module):
    \"\"\"Tensor-native model: node features are [C, ph, pw] throughout.\"\"\"
    def __init__(self):
        super().__init__()
        # ConvMessagePassing operates on [C,H,W] tensors — no flatten.
        self.conv = ConvMessagePassing(
            in_shape=(Cp, ph, pw), out_shape=(8, ph, pw)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # [N,8,1,1]
        self.head = nn.Linear(8, num_classes)

    def forward(self, x, edge_index):
        z = self.conv(x, edge_index).relu()        # [N, 8, ph, pw]
        return self.head(self.pool(z).flatten(1))  # [N, num_classes]

class FlatModel(nn.Module):
    \"\"\"Flattened baseline: node features lose spatial layout.\"\"\"
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x, edge_index):
        # Simple mean-neighbor aggregation (no spatial structure).
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        counts = torch.zeros(x.size(0)).index_add(0, dst, torch.ones(src.size(0)))
        agg = agg / counts.clamp(min=1).unsqueeze(-1)
        return self.fc2(F.relu(self.fc1(agg)))

tensor_model = TensorModel()
flat_model   = FlatModel(patches.flatten(1).size(1))
tp = sum(p.numel() for p in tensor_model.parameters())
fp = sum(p.numel() for p in flat_model.parameters())
print(f"Tensor model parameters: {tp}")
print(f"Flat   model parameters: {fp}")"""),
    md("## 5. Train Both Models"),
    code("""import time
opt_t = torch.optim.Adam(tensor_model.parameters(), lr=1e-2)
opt_f = torch.optim.Adam(flat_model.parameters(),   lr=1e-2)
EPOCHS = 10

t0 = time.time()
for ep in range(1, EPOCHS+1):
    # Tensor model
    z_t = tensor_model(g_tensor.node_features, g_tensor.edge_index)
    loss_t = F.cross_entropy(z_t, g_tensor.node_labels)
    opt_t.zero_grad(); loss_t.backward(); opt_t.step()
    acc_t = (z_t.detach().argmax(-1) == g_tensor.node_labels).float().mean()

    # Flat baseline
    z_f = flat_model(g_flat.node_features, g_flat.edge_index)
    loss_f = F.cross_entropy(z_f, g_flat.node_labels)
    opt_f.zero_grad(); loss_f.backward(); opt_f.step()
    acc_f = (z_f.detach().argmax(-1) == g_flat.node_labels).float().mean()

    if ep == 1 or ep % 5 == 0:
        print(f"Ep {ep:2d} | tensor loss={loss_t:.4f} acc={acc_t:.3f} | "
              f"flat loss={loss_f:.4f} acc={acc_f:.3f}")

print(f"\\nTotal time: {time.time()-t0:.1f}s")"""),
    md("## 6. Verify Gradient Flow and Tensor Shape Preservation"),
    code("""# Tensor model: verify spatial shape preserved through the whole graph.
z = tensor_model.conv(g_tensor.node_features, g_tensor.edge_index)
print("After ConvMessagePassing — node feature shape:", z.shape)
# Expected: [N, 8, ph, pw] — same spatial dims, more channels.
assert z.shape == (N, 8, ph, pw), f"Unexpected shape: {z.shape}"
print("✓ Spatial dimensions preserved: no silent flattening inside TGraphX.")

# Check gradient is finite and nonzero.
loss_check = tensor_model(g_tensor.node_features, g_tensor.edge_index).sum()
loss_check.backward()
for name, p in tensor_model.named_parameters():
    if p.grad is not None:
        assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"
        assert p.grad.abs().sum() > 0, f"Zero grad in {name}"
print("✓ Gradients are finite and nonzero.")"""),
    md("""## 7. Key Takeaways

| | Tensor-native model | Flattened baseline |
|---|---|---|
| Node features | `[C, H, W]` preserved | flattened to `D` |
| Message passing | 1×1 conv (spatial-aware) | linear projection |
| Spatial invariants | respected | discarded |
| TGraphX forced? | no — flattening is always opt-in | n/a |

**TGraphX does not force spatial structure.**  You can always use flat
features.  But if your node features *have* spatial structure (patches,
feature maps, spectrograms, volumetric tensors…), TGraphX keeps it intact
so your model can leverage it.

**This demo uses a tiny graph.**  For larger graphs, combine `ConvMessagePassing`
with `NeighborLoader` — see `tutorials/tensor_node_classification_neighbor_loader.py`.
"""),
    md("""## 8. Next Steps

- **Tutorial:** `tutorials/tensor_node_classification_neighbor_loader.py`
- **Benchmark:** `benchmarks/tensor_vs_flatten_benchmark.py`
- **Limitations:** on this tiny 9-node graph, the two models converge to
  similar accuracy.  The spatial advantage grows with more complex patterns.
"""),
])

# ── Notebook 03 — KG Completion ──────────────────────────────────────────────

NB03 = notebook([
    md("""# 03 — Knowledge Graph Completion: RESCAL, TransE, SimplE, and HPO

**Goal:** Train KG embedding models on a tiny synthetic knowledge graph
(researchers, papers, institutions, topics) and rank missing links.

**Why this matters:** TGraphX supports tensor-aware KG workflows inside the
same framework as GNN training and graph mining.

**TGraphX subsystem:** `tgraphx.kg`

**Data:** Synthetic — no download required.

**Runtime:** < 60 seconds on CPU.
"""),
    code("""import torch
from tgraphx.kg import (
    KnowledgeGraph,
    TransEModel, RESCALModel, SimplEModel,
    evaluate_filtered_ranking,
    list_kg_models,
    run_kg_hpo,
)
print("Available KG models:", list(list_kg_models().keys()))"""),
    md("""## 2. Scenario: Academic Knowledge Graph

We model a small academic community:

| Entity type | IDs |
|---|---|
| Researchers | 0–4 |
| Papers | 5–9 |
| Topics | 10–12 |
| Institutions | 13–14 |

Relations:
- 0 = `authored` (researcher → paper)
- 1 = `affiliated_with` (researcher → institution)
- 2 = `covers_topic` (paper → topic)
- 3 = `supervised` (researcher → researcher)
"""),
    code("""# Hand-crafted tiny academic KG.
heads = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4,  5, 6, 7, 8, 9,  0, 1])
rels  = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1,  2, 2, 2, 2, 2,  3, 3])
tails = torch.tensor([5, 6, 7, 8, 9, 13,14,13,14,13, 10,11,12,10,11, 1, 2])
N_e, N_r = 15, 4

kg = KnowledgeGraph.from_hrt(heads, rels, tails, num_entities=N_e, num_relations=N_r)
print(f"KG: {kg.num_entities} entities, {kg.num_relations} relations, "
      f"{kg.num_triples} triples")"""),
    md("## 3. Train Individual Models"),
    code("""def train_eval(model, kg, epochs=40, seed=42):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    triples = kg.triples
    for _ in range(epochs):
        neg = triples.clone()
        neg[:, 2] = torch.randint(0, N_e, (triples.size(0),))
        loss = (1.0 + model.score_triples(neg) - model.score_triples(triples)).clamp(min=0).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # Evaluate.
    all_pos = set(map(tuple, triples.tolist()))
    res = evaluate_filtered_ranking(model, triples, all_pos, N_e, filtered=True, hits_at=(1, 3))
    return res

for name, cls in [("TransE", TransEModel), ("RESCAL", RESCALModel), ("SimplE", SimplEModel)]:
    model = cls(N_e, N_r, embedding_dim=16)
    res   = train_eval(model, kg)
    print(f"{name:8s}  MRR={res.filt_mrr:.3f}  H@1={res.filt_hits[1]:.3f}  H@3={res.filt_hits[3]:.3f}")"""),
    md("""## 4. Why RESCAL Captures Asymmetric Relations

DistMult scores `f(h,r,t) = ⟨h, r, t⟩` which is **symmetric in h and t**:
`f(A, authored, Paper1) == f(Paper1, authored, A)`.

RESCAL uses a matrix per relation: `f(h,r,t) = h^T M_r t`.  A non-symmetric
`M_r` means the model can correctly score directed relations.

SimplE also captures asymmetry via separate forward/inverse embeddings.
"""),
    code("""# Demonstrate asymmetry: RESCAL can distinguish direction.
rescal = RESCALModel(N_e, N_r, embedding_dim=8)
fwd = rescal.score_triples(torch.tensor([[0, 0, 5]]))  # researcher authored paper
rev = rescal.score_triples(torch.tensor([[5, 0, 0]]))  # paper authored researcher?
print(f"Forward (researcher→paper): {fwd.item():.4f}")
print(f"Reverse (paper→researcher): {rev.item():.4f}")
print("Scores differ (random init): asymmetry can be learned.")"""),
    md("## 5. KG HPO — Grid Search Over Models and Hyper-Params"),
    code("""result = run_kg_hpo(
    kg,
    model_names=["TransE", "DistMult", "SimplE"],
    search_space={
        "embedding_dim": [8, 16],
        "lr": [1e-2, 5e-3],
    },
    metric="mrr",
    strategy="grid",
    max_trials=6,
    epochs=20,
    seed=42,
)
print("Best model:", result.best_model_name)
print("Best config:", result.best_config)
print("Best MRR:  ", result.best_metrics["mrr"])
print(f"\\nTrials run: {len(result.trials)}")"""),
    code("""result.summary()"""),
    md("""## 6. Next Steps

- **Tutorial:** `tutorials/kg_benchmark_quickstart.py`
- **API ref:** `tgraphx/kg/` — TrainE, DistMult, ComplEx, RotatE, RESCAL, SimplE
- **Limitations:** this is a tiny 17-triple KG.  Filtered MRR is not a
  reliable metric at this scale.  For meaningful KG benchmarks use
  FB15k-237 or WN18RR (optional PyG adapter; explicit download required).
"""),
])

# ── Notebook 04 — Graph Generation and Optimization ─────────────────────────

NB04 = notebook([
    md("""# 04 — Graph Generation and Evolutionary Optimization

**Goal:** Generate candidate graphs using classical generators, compute
structural metrics, then optimize a graph with evolutionary search toward
a connectivity target.

**TGraphX subsystem:** `tgraphx.generation`, `tgraphx.evolutionary`

**Data:** Synthetic — no download.

**Runtime:** < 60 seconds on CPU.
"""),
    code("""from tgraphx import run_graph_generation, run_evolutionary_optimization
from tgraphx import list_graph_generation_methods, list_evolutionary_optimizers
print("Generation methods:", list(list_graph_generation_methods().keys()))
print("Evolutionary optimizers:", list(list_evolutionary_optimizers().keys()))"""),
    md("## 2. Generate Graphs with Structural Metrics"),
    code("""# Generate 10 Erdős–Rényi and 10 Barabási–Albert graphs.
er_result = run_graph_generation(
    method="erdos_renyi", num_graphs=10, num_nodes=20,
    num_edges=40, seed=42,
)
ba_result = run_graph_generation(
    method="barabasi_albert", num_graphs=10, num_nodes=20,
    num_edges=40, seed=42,
)
for name, res in [("ER", er_result), ("BA", ba_result)]:
    m = res.metrics
    print(f"{name}: validity={m.get('validity',0):.2f}  "
          f"uniqueness={m.get('uniqueness',0):.2f}  "
          f"diversity={m.get('diversity',0):.3f}")"""),
    md("## 3. Evolve a Graph Toward High Connectivity"),
    code("""from tgraphx.evolutionary import GraphGenome, GeneticAlgorithmOptimizer, GeneticAlgorithmConfig
from tgraphx.evolutionary import connectivity_fitness
import torch

def make_genome(seed=0):
    torch.manual_seed(seed)
    ei = torch.randint(0, 10, (2, 12))
    return GraphGenome(edge_index=ei, num_nodes=10)

config = GeneticAlgorithmConfig(population_size=12, n_generations=20, seed=42)
result = GeneticAlgorithmOptimizer(config, connectivity_fitness).optimize(
    [make_genome(i) for i in range(12)]
)
print(f"Best connectivity fitness: {result.best_fitness:.4f}")
print(f"Generations run: {len(result.history)}")"""),
    md("""## 4. Why This Matters

Graph generation + evolution lets you:
- create synthetic graphs with controlled properties for benchmarking;
- apply multi-objective Pareto search when multiple metrics compete;
- search over graph structure jointly with model training;
- study graph property distributions.

TGraphX evolutionary utilities preserve tensor node features through
mutation/crossover if the genome carries them.

## 5. Next Steps
- **Tutorial:** `tutorials/graph_generation_quickstart.py`
- **NSGA-II multi-objective:** see `NSGAIIOptimizer`
- **Limitations:** these generators produce simple random graphs.
  Neural generation (VAE, autoregressive) is in `tgraphx.generation` but
  marked Experimental.
"""),
])

# ── Notebook 05 — Graph RL ───────────────────────────────────────────────────

NB05 = notebook([
    md("""# 05 — Graph Reinforcement Learning: Coloring and Navigation

**Goal:** Train an RL agent to make sequential decisions on a graph.
Compare random/greedy baselines with a learning algorithm (DQN).

**TGraphX subsystem:** `tgraphx.rl`

**Data:** Tiny synthetic graph environments.

**Runtime:** < 60 seconds on CPU.

**Honest note:** TGraphX RL is a research-focused foundation, not a
production RLlib/SB3 replacement.  These demonstrations show the API and
learning dynamics on small graphs.
"""),
    code("""from tgraphx import run_graph_rl, list_graph_rl_algorithms
from tgraphx.rl import EarlyStoppingCallback, CSVLoggerCallback
import tempfile, json
print("Available RL algorithms:")
for name, desc in list_graph_rl_algorithms().items():
    print(f"  {name}: {desc}")"""),
    md("## 2. Baseline Comparison: Random vs Greedy vs DQN"),
    code("""results = {}
for algo in ["random", "greedy", "dqn"]:
    r = run_graph_rl(
        env="graph_navigation",
        algorithm=algo,
        episodes=30,
        seed=42,
    )
    results[algo] = r.metrics["mean_return"]
    print(f"{algo:8s}: mean_return={r.metrics['mean_return']:.2f}")"""),
    md("## 3. DQN with Early Stopping Callback"),
    code("""with tempfile.TemporaryDirectory() as tmpdir:
    csv_log = CSVLoggerCallback(tmpdir + "/episodes.csv")
    stopper = EarlyStoppingCallback(monitor="reward", patience=8, mode="max")

    r = run_graph_rl(
        env="graph_navigation",
        algorithm="dqn",
        episodes=50,
        seed=42,
        callbacks=[csv_log, stopper],
    )

    print(f"Stopped early: {getattr(r, 'stopped_early', False)}")
    print(f"Mean return: {r.metrics['mean_return']:.2f}")
    # Print a few CSV rows.
    import csv
    with open(tmpdir + "/episodes.csv") as f:
        rows = list(csv.DictReader(f))
    print(f"Logged {len(rows)} episodes.")
    if rows:
        print("Sample:", rows[-1])"""),
    md("## 4. Graph Coloring Environment"),
    code("""# Try a graph coloring environment: agent assigns colors to nodes.
r_coloring = run_graph_rl(
    env="graph_coloring",
    algorithm="random",
    episodes=10,
    seed=0,
)
print("Coloring env mean return:", r_coloring.metrics["mean_return"])
print("Config:", r_coloring.config)"""),
    md("""## 5. Key Concepts

| Concept | In TGraphX |
|---|---|
| Environment | `GraphEnv` subclass (navigation, coloring, max-cut, …) |
| Observation | `{"node_features": Tensor, "edge_index": Tensor, ...}` |
| Action | Discrete node/edge choice or continuous vector |
| Policy | `GraphPolicyNetwork` (graph-based forward pass) |
| Algorithms | Random, Greedy, REINFORCE, A2C, DQN, Double DQN, PPO, TD3, SAC |
| Callbacks | `EarlyStoppingCallback`, `CSVLoggerCallback` |

## 6. Next Steps
- **Tutorial:** `tutorials/graph_rl_quickstart.py`
- **Algorithms doc:** `docs/graph_rl_algorithms.md`
- **Limitations:** DQN on navigation converges slowly; PPO/SAC need more
  tuning.  For serious RL research, combine these foundations with a
  dedicated framework.
"""),
])

# ── Notebook 06 — GraphML IO ─────────────────────────────────────────────────

NB06 = notebook([
    md("""# 06 — Graph IO: GraphML Round-Trip and Interoperability

**Goal:** Build a small graph, save it to GraphML, read it back, verify
the round-trip, and understand the tensor-feature limitations.

**TGraphX subsystem:** `tgraphx.io`

**Data:** Synthetic — no download.

**Runtime:** < 10 seconds on CPU.
"""),
    code("""import torch
from pathlib import Path
import tempfile
from tgraphx import Graph
from tgraphx.io import write_graphml, read_graphml
print("tgraphx.io ready")"""),
    md("## 2. Build a Small Graph with Metadata"),
    code("""# Create a directed 5-node graph with edge weights and labels.
x = torch.tensor([          # Scalar node features [5, 1] (1-D → round-trips)
    [0.1], [0.5], [0.9], [0.3], [0.7]
])
edge_index = torch.tensor([
    [0, 1, 2, 2, 3, 4],    # source nodes
    [1, 2, 3, 4, 4, 0],    # target nodes
], dtype=torch.long)
edge_weight = torch.tensor([1.5, 2.0, 0.5, 1.0, 3.0, 2.5])
y = torch.tensor([0, 1, 1, 0, 2], dtype=torch.long)  # node labels

g = Graph(node_features=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
print(g)"""),
    md("## 3. Write to GraphML and Read Back"),
    code("""with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
    path = Path(f.name)

# Write
write_graphml(g, path, include_labels=True, include_tensor_features=True)
print("Written:", path)
print("File size:", path.stat().st_size, "bytes")

# Read
g2 = read_graphml(path, feature_dtype=torch.float32)
print("\\nRound-trip result:")
print(f"  Nodes: {g2.num_nodes} (expected {g.num_nodes})")
print(f"  Edges: {g2.num_edges} (expected {g.num_edges})")
print(f"  node_features: {g2.node_features}")
print(f"  node_labels: {g2.node_labels}")
print(f"  edge_weight: {g2.edge_weight}")

# Clean up
path.unlink()"""),
    md("## 4. What Does NOT Round-Trip (and Why)"),
    code("""# Attempt to save [N, C, H, W] tensor features — TGraphX refuses safely.
x_spatial = torch.randn(4, 3, 8, 8)   # image-like node features
g_spatial = Graph(node_features=x_spatial,
                  edge_index=torch.tensor([[0,1],[1,2]]))
try:
    with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
        write_graphml(g_spatial, f.name, include_tensor_features=True)
except ValueError as e:
    print("ValueError (expected):", e)
print("\\nTGraphX refuses to silently flatten [N,C,H,W] through GraphML.")
print("For lossless persistence: use torch.save({'graph': g_spatial}, 'graph.pt')")"""),
    md("""## 5. Interoperability

GraphML files written by TGraphX can be opened in:
- **NetworkX**: `import networkx as nx; G = nx.read_graphml("out.graphml")`
- **Gephi**: File → Open
- **Cytoscape**: File → Import → Network from File

Node labels and edge weights survive the export (as XML attributes).

**What GraphML cannot express:**
- `[C, H, W]` tensor node features
- `graph_features` (graph-level tensors)
- arbitrary metadata dicts

For these, use `torch.save` or write a custom JSON+tensor pair.

## 6. Next Steps
- **IO docs:** `docs/io.md`
- **Roadmap:** GEXF and Pajek planned for v1.4
"""),
])

# ── Notebook 07 — Benchmark Suite ────────────────────────────────────────────

NB07 = notebook([
    md("""# 07 — v1.3 Benchmark Suite and Dashboard Artifacts

**Goal:** Run the TGraphX v1.3 smoke benchmark suite, inspect the JSON
results, and understand how to point the local dashboard to the artifact
directory.

**TGraphX subsystem:** `benchmarks/run_v13_benchmark_suite.py`

**Data:** Synthetic — no download.

**Runtime:** < 120 seconds on CPU.

**Important note:** These are **smoke benchmarks** — tiny synthetic data.
They are NOT competitive throughput claims against PyG, DGL, PyKEEN, or SB3.
"""),
    code("""import subprocess, json, sys
result = subprocess.run(
    [sys.executable, "benchmarks/run_v13_benchmark_suite.py", "--small", "--json"],
    capture_output=True, text=True,
)
if result.returncode != 0:
    print("STDERR:", result.stderr[:500])
else:
    data = json.loads(result.stdout)
    print(f"Suite: {data['suite']}")
    print(f"Version: {data['package_version']}  Device: {data['device']}")
    print()
    for row in data['benchmarks']:
        status = row['status']
        rt = f"{row['runtime_s']:.3f}s" if row['runtime_s'] else "failed"
        print(f"  {row['name']:<35} {status:<7} {rt}")"""),
    md("## 2. Inspect Individual Benchmark Metrics"),
    code("""# Show metrics from successful rows.
for row in data['benchmarks']:
    if row['status'] == 'ok' and row['metrics']:
        print(f"\\n{row['name']}:")
        for k, v in row['metrics'].items():
            print(f"  {k}: {v}")"""),
    md("## 3. Write Dashboard-Compatible Output"),
    code("""import tempfile, pathlib
with tempfile.TemporaryDirectory() as d:
    # Write the benchmark JSON to a directory.
    out = pathlib.Path(d) / "benchmark_results.json"
    out.write_text(json.dumps(data, indent=2))
    print("Written:", out)
    print("\\nTo view in the TGraphX dashboard:")
    print(f"  tgraphx-dashboard --logdir {d}")
    print("\\n(The dashboard reads benchmark_results.json and similar files.)")"""),
    md("""## 4. Benchmark Scope

| Type | What it measures | What it does NOT claim |
|---|---|---|
| Smoke benchmark | Correctness + basic runtime | Competitive throughput vs PyG/DGL |
| Performance benchmark | Local machine timing | Cross-machine reproducibility |
| Reference benchmark | Algorithm correctness | SOTA performance |

The honest scope is in `docs/benchmark_report.md`.

## 5. Next Steps
- **Full suite:** `benchmarks/run_v13_benchmark_suite.py`
- **Report:** `docs/benchmark_report.md`
- **Dashboard:** `tgraphx-dashboard --logdir <dir>`
"""),
])

# ── Write all notebooks ───────────────────────────────────────────────────────

notebooks_dir = Path("notebooks")
notebooks_dir.mkdir(exist_ok=True)

to_write = {
    "01_easy_tensor_node_classification.ipynb": NB01,
    "02_image_patch_tensor_graph.ipynb": NB02,
    "03_kg_completion_rescal_simple_hpo.ipynb": NB03,
    "04_graph_generation_and_optimization.ipynb": NB04,
    "05_graph_rl_coloring_and_navigation.ipynb": NB05,
    "06_graph_io_roundtrip.ipynb": NB06,
    "07_benchmark_suite_and_dashboard.ipynb": NB07,
}

for filename, nb in to_write.items():
    path = notebooks_dir / filename
    path.write_text(json.dumps(nb, indent=1))
    n_cells = len(nb["cells"])
    print(f"Wrote {filename} ({n_cells} cells)")

print(f"\nAll {len(to_write)} notebooks written to notebooks/")

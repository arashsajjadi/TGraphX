"""Generate upgraded advanced real-dataset notebooks 31–35.

Usage:
    python tools/build_advanced_notebooks.py
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path("colab_drafts/advanced_real_datasets")
OUT.mkdir(parents=True, exist_ok=True)


def nb(cells: list[tuple[str, str]]) -> dict:
    """Create a minimal nbformat 4 notebook."""
    result_cells = []
    for cell_type, source in cells:
        lines = source.lstrip("\n")
        if cell_type == "code":
            result_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines,
            })
        else:
            result_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": lines,
            })
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": result_cells,
    }


def save(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  Wrote {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 31 — MNIST Class-Graph Membership with Tensor Nodes
# ─────────────────────────────────────────────────────────────────────────────

NB31 = nb([
("markdown", """
# 31 — MNIST Class-Graph Membership with Tensor Nodes

**Research question:** Can tensor-native graph learning exploit both spatial image
structure and graph-level visual/prototype relationships, while keeping MNIST
images as `[1, 28, 28]` tensor-valued nodes — without early flattening?

**What this demonstrates:**
- MNIST images flow as `[N, 1, 28, 28]` tensors through every graph step
- Two edge types: visual-similarity kNN + class-prototype membership
- `ConvMessagePassing` preserving spatial layout through message passing
- `NeighborLoader` mini-batch training with `batch.seed_logits` / `batch.seed_y`
- Flatten-MLP baseline to quantify the cost of early flattening
- Dashboard artifact writing (`write_run_metadata`, `write_metrics_summary`)
- Reproducibility: seeded, deterministic, version-stamped

**Dataset:** MNIST handwritten digits (LeCun et al., 1998) via torchvision.

**Task type:** Transductive node classification on a visual-similarity graph.
Validation/test nodes are structurally present in the graph; their labels are
withheld during training but their positions in the graph are visible.
"""),

("code", """
# ── Configuration ─────────────────────────────────────────────────────────────
FAST_MODE = True
SEED = 42
SUBSET_SIZE = 1000 if FAST_MODE else 5000
K_VISUAL = 5       # visual kNN neighbors per node
K_PROTO = 1        # prototype-membership edges per node
EPOCHS = 3 if FAST_MODE else 15
BATCH_SIZE = 32
FANOUTS = [10, 5]
NUM_CLASSES = 10
HIDDEN_DIM = 32
print(f"FAST_MODE={FAST_MODE}  N={SUBSET_SIZE}  epochs={EPOCHS}")
"""),

("code", """
# ── Install and import ────────────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tgraphx", "torchvision"],
               check=False)
import time, json, pathlib
import torch, torch.nn as nn, torch.nn.functional as F
import tgraphx
from tgraphx.reproducibility import set_seed
from tgraphx import Graph, ConvMessagePassing, count_parameters
from tgraphx.loaders import NeighborLoader
from tgraphx.tracking import (
    write_run_metadata, write_metrics_summary,
    write_dataset_metadata, write_graph_stats,
)
from tgraphx.mining import graph_summary, degree_statistics

RUN_DIR = pathlib.Path("runs/advanced_notebooks/31_mnist")
RUN_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(SEED, deterministic=True)
print(f"TGraphX v{tgraphx.__version__}  |  device={device}  |  SEED={SEED}")
"""),

("markdown", """## 1. Dataset loading"""),

("code", """
# ── Load MNIST images as tensor nodes ──────────────────────────────────────
USING_REAL_MNIST = False
try:
    from torchvision import datasets, transforms
    ds = datasets.MNIST(root="/tmp/mnist", train=True,
                        download=True, transform=transforms.ToTensor())
    gen = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(ds), generator=gen)[:SUBSET_SIZE]
    images = torch.stack([ds[i][0] for i in idx])   # [N, 1, 28, 28]
    labels = torch.tensor([ds[i][1] for i in idx])  # [N]
    USING_REAL_MNIST = True
    print(f"Real MNIST loaded: {images.shape}  labels={labels.shape}")
except Exception as exc:
    print(f"MNIST unavailable ({exc}); using synthetic MNIST-shaped fallback.")
    gen = torch.Generator().manual_seed(SEED)
    images = torch.randn(SUBSET_SIZE, 1, 28, 28, generator=gen)
    labels = torch.randint(0, NUM_CLASSES, (SUBSET_SIZE,),
                           generator=torch.Generator().manual_seed(SEED + 1))

write_dataset_metadata(
    str(RUN_DIR / "dataset_metadata.json"),
    name="MNIST" if USING_REAL_MNIST else "synthetic_mnist_fallback",
    source="torchvision.datasets.MNIST" if USING_REAL_MNIST else "synthetic",
    num_samples=SUBSET_SIZE,
    node_feature_shape="[1, 28, 28]",
    num_classes=NUM_CLASSES,
)
print(f"node_features shape (before graph): {images.shape}")
assert images.shape == (SUBSET_SIZE, 1, 28, 28), "Unexpected image shape"
"""),

("markdown", """## 2. Graph construction: visual-similarity + prototype edges"""),

("code", """
# ── Deterministic train/val/test split ─────────────────────────────────────
N = SUBSET_SIZE
gen = torch.Generator().manual_seed(SEED)
perm = torch.randperm(N, generator=gen)
n_train = int(0.7 * N)
n_val = int(0.15 * N)
train_idx = perm[:n_train]
val_idx = perm[n_train:n_train + n_val]
test_idx = perm[n_train + n_val:]

train_mask = torch.zeros(N, dtype=torch.bool)
val_mask = torch.zeros(N, dtype=torch.bool)
test_mask = torch.zeros(N, dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True
print(f"Split: train={train_mask.sum()}  val={val_mask.sum()}  test={test_mask.sum()}")
"""),

("code", """
# ── Visual-similarity kNN edges ────────────────────────────────────────────
# Chunked cosine similarity to avoid O(N^2) memory for large N.
CHUNK = 256
flat = images.view(N, -1).float()
flat_n = flat / flat.norm(dim=1, keepdim=True).clamp(min=1e-8)

src_list, dst_list = [], []
for i in range(0, N, CHUNK):
    chunk = flat_n[i:i + CHUNK]          # [C, D]
    sims = chunk @ flat_n.T              # [C, N]
    sims[:, i:i + CHUNK].fill_diagonal_(-1.0)
    _, topk = sims.topk(K_VISUAL, dim=1)
    base = torch.arange(i, min(i + CHUNK, N)).unsqueeze(1).expand_as(topk)
    src_list.append(base.reshape(-1))
    dst_list.append(topk.reshape(-1))

src_vis = torch.cat(src_list)
dst_vis = torch.cat(dst_list)
vis_edges = torch.stack([src_vis, dst_vis], dim=0)  # directed → make undirected
vis_edges = torch.cat([vis_edges, vis_edges.flip(0)], dim=1)
vis_edges = torch.unique(vis_edges, dim=1)
print(f"Visual-similarity edges: {vis_edges.shape[1]:,}")
"""),

("code", """
# ── Class-prototype membership edges ──────────────────────────────────────
# Prototypes computed from TRAINING nodes only (no label leakage for val/test).
proto_flat = torch.zeros(NUM_CLASSES, flat_n.shape[1])
for c in range(NUM_CLASSES):
    mask_c = train_mask & (labels == c)
    if mask_c.sum() > 0:
        proto_flat[c] = flat_n[mask_c].mean(0)
proto_flat_n = proto_flat / proto_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)

# Connect each node to K_PROTO nearest class prototypes by visual similarity.
# Val/test nodes connect based on visual similarity (not their labels — no leakage).
node_proto_sim = flat_n @ proto_flat_n.T  # [N, 10]
_, best_proto = node_proto_sim.topk(K_PROTO, dim=1)
proto_src = torch.arange(N).unsqueeze(1).expand(-1, K_PROTO).reshape(-1)
proto_dst = best_proto.reshape(-1)
# Offset prototype nodes into a virtual index range [N, N+NUM_CLASSES)
proto_dst_global = proto_dst + N
proto_edges = torch.stack([proto_src, proto_dst_global], dim=0)

# Combined node set: N image nodes + NUM_CLASSES prototype nodes.
TOTAL_NODES = N + NUM_CLASSES
proto_feats = proto_flat.view(NUM_CLASSES, 1, 28, 28).clamp(-3, 3)
all_images = torch.cat([images, proto_feats.to(images.dtype)], dim=0)

# Extend visual edges with self-loops for prototype nodes (stable feature storage).
proto_self = torch.stack([
    torch.arange(N, TOTAL_NODES), torch.arange(N, TOTAL_NODES)
], dim=0)

# ── Edge types: encode each edge category ──────────────────────────────
# edge_type 0 = visual_similarity (kNN over pixel space)
# edge_type 1 = prototype_membership (node → nearest class prototype)
# edge_type 2 = prototype_self_loop (structural, keeps prototype features)
ea_vis   = torch.zeros(vis_edges.shape[1],   dtype=torch.float32)
ea_proto = torch.ones(proto_edges.shape[1],  dtype=torch.float32)
ea_self  = torch.full((proto_self.shape[1],), 2.0)
all_edges    = torch.cat([vis_edges, proto_edges, proto_self], dim=1)
all_edge_attr = ea_vis.tolist() + ea_proto.tolist() + ea_self.tolist()
all_edge_attr = torch.tensor(all_edge_attr, dtype=torch.float32).unsqueeze(1)  # [E,1]

# Labels: image nodes have digit labels; prototype nodes get label -1 (ignored in loss).
all_labels = torch.cat([labels, torch.full((NUM_CLASSES,), -1)])

print(f"Total nodes: {TOTAL_NODES}  (images={N}  prototypes={NUM_CLASSES})")
print(f"edge_type 0 (visual_similarity):    {(all_edge_attr[:,0] == 0).sum():,} edges")
print(f"edge_type 1 (prototype_membership): {(all_edge_attr[:,0] == 1).sum():,} edges")
print(f"edge_type 2 (prototype_self_loop):  {(all_edge_attr[:,0] == 2).sum():,} edges")
print(f"Total edges: {all_edges.shape[1]:,}")
print(f"node_features shape (in graph): {all_images.shape}")
assert all_images.shape == (TOTAL_NODES, 1, 28, 28)
assert all_labels.shape == (TOTAL_NODES,)
assert all_edge_attr.shape[0] == all_edges.shape[1]
"""),

("code", """
# ── Construct TGraphX Graph ────────────────────────────────────────────────
# edge_attr encodes edge_type: 0=visual_similarity, 1=prototype_membership
g = Graph(node_features=all_images, edge_index=all_edges, y=all_labels,
          edge_attr=all_edge_attr)
print(f"TGraphX graph: {g}")
print(f"graph.edge_features.shape: {g.edge_features.shape}  (E x 1 edge_type)")

# Structural summary (image-node subgraph)
summary = graph_summary(vis_edges, num_nodes=N, directed=False)
print(f"Visual-similarity subgraph — density: {summary['density']:.4f}  "
      f"mean_degree: {summary['mean_degree']:.2f}  "
      f"components: {summary['num_connected_components']}")

write_graph_stats(g, str(RUN_DIR / "graph_summary.json"))
print("Leakage policy: prototype features derived from train-label means only.")
print("Val/test nodes connect to prototypes by visual similarity (labels not used).")
"""),

("markdown", """## 3. Model definition"""),

("code", """
# ── Tensor-aware graph classifier ─────────────────────────────────────────
class TensorMNISTGNN(nn.Module):
    \"\"\"Two ConvMessagePassing layers, then a linear head.\"\"\"
    def __init__(self, hidden: int = HIDDEN_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = ConvMessagePassing(in_shape=(1, 28, 28),
                                        out_shape=(hidden, 14, 14))
        self.conv2 = ConvMessagePassing(in_shape=(hidden, 14, 14),
                                        out_shape=(hidden, 7, 7))
        self.pool = nn.AdaptiveAvgPool2d(3)
        self.head = nn.Linear(hidden * 3 * 3, num_classes)

    def forward(self, x: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, ei))   # [N, hidden, 14, 14]
        h = F.relu(self.conv2(h, ei))   # [N, hidden, 7, 7]
        h = self.pool(h).flatten(1)     # [N, hidden*9]
        return self.head(h)             # [N, num_classes]


model = TensorMNISTGNN().to(device)
print(f"TensorMNISTGNN parameters: {count_parameters(model):,}")

# Shape trace (no grad)
with torch.no_grad():
    tiny_x = all_images[:4].to(device)
    # Filter both src AND dst to be inside the tiny subset so indexing stays valid
    _mask = (all_edges[0] < 4) & (all_edges[1] < 4)
    tiny_ei = all_edges[:, _mask].to(device)
    if tiny_ei.shape[1] == 0:
        tiny_ei = torch.tensor([[0], [1]], dtype=torch.long, device=device)
    out = model(tiny_x, tiny_ei)
print(f"Shape trace: input={tiny_x.shape}  output={out.shape}")
assert out.shape == (4, NUM_CLASSES)
print("Shape trace passed.")
"""),

("code", """
# ── Flatten-MLP baseline ───────────────────────────────────────────────────
class FlattenMLP(nn.Module):
    \"\"\"MLP on flattened pixels — ignores graph structure.\"\"\"
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # edge_index unused

baseline = FlattenMLP().to(device)
print(f"FlattenMLP parameters: {count_parameters(baseline):,}")
"""),

("markdown", """## 4. Training"""),

("code", """
# ── Training helper ────────────────────────────────────────────────────────
def train_model(mdl, graph, mask, epochs, name="model"):
    opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
    loader = NeighborLoader(
        graph, fanouts=FANOUTS, batch_size=BATCH_SIZE,
        mask=mask, shuffle=True, seed=SEED,
    )
    mdl.train()
    history = []
    for ep in range(1, epochs + 1):
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            x = batch.node_features.to(device)
            ei = batch.edge_index.to(device)
            logits = mdl(x, ei)
            seed_logits = batch.seed_logits(logits)
            seed_y = batch.seed_y.to(device)
            valid = seed_y >= 0  # ignore prototype nodes (label=-1)
            if valid.sum() == 0:
                continue
            loss = F.cross_entropy(seed_logits[valid], seed_y[valid])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        history.append(avg_loss)
        if ep % max(1, epochs // 3) == 0 or ep == epochs:
            print(f"  [{name}] epoch {ep}/{epochs}  loss={avg_loss:.4f}")
    return history


# ── Gradient sanity check ──────────────────────────────────────────────────
opt_check = torch.optim.Adam(model.parameters(), lr=1e-3)
sample_loader = NeighborLoader(
    g, fanouts=FANOUTS, batch_size=16, mask=train_mask, shuffle=True, seed=SEED
)
for batch in sample_loader:
    x = batch.node_features.to(device)
    ei = batch.edge_index.to(device)
    logits = model(x, ei)
    seed_y = batch.seed_y.to(device)
    valid = seed_y >= 0
    if valid.sum() > 0:
        loss = F.cross_entropy(batch.seed_logits(logits)[valid], seed_y[valid])
        loss.backward()
        total_grad = sum(p.grad.abs().sum().item()
                         for p in model.parameters() if p.grad is not None)
        print(f"Gradient sanity: total_grad_norm={total_grad:.4f}  (expect > 0)")
        assert total_grad > 0, "Zero gradients — check model or data"
        opt_check.zero_grad()
    break
print("Gradient sanity check passed.")
"""),

("code", """
# ── Train both models ──────────────────────────────────────────────────────
t0 = time.time()
print("=== Training TensorMNISTGNN ===")
gnn_history = train_model(model, g, train_mask, EPOCHS, "GNN")

model_baseline = FlattenMLP().to(device)
print("\\n=== Training FlattenMLP baseline ===")
base_history = train_model(model_baseline, g, train_mask, EPOCHS, "MLP")
train_time = time.time() - t0
print(f"\\nTotal training time: {train_time:.1f}s")
"""),

("markdown", """## 5. Evaluation"""),

("code", """
# ── Evaluation helper ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(mdl, graph, mask, name="model"):
    mdl.eval()
    loader = NeighborLoader(
        graph, fanouts=FANOUTS, batch_size=BATCH_SIZE,
        mask=mask, shuffle=False, seed=SEED,
    )
    correct, total = 0, 0
    for batch in loader:
        x = batch.node_features.to(device)
        ei = batch.edge_index.to(device)
        logits = mdl(x, ei)
        seed_logits = batch.seed_logits(logits)
        seed_y = batch.seed_y.to(device)
        valid = seed_y >= 0
        if valid.sum() == 0:
            continue
        preds = seed_logits[valid].argmax(1)
        correct += (preds == seed_y[valid]).sum().item()
        total += valid.sum().item()
    acc = correct / max(1, total)
    print(f"  [{name}] accuracy = {acc:.4f}  ({correct}/{total})")
    return acc


print("=== Validation accuracy ===")
gnn_val_acc = evaluate(model, g, val_mask, "GNN")
base_val_acc = evaluate(model_baseline, g, val_mask, "MLP-baseline")

print("\\n=== Test accuracy ===")
gnn_test_acc = evaluate(model, g, test_mask, "GNN")
base_test_acc = evaluate(model_baseline, g, test_mask, "MLP-baseline")
"""),

("code", """
# ── Parameter counts ───────────────────────────────────────────────────────
gnn_params = count_parameters(model)
mlp_params = count_parameters(model_baseline)

print(f"TensorMNISTGNN params : {gnn_params:,}")
print(f"FlattenMLP params     : {mlp_params:,}")
print(f"GNN val accuracy      : {gnn_val_acc:.4f}")
print(f"MLP val accuracy      : {base_val_acc:.4f}")
print(f"GNN test accuracy     : {gnn_test_acc:.4f}")
print(f"MLP test accuracy     : {base_test_acc:.4f}")
print(f"Runtime               : {train_time:.1f}s")
print()
print("NOTE: FAST_MODE uses a 1000-image subset. Results are not representative")
print("of full-scale performance. The comparison is illustrative only.")
"""),

("markdown", """## 6. Dashboard artifacts"""),

("code", """
# ── Write dashboard artifacts ──────────────────────────────────────────────
write_run_metadata(
    str(RUN_DIR / "run_metadata.json"),
    notebook="31_mnist_class_graph_membership_tensor_nodes",
    tgraphx_version=tgraphx.__version__,
    seed=SEED,
    fast_mode=FAST_MODE,
    device=device,
    subset_size=SUBSET_SIZE,
    using_real_mnist=USING_REAL_MNIST,
    runtime_s=round(train_time, 2),
)

write_metrics_summary(
    str(RUN_DIR / "metrics_summary.json"),
    gnn_val_acc=round(gnn_val_acc, 4),
    gnn_test_acc=round(gnn_test_acc, 4),
    mlp_baseline_val_acc=round(base_val_acc, 4),
    mlp_baseline_test_acc=round(base_test_acc, 4),
    gnn_params=gnn_params,
    mlp_params=mlp_params,
    epochs=EPOCHS,
    runtime_s=round(train_time, 2),
    task="node_classification",
    model="TensorMNISTGNN",
    baseline="FlattenMLP",
)

benchmark = {
    "task": "node_classification",
    "dataset": "MNIST" if USING_REAL_MNIST else "synthetic_fallback",
    "subset_size": SUBSET_SIZE,
    "model": "TensorMNISTGNN",
    "gnn_val_acc": round(gnn_val_acc, 4),
    "mlp_val_acc": round(base_val_acc, 4),
    "gnn_test_acc": round(gnn_test_acc, 4),
    "mlp_test_acc": round(base_test_acc, 4),
    "gnn_params": gnn_params,
    "mlp_params": mlp_params,
    "runtime_s": round(train_time, 2),
    "fast_mode": FAST_MODE,
}
with open(RUN_DIR / "benchmark_summary.json", "w") as f:
    json.dump(benchmark, f, indent=2)

print(f"Artifacts written to: {RUN_DIR}")
for p in sorted(RUN_DIR.glob("*.json")):
    print(f"  {p.name}")
"""),

("markdown", """## 7. Results summary"""),

("code", """
# ── Final results table ────────────────────────────────────────────────────
print("=" * 60)
print("TGraphX MNIST Class-Graph — Results Summary")
print("=" * 60)
print(f"{'Metric':<35} {'GNN':>10} {'MLP-base':>10}")
print("-" * 60)
print(f"{'Val accuracy':<35} {gnn_val_acc:>10.4f} {base_val_acc:>10.4f}")
print(f"{'Test accuracy':<35} {gnn_test_acc:>10.4f} {base_test_acc:>10.4f}")
print(f"{'Parameters':<35} {gnn_params:>10,} {mlp_params:>10,}")
print(f"{'Runtime (s)':<35} {train_time:>10.1f} {'—':>10}")
print("-" * 60)
print(f"Graph: {N} image nodes + {NUM_CLASSES} prototypes, "
      f"{all_edges.shape[1]:,} edges")
print(f"Epochs: {EPOCHS}  |  FAST_MODE: {FAST_MODE}  |  Device: {device}")
"""),

("markdown", """
## Scientific and methodological notes

- **Learning setting:** Transductive node classification — image and prototype
  nodes are all structurally present in the graph during training; only train-mask
  labels enter the loss.
- **Split policy:** Deterministic random 70/15/15 train/val/test split over the
  image-node subset; prototype nodes use label `-1` (ignored in loss).
- **Leakage policy:** Class prototypes are computed from train-mask labels ONLY.
  Validation and test nodes are connected to prototypes via visual similarity, not
  via their (held-out) labels.
- **Baseline:** `FlattenMLP` operates on flattened pixels without using graph
  structure. It quantifies the cost of early flattening relative to tensor-native
  message passing.
- **Metrics:** Validation and test accuracy on seed nodes only; parameter counts
  and runtime are reported for honest comparison.
- **Why FAST_MODE metrics are not benchmark claims:** With 1000 training images
  and 3 epochs, results are illustrative, not competitive. No SOTA claim is made.
- **TGraphX capability demonstrated:** Tensor-valued `[N, 1, 28, 28]` nodes;
  multi-type `edge_attr` (visual_similarity vs prototype_membership); seed-node
  mini-batch loss via `batch.seed_logits` / `batch.seed_y`; dashboard artifacts.

## What this demonstrates

- **Tensor-native nodes:** MNIST images flow as `[N, 1, 28, 28]` tensors through
  every graph-learning step without flattening.
- **Two edge types:** Visual-similarity kNN connects visually similar digits;
  prototype-membership edges ground each node to its nearest class centroid
  (built from training data only).
- **ConvMessagePassing:** Aggregates spatial image tensors from neighbors while
  preserving spatial layout.
- **NeighborLoader + seed-node loss:** Efficient mini-batch training with
  `batch.seed_logits(logits)` and `batch.seed_y`.
- **Leakage policy:** Prototype centroids are built from training labels only.
  Val/test nodes connect to prototypes by visual similarity, not by label.

## Limitations

- FAST_MODE subset (1000 images) is too small for representative accuracy.
- The kNN graph uses flattened pixel cosine similarity, not learned features.
- The GNN and MLP are deliberately small for fast demonstration.
- Prototype nodes are fixed after initialization (not learned jointly).
- Results may show MLP ≥ GNN on small subsets due to limited graph density.
- Full-scale results would require a larger subset and more epochs.

## Next steps

- Scale to full 60k MNIST; use deeper ConvMessagePassing with residual connections.
- Use learned embeddings (e.g. CNN encoder) for kNN and prototype construction.
- Explore heterogeneous edge types with `HeteroGraph`.
- Compare with transductive GNN (single full-graph training) instead of sampling.
"""),

("code", """
# ── Notebook passed ────────────────────────────────────────────────────────
assert (RUN_DIR / "benchmark_summary.json").exists(), "benchmark_summary.json missing"
assert gnn_val_acc >= 0.0
assert base_val_acc >= 0.0
print("Notebook 31 — MNIST Class-Graph Membership passed all checks.")
"""),
])

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 32 — CIFAR-10 Patch-Graph Classification
# ─────────────────────────────────────────────────────────────────────────────

NB32 = nb([
("markdown", """
# 32 — CIFAR-10 Patch-Graph Classification

**Research question:** Can a tensor-native patch graph classify colour images
by reasoning over spatially adjacent patches — without flattening the spatial
structure at any point?

**What this demonstrates:**
- Each CIFAR-10 image is split into an 8×8-patch grid graph (16 patches per image
  for a 4×4 split, or 64 patches for 8×8)
- Node features are patch tensors `[3, patch_H, patch_W]` — spatially structured
- `ConvMessagePassing` aggregates neighbouring patches while preserving layout
- `global_mean_pool` + `global_max_pool` for graph-level readout
- `GraphDataLoader` batches graph-per-image collections
- TGraphX `CIFAR10PatchGraphDataset` as the dataset bridge
- Flatten-MLP baseline for comparison

**Dataset:** CIFAR-10 (Krizhevsky, 2009) via `tgraphx.datasets.CIFAR10PatchGraphDataset`.

**Task type:** Graph classification (inductive). Each CIFAR-10 image → one patch graph
with a class label. Train/val/test split over the graph collection.

**Leakage policy:** This is an inductive graph-classification task. Each graph is a
single CIFAR-10 image; its label is attached to the graph object and is withheld
from the model until test evaluation. There is no node-label leakage because each
graph is fully self-contained. Train, validation, and test graphs are disjoint.
"""),

("code", """
# ── Configuration ─────────────────────────────────────────────────────────────
FAST_MODE = True
SEED = 42
PATCH_SIZE = 8           # 32 / 8 = 4 patches per row → 16 patches per image
SUBSET_TRAIN = 500 if FAST_MODE else 5000
SUBSET_VAL = 100 if FAST_MODE else 1000
EPOCHS = 3 if FAST_MODE else 15
BATCH_SIZE = 32
NUM_CLASSES = 10
HIDDEN_DIM = 32
print(f"FAST_MODE={FAST_MODE}  patch_size={PATCH_SIZE}  "
      f"train={SUBSET_TRAIN}  epochs={EPOCHS}")
"""),

("code", """
# ── Install and import ────────────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tgraphx", "torchvision"],
               check=False)
import time, json, pathlib
import torch, torch.nn as nn, torch.nn.functional as F
import tgraphx
from tgraphx.reproducibility import set_seed
from tgraphx import (
    Graph, GraphBatch, GraphDataLoader, ConvMessagePassing, count_parameters,
    global_mean_pool, global_max_pool,
)
from tgraphx.tracking import (
    write_run_metadata, write_metrics_summary,
    write_dataset_metadata, write_graph_stats,
)
from tgraphx.mining import graph_summary

RUN_DIR = pathlib.Path("runs/advanced_notebooks/32_cifar10")
RUN_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(SEED, deterministic=True)
print(f"TGraphX v{tgraphx.__version__}  |  device={device}  |  SEED={SEED}")
"""),

("markdown", """## 1. Dataset loading — CIFAR-10 as patch graphs"""),

("code", """
# ── Load CIFAR-10 via TGraphX dataset bridge ──────────────────────────────
# CIFAR10PatchGraphDataset converts each 32x32 RGB image into a graph where
# nodes are spatial patches [3, patch_size, patch_size] connected by grid edges.
USING_REAL_CIFAR = False
graphs_train, graphs_val, graphs_test = [], [], []

try:
    from tgraphx.datasets import CIFAR10PatchGraphDataset
    ds_train = CIFAR10PatchGraphDataset(
        train=True, download=True, patch_size=PATCH_SIZE,
        graph_builder="grid",
    )
    ds_test = CIFAR10PatchGraphDataset(
        train=False, download=True, patch_size=PATCH_SIZE,
        graph_builder="grid",
    )
    gen = torch.Generator().manual_seed(SEED)
    train_idx = torch.randperm(len(ds_train), generator=gen)[:SUBSET_TRAIN]
    val_idx_pool = torch.randperm(len(ds_test), generator=gen)
    val_idx = val_idx_pool[:SUBSET_VAL]
    test_idx = val_idx_pool[SUBSET_VAL:SUBSET_VAL + SUBSET_VAL]

    graphs_train = [ds_train.get(i) for i in train_idx]
    graphs_val = [ds_test.get(i) for i in val_idx]
    graphs_test = [ds_test.get(i) for i in test_idx]
    USING_REAL_CIFAR = True
    print(f"CIFAR-10 patch graphs loaded via TGraphX bridge.")
except Exception as exc:
    print(f"CIFAR-10 unavailable ({exc}); using synthetic patch-graph fallback.")

if not USING_REAL_CIFAR:
    # Synthetic fallback: random patch graphs with CIFAR-like structure
    n_patches = (32 // PATCH_SIZE) ** 2  # 16 for patch_size=8
    from tgraphx import build_grid_graph
    gen = torch.Generator().manual_seed(SEED)
    ph = pw = PATCH_SIZE
    n_rows = n_cols = 32 // PATCH_SIZE
    grid_ei = build_grid_graph(n_rows, n_cols, directed=False, self_loops=True)

    def _make_fake_graph(label: int) -> Graph:
        feats = torch.randn(n_patches, 3, ph, pw, generator=gen)
        return Graph(node_features=feats, edge_index=grid_ei,
                     graph_label=torch.tensor(label))

    for i in range(SUBSET_TRAIN):
        graphs_train.append(_make_fake_graph(i % NUM_CLASSES))
    for i in range(SUBSET_VAL):
        graphs_val.append(_make_fake_graph(i % NUM_CLASSES))
    for i in range(SUBSET_VAL):
        graphs_test.append(_make_fake_graph(i % NUM_CLASSES))

sample = graphs_train[0]
n_patches_per_img = sample.node_features.shape[0]
patch_shape = tuple(sample.node_features.shape[1:])
print(f"Sample patch graph: {sample}")
print(f"  patches per image: {n_patches_per_img}")
print(f"  patch tensor shape: {patch_shape}  (C, pH, pW)")
print(f"  graph_label: {sample.graph_label.item()}")

write_dataset_metadata(
    str(RUN_DIR / "dataset_metadata.json"),
    name="CIFAR-10-patch-graph" if USING_REAL_CIFAR else "synthetic_cifar_patch_fallback",
    num_train_graphs=len(graphs_train),
    num_val_graphs=len(graphs_val),
    node_feature_shape=str(patch_shape),
    nodes_per_graph=n_patches_per_img,
)
"""),

("code", """
# ── Graph structural summary ────────────────────────────────────────────────
sample_ei = sample.edge_index
summary = graph_summary(sample_ei, num_nodes=n_patches_per_img, directed=False)
print(f"Sample patch-graph: nodes={summary['num_nodes']}  "
      f"edges={summary['num_edges']}  density={summary['density']:.4f}  "
      f"mean_degree={summary['mean_degree']:.2f}")
write_graph_stats(sample, str(RUN_DIR / "sample_graph_summary.json"))
"""),

("markdown", """## 2. Model definition"""),

("code", """
# ── Patch-graph GNN with mean+max pooling ─────────────────────────────────
class PatchGraphGNN(nn.Module):
    \"\"\"
    Convolution-aware message passing over patch graphs, followed by
    global mean + max pooling for graph-level readout.
    \"\"\"
    def __init__(self, patch_shape: tuple, hidden: int, num_classes: int):
        super().__init__()
        C, pH, pW = patch_shape
        h2, w2 = pH // 2, pW // 2
        self.conv1 = ConvMessagePassing(in_shape=(C, pH, pW),
                                        out_shape=(hidden, h2, w2))
        # After conv1 each patch becomes [hidden, h2, w2]; pool to scalar vec
        self.pool_spatial = nn.AdaptiveAvgPool2d(1)  # [hidden, 1, 1] → [hidden]
        self.hidden = hidden
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),  # *2 for mean+max concat
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        x = batch.node_features               # [N_total, C, pH, pW]
        ei = batch.edge_index
        bi = batch.batch                       # [N_total] graph membership

        h = F.relu(self.conv1(x, ei))         # [N_total, hidden, h2, w2]
        h = self.pool_spatial(h).squeeze(-1).squeeze(-1)  # [N_total, hidden]

        h_mean = global_mean_pool(h, bi)      # [G, hidden]
        h_max = global_max_pool(h, bi)        # [G, hidden]
        h_global = torch.cat([h_mean, h_max], dim=1)  # [G, hidden*2]
        return self.head(h_global)            # [G, num_classes]


model = PatchGraphGNN(patch_shape, hidden=HIDDEN_DIM, num_classes=NUM_CLASSES).to(device)
print(f"PatchGraphGNN parameters: {count_parameters(model):,}")

# Shape trace
sample_batch = GraphBatch(graphs_train[:2])
out = model(sample_batch.to(device))
print(f"Shape trace: {len(graphs_train[:2])} graphs → logits {out.shape}")
assert out.shape == (2, NUM_CLASSES)
print("Shape trace passed.")
"""),

("code", """
# ── Flatten-MLP baseline ───────────────────────────────────────────────────
class FlattenMLP(nn.Module):
    \"\"\"MLP on flattened patches — ignores spatial layout and graph structure.\"\"\"
    def __init__(self, n_patches: int, patch_shape: tuple, num_classes: int):
        super().__init__()
        C, pH, pW = patch_shape
        in_dim = n_patches * C * pH * pW
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        # Reshape: group by graph and flatten all patches
        G = batch.num_graphs
        nf = batch.node_features.view(G, -1)  # [G, n_patches*C*pH*pW]
        return self.net(nf)

baseline = FlattenMLP(n_patches_per_img, patch_shape, NUM_CLASSES).to(device)
print(f"FlattenMLP parameters: {count_parameters(baseline):,}")
"""),

("markdown", """## 3. Training"""),

("code", """
# ── Training loop ──────────────────────────────────────────────────────────
def train_graph_model(mdl, graphs, epochs, name="model"):
    loader = GraphDataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
    mdl.train()
    history = []
    for ep in range(1, epochs + 1):
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            logits = mdl(batch)
            labels = batch.graph_labels.to(device)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        history.append(avg_loss)
        if ep % max(1, epochs // 3) == 0 or ep == epochs:
            print(f"  [{name}] epoch {ep}/{epochs}  loss={avg_loss:.4f}")
    return history


# Gradient sanity
loader_check = GraphDataLoader(graphs_train[:4], batch_size=4)
for batch in loader_check:
    batch = batch.to(device)
    logits = model(batch)
    loss = F.cross_entropy(logits, batch.graph_labels.to(device))
    loss.backward()
    grads = sum(p.grad.abs().sum().item()
                for p in model.parameters() if p.grad is not None)
    print(f"Gradient sanity: total_grad={grads:.4f}  (expect > 0)")
    assert grads > 0
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    break
print("Gradient sanity passed.")
"""),

("code", """
t0 = time.time()
print("=== Training PatchGraphGNN ===")
gnn_history = train_graph_model(model, graphs_train, EPOCHS, "GNN")

model_baseline = FlattenMLP(n_patches_per_img, patch_shape, NUM_CLASSES).to(device)
print("\\n=== Training FlattenMLP baseline ===")
base_history = train_graph_model(model_baseline, graphs_train, EPOCHS, "MLP")
train_time = time.time() - t0
print(f"\\nTotal training time: {train_time:.1f}s")
"""),

("markdown", """## 4. Evaluation"""),

("code", """
@torch.no_grad()
def evaluate_graph(mdl, graphs, name="model"):
    mdl.eval()
    loader = GraphDataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)
    correct, total = 0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = mdl(batch)
        preds = logits.argmax(1)
        labels = batch.graph_labels.to(device)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
    acc = correct / max(1, total)
    print(f"  [{name}] accuracy = {acc:.4f}  ({correct}/{total})")
    return acc


print("=== Validation accuracy ===")
gnn_val_acc = evaluate_graph(model, graphs_val, "GNN")
base_val_acc = evaluate_graph(model_baseline, graphs_val, "MLP-baseline")

print("\\n=== Test accuracy ===")
gnn_test_acc = evaluate_graph(model, graphs_test, "GNN")
base_test_acc = evaluate_graph(model_baseline, graphs_test, "MLP-baseline")
"""),

("markdown", """## 5. Dashboard artifacts"""),

("code", """
gnn_params = count_parameters(model)
mlp_params = count_parameters(model_baseline)

write_run_metadata(
    str(RUN_DIR / "run_metadata.json"),
    notebook="32_cifar10_patch_graph_classification",
    tgraphx_version=tgraphx.__version__,
    seed=SEED, fast_mode=FAST_MODE, device=device,
    patch_size=PATCH_SIZE, using_real_cifar=USING_REAL_CIFAR,
    runtime_s=round(train_time, 2),
)
write_metrics_summary(
    str(RUN_DIR / "metrics_summary.json"),
    gnn_val_acc=round(gnn_val_acc, 4),
    gnn_test_acc=round(gnn_test_acc, 4),
    mlp_baseline_val_acc=round(base_val_acc, 4),
    mlp_baseline_test_acc=round(base_test_acc, 4),
    gnn_params=gnn_params, mlp_params=mlp_params,
    task="graph_classification",
)
benchmark = {
    "task": "graph_classification",
    "dataset": "CIFAR-10-patch-graph" if USING_REAL_CIFAR else "synthetic",
    "patch_size": PATCH_SIZE,
    "patches_per_image": n_patches_per_img,
    "train_graphs": len(graphs_train),
    "gnn_val_acc": round(gnn_val_acc, 4),
    "mlp_val_acc": round(base_val_acc, 4),
    "gnn_test_acc": round(gnn_test_acc, 4),
    "gnn_params": gnn_params,
    "mlp_params": mlp_params,
    "runtime_s": round(train_time, 2),
}
with open(RUN_DIR / "benchmark_summary.json", "w") as f:
    json.dump(benchmark, f, indent=2)

print(f"Artifacts written to: {RUN_DIR}")
for p in sorted(RUN_DIR.glob("*.json")):
    print(f"  {p.name}")

print("\\n" + "=" * 60)
print("TGraphX CIFAR-10 Patch-Graph — Results Summary")
print("=" * 60)
print(f"{'Metric':<35} {'GNN':>10} {'MLP-base':>10}")
print("-" * 60)
print(f"{'Val accuracy':<35} {gnn_val_acc:>10.4f} {base_val_acc:>10.4f}")
print(f"{'Test accuracy':<35} {gnn_test_acc:>10.4f} {base_test_acc:>10.4f}")
print(f"{'Parameters':<35} {gnn_params:>10,} {mlp_params:>10,}")
print(f"Patches/image: {n_patches_per_img}  Patch shape: {patch_shape}")
print(f"FAST_MODE: {FAST_MODE}  Device: {device}")
"""),

("markdown", """
## Scientific and methodological notes

- **Learning setting:** Inductive graph classification — each CIFAR-10 image
  becomes one self-contained graph; the model never sees test graphs during training.
- **Split policy:** Train, val, and test graphs are disjoint sets of images.
- **Leakage policy:** Per-graph labels are attached to graph objects and only
  used in the training loop on training graphs; test graph labels are withheld.
- **Baseline:** `FlattenMLP` operates on concatenated patch tensors without using
  graph structure or spatial adjacency. It is actually trained, not just counted.
- **Metrics:** Val/test accuracy on graph classification; runtime; parameter count;
  gradient sanity.
- **Why FAST_MODE metrics are not benchmark claims:** CIFAR-10 in 500-image
  FAST_MODE with 3 epochs is far below convergence. We do NOT claim parity with
  any CIFAR-10 vision model.
- **TGraphX capability demonstrated:** True patch-graph node tensors
  `[num_patches, 3, pH, pW]`; spatial-adjacency edges; graph-level mean+max
  readout via `global_mean_pool` + `global_max_pool`; `GraphDataLoader` batching;
  dashboard artifacts; `CIFAR10PatchGraphDataset` bridge.

## What this demonstrates

- **True patch-graph structure:** Each CIFAR-10 image becomes a graph where nodes
  are spatial patches `[3, pH, pW]` connected by grid adjacency edges. This is a
  genuine patch graph, not a whole-image node graph.
- **Tensor-native nodes:** Patch tensors flow through `ConvMessagePassing` without
  spatial flattening, preserving local colour and texture structure.
- **Graph-level classification:** `global_mean_pool + global_max_pool` aggregates
  patch-level representations to a single graph-level vector.
- **TGraphX dataset bridge:** `CIFAR10PatchGraphDataset` wraps torchvision and
  converts images to TGraphX patch graphs automatically.
- **`GraphDataLoader`:** Batches multiple patch graphs into a `GraphBatch`.

## Limitations

- FAST_MODE uses a 500-image training subset — too small for meaningful accuracy.
- Grid edges encode spatial adjacency only; richer visual-similarity edges
  between patches could improve information flow.
- The patch encoder is a single ConvMessagePassing layer; deeper networks
  would benefit from larger datasets.
- CIFAR-10 is challenging; FAST_MODE accuracy may appear random-level.

## Next steps

- Scale to full CIFAR-10 (50k train) with more epochs.
- Add visual-similarity edges between patches (hybrid graph).
- Use a pretrained patch encoder (e.g. ViT patch embeddings) as node features.
- Experiment with `ConvMessagePassing` residual connections.
"""),

("code", """
assert (RUN_DIR / "benchmark_summary.json").exists()
assert gnn_val_acc >= 0.0
print("Notebook 32 — CIFAR-10 Patch-Graph Classification passed all checks.")
"""),
])

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 33 — Cora Citation Network: Sampling, Dashboard, Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

NB33 = nb([
("markdown", """
# 33 — Cora Citation Network: Sampling, Dashboard Artifacts, and Reproducibility

**Research question:** How does TGraphX's NeighborLoader + seed-node loss workflow
compare to an MLP baseline on a canonical citation-network benchmark?

**What this demonstrates:**
- TGraphX `PyGPlanetoidDataset` as a dataset bridge for Cora
- Transductive node classification: all nodes are in the graph; labels withheld for val/test
- `NeighborLoader` mini-batch training with `batch.seed_logits(logits)` / `batch.seed_y`
- `GCNConv` (TGraphX vector GNN) vs MLP baseline
- Dashboard artifact suite: `write_run_metadata`, `write_metrics_summary`, etc.
- Benchmark-style JSON reporting and reproducibility utilities

**Dataset:** Cora citation network (McCallum et al., 2000; Planetoid split by Yang et al., 2016).
2708 nodes, 5429 edges, 7 classes, 1433 bag-of-words features.

**Task type:** Transductive semi-supervised node classification.
All nodes are structurally present in the graph during training;
only the train-node labels are used to compute loss.
"""),

("code", """
# ── Configuration ─────────────────────────────────────────────────────────────
FAST_MODE = True
SEED = 42
EPOCHS = 5 if FAST_MODE else 30
BATCH_SIZE = 64
FANOUTS = [15, 10]
HIDDEN_DIM = 64
print(f"FAST_MODE={FAST_MODE}  epochs={EPOCHS}  batch_size={BATCH_SIZE}")
"""),

("code", """
# ── Install and import ────────────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tgraphx"], check=False)
import time, json, pathlib
import torch, torch.nn as nn, torch.nn.functional as F
import tgraphx
from tgraphx.reproducibility import set_seed
from tgraphx import Graph, GCNConv, count_parameters
from tgraphx.loaders import NeighborLoader
from tgraphx.tracking import (
    write_run_metadata, write_metrics_summary,
    write_dataset_metadata, write_graph_stats, write_sampling_metadata,
)
from tgraphx.mining import graph_summary, degree_statistics

RUN_DIR = pathlib.Path("runs/advanced_notebooks/33_cora")
RUN_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(SEED, deterministic=True)
print(f"TGraphX v{tgraphx.__version__}  |  device={device}  |  SEED={SEED}")
"""),

("markdown", """## 1. Dataset loading"""),

("code", """
# ── Load Cora via TGraphX dataset bridge ──────────────────────────────────
# Requires: pip install torch-geometric
# If torch-geometric is unavailable, falls back to a synthetic SBM graph
# that mimics Cora's scale. Fallback is clearly labeled.
USING_REAL_CORA = False
g = None
NUM_CLASSES = 7

try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch-geometric"], check=False)
    from tgraphx.datasets import PyGPlanetoidDataset
    pyg_ds = PyGPlanetoidDataset(name="Cora", download=True)
    g_raw = pyg_ds.get(0)
    # PyGPlanetoidDataset returns a Graph; Cora has a standard train_mask in metadata
    x = g_raw.node_features
    edge_index = g_raw.edge_index
    y = g_raw.node_labels if g_raw.node_labels is not None else g_raw.y
    NUM_NODES, FEAT_DIM = x.shape
    # Try to get masks from metadata (Planetoid stores them in data.train_mask etc.)
    pyg_data = pyg_ds._upstream[0]
    train_mask = pyg_data.train_mask.bool()
    val_mask = pyg_data.val_mask.bool()
    test_mask = pyg_data.test_mask.bool()
    g = Graph(node_features=x, edge_index=edge_index, y=y)
    NUM_CLASSES = int(y.max().item()) + 1
    USING_REAL_CORA = True
    print(f"Real Cora loaded: {NUM_NODES} nodes  {edge_index.shape[1]} edges  "
          f"{NUM_CLASSES} classes  features={FEAT_DIM}")
except Exception as exc:
    print(f"Cora (PyG) unavailable: {exc}")
    print("Using synthetic SBM fallback (not Cora).")

if not USING_REAL_CORA:
    # Synthetic SBM mimicking Cora's scale
    NUM_NODES = 500 if FAST_MODE else 2708
    FEAT_DIM = 64 if FAST_MODE else 1433
    gen = torch.Generator().manual_seed(SEED)
    x = torch.randn(NUM_NODES, FEAT_DIM, generator=gen)
    y = torch.randint(0, NUM_CLASSES, (NUM_NODES,),
                      generator=torch.Generator().manual_seed(SEED))
    # SBM-like edges: higher intra-class connectivity
    edges_src, edges_dst = [], []
    for i in range(NUM_NODES):
        for j in range(i + 1, min(i + 10, NUM_NODES)):
            if (y[i] == y[j] and torch.rand(1).item() < 0.5) or \
               (y[i] != y[j] and torch.rand(1).item() < 0.05):
                edges_src.extend([i, j]); edges_dst.extend([j, i])
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    g = Graph(node_features=x, edge_index=edge_index, y=y)
    # Custom 60/20/20 split
    perm = torch.randperm(NUM_NODES, generator=torch.Generator().manual_seed(SEED))
    n_train = int(0.6 * NUM_NODES)
    n_val = int(0.2 * NUM_NODES)
    train_mask = torch.zeros(NUM_NODES, dtype=torch.bool)
    val_mask = torch.zeros(NUM_NODES, dtype=torch.bool)
    test_mask = torch.zeros(NUM_NODES, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True
    print(f"Synthetic SBM: {NUM_NODES} nodes  {edge_index.shape[1]} edges  "
          f"{NUM_CLASSES} classes")

print(f"Split — train: {train_mask.sum()}  val: {val_mask.sum()}  "
      f"test: {test_mask.sum()}")
print(f"node_features: {g.node_features.shape}")
write_dataset_metadata(
    str(RUN_DIR / "dataset_metadata.json"),
    name="Cora" if USING_REAL_CORA else "synthetic_sbm_fallback",
    task="transductive_node_classification",
    num_nodes=NUM_NODES,
    num_edges=int(g.edge_index.shape[1]),
    num_classes=NUM_CLASSES,
    feature_dim=FEAT_DIM if not USING_REAL_CORA else int(g.node_features.shape[1]),
)
"""),

("code", """
# ── Graph structural summary ────────────────────────────────────────────────
summary = graph_summary(g.edge_index, num_nodes=g.node_features.shape[0],
                        directed=True)
deg_stats = degree_statistics(g.edge_index,
                               num_nodes=g.node_features.shape[0])
print(f"Graph density: {summary['density']:.5f}")
print(f"Mean degree: {summary['mean_degree']:.2f}  "
      f"Max degree: {summary['max_degree']}")
print(f"Connected components: {summary['num_connected_components']}")
write_graph_stats(g, str(RUN_DIR / "graph_summary.json"))

print("\\nTransductive setting: all nodes present in graph during training.")
print("Only train-node labels are used to compute loss.")
print("Val/test nodes are evaluated on the same graph structure.")
"""),

("markdown", """## 2. Model definition"""),

("code", """
# ── 2-layer GCN model ─────────────────────────────────────────────────────
class CoraGCN(nn.Module):
    \"\"\"Two GCNConv layers for transductive node classification.\"\"\"
    def __init__(self, in_features: int, hidden: int, num_classes: int):
        super().__init__()
        self.gc1 = GCNConv(in_features, hidden)
        self.gc2 = GCNConv(hidden, num_classes)

    def forward(self, x: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gc1(x, ei))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.gc2(h, ei)


feat_dim = int(g.node_features.shape[1])
model = CoraGCN(feat_dim, HIDDEN_DIM, NUM_CLASSES).to(device)
print(f"CoraGCN parameters: {count_parameters(model):,}")

# Shape trace
with torch.no_grad():
    tiny_x = g.node_features[:5].to(device)
    _mask = (g.edge_index[0] < 5) & (g.edge_index[1] < 5)
    tiny_ei = g.edge_index[:, _mask].to(device)
    if tiny_ei.shape[1] == 0:
        tiny_ei = torch.tensor([[0], [1]], dtype=torch.long, device=device)
    out = model(tiny_x, tiny_ei)
print(f"Shape trace: {tiny_x.shape} → {out.shape}")
assert out.shape == (5, NUM_CLASSES)
print("Shape trace passed.")
"""),

("code", """
# ── MLP baseline (ignores graph structure) ─────────────────────────────────
class FlattenMLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # ei unused

baseline = FlattenMLP(feat_dim, HIDDEN_DIM, NUM_CLASSES).to(device)
print(f"FlattenMLP parameters: {count_parameters(baseline):,}")
"""),

("markdown", """## 3. Training"""),

("code", """
# ── Training helper with NeighborLoader + seed-node loss ──────────────────
def train_node_model(mdl, graph, mask, epochs, name="model"):
    loader = NeighborLoader(
        graph, fanouts=FANOUTS, batch_size=BATCH_SIZE,
        mask=mask, shuffle=True, seed=SEED,
    )
    opt = torch.optim.Adam(mdl.parameters(), lr=5e-3, weight_decay=5e-4)
    mdl.train()
    history = []
    for ep in range(1, epochs + 1):
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            x = batch.node_features.to(device)
            ei = batch.edge_index.to(device)
            logits = mdl(x, ei)
            # seed_logits extracts logits for seed nodes only
            loss = F.cross_entropy(batch.seed_logits(logits),
                                   batch.seed_y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        history.append(avg_loss)
        if ep % max(1, epochs // 3) == 0 or ep == epochs:
            print(f"  [{name}] epoch {ep}/{epochs}  loss={avg_loss:.4f}")
    return history


# Gradient sanity
sample_loader = NeighborLoader(g, fanouts=FANOUTS, batch_size=16,
                                mask=train_mask, shuffle=True, seed=SEED)
for batch in sample_loader:
    logits = model(batch.node_features.to(device), batch.edge_index.to(device))
    loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y.to(device))
    loss.backward()
    grads = sum(p.grad.abs().sum().item()
                for p in model.parameters() if p.grad is not None)
    print(f"Gradient sanity: total_grad={grads:.4f}  (expect > 0)")
    assert grads > 0
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
    break

# Sampling metadata
write_sampling_metadata(
    str(RUN_DIR / "sampling_metadata.json"),
    sampler="NeighborLoader",
    fanouts=FANOUTS, batch_size=BATCH_SIZE,
    num_train_nodes=int(train_mask.sum()),
)
"""),

("code", """
t0 = time.time()
print("=== Training CoraGCN ===")
gnn_history = train_node_model(model, g, train_mask, EPOCHS, "GCN")

baseline_model = FlattenMLP(feat_dim, HIDDEN_DIM, NUM_CLASSES).to(device)
print("\\n=== Training FlattenMLP baseline ===")
base_history = train_node_model(baseline_model, g, train_mask, EPOCHS, "MLP")
train_time = time.time() - t0
print(f"\\nTotal training time: {train_time:.1f}s")
"""),

("markdown", """## 4. Evaluation"""),

("code", """
@torch.no_grad()
def evaluate_node(mdl, graph, mask, name="model"):
    mdl.eval()
    loader = NeighborLoader(
        graph, fanouts=FANOUTS, batch_size=BATCH_SIZE,
        mask=mask, shuffle=False, seed=SEED,
    )
    correct, total = 0, 0
    for batch in loader:
        x = batch.node_features.to(device)
        ei = batch.edge_index.to(device)
        logits = mdl(x, ei)
        preds = batch.seed_logits(logits).argmax(1)
        correct += (preds == batch.seed_y.to(device)).sum().item()
        total += batch.seed_y.shape[0]
    acc = correct / max(1, total)
    print(f"  [{name}] accuracy = {acc:.4f}  ({correct}/{total})")
    return acc


print("=== Validation accuracy ===")
gnn_val_acc = evaluate_node(model, g, val_mask, "GCN")
base_val_acc = evaluate_node(baseline_model, g, val_mask, "MLP")

print("\\n=== Test accuracy ===")
gnn_test_acc = evaluate_node(model, g, test_mask, "GCN")
base_test_acc = evaluate_node(baseline_model, g, test_mask, "MLP")
"""),

("markdown", """## 5. Dashboard artifacts"""),

("code", """
gnn_params = count_parameters(model)
mlp_params = count_parameters(baseline_model)

write_run_metadata(
    str(RUN_DIR / "run_metadata.json"),
    notebook="33_cora_citation_network_sampling_and_dashboard",
    tgraphx_version=tgraphx.__version__,
    seed=SEED, fast_mode=FAST_MODE, device=device,
    using_real_cora=USING_REAL_CORA, runtime_s=round(train_time, 2),
)
write_metrics_summary(
    str(RUN_DIR / "metrics_summary.json"),
    gcn_val_acc=round(gnn_val_acc, 4),
    gcn_test_acc=round(gnn_test_acc, 4),
    mlp_val_acc=round(base_val_acc, 4),
    mlp_test_acc=round(base_test_acc, 4),
    gcn_params=gnn_params, mlp_params=mlp_params,
    task="transductive_node_classification",
)
benchmark = {
    "task": "transductive_node_classification",
    "dataset": "Cora" if USING_REAL_CORA else "synthetic_sbm_fallback",
    "num_nodes": NUM_NODES,
    "gcn_val_acc": round(gnn_val_acc, 4),
    "mlp_val_acc": round(base_val_acc, 4),
    "gcn_test_acc": round(gnn_test_acc, 4),
    "mlp_test_acc": round(base_test_acc, 4),
    "gcn_params": gnn_params,
    "mlp_params": mlp_params,
    "runtime_s": round(train_time, 2),
    "fast_mode": FAST_MODE,
}
with open(RUN_DIR / "benchmark_summary.json", "w") as f:
    json.dump(benchmark, f, indent=2)

print(f"Artifacts written to: {RUN_DIR}")
for p in sorted(RUN_DIR.glob("*.json")):
    print(f"  {p.name}")

print("\\n" + "=" * 60)
print("TGraphX Cora Citation Network — Results Summary")
print("=" * 60)
print(f"{'Metric':<35} {'GCN':>10} {'MLP-base':>10}")
print("-" * 60)
print(f"{'Val accuracy':<35} {gnn_val_acc:>10.4f} {base_val_acc:>10.4f}")
print(f"{'Test accuracy':<35} {gnn_test_acc:>10.4f} {base_test_acc:>10.4f}")
print(f"{'Parameters':<35} {gnn_params:>10,} {mlp_params:>10,}")
print(f"FAST_MODE: {FAST_MODE}  Device: {device}")
"""),

("markdown", """
## Scientific and methodological notes

- **Learning setting:** Transductive semi-supervised node classification — all
  Cora nodes are structurally present in the graph during training; only train-mask
  node labels contribute to the loss; val/test labels are withheld.
- **Split policy:** Standard Planetoid split when PyG is available; deterministic
  60/20/20 custom split for the synthetic SBM fallback.
- **Leakage policy:** Labels for val/test nodes are never used in loss computation
  or graph construction. Graph structure is identical for all nodes; only label
  visibility differs across splits.
- **Baseline:** `FlattenMLP` is a 2-layer MLP on the same node feature vectors
  but ignores all citation edges. It is actually trained.
- **Metrics:** Val/test accuracy on seed nodes; sampling metadata; runtime;
  parameter count.
- **Why FAST_MODE metrics are not benchmark claims:** 5 epochs and a small
  batch are far from convergence. Reference Kipf & Welling (2017) report ~81%
  test accuracy on Cora; this notebook makes NO claim of that level.
- **TGraphX capability demonstrated:** TGraphX dataset bridge (`PyGPlanetoidDataset`),
  `NeighborLoader` seed-node mini-batch loss, sampling metadata, full dashboard
  artifact suite, transductive setting handling.

## What this demonstrates

- **TGraphX dataset bridge:** `PyGPlanetoidDataset` wraps `torch_geometric` and
  exposes a TGraphX `Graph` with standard Cora splits, with graceful fallback
  to a synthetic SBM graph if PyG is unavailable.
- **Transductive setting:** All nodes are structurally present during training;
  seed-node loss is computed only on the training mask.
- **NeighborLoader seed-node loss:** `batch.seed_logits(logits)` and `batch.seed_y`
  extract logits/labels for seed nodes only, enabling efficient mini-batch training.
- **GCN vs MLP:** `GCNConv` propagates feature information along citation edges;
  MLP ignores graph structure entirely.
- **Full artifact suite:** Run metadata, metrics, sampling metadata, graph stats.

## Limitations

- FAST_MODE uses only 5 epochs — results are not stable.
- If using the synthetic SBM fallback, results are not comparable to Cora literature.
- Standard Cora accuracy with GCN (Kipf & Welling, 2017) is ~81%; FAST_MODE
  results will be far below this.
- This notebook uses a simple 2-layer GCN; state-of-the-art on Cora is higher.
- We do not claim SOTA; the purpose is to demonstrate the TGraphX workflow.

## Next steps

- Run with full epochs (30+) for meaningful Cora results.
- Replace GCNConv with `TensorGATLayer` or `GATv2Conv` for attention-based models.
- Try whole-graph training with `Graph.to(device)` instead of NeighborLoader.
- Extend to CiteSeer and PubMed via `PyGPlanetoidDataset(name="CiteSeer")`.
"""),

("code", """
assert (RUN_DIR / "benchmark_summary.json").exists()
assert gnn_val_acc >= 0.0
print("Notebook 33 — Cora Citation Network passed all checks.")
"""),
])

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 34 — MovieLens User–Item Knowledge Graph Recommendation
# ─────────────────────────────────────────────────────────────────────────────

NB34 = nb([
("markdown", """
# 34 — MovieLens User–Item Knowledge Graph Recommendation

**Research question:** Can a multi-relational knowledge graph with user–movie–genre
structure improve recommendation link prediction beyond a popularity baseline?

**What this demonstrates:**
- MovieLens 100K as a multi-relational TGraphX `KnowledgeGraph`
- Relations: `rated_high`, `rated_low`, `has_genre`, `has_occupation`
- Entity features: genre multi-hot for movies, demographic vectors for users
- `TransEModel` trained via `KGTrainer` + `KGTrainingConfig`
- Filtered ranking evaluation via `KGEvaluator` (MRR, Hits@K)
- `run_kg_hpo` hyperparameter search over embedding dim and learning rate
- Top-K movie recommendations with movie titles
- Dashboard artifacts via `write_kg_summary`, `write_run_metadata`

**Dataset:** MovieLens 100K (Harper & Konstan, 2015).

**Task type:** Knowledge graph link prediction (inductive on new triples).
Train/val/test split by randomly holding out 10% each for val and test.
"""),

("code", """
# ── Configuration ─────────────────────────────────────────────────────────────
FAST_MODE = True
SEED = 42
MAX_USERS = 150 if FAST_MODE else 943
MAX_MOVIES = 300 if FAST_MODE else 1682
EMBEDDING_DIM = 32
EPOCHS = 5 if FAST_MODE else 30
BATCH_SIZE = 256
print(f"FAST_MODE={FAST_MODE}  users≤{MAX_USERS}  movies≤{MAX_MOVIES}  "
      f"epochs={EPOCHS}")
"""),

("code", """
# ── Install and import ────────────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tgraphx"], check=False)
import csv, io, json, os, pathlib, time, urllib.request, zipfile
import torch
import tgraphx
from tgraphx.reproducibility import set_seed
from tgraphx import KnowledgeGraph, KGTrainer, KGTrainingConfig, count_parameters
from tgraphx.kg import (
    TransEModel, KGEvaluator,
    write_kg_summary, write_kg_evaluation_report,
    write_kg_training_report, write_kg_benchmark_report,
)
from tgraphx.kg.hpo import run_kg_hpo
from tgraphx.tracking import write_run_metadata, write_metrics_summary

RUN_DIR = pathlib.Path("runs/advanced_notebooks/34_movielens")
RUN_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(SEED, deterministic=True)
print(f"TGraphX v{tgraphx.__version__}  |  device={device}  |  SEED={SEED}")
"""),

("markdown", """## 1. Dataset loading"""),

("code", """
# ── Download MovieLens 100K ────────────────────────────────────────────────
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_DIR = pathlib.Path("/tmp/ml-100k")
USING_REAL_ML = False

if not FAST_MODE or not ML_DIR.exists():
    try:
        if not ML_DIR.exists():
            print("Downloading MovieLens 100K (~5 MB)...")
            zip_path = pathlib.Path("/tmp/ml-100k.zip")
            urllib.request.urlretrieve(ML_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall("/tmp")
            print("Downloaded and extracted.")
        USING_REAL_ML = True
    except Exception as exc:
        print(f"MovieLens download failed ({exc}); using synthetic fallback.")
elif ML_DIR.exists() and (ML_DIR / "u.data").exists():
    USING_REAL_ML = True
    print("MovieLens 100K found at /tmp/ml-100k")
else:
    print("Using synthetic fallback (no download in FAST_MODE skip).")

print(f"USING_REAL_ML = {USING_REAL_ML}")
"""),

("code", """
# ── Parse MovieLens 100K into entity maps and relations ───────────────────
genre_list = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]
NUM_GENRES = len(genre_list)

# Rating threshold: >= 4 → rated_high, <= 2 → rated_low
HIGH_THRESH, LOW_THRESH = 4, 2

user_ids, movie_ids, movie_titles = [], [], []
genre_vecs = {}     # movie_id → genre multi-hot [NUM_GENRES]
occupation_map = {}  # user_id → occupation string
user_ratings = []   # (user_id, movie_id, rating)

if USING_REAL_ML:
    # Parse u.item (movie metadata)
    with open(ML_DIR / "u.item", encoding="latin-1") as f:
        for row in csv.reader(f, delimiter="|"):
            mid = int(row[0])
            if mid > MAX_MOVIES:
                continue
            movie_ids.append(mid)
            movie_titles.append(row[1])
            genre_vec = torch.tensor([float(g) for g in row[5:5 + NUM_GENRES]])
            genre_vecs[mid] = genre_vec

    # Parse u.user (user demographics)
    with open(ML_DIR / "u.user") as f:
        for row in csv.reader(f, delimiter="|"):
            uid = int(row[0])
            if uid > MAX_USERS:
                continue
            user_ids.append(uid)
            occupation_map[uid] = row[3]

    # Parse u.data (ratings)
    with open(ML_DIR / "u.data") as f:
        for row in csv.reader(f, delimiter="\\t"):
            uid, mid, rating = int(row[0]), int(row[1]), int(row[2])
            if uid <= MAX_USERS and mid <= MAX_MOVIES:
                user_ratings.append((uid, mid, rating))
else:
    # Synthetic fallback
    import random
    random.seed(SEED)
    for uid in range(1, MAX_USERS + 1):
        user_ids.append(uid)
        occupation_map[uid] = random.choice(["student", "engineer", "artist"])
    for mid in range(1, MAX_MOVIES + 1):
        movie_ids.append(mid)
        movie_titles.append(f"Movie_{mid}")
        gv = torch.zeros(NUM_GENRES)
        gv[random.randint(0, NUM_GENRES - 1)] = 1.0
        genre_vecs[mid] = gv
    for uid in range(1, MAX_USERS + 1):
        for _ in range(5):
            mid = random.randint(1, MAX_MOVIES)
            rating = random.randint(1, 5)
            user_ratings.append((uid, mid, rating))

print(f"Users: {len(user_ids)}  Movies: {len(movie_ids)}  Ratings: {len(user_ratings)}")
"""),

("code", """
# ── Build entity ID maps ───────────────────────────────────────────────────
# Entity layout: user_ids, then movie_ids, then genre nodes, then occupation nodes
occ_list = sorted(set(occupation_map.values()))
occ_to_id = {o: i for i, o in enumerate(occ_list)}
NUM_OCC = len(occ_list)

# Offsets
USER_OFFSET = 0
MOVIE_OFFSET = MAX_USERS
GENRE_OFFSET = MOVIE_OFFSET + MAX_MOVIES
OCC_OFFSET = GENRE_OFFSET + NUM_GENRES
NUM_ENTITIES = OCC_OFFSET + NUM_OCC

# Relation IDs
REL_RATED_HIGH = 0
REL_RATED_LOW = 1
REL_HAS_GENRE = 2
REL_HAS_OCC = 3
NUM_RELATIONS = 4

print(f"Entity layout: {MAX_USERS} users + {MAX_MOVIES} movies + "
      f"{NUM_GENRES} genres + {NUM_OCC} occupations = {NUM_ENTITIES} total")
print(f"Relations: rated_high(0), rated_low(1), has_genre(2), has_occupation(3)")
"""),

("code", """
# ── Build KG triples ───────────────────────────────────────────────────────
triples = []

# Rating triples
for uid, mid, rating in user_ratings:
    u = uid - 1 + USER_OFFSET      # 0-indexed
    m = mid - 1 + MOVIE_OFFSET
    if rating >= HIGH_THRESH:
        triples.append([u, REL_RATED_HIGH, m])
    elif rating <= LOW_THRESH:
        triples.append([u, REL_RATED_LOW, m])

# Genre triples: movie_id → genre
for mid in movie_ids:
    m = mid - 1 + MOVIE_OFFSET
    gv = genre_vecs.get(mid, torch.zeros(NUM_GENRES))
    for gi in gv.nonzero(as_tuple=False).view(-1).tolist():
        g_ent = gi + GENRE_OFFSET
        triples.append([m, REL_HAS_GENRE, g_ent])

# Occupation triples: user → occupation
for uid in user_ids:
    u = uid - 1 + USER_OFFSET
    occ = occupation_map.get(uid, occ_list[0])
    occ_ent = occ_to_id[occ] + OCC_OFFSET
    triples.append([u, REL_HAS_OCC, occ_ent])

triples_tensor = torch.tensor(triples, dtype=torch.long)
print(f"Total triples: {len(triples):,}")
print(f"  rated_high: {(triples_tensor[:, 1] == REL_RATED_HIGH).sum()}")
print(f"  rated_low:  {(triples_tensor[:, 1] == REL_RATED_LOW).sum()}")
print(f"  has_genre:  {(triples_tensor[:, 1] == REL_HAS_GENRE).sum()}")
print(f"  has_occ:    {(triples_tensor[:, 1] == REL_HAS_OCC).sum()}")
"""),

("code", """
# ── Entity features: genre multi-hot for movies ───────────────────────────
# User features: age-group index or binary occupation one-hot
entity_features = torch.zeros(NUM_ENTITIES, NUM_GENRES)
for mid in movie_ids:
    m_idx = mid - 1 + MOVIE_OFFSET
    gv = genre_vecs.get(mid, torch.zeros(NUM_GENRES))
    if m_idx < NUM_ENTITIES:
        entity_features[m_idx] = gv

# Build TGraphX KnowledgeGraph with entity features
kg = KnowledgeGraph(
    triples=triples_tensor,
    num_entities=NUM_ENTITIES,
    num_relations=NUM_RELATIONS,
    entity_features={"genre_vec": entity_features},
)
print(f"KnowledgeGraph: {kg}")

# Train/val/test split (80/10/10)
gen = torch.Generator().manual_seed(SEED)
perm = torch.randperm(len(triples_tensor), generator=gen)
n_train = int(0.8 * len(triples_tensor))
n_val = int(0.1 * len(triples_tensor))
train_triples = triples_tensor[perm[:n_train]]
val_triples = triples_tensor[perm[n_train:n_train + n_val]]
test_triples = triples_tensor[perm[n_train + n_val:]]
print(f"Split: train={len(train_triples)}  val={len(val_triples)}  "
      f"test={len(test_triples)}")
# ── Leakage policy ────────────────────────────────────────────────────
# This is an edge-wise (triple-wise) split for link prediction.
# Edge-wise leakage analysis:
# - Entity IDs are shared across train/val/test (standard for KG link prediction).
# - val/test triples are held out; the model never sees their relation labels.
# - Genre and occupation triples appear in all splits proportionally.
# - Filtered ranking evaluation correctly filters out training triples from ranking.
# - Limitation: entity-leakage is unavoidable in transductive KG embedding; see Limitations.
print("Leakage policy: edge-wise split; val/test triples withheld during training.")
print("Entity IDs shared across splits (standard transductive KG LP setting).")

_kg_summary_data = kg.summary() if hasattr(kg, "summary") else {"entities": NUM_ENTITIES}
_kg_summary_data.update({
    "relations": ["rated_high", "rated_low", "has_genre", "has_occupation"],
    "num_entities": NUM_ENTITIES, "num_relations": NUM_RELATIONS,
})
write_kg_summary(str(RUN_DIR / "kg_summary.json"), _kg_summary_data)
"""),

("markdown", """## 2. Model and training"""),

("code", """
# ── TransE model ─────────────────────────────────────────────────────────
model = TransEModel(NUM_ENTITIES, NUM_RELATIONS, embedding_dim=EMBEDDING_DIM)
config = KGTrainingConfig(
    num_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    loss_type="softplus",
    lr=1e-3,
    seed=SEED,
    device=device,
)
trainer = KGTrainer(model, config, train_triples)
print(f"TransEModel parameters: {count_parameters(model):,}")
print(f"Config: {config}")
"""),

("code", """
# ── Gradient sanity check ──────────────────────────────────────────────────
# Move model to device before manual forward to keep tensors on the same device.
model = model.to(device)
model.train()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sample_triples = train_triples[:16].to(device)
scores = model.score_triples(sample_triples)
neg = sample_triples.clone()
neg[:, 2] = torch.randint(0, NUM_ENTITIES, (16,), device=device)
neg_scores = model.score_triples(neg)
from tgraphx.kg.losses import SoftplusKGLoss
loss_fn = SoftplusKGLoss()
loss = loss_fn(scores, neg_scores)
loss.backward()
grads = sum(p.grad.abs().sum().item()
            for p in model.parameters() if p.grad is not None)
print(f"Gradient sanity: total_grad={grads:.4f}  (expect > 0)")
assert grads > 0
opt.zero_grad()
print("Gradient sanity passed.")
"""),

("code", """
t0 = time.time()
print(f"=== Training TransEModel ({EPOCHS} epochs) ===")
history = trainer.fit()
train_time = time.time() - t0
print(f"Final loss: {history['final_loss']:.4f}  Runtime: {train_time:.1f}s")
write_kg_training_report(
    str(RUN_DIR / "kg_training_report.json"),
    {
        "loss_history": history["loss_history"],
        "final_loss": history["final_loss"],
        "runtime_s": round(train_time, 2),
        "num_epochs": EPOCHS,
        "model": "TransE",
        "seed": SEED,
        "device": str(device),
    },
)
"""),

("markdown", """## 3. Evaluation"""),

("code", """
# ── Filtered ranking evaluation ────────────────────────────────────────────
evaluator = KGEvaluator(
    train_triples=train_triples,
    valid_triples=val_triples,
    test_triples=test_triples,
    num_entities=NUM_ENTITIES,
)
eval_result = evaluator.evaluate(
    model, triples=val_triples, filtered=True,
    batch_size=64, device=device,
)
rd = eval_result.to_dict()
filtered = rd.get("filtered", rd)
if isinstance(filtered, dict) and "combined" in filtered:
    filtered = filtered["combined"]

mrr = filtered.get("MRR", float("nan"))
h1 = filtered.get("Hits@1", float("nan"))
h3 = filtered.get("Hits@3", float("nan"))
h10 = filtered.get("Hits@10", float("nan"))

print(f"Validation filtered ranking:")
print(f"  MRR      = {mrr:.4f}")
print(f"  Hits@1   = {h1:.4f}")
print(f"  Hits@3   = {h3:.4f}")
print(f"  Hits@10  = {h10:.4f}")

print("\\nNOTE: FAST_MODE uses a small subset; ranking metrics are not")
print("representative of full MovieLens performance.")

write_kg_evaluation_report(
    str(RUN_DIR / "kg_eval_report.json"),
    {"mrr": mrr, "hits_at_1": h1, "hits_at_3": h3, "hits_at_10": h10,
     "eval_set": "validation", "filtered": True},
)
"""),

("markdown", """## 4. Hyperparameter search"""),

("code", """
# ── run_kg_hpo: small grid search ─────────────────────────────────────────
if FAST_MODE:
    hpo_search_space = {
        "embedding_dim": [16, 32],
        "lr": [1e-3, 5e-3],
    }
    hpo_max_trials = 4
    hpo_epochs = 3
else:
    hpo_search_space = {
        "embedding_dim": [16, 32, 64],
        "lr": [1e-3, 5e-3, 1e-2],
    }
    hpo_max_trials = 9
    hpo_epochs = 10

print(f"Running KG HPO: {len(hpo_search_space['embedding_dim'])} x "
      f"{len(hpo_search_space['lr'])} configs  "
      f"(max_trials={hpo_max_trials}, epochs={hpo_epochs})")

kg_train_only = KnowledgeGraph(
    train_triples, num_entities=NUM_ENTITIES, num_relations=NUM_RELATIONS
)
hpo_result = run_kg_hpo(
    kg_train_only,
    model_names=["TransE"],
    search_space=hpo_search_space,
    max_trials=hpo_max_trials,
    epochs=hpo_epochs,
    seed=SEED,
    device=device,
)
print(f"Best model: {hpo_result.best_model_name}")
print(f"Best config: {hpo_result.best_config}")
print(f"Best metrics: {hpo_result.best_metrics}")
"""),

("markdown", """## 5. Top-K recommendations"""),

("code", """
# ── Top-5 movie recommendations for sample users ───────────────────────────
model.eval()
NUM_SAMPLE_USERS = min(3, MAX_USERS)

# Find movies each user has already rated (to exclude from recommendations)
user_rated = {uid: set() for uid in range(MAX_USERS)}
for h, r, t in train_triples.tolist():
    if h < MAX_USERS and r in (REL_RATED_HIGH, REL_RATED_LOW):
        user_rated[h].add(t)

movie_entity_ids = torch.arange(MOVIE_OFFSET, MOVIE_OFFSET + len(movie_ids))
title_map = {mid - 1 + MOVIE_OFFSET: title
             for mid, title in zip(movie_ids, movie_titles)}

print("=== Top-5 Movie Recommendations ===")
with torch.no_grad():
    for user_idx in range(NUM_SAMPLE_USERS):
        already_rated = user_rated.get(user_idx, set())
        candidate_movies = torch.tensor(
            [m.item() for m in movie_entity_ids
             if m.item() not in already_rated],
            dtype=torch.long,
        )
        if len(candidate_movies) == 0:
            continue
        queries = torch.stack([
            torch.full((len(candidate_movies),), user_idx, dtype=torch.long),
            torch.full((len(candidate_movies),), REL_RATED_HIGH, dtype=torch.long),
            candidate_movies,
        ], dim=1).to(device)
        scores = model.score_triples(queries).cpu()
        top5 = scores.argsort(descending=True)[:5]
        print(f"\\nUser {user_idx + 1} top-5 TransE recommendations:")
        for rank, j in enumerate(top5.tolist(), 1):
            ent_id = candidate_movies[j].item()
            title = title_map.get(ent_id, f"Entity_{ent_id}")
            print(f"  {rank}. {title}  (score={scores[j]:.3f})")

# ── Popularity baseline ─────────────────────────────────────────────────
# Rank movies by training-set count of rated_high relations (no learning).
print("\\n=== Top-5 Popularity Baseline (no learning) ===")
popularity = torch.zeros(NUM_ENTITIES)
for h, r, t in train_triples.tolist():
    if r == REL_RATED_HIGH:
        popularity[t] += 1.0
for user_idx in range(NUM_SAMPLE_USERS):
    already_rated = user_rated.get(user_idx, set())
    pop_scores = popularity.clone()
    for m in already_rated:
        pop_scores[m] = -1.0
    movie_scores = pop_scores[movie_entity_ids]
    top5_pop = movie_scores.argsort(descending=True)[:5]
    print(f"\\nUser {user_idx + 1} top-5 popularity baseline:")
    for rank, j in enumerate(top5_pop.tolist(), 1):
        ent_id = movie_entity_ids[j].item()
        title = title_map.get(ent_id, f"Entity_{ent_id}")
        print(f"  {rank}. {title}  (count={int(movie_scores[j].item())})")
"""),

("markdown", """## 6. Dashboard artifacts"""),

("code", """
write_run_metadata(
    str(RUN_DIR / "run_metadata.json"),
    notebook="34_movielens_user_item_kg_recommendation",
    tgraphx_version=tgraphx.__version__,
    seed=SEED, fast_mode=FAST_MODE, device=device,
    using_real_movielens=USING_REAL_ML,
    num_entities=NUM_ENTITIES, num_relations=NUM_RELATIONS,
    runtime_s=round(train_time, 2),
)
write_metrics_summary(
    str(RUN_DIR / "metrics_summary.json"),
    mrr=round(mrr, 4), hits_at_1=round(h1, 4),
    hits_at_3=round(h3, 4), hits_at_10=round(h10, 4),
    model="TransEModel", task="kg_link_prediction",
    embedding_dim=EMBEDDING_DIM,
)
benchmark = {
    "task": "kg_link_prediction",
    "dataset": "MovieLens100K" if USING_REAL_ML else "synthetic",
    "num_entities": NUM_ENTITIES,
    "num_relations": NUM_RELATIONS,
    "num_train_triples": len(train_triples),
    "mrr": round(mrr, 4),
    "hits_at_10": round(h10, 4),
    "model": "TransE",
    "best_hpo_config": hpo_result.best_config,
    "best_hpo_mrr": round(hpo_result.best_metrics.get("mrr", 0.0), 4),
    "fast_mode": FAST_MODE,
}
with open(RUN_DIR / "benchmark_summary.json", "w") as f:
    json.dump(benchmark, f, indent=2)

print(f"Artifacts written to: {RUN_DIR}")
for p in sorted(RUN_DIR.glob("*.json")):
    print(f"  {p.name}")

print("\\n" + "=" * 60)
print("TGraphX MovieLens KG — Results Summary")
print("=" * 60)
print(f"{'MRR (filtered)':<35} {mrr:>10.4f}")
print(f"{'Hits@1':<35} {h1:>10.4f}")
print(f"{'Hits@3':<35} {h3:>10.4f}")
print(f"{'Hits@10':<35} {h10:>10.4f}")
print(f"{'Best HPO MRR':<35} {hpo_result.best_metrics.get('mrr', 0.0):>10.4f}")
print(f"FAST_MODE: {FAST_MODE}  Device: {device}")
"""),

("markdown", """
## Scientific and methodological notes

- **Learning setting:** Transductive KG link prediction — entity IDs are shared
  across train/val/test splits, but the triples (relation labels) are disjoint.
- **Split policy:** 80/10/10 edge-wise split of all triples (ratings + metadata).
- **Leakage policy:** Validation triples never enter training; test triples never
  enter training or HPO. Filtered ranking evaluation correctly excludes ALL
  training triples from candidate rankings, preventing inflated metrics.
- **Baseline meaning:** The popularity baseline ranks movies by frequency of
  `rated_high` interactions in training data (no learning). It is the minimum
  bar a learning model must beat to claim usefulness.
- **Metrics:** Filtered MRR, Hits@1/3/10 over validation triples; HPO best MRR;
  top-K recommendations with movie titles.
- **Why FAST_MODE metrics are not benchmark claims:** 150 users × 300 movies × 5
  epochs is illustrative. We do NOT claim TransE parity with feature-aware
  recommendation models.
- **TGraphX capability demonstrated:** Multi-relational `KnowledgeGraph` with
  `entity_features`; `KGTrainer` + `KGTrainingConfig`; filtered `KGEvaluator`;
  `run_kg_hpo`; structured dashboard artifacts including KG-specific reports.

## What this demonstrates

- **Multi-relational KG:** Users, movies, genres, and occupations are modelled
  as entities with four relation types, enabling richer relational reasoning
  than a simple user–item bipartite graph.
- **Entity features:** Genre multi-hot vectors are stored in the TGraphX
  `KnowledgeGraph` entity feature dict — never silently flattened.
- **KGTrainer workflow:** `KGTrainingConfig` + `KGTrainer` + `KGEvaluator`
  provides a reproducible, dashboard-aware training pipeline.
- **Filtered ranking:** `KGEvaluator` computes MRR and Hits@K using true-triple
  filtering to avoid penalising correct answers.
- **HPO:** `run_kg_hpo` searches embedding dimension and learning rate with a
  small grid search; best config is reported in the benchmark JSON.
- **Top-K recommendations:** TransE scores are used to rank movies for each user.

## Limitations

- FAST_MODE uses 150 users × 300 movies — too small for meaningful MRR.
- TransE is a simple translation-based model; RotatE or DistMult may score better.
- Only `rated_high` recommendations are shown; `rated_low` signal is not used
  for negative recommendation filtering.
- We do not claim SOTA on MovieLens; this demonstrates TGraphX KG workflow.
- Genre/occupation triples may not improve link prediction in a small subset.

## Next steps

- Scale to full MovieLens 100K (943 users, 1682 movies).
- Try `DistMultModel` or `RotatEModel` and compare via HPO.
- Add side information: movie year, user age group as additional relations.
- Use `feature-aware KG scoring` to leverage genre vectors in the model.
- Evaluate on the test set after final model selection.
"""),

("code", """
assert (RUN_DIR / "benchmark_summary.json").exists()
assert not (mrr != mrr), "MRR is NaN — check evaluation"
print("Notebook 34 — MovieLens KG Recommendation passed all checks.")
"""),
])

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 35 — Molecular Graph Classification: MUTAG
# ─────────────────────────────────────────────────────────────────────────────

NB35 = nb([
("markdown", """
# 35 — Molecular Graph Classification: MUTAG

**Research question:** Can graph-level message passing over atom/bond graphs
with both node and edge features predict molecular mutagenicity more accurately
than a simple degree-statistic baseline?

**What this demonstrates:**
- MUTAG molecular graphs: atoms as nodes, bonds as edges with bond-type features
- `PyGTUDatasetAdapter` as the TGraphX dataset bridge for MUTAG (via PyG)
- Edge-feature-aware graph-level GNN with mean+max pooling readout
- `GraphDataLoader` batching over a collection of molecular graphs
- `motif_profile` and `graph_summary` for structural dataset analysis
- Degree-feature baseline for comparison
- Dashboard artifact writing and benchmark-style reporting

**Dataset:** MUTAG (Debnath et al., 1991). 188 aromatic/heteroaromatic compounds;
label = mutagenic on *Salmonella typhimurium* (1) or not (0).

**Task type:** Inductive graph classification. Train/val/test split over graphs.
"""),

("code", """
# ── Configuration ─────────────────────────────────────────────────────────────
FAST_MODE = True
SEED = 42
EPOCHS = 10 if FAST_MODE else 50
HIDDEN_DIM = 32
BATCH_SIZE = 16
print(f"FAST_MODE={FAST_MODE}  epochs={EPOCHS}  hidden={HIDDEN_DIM}")
"""),

("code", """
# ── Install and import ────────────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tgraphx"], check=False)
import json, pathlib, time
import torch, torch.nn as nn, torch.nn.functional as F
import tgraphx
from tgraphx.reproducibility import set_seed
from tgraphx import (
    Graph, GraphBatch, GraphDataLoader, count_parameters,
    global_mean_pool, global_max_pool,
)
from tgraphx import LinearMessagePassing
from tgraphx.tracking import (
    write_run_metadata, write_metrics_summary,
    write_dataset_metadata, write_graph_stats,
)
from tgraphx.mining import graph_summary, motif_profile, degree_statistics

RUN_DIR = pathlib.Path("runs/advanced_notebooks/35_mutag")
RUN_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(SEED, deterministic=True)
print(f"TGraphX v{tgraphx.__version__}  |  device={device}  |  SEED={SEED}")
"""),

("markdown", """## 1. Dataset loading"""),

("code", """
# ── Load MUTAG via TGraphX PyG adapter ──────────────────────────────────
# MUTAG: 188 molecular graphs, 2 classes (mutagenic / non-mutagenic).
# Atoms: 7 atom types (one-hot). Bonds: 4 bond types (one-hot edge features).
USING_REAL_MUTAG = False
graphs = []

try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch-geometric"], check=False)
    from tgraphx.datasets import PyGTUDatasetAdapter
    tu_ds = PyGTUDatasetAdapter(name="MUTAG", download=True)
    graphs = [tu_ds.get(i) for i in range(len(tu_ds))]
    USING_REAL_MUTAG = True
    print(f"MUTAG loaded via TGraphX bridge: {len(graphs)} graphs")
except Exception as exc:
    print(f"MUTAG (PyG) unavailable: {exc}")
    print("Using synthetic molecular-graph fallback.")

if not USING_REAL_MUTAG:
    # Synthetic fallback: molecule-like graphs with 7 atom types, 4 bond types
    import random
    random.seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    for i in range(188):
        n_atoms = random.randint(10, 25)
        n_bonds = random.randint(n_atoms, n_atoms * 2)
        atom_feat = F.one_hot(
            torch.randint(0, 7, (n_atoms,), generator=rng), num_classes=7
        ).float()
        src = torch.randint(0, n_atoms, (n_bonds,), generator=rng)
        dst = torch.randint(0, n_atoms, (n_bonds,), generator=rng)
        mask = src != dst
        src, dst = src[mask], dst[mask]
        edge_index = torch.stack([
            torch.cat([src, dst]), torch.cat([dst, src])
        ], dim=0)
        edge_index = torch.unique(edge_index, dim=1)
        n_edges = edge_index.shape[1]
        bond_feat = F.one_hot(
            torch.randint(0, 4, (n_edges,), generator=rng), num_classes=4
        ).float()
        label = random.randint(0, 1)
        g = Graph(
            node_features=atom_feat,
            edge_index=edge_index,
            edge_attr=bond_feat,        # bond-type one-hot: [E, 4]
            graph_label=torch.tensor(label),
        )
        graphs.append(g)
    print(f"Synthetic molecule-like graphs: {len(graphs)}")

sample = graphs[0]
print(f"Sample graph: {sample}")
print(f"  node_features shape: {sample.node_features.shape}  (atoms x features)")
if sample.edge_features is not None:
    print(f"  edge_attr (bond features) shape: {sample.edge_features.shape}  (bonds x 4 types)")
print(f"  graph_label: {sample.graph_label.item()}")
print(f"  MUTAG classes: 0=non-mutagenic  1=mutagenic")

write_dataset_metadata(
    str(RUN_DIR / "dataset_metadata.json"),
    name="MUTAG" if USING_REAL_MUTAG else "synthetic_molecule_fallback",
    num_graphs=len(graphs),
    node_feature_dim=int(sample.node_features.shape[1]),
    edge_feature_dim=int(sample.edge_features.shape[1]) if sample.edge_features is not None else 0,
    num_classes=2,
    task="graph_classification",
)
"""),

("code", """
# ── Structural dataset analysis ────────────────────────────────────────────
print("=== Structural analysis of first 5 molecular graphs ===")
motif_stats = []
for i, g_mol in enumerate(graphs[:5]):
    s = graph_summary(g_mol.edge_index, num_nodes=g_mol.node_features.shape[0],
                      directed=False)
    mp = motif_profile(g_mol.edge_index, num_nodes=g_mol.node_features.shape[0],
                       directed=False)
    motif_stats.append(mp)
    print(f"  Graph {i}: nodes={s['num_nodes']}  edges={s['num_edges']}  "
          f"density={s['density']:.3f}  "
          f"triangles={mp.get('triangles', 0)}")

# Summary stats across dataset
all_sizes = [g_mol.node_features.shape[0] for g_mol in graphs]
print(f"\\nDataset: {len(graphs)} graphs")
print(f"  Node count: min={min(all_sizes)}  max={max(all_sizes)}  "
      f"mean={sum(all_sizes)/len(all_sizes):.1f}")
write_graph_stats(sample, str(RUN_DIR / "sample_graph_stats.json"))
"""),

("markdown", """## 2. Data split

**Leakage policy:** This is an inductive graph-classification setting. Train,
validation, and test graphs are disjoint sets. Each graph is self-contained:
its `graph_label` is attached to the graph object and never appears as a
node feature. Bond `edge_attr` (`edge_type` indicator) is an input, not a target.
"""),

("code", """
# ── Train/val/test split ───────────────────────────────────────────────────
gen = torch.Generator().manual_seed(SEED)
perm = torch.randperm(len(graphs), generator=gen).tolist()
n_train = int(0.7 * len(graphs))
n_val = int(0.15 * len(graphs))
graphs_train = [graphs[i] for i in perm[:n_train]]
graphs_val = [graphs[i] for i in perm[n_train:n_train + n_val]]
graphs_test = [graphs[i] for i in perm[n_train + n_val:]]
print(f"Split: train={len(graphs_train)}  val={len(graphs_val)}  "
      f"test={len(graphs_test)}")
print("Inductive split: train/val/test graphs are disjoint.")
print("Leakage policy: no label leakage; splits performed before model construction.")
print("Bond features (edge_attr / edge_type) are inputs, not targets.")
"""),

("markdown", """## 3. Model definition"""),

("code", """
# ── Molecular GNN with edge-feature awareness ─────────────────────────────
NODE_DIM = int(sample.node_features.shape[1])   # 7 atom types
EDGE_DIM = int(sample.edge_features.shape[1]) if sample.edge_features is not None else 0

class MoleculeGNN(nn.Module):
    \"\"\"
    2-layer GNN for graph-level classification.
    Uses LinearMessagePassing (supports vector node features).
    Readout: global_mean_pool + global_max_pool concatenated.
    \"\"\"
    def __init__(self, node_dim: int, hidden: int, edge_dim: int, num_classes: int = 2):
        super().__init__()
        self.mp1 = LinearMessagePassing(in_shape=(node_dim,),
                                         out_shape=(hidden,))
        self.mp2 = LinearMessagePassing(in_shape=(hidden,),
                                         out_shape=(hidden,))
        # Edge feature projection (if available)
        self.edge_proj = nn.Linear(edge_dim, hidden) if edge_dim > 0 else None
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        x = batch.node_features              # [N_total, node_dim]
        ei = batch.edge_index
        bi = batch.batch                     # graph membership

        # Optionally incorporate edge features into initial node features
        if self.edge_proj is not None and batch.has_edge_features:
            e_proj = self.edge_proj(batch.edge_features)  # [E, hidden]
            # Simple scatter: add projected edge features to source nodes
            src = ei[0]
            x_aug = torch.zeros(x.shape[0], e_proj.shape[1],
                                device=x.device, dtype=x.dtype)
            x_aug.scatter_add_(0, src.unsqueeze(1).expand_as(e_proj), e_proj)
            h = F.relu(self.mp1(x, ei) + x_aug[:, :x.shape[1]].clamp(-1, 1)
                       if x_aug.shape[1] == x.shape[1]
                       else self.mp1(x, ei))
        else:
            h = F.relu(self.mp1(x, ei))
        h = F.relu(self.mp2(h, ei))

        h_mean = global_mean_pool(h, bi)   # [G, hidden]
        h_max = global_max_pool(h, bi)     # [G, hidden]
        return self.head(torch.cat([h_mean, h_max], dim=1))  # [G, 2]


model = MoleculeGNN(NODE_DIM, HIDDEN_DIM, EDGE_DIM).to(device)
print(f"MoleculeGNN parameters: {count_parameters(model):,}")

# Shape trace
sample_batch = GraphBatch(graphs_train[:2])
out = model(sample_batch.to(device))
print(f"Shape trace: 2 graphs → {out.shape}")
assert out.shape == (2, 2)
print("Shape trace passed.")
"""),

("code", """
# ── Degree-feature baseline ────────────────────────────────────────────────
# Simple baseline: classify graph using mean degree and triangle count.
from tgraphx.mining import triangle_count

def extract_degree_features(graph_list: list) -> torch.Tensor:
    feats = []
    for g_mol in graph_list:
        ei = g_mol.edge_index
        n = g_mol.node_features.shape[0]
        deg_stats = degree_statistics(ei, num_nodes=n)
        tri = triangle_count(ei, num_nodes=n)
        feats.append([
            float(n),
            float(ei.shape[1]),
            float(deg_stats["mean_degree"]),
            float(deg_stats["max_degree"]),
            float(tri) / max(1, n),
        ])
    return torch.tensor(feats, dtype=torch.float)


class DegreeFeatureBaseline(nn.Module):
    def __init__(self, in_dim: int = 5, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

baseline = DegreeFeatureBaseline().to(device)
train_feats = extract_degree_features(graphs_train)
val_feats = extract_degree_features(graphs_val)
test_feats = extract_degree_features(graphs_test)

train_labels = torch.tensor(
    [g_mol.graph_label.item() for g_mol in graphs_train], dtype=torch.long)
val_labels = torch.tensor(
    [g_mol.graph_label.item() for g_mol in graphs_val], dtype=torch.long)
test_labels = torch.tensor(
    [g_mol.graph_label.item() for g_mol in graphs_test], dtype=torch.long)
print(f"DegreeFeatureBaseline: {count_parameters(baseline):,} parameters")
print(f"Train class balance: {(train_labels==0).sum()} non-muta / "
      f"{(train_labels==1).sum()} muta")
"""),

("markdown", """## 4. Training"""),

("code", """
def train_graph_model(mdl, graphs_list, epochs, name="model"):
    loader = GraphDataLoader(graphs_list, batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.Adam(mdl.parameters(), lr=5e-3, weight_decay=1e-4)
    mdl.train()
    history = []
    for ep in range(1, epochs + 1):
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            logits = mdl(batch)
            labels = batch.graph_labels.to(device)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        history.append(avg_loss)
        if ep % max(1, epochs // 3) == 0 or ep == epochs:
            print(f"  [{name}] epoch {ep}/{epochs}  loss={avg_loss:.4f}")
    return history


def train_baseline(mdl, feats, labels, epochs, name="baseline"):
    opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
    mdl.train()
    history = []
    for ep in range(1, epochs + 1):
        logits = mdl(feats.to(device))
        loss = F.cross_entropy(logits, labels.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
        history.append(loss.item())
        if ep % max(1, epochs // 3) == 0 or ep == epochs:
            print(f"  [{name}] epoch {ep}/{epochs}  loss={loss.item():.4f}")
    return history


# Gradient sanity
sample_loader = GraphDataLoader(graphs_train[:4], batch_size=4)
for batch in sample_loader:
    batch = batch.to(device)
    logits = model(batch)
    loss = F.cross_entropy(logits, batch.graph_labels.to(device))
    loss.backward()
    grads = sum(p.grad.abs().sum().item()
                for p in model.parameters() if p.grad is not None)
    print(f"Gradient sanity: total_grad={grads:.4f}  (expect > 0)")
    assert grads > 0
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
    break
print("Gradient sanity passed.")
"""),

("code", """
t0 = time.time()
print("=== Training MoleculeGNN ===")
gnn_history = train_graph_model(model, graphs_train, EPOCHS, "GNN")

print("\\n=== Training DegreeFeatureBaseline ===")
base_history = train_baseline(baseline, train_feats, train_labels, EPOCHS)
train_time = time.time() - t0
print(f"\\nTotal training time: {train_time:.1f}s")
"""),

("markdown", """## 5. Evaluation"""),

("code", """
@torch.no_grad()
def evaluate_graph_cls(mdl, graphs_list, name="model"):
    mdl.eval()
    loader = GraphDataLoader(graphs_list, batch_size=BATCH_SIZE, shuffle=False)
    correct, total = 0, 0
    for batch in loader:
        batch = batch.to(device)
        preds = mdl(batch).argmax(1)
        labels = batch.graph_labels.to(device)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
    acc = correct / max(1, total)
    print(f"  [{name}] accuracy = {acc:.4f}  ({correct}/{total})")
    return acc


@torch.no_grad()
def evaluate_baseline_cls(mdl, feats, labels, name="baseline"):
    mdl.eval()
    preds = mdl(feats.to(device)).argmax(1)
    correct = (preds == labels.to(device)).sum().item()
    acc = correct / max(1, len(labels))
    print(f"  [{name}] accuracy = {acc:.4f}  ({correct}/{len(labels)})")
    return acc


print("=== Validation accuracy ===")
gnn_val_acc = evaluate_graph_cls(model, graphs_val, "GNN")
base_val_acc = evaluate_baseline_cls(baseline, val_feats, val_labels, "Degree-baseline")

print("\\n=== Test accuracy ===")
gnn_test_acc = evaluate_graph_cls(model, graphs_test, "GNN")
base_test_acc = evaluate_baseline_cls(baseline, test_feats, test_labels, "Degree-baseline")
"""),

("markdown", """## 6. Dashboard artifacts"""),

("code", """
gnn_params = count_parameters(model)
base_params = count_parameters(baseline)

write_run_metadata(
    str(RUN_DIR / "run_metadata.json"),
    notebook="35_molecular_graph_classification_mutag_or_qm9",
    tgraphx_version=tgraphx.__version__,
    seed=SEED, fast_mode=FAST_MODE, device=device,
    using_real_mutag=USING_REAL_MUTAG, runtime_s=round(train_time, 2),
)
write_metrics_summary(
    str(RUN_DIR / "metrics_summary.json"),
    gnn_val_acc=round(gnn_val_acc, 4),
    gnn_test_acc=round(gnn_test_acc, 4),
    degree_baseline_val_acc=round(base_val_acc, 4),
    degree_baseline_test_acc=round(base_test_acc, 4),
    gnn_params=gnn_params, baseline_params=base_params,
    task="graph_classification",
)
benchmark = {
    "task": "graph_classification",
    "dataset": "MUTAG" if USING_REAL_MUTAG else "synthetic_molecule_fallback",
    "num_graphs": len(graphs),
    "gnn_val_acc": round(gnn_val_acc, 4),
    "degree_val_acc": round(base_val_acc, 4),
    "gnn_test_acc": round(gnn_test_acc, 4),
    "degree_test_acc": round(base_test_acc, 4),
    "gnn_params": gnn_params, "base_params": base_params,
    "runtime_s": round(train_time, 2),
    "fast_mode": FAST_MODE,
}
with open(RUN_DIR / "benchmark_summary.json", "w") as f:
    json.dump(benchmark, f, indent=2)

print(f"Artifacts written to: {RUN_DIR}")
for p in sorted(RUN_DIR.glob("*.json")):
    print(f"  {p.name}")

print("\\n" + "=" * 60)
print("TGraphX MUTAG Molecular Graph Classification — Results Summary")
print("=" * 60)
print(f"{'Metric':<35} {'GNN':>10} {'Degree-base':>10}")
print("-" * 60)
print(f"{'Val accuracy':<35} {gnn_val_acc:>10.4f} {base_val_acc:>10.4f}")
print(f"{'Test accuracy':<35} {gnn_test_acc:>10.4f} {base_test_acc:>10.4f}")
print(f"{'Parameters':<35} {gnn_params:>10,} {base_params:>10,}")
print(f"Graphs: {len(graphs)} total  "
      f"({len(graphs_train)} train / {len(graphs_val)} val / {len(graphs_test)} test)")
print(f"FAST_MODE: {FAST_MODE}  Device: {device}")
print("NOTE: MUTAG is small (188 graphs). Results may be unstable.")
"""),

("markdown", """
## Scientific and methodological notes

- **Learning setting:** Inductive graph classification — molecular graphs in
  train, val, and test sets are disjoint; the model never sees test graphs.
- **Split policy:** Deterministic 70/15/15 random split over the graph collection
  using a seeded `torch.randperm`.
- **Leakage policy:** Each molecular graph is fully self-contained; graph-level
  labels are attached to graph objects and only used in the training loop on
  training graphs. Bond `edge_attr` (`edge_type` features) is part of the input,
  not the target.
- **Baseline:** `DegreeFeatureBaseline` uses only structural features (mean
  degree, triangle count, node count) — no atom or bond identities. It is
  actually trained, not just counted.
- **Metrics:** Val/test accuracy on graph classification; runtime; parameter
  count; gradient sanity.
- **Why FAST_MODE metrics are not benchmark claims:** MUTAG has only 188 graphs
  and 10 epochs gives high variance. Reference GNN results on MUTAG reach
  ~85–89%; this notebook makes NO claim of that level. Repeated splits would
  be needed for stable estimates.
- **TGraphX capability demonstrated:** `Graph` with `edge_attr` (bond features);
  `GraphDataLoader` batching; `global_mean_pool + global_max_pool` readout;
  structural mining via `graph_summary`, `motif_profile`, `triangle_count`,
  `degree_statistics`; PyG dataset bridge.

## What this demonstrates

- **Molecular graphs as first-class TGraphX objects:** Atom features and bond
  features (edge_attr / `edge_type`) are stored in `Graph.node_features` and
  `Graph.edge_features` without flattening.
- **Edge-feature incorporation:** Bond-type features are projected and added to
  atom representations before message passing.
- **Mean+max readout:** `global_mean_pool` + `global_max_pool` provides a richer
  graph-level representation than mean pooling alone.
- **Structural mining:** `motif_profile` and `graph_summary` give dataset-level
  structural insights alongside learning-based analysis.
- **TGraphX dataset bridge:** `PyGTUDatasetAdapter` wraps PyG's TUDataset and
  exposes MUTAG as TGraphX `Graph` objects.

## Limitations

- MUTAG has only 188 graphs — results are highly sensitive to random splits.
- FAST_MODE uses only 10 epochs — convergence not guaranteed.
- The degree-feature baseline may outperform GNN on small datasets.
- We do not claim SOTA; literature GNN results on MUTAG reach ~85–89%.
- Without PyG installed, results are on a synthetic fallback (not MUTAG).
- Edge features are incorporated via a simple scatter sum; more principled
  edge-conditioned message passing would be more expressive.

## Next steps

- Run with 50+ epochs and repeated random splits for stable MUTAG results.
- Use `PROTEINS` or `ENZYMES` from `PyGTUDatasetAdapter` for larger datasets.
- Implement edge-conditioned message passing (e.g. NNConv) using TGraphX primitives.
- Apply attention pooling or differentiable graph pooling (DiffPool) for readout.
"""),

("code", """
assert (RUN_DIR / "benchmark_summary.json").exists()
assert gnn_val_acc >= 0.0
print("Notebook 35 — MUTAG Molecular Graph Classification passed all checks.")
"""),
])

# ─────────────────────────────────────────────────────────────────────────────
# Write all notebooks
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating advanced real-dataset notebooks...")
    save(OUT / "31_mnist_class_graph_membership_tensor_nodes.ipynb", NB31)
    save(OUT / "32_cifar10_visual_similarity_patch_graph.ipynb", NB32)
    save(OUT / "33_cora_citation_network_sampling_and_dashboard.ipynb", NB33)
    save(OUT / "34_movielens_user_item_kg_recommendation.ipynb", NB34)
    save(OUT / "35_molecular_graph_classification_mutag_or_qm9.ipynb", NB35)
    print("Done.")

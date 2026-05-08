"""v024_new_features.py — demo of all major v0.2.4 features.

Covers:
1. GAT channel attention mode (experimental)
2. Patch helper padding="auto"
3. Learned graph helpers
4. HeteroGraph container (experimental)
5. TemporalGraphSequence container (experimental)
6. GraphTransformerLayer (experimental, vector-only)
7. Optional PyG/DGL converters (skips if not installed)
8. MLflowLogger (skips if mlflow not installed)
"""
import torch
from tgraphx import Graph
from tgraphx.graph_builders import build_grid_graph, image_to_patches, volume_to_patches
from tgraphx.layers import TensorGATLayer

torch.manual_seed(0)
print("=" * 60)
print("TGraphX v0.2.4 — new features demo")
print("=" * 60)

# ── 1. GAT channel attention mode ─────────────────────────────────────────────
print("\n1. TensorGATLayer attention_mode='channel'")
N, C, H, W = 9, 4, 4, 4
x = torch.randn(N, C, H, W)
ei = build_grid_graph(3, 3, directed=False, self_loops=True)
l_ch = TensorGATLayer(C, C, num_heads=2, attention_mode="channel").eval()
with torch.no_grad():
    out, attn = l_ch(x, ei, return_attention=True)
print(f"  output shape:  {tuple(out.shape)}")
print(f"  attn shape:    {tuple(attn.shape)}  (per head per channel)")
print(f"  finite output: {torch.isfinite(out).all()}")

# ── 2. Patch padding ───────────────────────────────────────────────────────────
print("\n2. image_to_patches(padding='auto')")
imgs = torch.randn(2, 3, 9, 9)   # 9×9 not divisible by 4
patches = image_to_patches(imgs, patch_size=4, padding="auto")
print(f"  Input image:   {tuple(imgs.shape)}")
print(f"  Output patches:{tuple(patches.shape)}  (padded to 12×12 → 3×3 grid)")

# ── 3. Learned graph helpers ───────────────────────────────────────────────────
print("\n3. Learned graph construction helpers")
from tgraphx.learned_graph import (
    soft_adjacency_from_embeddings,
    top_k_edges_from_scores,
    build_knn_graph_from_embeddings,
    EdgeScorer,
)
z = torch.randn(10, 16)
A = soft_adjacency_from_embeddings(z)
print(f"  soft_adjacency:   {tuple(A.shape)}  range [{A.min():.3f}, {A.max():.3f}]")
ei_topk, scores = top_k_edges_from_scores(A, k=3)
print(f"  top_k_edges:      {tuple(ei_topk.shape)}")
ei_knn = build_knn_graph_from_embeddings(z, k=3)
print(f"  knn from embed:   {tuple(ei_knn.shape)}")
scorer = EdgeScorer(in_dim=16, hidden_dim=8)
s = scorer(z, ei_knn)
print(f"  EdgeScorer output:{tuple(s.shape)}")

# ── 4. HeteroGraph container ───────────────────────────────────────────────────
print("\n4. HeteroGraph (🧪 Experimental)")
from tgraphx.core.hetero_graph import HeteroGraph
hg = HeteroGraph(
    node_stores={
        "paper": torch.randn(5, 16),
        "author": torch.randn(3, 8),
    },
    edge_stores={
        ("author", "writes", "paper"): torch.tensor(
            [[0, 1, 2], [0, 1, 2]], dtype=torch.long
        ),
    },
)
print(f"  Node types: {hg.node_types}")
print(f"  Edge types: {hg.edge_types}")
print(f"  paper nodes: {hg.num_nodes('paper')}")

# ── 5. TemporalGraphSequence ───────────────────────────────────────────────────
print("\n5. TemporalGraphSequence (🧪 Experimental)")
from tgraphx.core.temporal import TemporalGraphSequence
graphs = [Graph(torch.randn(4, 8), None) for _ in range(3)]
seq = TemporalGraphSequence(graphs, timestamps=[0.0, 1.0, 2.0])
print(f"  Snapshots: {seq.num_snapshots}")
for t, g in seq:
    pass  # iterate over (timestamp, Graph) pairs
print(f"  Iteration: OK  (timestamp/graph pairs)")

# ── 6. GraphTransformerLayer ───────────────────────────────────────────────────
print("\n6. GraphTransformerLayer (🧪 Experimental, vector-only)")
from tgraphx.layers.graph_transformer import GraphTransformerLayer
layer_gt = GraphTransformerLayer(32, 32, num_heads=4, dropout=0.0).eval()
x_vec = torch.randn(10, 32)
with torch.no_grad():
    out_gt = layer_gt(x_vec)
print(f"  Input:  {tuple(x_vec.shape)}")
print(f"  Output: {tuple(out_gt.shape)}")
print(f"  Finite: {torch.isfinite(out_gt).all()}")

# ── 7. Optional PyG/DGL converters ────────────────────────────────────────────
print("\n7. PyG/DGL converters (optional)")
try:
    from tgraphx.interop import to_pyg_data
    g = Graph(torch.randn(5, 16),
              torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long))
    data = to_pyg_data(g)
    print(f"  to_pyg_data: {data}")
except ImportError:
    print("  PyG not installed — skipping (pip install torch-geometric)")
try:
    from tgraphx.interop import to_dgl_graph
    g = Graph(torch.randn(5, 16),
              torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long))
    dgl_g = to_dgl_graph(g)
    print(f"  to_dgl_graph: {dgl_g}")
except ImportError:
    print("  DGL not installed — skipping (pip install dgl)")

# ── 8. MLflowLogger ───────────────────────────────────────────────────────────
print("\n8. MLflowLogger (optional)")
try:
    from tgraphx.tracking import MLflowLogger
    print("  MLflowLogger class importable: OK")
    print("  (Start with: with MLflowLogger(run_name='run') as ml: ml.log(epoch=1, loss=0.5))")
except Exception as e:
    print(f"  Skipping: {e}")

print("\n" + "=" * 60)
print("All demos completed.")

"""Easy tensor node classification — no direct torch import required.

This example shows how to train a tensor-aware GNN for node classification
using only the TGraphX easy-mode API.  No ``import torch`` is needed.

Advanced users can always access the underlying PyTorch objects via the
result object (see the bottom of this file).

Usage::

    python examples/easy_tensor_node_classification_no_torch.py
"""

import tgraphx as tgx

# ── Step 1: Create synthetic data ─────────────────────────────────────────────
# Node features are image-like tensors: [N, C, H, W].
# No import torch required.
data = tgx.easy.synthetic_tensor_node_classification(
    num_nodes=500,
    node_shape=(8, 6, 6),
    num_classes=5,
    num_edges=2500,
    seed=42,
)
print(f"Graph: {data}")

# ── Step 2: Train a node classifier ──────────────────────────────────────────
result = tgx.easy.train_node_classifier(
    data,
    model="tensor_gcn",
    sampler="neighbor",
    fanouts=[10, 5],
    batch_size=32,
    epochs=3,
    seed=42,
)

# ── Step 3: Inspect results ───────────────────────────────────────────────────
print(f"\nFinal metrics: {result.metrics}")
print(f"Training elapsed: {result.elapsed:.1f}s")
result.summary()

# ── Step 4: Access low-level objects if needed ────────────────────────────────
# Advanced users can drop down to PyTorch at any point:
model = result.model         # nn.Module
graph = result.graph         # tgraphx.Graph with PyTorch tensors
optimizer = result.optimizer  # torch.optim.Adam

print("\nLow-level escape hatch:")
print(f"  model: {type(model).__name__}")
print(f"  graph.node_features: {graph.node_features.shape}")
print(f"  optimizer: {type(optimizer).__name__}")

print("\nExample PASSED")

"""Tensor node classification with NeighborLoader — canonical TGraphX tutorial.

This is the recommended first tutorial for users coming from PyG/DGL or new to
TGraphX.  It demonstrates:

1. Building a Graph with image-like ([N, C, H, W]) node features and labels.
2. Using NeighborLoader to sample mini-batches.
3. Training with ConvMessagePassing layers.
4. Computing seed-node loss correctly via batch.seed_logits() and batch.seed_y.
5. Evaluating accuracy.

The example is:
- CPU-runnable (no GPU required).
- CUDA-optional (add --device cuda to use a GPU).
- Deterministic (seed=42).
- Fast (< 30 seconds on CPU).

Usage::

    python tutorials/tensor_node_classification_neighbor_loader.py
    python tutorials/tensor_node_classification_neighbor_loader.py --device cuda
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from tgraphx import (
    Graph,
    ConvMessagePassing,
    NeighborLoader,
)
from tgraphx.reproducibility import set_seed


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Tensor node classification tutorial")
    p.add_argument("--device", default="cpu", help="'cpu' or 'cuda' (default: cpu)")
    p.add_argument("--num-nodes", type=int, default=500, help="Number of nodes (default: 500)")
    p.add_argument("--num-edges", type=int, default=3000, help="Number of edges (default: 3000)")
    p.add_argument("--channels", type=int, default=8, help="Node feature channels (default: 8)")
    p.add_argument("--height", type=int, default=6, help="Node feature height (default: 6)")
    p.add_argument("--width", type=int, default=6, help="Node feature width (default: 6)")
    p.add_argument("--num-classes", type=int, default=4, help="Number of classes (default: 4)")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    p.add_argument("--batch-size", type=int, default=32, help="Seed nodes per batch (default: 32)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()


# ── Model ─────────────────────────────────────────────────────────────────────

class TensorNodeClassifier(nn.Module):
    """Two-layer ConvMessagePassing → global pool → linear classifier.

    Designed for node features shaped [N, C, H, W].

    Shape contract:
        ConvMessagePassing preserves spatial dimensions (H, W).
        Only the channel count changes between layers.
        Use AdaptiveAvgPool2d to collapse spatial dimensions before the head.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 height: int, width: int, num_classes: int) -> None:
        super().__init__()
        # Layer 1: in_channels → hidden_channels, spatial preserved.
        self.conv1 = ConvMessagePassing(
            in_shape=(in_channels, height, width),
            out_shape=(hidden_channels, height, width),
        )
        # Layer 2: hidden_channels → hidden_channels, spatial preserved.
        self.conv2 = ConvMessagePassing(
            in_shape=(hidden_channels, height, width),
            out_shape=(hidden_channels, height, width),
        )
        # Collapse spatial dims: [N, hidden_channels, H, W] → [N, hidden_channels]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, C, H, W] node features.
            edge_index: [2, E] edge index.

        Returns:
            [N, num_classes] logits.
        """
        z = self.conv1(x, edge_index).relu()         # [N, hidden, H, W]
        z = self.conv2(z, edge_index).relu()          # [N, hidden, H, W]
        z = self.pool(z).flatten(1)                   # [N, hidden]
        return self.head(z)                            # [N, num_classes]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # ── Reproducibility ───────────────────────────────────────────────────
    set_seed(args.seed)

    N = args.num_nodes
    C = args.channels
    H = args.height
    W = args.width
    num_classes = args.num_classes

    print(f"TGraphX — Tensor Node Classification Tutorial")
    print(f"  device: {device} | nodes: {N} | edges: {args.num_edges}")
    print(f"  node shape: [{N}, {C}, {H}, {W}] | classes: {num_classes}")
    print()

    # ── Synthetic data ────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    x = torch.randn(N, C, H, W)
    edge_index = torch.randint(0, N, (2, args.num_edges))
    y = torch.randint(0, num_classes, (N,))

    # Build graph with labels — the canonical TGraphX way.
    g = Graph(node_features=x, edge_index=edge_index, y=y)
    print(f"Graph: {g}")

    # ── Loader ───────────────────────────────────────────────────────────
    # NeighborLoader yields GraphMiniBatch objects with direct attribute access.
    loader = NeighborLoader(
        g,
        fanouts=[15, 10],
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    hidden = 16
    model = TensorNodeClassifier(
        in_channels=C, hidden_channels=hidden,
        height=H, width=W, num_classes=num_classes,
    ).to(device)
    opt = Adam(model.parameters(), lr=1e-3)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # ── Training loop ─────────────────────────────────────────────────────
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seeds = 0

        for batch in loader:
            # Move batch to device.
            batch.to(device)

            # Forward pass over the sampled subgraph.
            logits = model(batch.node_features, batch.edge_index)

            # Compute loss only on seed (supervision) nodes.
            # batch.seed_logits() extracts the logits for seed nodes.
            # batch.seed_y returns the ground-truth labels for seed nodes.
            loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss) * batch.batch_size
            preds = batch.seed_logits(logits).argmax(dim=-1)
            total_correct += int((preds == batch.seed_y).sum())
            total_seeds += batch.batch_size

        avg_loss = total_loss / max(total_seeds, 1)
        acc = total_correct / max(total_seeds, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  acc={acc:.4f}  ({elapsed:.1f}s)")

    # ── Final eval ────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        eval_loader = NeighborLoader(
            g,
            fanouts=[15, 10],
            batch_size=args.batch_size,
            shuffle=False,
            seed=0,
        )
        correct = 0
        total = 0
        for batch in eval_loader:
            batch.to(device)
            logits = model(batch.node_features, batch.edge_index)
            preds = batch.seed_logits(logits).argmax(dim=-1)
            correct += int((preds == batch.seed_y).sum())
            total += batch.batch_size

    final_acc = correct / max(total, 1)
    print()
    print(f"Final evaluation accuracy: {final_acc:.4f}")
    print(f"Total time: {time.time() - t0:.1f}s")

    # Quick sanity checks so this can be run as a smoke test.
    assert final_acc >= 0.0, "Accuracy must be non-negative"
    assert time.time() - t0 < 300, "Tutorial took too long (>300s)"
    print()
    print("Tutorial PASSED")


if __name__ == "__main__":
    main()

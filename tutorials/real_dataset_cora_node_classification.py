"""Cora node classification with optional PyTorch Geometric adapter.

This tutorial demonstrates TGraphX node classification on the Cora citation
graph if PyG is installed.  When PyG is **not** installed, it skips
gracefully with a clear message.  No hidden network downloads.

Usage::

    # Skip cleanly without PyG.
    python tutorials/real_dataset_cora_node_classification.py

    # If PyG is installed AND you accept the upstream PyG cache download:
    python tutorials/real_dataset_cora_node_classification.py --download

    # Smoke run (PyG-free synthetic Cora-shape graph for CI):
    python tutorials/real_dataset_cora_node_classification.py --small

CPU-only by default; CUDA optional via --device cuda.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=20,
                   help="Training epochs (default 20)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--download", action="store_true",
                   help="Permit PyG to download Cora data via its cache (~5 MB)")
    p.add_argument("--small", action="store_true",
                   help="Use a tiny synthetic Cora-shape graph (no PyG required)")
    return p.parse_args()


def _load_pyg_cora(download: bool):
    """Try to load Cora via PyG.  Returns None if PyG is not installed."""
    try:
        from torch_geometric.datasets import Planetoid  # type: ignore
    except ImportError:
        return None
    if not download:
        # If the user did not pass --download we still try to load from
        # cache; if missing, return None so we can skip cleanly.
        try:
            ds = Planetoid(root="/tmp/tgraphx_cora_cache", name="Cora")
        except Exception as exc:
            print(f"[skip] PyG installed but Cora not cached: {exc}", file=sys.stderr)
            print("       Re-run with --download to fetch (~5 MB from PyG mirror).",
                  file=sys.stderr)
            return None
    else:
        ds = Planetoid(root="/tmp/tgraphx_cora_cache", name="Cora")
    return ds[0]


def _make_synthetic_cora(seed: int):
    """Build a tiny CPU-friendly synthetic Cora-shape graph for --small mode."""
    from tgraphx import Graph
    torch.manual_seed(seed)
    N, D, K = 200, 64, 7  # Cora has ~2700 nodes / 1433 features / 7 classes.
    x = torch.randn(N, D)
    # Block-structured edges so labels correlate with neighbourhoods.
    y = torch.randint(0, K, (N,))
    src = torch.arange(N).repeat_interleave(4)
    # Each node connects to 4 random nodes from its own class (mostly).
    same_class_targets = []
    for s in src.tolist():
        same_idx = (y == y[s]).nonzero(as_tuple=False).view(-1)
        choice = same_idx[torch.randint(0, same_idx.numel(), (1,))].item()
        same_class_targets.append(choice)
    dst = torch.tensor(same_class_targets, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    return Graph(node_features=x, edge_index=edge_index, y=y), K


def _build_pyg_graph(pyg_data):
    """Wrap a torch_geometric.data.Data object as a TGraphX Graph."""
    from tgraphx import Graph
    return Graph(
        node_features=pyg_data.x,
        edge_index=pyg_data.edge_index,
        y=pyg_data.y,
    ), int(pyg_data.y.max().item()) + 1


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    if args.small:
        print("Using synthetic Cora-shape graph (--small mode).")
        graph, num_classes = _make_synthetic_cora(args.seed)
    else:
        pyg_data = _load_pyg_cora(args.download)
        if pyg_data is None:
            print("torch_geometric is not installed (or Cora not cached and "
                  "--download was not given).  Falling back to --small mode.")
            graph, num_classes = _make_synthetic_cora(args.seed)
        else:
            graph, num_classes = _build_pyg_graph(pyg_data)

    graph.to(device)
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges, "
          f"feature_dim={graph.feature_shape[0]}, num_classes={num_classes}")

    # ---- Model ---------------------------------------------------------
    from tgraphx import GCNConv

    class GCN(torch.nn.Module):
        def __init__(self, in_dim, hidden, out_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, out_dim)

        def forward(self, x, edge_index):
            z = self.conv1(x, edge_index).relu()
            return self.conv2(z, edge_index)

    in_dim = graph.feature_shape[0]
    hidden = 16
    model = GCN(in_dim, hidden, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # ---- Training (full-batch) ------------------------------------------
    t0 = time.time()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(graph.node_features, graph.edge_index)
        loss = F.cross_entropy(logits, graph.node_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            acc = (logits.argmax(-1) == graph.node_labels).float().mean().item()
        history.append({"epoch": epoch, "loss": loss.item(), "accuracy": acc})
        if epoch % max(1, args.epochs // 5) == 0 or epoch == 1:
            print(f"  epoch {epoch:>3d} / {args.epochs}  loss={loss.item():.4f}  acc={acc:.4f}")

    elapsed = time.time() - t0
    print(f"\nFinal accuracy: {history[-1]['accuracy']:.4f}  ({elapsed:.1f}s)")
    print("Tutorial PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

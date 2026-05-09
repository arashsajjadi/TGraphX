"""Benchmark: easy-mode API vs manual PyTorch/TGraphX code.

Measures wall-clock overhead of the easy-mode training wrapper compared with
an equivalent manual implementation.

Usage::

    python benchmarks/ux/benchmark_easy_vs_manual.py --small --json
    python benchmarks/ux/benchmark_easy_vs_manual.py --epochs 5

Output (JSON):
    {
      "task": "tensor_node_classification",
      "runtime_easy_s": 1.23,
      "runtime_manual_s": 1.19,
      "overhead_percent": 3.36,
      "status": "PASS",
      "notes": "overhead < 10%"
    }
"""

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--small", action="store_true", help="Use tiny graph for CI")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--json", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_data(num_nodes, node_shape, num_classes, num_edges, seed):
    from tgraphx import Graph
    gen = torch.Generator()
    gen.manual_seed(seed)
    x = torch.randn(num_nodes, *node_shape, generator=gen)
    ei = torch.randint(0, num_nodes, (2, num_edges), generator=gen)
    y = torch.randint(0, num_classes, (num_nodes,), generator=gen)
    return Graph(node_features=x, edge_index=ei, y=y)


def run_manual(graph, epochs, batch_size, fanouts, seed):
    """Manual TGraphX training loop."""
    from tgraphx import NeighborLoader, ConvMessagePassing
    C, H, W = graph.feature_shape
    num_classes = int(graph.node_labels.max().item()) + 1

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = ConvMessagePassing((C, H, W), (16, H, W))
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Linear(16, num_classes)

        def forward(self, x, ei):
            z = self.conv(x, ei).relu()
            return self.head(self.pool(z).flatten(1))

    model = Model()
    opt = Adam(model.parameters(), lr=1e-3)
    loader = NeighborLoader(graph, fanouts=fanouts, batch_size=batch_size, seed=seed)

    t0 = time.perf_counter()
    for _ in range(epochs):
        for batch in loader:
            logits = model(batch.node_features, batch.edge_index)
            loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)
            opt.zero_grad(); loss.backward(); opt.step()
    return time.perf_counter() - t0


def run_easy(graph, epochs, batch_size, fanouts, seed):
    """Easy-mode training."""
    import tgraphx as tgx
    t0 = time.perf_counter()
    tgx.easy.train_node_classifier(
        graph, model="tensor_gcn", sampler="neighbor",
        fanouts=fanouts, batch_size=batch_size, epochs=epochs,
        seed=seed, verbose=False,
    )
    return time.perf_counter() - t0


def main():
    args = parse_args()

    if args.small:
        num_nodes, node_shape, num_classes, num_edges = 100, (4, 4, 4), 3, 300
        batch_size, fanouts = 16, [5, 3]
    else:
        num_nodes, node_shape, num_classes, num_edges = 1000, (8, 6, 6), 5, 5000
        batch_size, fanouts = 64, [10, 5]

    graph = make_data(num_nodes, node_shape, num_classes, num_edges, seed=args.seed)

    # Warm up.
    run_easy(graph, epochs=1, batch_size=batch_size, fanouts=fanouts, seed=args.seed)
    run_manual(graph, epochs=1, batch_size=batch_size, fanouts=fanouts, seed=args.seed)

    t_easy = run_easy(graph, epochs=args.epochs, batch_size=batch_size, fanouts=fanouts, seed=args.seed)
    t_manual = run_manual(graph, epochs=args.epochs, batch_size=batch_size, fanouts=fanouts, seed=args.seed)

    overhead = (t_easy - t_manual) / max(t_manual, 1e-9) * 100
    status = "PASS" if overhead < 15 else "WARN"
    notes = f"overhead={overhead:.1f}% {'< 15% threshold' if overhead < 15 else '> 15% threshold'}"

    result = {
        "task": "tensor_node_classification",
        "epochs": args.epochs,
        "num_nodes": num_nodes,
        "node_shape": list(node_shape),
        "runtime_easy_s": round(t_easy, 4),
        "runtime_manual_s": round(t_manual, 4),
        "overhead_percent": round(overhead, 2),
        "status": status,
        "notes": notes,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Easy mode : {t_easy:.3f}s")
        print(f"Manual    : {t_manual:.3f}s")
        print(f"Overhead  : {overhead:.1f}%")
        print(f"Status    : {status}")
        print(f"Notes     : {notes}")


if __name__ == "__main__":
    main()

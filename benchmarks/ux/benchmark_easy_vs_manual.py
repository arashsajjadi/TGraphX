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
    # Force the same device for both branches so the comparison is fair.
    # Without this, easy="auto" picks CUDA while manual stays on CPU and the
    # measured "overhead" reflects the device gap, not wrapper cost.
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Device used by BOTH easy and manual paths (default: cpu).")
    return p.parse_args()


def make_data(num_nodes, node_shape, num_classes, num_edges, seed):
    from tgraphx import Graph
    gen = torch.Generator()
    gen.manual_seed(seed)
    x = torch.randn(num_nodes, *node_shape, generator=gen)
    ei = torch.randint(0, num_nodes, (2, num_edges), generator=gen)
    y = torch.randint(0, num_classes, (num_nodes,), generator=gen)
    return Graph(node_features=x, edge_index=ei, y=y)


def run_manual(graph, epochs, batch_size, fanouts, seed, device="cpu"):
    """Manual TGraphX training loop.

    Calls ``set_seed`` so it pays the same reproducibility cost as easy mode
    (which calls it implicitly when ``seed`` is provided).
    """
    from tgraphx import NeighborLoader, ConvMessagePassing
    from tgraphx.reproducibility import set_seed as _set_seed
    if seed is not None:
        _set_seed(int(seed))
    C, H, W = graph.feature_shape
    num_classes = int(graph.node_labels.max().item()) + 1
    dev = torch.device(device)

    class Model(nn.Module):
        # Match make_tensor_node_classifier() in tgraphx.easy:
        # two ConvMessagePassing layers + adaptive pool + linear head.
        def __init__(self):
            super().__init__()
            self.conv1 = ConvMessagePassing((C, H, W), (16, H, W))
            self.conv2 = ConvMessagePassing((16, H, W), (16, H, W))
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Linear(16, num_classes)

        def forward(self, x, ei):
            z = self.conv1(x, ei).relu()
            z = self.conv2(z, ei).relu()
            return self.head(self.pool(z).flatten(1))

    model = Model().to(dev)
    opt = Adam(model.parameters(), lr=1e-3)
    if graph.device != dev:
        graph = graph.clone()
        graph.to(dev)
    loader = NeighborLoader(graph, fanouts=fanouts, batch_size=batch_size, seed=seed)

    t0 = time.perf_counter()
    for _ in range(epochs):
        for batch in loader:
            batch.to(dev)
            logits = model(batch.node_features, batch.edge_index)
            loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)
            opt.zero_grad(); loss.backward(); opt.step()
    return time.perf_counter() - t0


def run_easy(graph, epochs, batch_size, fanouts, seed, device="cpu"):
    """Easy-mode training."""
    import tgraphx as tgx
    t0 = time.perf_counter()
    tgx.easy.train_node_classifier(
        graph, model="tensor_gcn", sampler="neighbor",
        fanouts=fanouts, batch_size=batch_size, epochs=epochs,
        device=device,
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

    # Warm up (same device for both branches).
    run_easy(graph, epochs=1, batch_size=batch_size, fanouts=fanouts,
             seed=args.seed, device=args.device)
    run_manual(graph, epochs=1, batch_size=batch_size, fanouts=fanouts,
               seed=args.seed, device=args.device)

    # Take the median of 3 repeats so a single noisy run does not dominate.
    repeats = 3
    easy_times = [
        run_easy(graph, epochs=args.epochs, batch_size=batch_size,
                 fanouts=fanouts, seed=args.seed, device=args.device)
        for _ in range(repeats)
    ]
    manual_times = [
        run_manual(graph, epochs=args.epochs, batch_size=batch_size,
                   fanouts=fanouts, seed=args.seed, device=args.device)
        for _ in range(repeats)
    ]
    easy_times.sort()
    manual_times.sort()
    t_easy = easy_times[len(easy_times) // 2]
    t_manual = manual_times[len(manual_times) // 2]

    overhead = (t_easy - t_manual) / max(t_manual, 1e-9) * 100
    # Tiny graphs amortize setup overhead poorly; relax the threshold there.
    threshold = 50 if args.small else 15
    status = "PASS" if overhead < threshold else "WARN"
    notes = (
        f"overhead={overhead:.1f}% "
        f"({'< ' if overhead < threshold else '> '}{threshold}% threshold; "
        f"device={args.device}, repeats={repeats}, median)"
    )

    result = {
        "task": "tensor_node_classification",
        "epochs": args.epochs,
        "num_nodes": num_nodes,
        "node_shape": list(node_shape),
        "device": args.device,
        "repeats": repeats,
        "runtime_easy_s": round(t_easy, 4),
        "runtime_manual_s": round(t_manual, 4),
        "easy_times_s": [round(t, 4) for t in easy_times],
        "manual_times_s": [round(t, 4) for t in manual_times],
        "overhead_percent": round(overhead, 2),
        "threshold_percent": threshold,
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

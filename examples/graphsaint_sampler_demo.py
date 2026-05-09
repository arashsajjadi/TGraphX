"""GraphSAINT sampler demonstration (CPU-only, fast).

Builds a small synthetic graph, runs all three GraphSAINT samplers,
and writes a dashboard-compatible ``graphsaint_sampler_report.json``.
"""
from __future__ import annotations

import argparse
import os

import torch

from tgraphx import (
    Graph, GraphSAINTNodeSampler, GraphSAINTEdgeSampler,
    GraphSAINTRandomWalkSampler, GraphSAINTLoader,
)
from tgraphx.mining.reports import write_graphsaint_sampler_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=200)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-run-dir", default="logs/graphsaint_demo")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    N = args.num_nodes
    src = torch.arange(N).repeat_interleave(2)
    dst = torch.cat([(torch.arange(N) + 1) % N, (torch.arange(N) + 2) % N])
    ei = torch.stack([src, dst], dim=0)
    g = Graph(node_features=torch.randn(N, 8), edge_index=ei,
              edge_weight=torch.rand(ei.size(1)))

    samplers = {
        "node": GraphSAINTNodeSampler(g, args.budget, args.num_steps, seed=args.seed),
        "edge": GraphSAINTEdgeSampler(g, args.budget, args.num_steps, seed=args.seed),
        "random_walk": GraphSAINTRandomWalkSampler(
            g, num_roots=8, walk_length=5, num_steps=args.num_steps, seed=args.seed,
        ),
    }
    report = {"samplers": {}}
    for name, sampler in samplers.items():
        loader = GraphSAINTLoader(sampler, attach_norm=True, num_norm_samples=10)
        sizes_n, sizes_e = [], []
        for sub in loader:
            sizes_n.append(int(sub.num_nodes))
            sizes_e.append(int(sub.num_edges))
        report["samplers"][name] = {
            "num_steps": int(args.num_steps),
            "mean_nodes": float(sum(sizes_n) / len(sizes_n)) if sizes_n else 0.0,
            "mean_edges": float(sum(sizes_e) / len(sizes_e)) if sizes_e else 0.0,
            "min_nodes": min(sizes_n) if sizes_n else 0,
            "max_nodes": max(sizes_n) if sizes_n else 0,
        }

    os.makedirs(args.output_run_dir, exist_ok=True)
    out = os.path.join(args.output_run_dir, "graphsaint_sampler_report.json")
    write_graphsaint_sampler_report(out, report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

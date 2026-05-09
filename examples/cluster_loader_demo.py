"""Cluster-GCN partitioner + ClusterLoader demonstration."""
from __future__ import annotations

import argparse
import os

import torch

from tgraphx import (
    Graph, RandomBalancedPartitioner, BFSPartitioner,
    ConnectedComponentPartitioner, ClusterLoader,
)
from tgraphx.mining.reports import write_cluster_partition_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=120)
    parser.add_argument("--num-partitions", type=int, default=4)
    parser.add_argument("--algorithm", choices=("random", "bfs", "components"),
                        default="components")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-run-dir", default="logs/cluster_gcn_demo")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    N = args.num_nodes
    src = torch.arange(N).repeat_interleave(2)
    dst = torch.cat([(torch.arange(N) + 1) % N, (torch.arange(N) + 2) % N])
    ei = torch.stack([src, dst], dim=0)
    g = Graph(node_features=torch.randn(N, 8), edge_index=ei)

    if args.algorithm == "random":
        part = RandomBalancedPartitioner(args.num_partitions, args.seed).fit(g)
    elif args.algorithm == "bfs":
        part = BFSPartitioner(args.num_partitions, args.seed).fit(g)
    else:
        part = ConnectedComponentPartitioner(seed=args.seed).fit(g)

    loader = ClusterLoader(g, part, num_clusters_per_batch=1, shuffle=False)
    n_batches = sum(1 for _ in loader)

    os.makedirs(args.output_run_dir, exist_ok=True)
    report = part.to_dict()
    report["num_batches"] = int(n_batches)
    out = os.path.join(args.output_run_dir, "cluster_partition_report.json")
    write_cluster_partition_report(out, report)
    print(f"wrote {out}: {n_batches} batches, balance={part.balance_ratio:.3f}, cut_edges={part.cut_edges}")


if __name__ == "__main__":
    main()

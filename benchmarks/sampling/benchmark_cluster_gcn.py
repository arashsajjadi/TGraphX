"""Cluster-GCN partitioner benchmark.

Usage:
    python benchmarks/sampling/benchmark_cluster_gcn.py --small --json
    python benchmarks/sampling/benchmark_cluster_gcn.py \\
        --partitioner all --num-nodes 3000 --num-partitions 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import tgraphx
from tgraphx import Graph
from tgraphx.cluster_gcn import (
    RandomBalancedPartitioner,
    BFSPartitioner,
    ConnectedComponentPartitioner,
    ClusterLoader,
    PartitionResult,
)
from tgraphx.mining.reports import write_cluster_partition_report


_SMALL_N = 200
_DEFAULT_N = 2000
_DEFAULT_K = 8


def _make_graph(n: int, edge_factor: int, seed: int) -> Graph:
    torch.manual_seed(seed)
    src = torch.randint(n, (n * edge_factor,))
    dst = torch.randint(n, (n * edge_factor,))
    keep = src != dst
    src, dst = src[keep], dst[keep]
    ei = torch.unique(torch.stack([src, dst], dim=0), dim=1)
    x = torch.randn(n, 8)
    return Graph(node_features=x, edge_index=ei)


def _bench_partition(
    name: str,
    graph: Graph,
    k: int,
    seed: int,
    clusters_per_batch: int = 1,
) -> dict:
    t0 = time.perf_counter()
    if name == "random":
        part = RandomBalancedPartitioner(k, seed).fit(graph)
    elif name == "bfs":
        part = BFSPartitioner(k, seed).fit(graph)
    elif name == "connected":
        part = ConnectedComponentPartitioner(seed=seed).fit(graph)
    else:
        raise ValueError(name)
    dt_partition = (time.perf_counter() - t0) * 1000

    # Loader pass.
    loader = ClusterLoader(graph, part, num_clusters_per_batch=clusters_per_batch,
                           shuffle=False)
    node_counts, edge_counts = [], []
    t0 = time.perf_counter()
    for sub, _ in loader:
        node_counts.append(int(sub.num_nodes))
        edge_counts.append(int(sub.num_edges))
    dt_loader = (time.perf_counter() - t0) * 1000

    # Validate: all nodes assigned exactly once.
    all_assigned = bool((part.partition_sizes.sum().item() == graph.num_nodes))
    max_pid = int(part.partition_id.max().item())
    valid_range = bool(max_pid < part.num_partitions)

    return {
        "partitioner": name,
        "num_partitions": int(part.num_partitions),
        "balance_ratio": round(float(part.balance_ratio), 4),
        "cut_edges": int(part.cut_edges),
        "cut_edge_ratio": round(
            float(part.cut_edges) / max(1, int(graph.num_edges)), 4
        ),
        "partition_size_min": int(part.partition_sizes.min().item()),
        "partition_size_max": int(part.partition_sizes.max().item()),
        "partition_size_mean": round(float(part.partition_sizes.float().mean().item()), 2),
        "avg_batch_nodes": round(sum(node_counts) / max(1, len(node_counts)), 2),
        "avg_batch_edges": round(sum(edge_counts) / max(1, len(edge_counts)), 2),
        "partition_runtime_ms": round(dt_partition, 4),
        "loader_pass_runtime_ms": round(dt_loader, 4),
        "all_nodes_assigned": all_assigned,
        "valid_partition_range": valid_range,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-GCN partitioner benchmark")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--num-partitions", type=int, default=None)
    parser.add_argument("--partitioner", default="all",
                        choices=["random", "bfs", "connected", "all"])
    parser.add_argument("--clusters-per-batch", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    n = args.num_nodes or (_SMALL_N if args.small else _DEFAULT_N)
    k = args.num_partitions or (4 if args.small else _DEFAULT_K)
    ef = 3 if args.small else 6

    g = _make_graph(n, ef, args.seed)

    names = ["random", "bfs", "connected"] if args.partitioner == "all" else [args.partitioner]
    results = [_bench_partition(nm, g, k, args.seed, args.clusters_per_batch)
               for nm in names]

    report = {
        "package_version": tgraphx.__version__,
        "seed": int(args.seed),
        "num_nodes": int(g.num_nodes),
        "num_edges": int(g.num_edges),
        "requested_partitions": int(k),
        "clusters_per_batch": int(args.clusters_per_batch),
        "partitioner_results": results,
        "limitation_notes": [
            "SpectralPartitioner is not benchmarked here; "
            "it is O(N^3) and restricted to graphs <= 4096 nodes.",
            "No real dataset used; this is a synthetic benchmark.",
        ],
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_cluster_partition_report(args.output, report)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for r in results:
            print(f"[{r['partitioner']:>10}] K={r['num_partitions']} "
                  f"balance={r['balance_ratio']:.3f} "
                  f"cut={r['cut_edges']} "
                  f"rt={r['partition_runtime_ms']:.2f}ms")
        if all(r["all_nodes_assigned"] for r in results):
            print("all_nodes_assigned: PASSED")


if __name__ == "__main__":
    main()

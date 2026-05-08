"""Benchmark classical link prediction scoring.

Usage::

    python benchmarks/mining/benchmark_link_prediction.py --small
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from _common import make_parser, make_synthetic_graph, timer, print_result
import tgraphx
import torch


def run(args):
    torch.manual_seed(args.seed)
    N = 50 if args.small else (args.num_nodes or 200)
    ei, num_nodes = make_synthetic_graph(N, density=0.1, seed=args.seed)
    P = 100 if args.small else 500  # candidate pairs
    pairs = torch.stack([
        torch.randint(N, (P,)), torch.randint(N, (P,))
    ], dim=0)

    from tgraphx.mining import (
        common_neighbors_score, jaccard_score, adamic_adar_score,
        resource_allocation_score, preferential_attachment_score,
    )

    scorers = {
        "common_neighbors": common_neighbors_score,
        "jaccard": jaccard_score,
        "adamic_adar": adamic_adar_score,
        "resource_allocation": resource_allocation_score,
        "preferential_attachment": preferential_attachment_score,
    }

    timings = {}
    for name, fn in scorers.items():
        t, scores = timer(fn, ei, pairs, num_nodes=num_nodes, n_runs=3)
        timings[name + "_time_s"] = round(t, 6)
        timings[name + "_max"] = round(float(scores.max().item()), 4)

    result = {
        "benchmark": "link_prediction",
        "tgraphx_version": tgraphx.__version__,
        "num_nodes": num_nodes,
        "num_edges": int(ei.size(1)),
        "num_pairs": P,
        "device": str(args.device),
        "seed": args.seed,
        **timings,
    }
    print_result(result, args.json)
    return result


if __name__ == "__main__":
    parser = make_parser("benchmark_link_prediction", "Link prediction scoring benchmark.")
    run(parser.parse_args())

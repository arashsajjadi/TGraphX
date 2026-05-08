"""Benchmark motif/structural mining operations.

Usage::

    python benchmarks/mining/benchmark_motifs.py --small
    python benchmarks/mining/benchmark_motifs.py --json
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from _common import make_parser, make_synthetic_graph, timer, print_result
import tgraphx
import torch


def run(args):
    N = 50 if args.small else (args.num_nodes or 200)
    ei, num_nodes = make_synthetic_graph(N, density=0.1, seed=args.seed)

    from tgraphx.mining import (
        triangle_count, wedge_count, local_clustering_coefficient,
        motif_counts, motif_features,
    )

    t_tri, tri = timer(triangle_count, ei, num_nodes, directed=False)
    t_wedge, wed = timer(wedge_count, ei, num_nodes, directed=False)
    t_cc, cc = timer(local_clustering_coefficient, ei, num_nodes, directed=False)
    t_mc, mc = timer(motif_counts, ei, num_nodes, directed=False)
    t_mf, mf = timer(motif_features, ei, num_nodes, directed=False)

    result = {
        "benchmark": "motifs",
        "tgraphx_version": tgraphx.__version__,
        "num_nodes": num_nodes,
        "num_edges": int(ei.size(1)),
        "device": str(args.device),
        "seed": args.seed,
        "triangle_count": int(tri),
        "wedge_count": int(wed),
        "mean_clustering_coeff": round(float(cc.mean().item()), 4),
        "triangle_time_s": round(t_tri, 6),
        "wedge_time_s": round(t_wedge, 6),
        "clustering_coeff_time_s": round(t_cc, 6),
        "motif_counts_time_s": round(t_mc, 6),
        "motif_features_time_s": round(t_mf, 6),
        "motif_features_shape": list(mf.shape),
    }
    print_result(result, args.json)
    return result


if __name__ == "__main__":
    parser = make_parser("benchmark_motifs", "Motif counting benchmark.")
    run(parser.parse_args())

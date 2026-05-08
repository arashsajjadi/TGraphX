"""Benchmark graph similarity / WL kernel operations.

Usage::

    python benchmarks/mining/benchmark_graph_similarity.py --small
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from _common import make_parser, make_synthetic_graph, timer, print_result
import tgraphx
import torch


def run(args):
    torch.manual_seed(args.seed)
    G = 10 if args.small else 30
    N = 20 if args.small else (args.num_nodes or 50)

    graphs = []
    for i in range(G):
        ei, num_nodes = make_synthetic_graph(N, density=0.1, seed=args.seed + i)
        graphs.append({"edge_index": ei, "num_nodes": num_nodes})

    from tgraphx.mining import wl_kernel_matrix, pairwise_graph_similarity

    def _wl():
        return wl_kernel_matrix(graphs, num_iterations=2, normalize=True)

    def _deg():
        return pairwise_graph_similarity(graphs, method="degree")

    t_wl, K_wl = timer(_wl, n_runs=3)
    t_deg, K_deg = timer(_deg, n_runs=3)

    result = {
        "benchmark": "graph_similarity",
        "tgraphx_version": tgraphx.__version__,
        "num_graphs": G,
        "num_nodes_per_graph": N,
        "device": str(args.device),
        "seed": args.seed,
        "wl_kernel_time_s": round(t_wl, 6),
        "degree_similarity_time_s": round(t_deg, 6),
        "wl_kernel_shape": list(K_wl.shape),
        "wl_symmetric": bool(torch.allclose(K_wl, K_wl.t(), atol=1e-5).item()),
    }
    print_result(result, args.json)
    return result


if __name__ == "__main__":
    parser = make_parser("benchmark_graph_similarity", "Graph similarity benchmark.")
    run(parser.parse_args())

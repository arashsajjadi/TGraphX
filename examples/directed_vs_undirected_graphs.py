"""directed_vs_undirected_graphs.py — graph builder directedness demo.

Shows how directed=True/False and self_loops=True/False affect edge counts
and wraps the result in a Graph object.
"""
import torch
from tgraphx import Graph
from tgraphx.graph_builders import (
    build_fully_connected_graph,
    build_grid_graph,
    build_knn_graph,
)


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    # ------------------------------------------------------------------ #
    # 2-D Grid graph                                                       #
    # ------------------------------------------------------------------ #
    section("3×3 Grid graph")

    ei_dir = build_grid_graph(3, 3, directed=True, self_loops=False)
    ei_undir = build_grid_graph(3, 3, directed=False, self_loops=False)
    ei_undir_sl = build_grid_graph(3, 3, directed=False, self_loops=True)

    print(f"  directed=True,  self_loops=False : {ei_dir.shape[1]:3d} edges")
    print(f"  directed=False, self_loops=False : {ei_undir.shape[1]:3d} edges")
    print(f"  directed=False, self_loops=True  : {ei_undir_sl.shape[1]:3d} edges")

    # Wrap in a Graph object
    nf = torch.randn(9, 4)          # 9 nodes, [N, D] features
    g = Graph(nf, edge_index=ei_undir_sl)
    print(f"\n  Graph: {g}")

    # Verify reciprocal edges
    edge_set = set(zip(ei_undir.tolist()[0], ei_undir.tolist()[1]))
    has_all_reciprocal = all((v, u) in edge_set for (u, v) in edge_set)
    print(f"  All reciprocal edges present: {has_all_reciprocal}")

    # Self-loops
    sl_count = int((ei_undir_sl[0] == ei_undir_sl[1]).sum())
    print(f"  Self-loop count (9 nodes): {sl_count}")

    # ------------------------------------------------------------------ #
    # Fully connected graph                                                #
    # ------------------------------------------------------------------ #
    section("Fully connected graph (N=5)")

    N = 5
    ei_fc_dir = build_fully_connected_graph(N, directed=True, self_loops=False)
    ei_fc_sl = build_fully_connected_graph(N, self_loops=True)

    print(f"  self_loops=False : {ei_fc_dir.shape[1]} edges  (expected {N*(N-1)})")
    print(f"  self_loops=True  : {ei_fc_sl.shape[1]} edges  (expected {N*N})")

    # ------------------------------------------------------------------ #
    # kNN graph (directed vs undirected)                                   #
    # ------------------------------------------------------------------ #
    section("kNN graph — 6 nodes on a line, k=2")

    coords = torch.arange(6).float().unsqueeze(1)
    ei_knn_d = build_knn_graph(coords, k=2, directed=True,  self_loops=False)
    ei_knn_u = build_knn_graph(coords, k=2, directed=False, self_loops=False)

    print(f"  directed=True,  self_loops=False : {ei_knn_d.shape[1]} edges")
    print(f"  directed=False, self_loops=False : {ei_knn_u.shape[1]} edges")

    # Use undirected graph with self-loops
    ei_knn = build_knn_graph(coords, k=2, directed=False, self_loops=True)
    nf_knn = torch.randn(6, 8)
    g_knn = Graph(nf_knn, edge_index=ei_knn)
    print(f"\n  kNN Graph: {g_knn}")

    print("\nDone.")


if __name__ == "__main__":
    main()

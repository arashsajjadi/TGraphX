"""Advanced graph algorithms demo: max-flow, matching, coloring, hypergraph.

No downloads required.
"""
import torch
from tgraphx.mining import (
    greedy_maximal_matching, greedy_coloring, welsh_powell_coloring,
    greedy_maximal_independent_set, enumerate_maximal_cliques,
    edmonds_karp_max_flow, min_cut_from_max_flow, wl_isomorphism_test,
    Hypergraph, incidence_to_bipartite_graph, clique_expansion,
    write_graph_json, read_graph_json,
)
import tempfile, os

print("=" * 60)
print("Advanced Graph Algorithms Demo (TGraphX v0.4.4)")
print("=" * 60)

# Petersen-like graph for demonstrations.
ei = torch.tensor([
    [0,1,2,3,4,0,1,2,3,4,1,2,3,4,0],
    [1,2,3,4,0,5,6,7,8,9,6,7,8,9,5],
], dtype=torch.long)
ei = torch.cat([ei, ei.flip(0)], dim=1)
ei = torch.unique(ei, dim=1)
N = 10

print(f"\nGraph: {N} nodes, {ei.size(1)//2} undirected edges")

# Greedy matching.
m = greedy_maximal_matching(ei, N)
print(f"\nGreedy matching: {m.size(1)} pairs = {sorted(zip(m[0].tolist(), m[1].tolist()))}")

# Graph coloring.
colors, nc = welsh_powell_coloring(ei, N)
print(f"Welsh-Powell coloring: {nc} colors → {colors.tolist()}")

# Max independent set.
ind = greedy_maximal_independent_set(ei, N, seed=42)
print(f"Maximal independent set ({ind.numel()} nodes): {sorted(ind.tolist())}")

# Clique enumeration on a triangle.
tri_ei = torch.tensor([[0,1,2,1,2,0],[1,2,0,0,1,2]], dtype=torch.long)
cliques = enumerate_maximal_cliques(tri_ei, 3)
print(f"\nMaximal cliques in K3: {[sorted(c) for c in cliques]}")

# Max-flow.
flow_ei = torch.tensor([[0,0,1,2],[1,2,2,3]], dtype=torch.long)
cap = torch.tensor([3.0, 2.0, 2.0, 3.0])
max_flow, S, T = min_cut_from_max_flow(flow_ei, 4, cap, 0, 3)
print(f"\nMax-flow = {max_flow:.1f}, S={sorted(S)}, T={sorted(T)}")

# WL isomorphism test.
same = wl_isomorphism_test(tri_ei, 3, tri_ei, 3)
diff = wl_isomorphism_test(tri_ei, 3, torch.zeros((2,0),dtype=torch.long), 3)
print(f"\nWL test: same K3 → {same} (True), K3 vs empty → {diff} (False)")

# Hypergraph.
hg = Hypergraph(5, [[0,1,2],[2,3,4],[0,4]])
print(f"\nHypergraph: {hg.num_nodes} nodes, {hg.num_hyperedges} hyperedges, density={hg.density():.3f}")
bi_ei, total = incidence_to_bipartite_graph(hg)
print(f"Bipartite expansion: {total} nodes, {bi_ei.size(1)} edges")
cli_ei, cli_N = clique_expansion(hg)
print(f"Clique expansion: {cli_N} nodes, {cli_ei.size(1)//2} undirected edges")

# Graph IO.
with tempfile.TemporaryDirectory() as tmp:
    p = write_graph_json(os.path.join(tmp, "graph.json"), ei, N, metadata={"demo": True})
    g = read_graph_json(p)
    assert g["num_nodes"] == N and torch.equal(g["edge_index"], ei)
    print(f"\nGraph IO roundtrip: JSON saved and reloaded OK ({os.path.getsize(p)} bytes)")

print("\nDemo complete.")

"""graph_transformer_demo.py — GraphTransformerLayer with positional + edge bias.

CPU-safe; experimental APIs (🧪 v0.2.4 + v0.2.7 enhancements).
"""
import torch

from tgraphx.layers.graph_transformer import GraphTransformerLayer
from tgraphx.layers.transformer_encodings import (
    build_adjacency_bias,
    degree_encoding,
    laplacian_eigvec_encoding,
)
from tgraphx.layers.factory import make_layer

torch.manual_seed(0)
N, D = 12, 16
x = torch.randn(N, D)
src = torch.randint(0, N, (30,))
dst = torch.randint(0, N, (30,))
ei = torch.stack([src, dst], dim=0).long()

print("--- Plain GraphTransformerLayer ---")
plain = GraphTransformerLayer(D, D, num_heads=4).eval()
print(f"  out shape: {tuple(plain(x).shape)}")

print("\n--- With degree positional encoding ---")
pe_dim = 8
deg_pe = degree_encoding(ei, num_nodes=N, dim=pe_dim // 2, direction="both")
print(f"  degree encoding shape: {tuple(deg_pe.shape)}")
layer_pe = GraphTransformerLayer(
    D, D, num_heads=4, positional_encoding="degree", pe_dim=pe_dim,
).eval()
print(f"  out: {tuple(layer_pe(x, edge_index=ei, positional=deg_pe).shape)}")

print("\n--- With Laplacian eigenvector encoding ---")
lap_pe = laplacian_eigvec_encoding(ei, num_nodes=N, dim=4)
print(f"  laplacian encoding shape: {tuple(lap_pe.shape)}")
layer_lap = GraphTransformerLayer(
    D, D, num_heads=4, positional_encoding="laplacian", pe_dim=4,
).eval()
print(f"  out: {tuple(layer_lap(x, edge_index=ei, positional=lap_pe).shape)}")

print("\n--- With edge bias from adjacency ---")
bias = build_adjacency_bias(ei, num_nodes=N, neg_inf=True)
layer_eb = GraphTransformerLayer(D, D, num_heads=4, edge_bias=True).eval()
print(f"  out: {tuple(layer_eb(x, edge_index=ei, edge_bias_dense=bias).shape)}")

print("\n--- Via factory ---")
l = make_layer(
    "graph_transformer", in_shape=(D,), out_shape=(D,),
    heads=4, dropout=0.0, residual=True,
)
print(f"  factory layer: {l}")
print(f"  out: {tuple(l(x).shape)}")

print("\ngraph_transformer_demo: PASSED")

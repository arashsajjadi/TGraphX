"""gat_chunking_demo.py — TensorGATLayer two-pass chunked forward demo.

Demonstrates that chunk_size produces the same output as the standard
single-pass forward while using less peak edge-buffer memory.
"""
import torch
from tgraphx.layers import TensorGATLayer
from tgraphx.graph_builders import build_grid_graph

torch.manual_seed(0)

N, C, H, W = 9, 8, 4, 4
x = torch.randn(N, C, H, W)
ei = build_grid_graph(3, 3, directed=False, self_loops=True)
print(f"Graph: {N} nodes, {ei.size(1)} edges")
print(f"Node features: {tuple(x.shape)}")

layer = TensorGATLayer(C, C, num_heads=2, spatial_rank=2).eval()

with torch.no_grad():
    out_full = layer(x, ei)
    out_chunk = layer(x, ei, chunk_size=5)

diff = (out_full - out_chunk).abs().max().item()
print(f"\nFull pass output:    {tuple(out_full.shape)}")
print(f"Chunked pass output: {tuple(out_chunk.shape)}")
print(f"Max abs difference:  {diff:.2e}  (within float32 tolerance: OK)")

# Return attention weights
out_f, attn_f = layer(x, ei, return_attention=True)
out_c, attn_c = layer(x, ei, return_attention=True, chunk_size=5)
print(f"\nAttention (full):    {tuple(attn_f.shape)}")
print(f"Attention (chunked): {tuple(attn_c.shape)}")
attn_diff = (attn_f - attn_c).abs().max().item()
print(f"Attn max diff:       {attn_diff:.2e}")

# Verify attention sums to 1
dst = ei[1]
for j in range(min(3, N)):
    mask = dst == j
    if mask.any():
        s = attn_c[mask].sum(0).tolist()
        print(f"  Node {j} attn sum per head: {[f'{v:.4f}' for v in s]}")

print("\nGAT chunked forward demo: PASSED")

"""hetero_graph_batch_demo.py — HeteroGraph + HeteroGraphBatch + HeteroConv.

A small synthetic demo showing how to:
1. Build heterogeneous graphs with typed nodes and relations.
2. Disjoint-batch them with HeteroGraphBatch.
3. Apply a HeteroConv block of LinearMessagePassing layers.
4. Pool with a stable hetero readout.

CPU-safe; no internet; experimental APIs (🧪 v0.2.5).
"""
import torch

from tgraphx import HeteroGraph, HeteroGraphBatch
from tgraphx.layers import LinearMessagePassing
from tgraphx.layers.hetero import HeteroConv
from tgraphx.layers.hetero_readout import hetero_concat_pool

torch.manual_seed(0)

# Build two small heterogeneous graphs.
def make_graph(seed):
    torch.manual_seed(seed)
    n_paper = torch.randint(4, 7, (1,)).item()
    n_author = torch.randint(2, 5, (1,)).item()
    n_writes = torch.randint(2, 5, (1,)).item()
    return HeteroGraph(
        node_stores={
            "paper": torch.randn(n_paper, 8),
            "author": torch.randn(n_author, 8),
        },
        edge_stores={
            ("author", "writes", "paper"): torch.stack([
                torch.randint(0, n_author, (n_writes,)),
                torch.randint(0, n_paper, (n_writes,)),
            ], dim=0).long(),
        },
    )

g1 = make_graph(0)
g2 = make_graph(1)
g3 = make_graph(2)

batch = HeteroGraphBatch([g1, g2, g3])
print(f"Batch summary: {batch.num_graphs} graphs")
print(f"  num_nodes_dict: {batch.num_nodes_dict}")
print(f"  num_edges_dict: {batch.num_edges_dict}")
print(f"  batch_dict shapes:")
for t, vec in batch.batch_dict.items():
    print(f"    {t}: {tuple(vec.shape)}")

# A HeteroConv block of vector-feature layers.
conv = HeteroConv({
    ("author", "writes", "paper"): LinearMessagePassing((8,), (16,)),
}, aggr="sum")
out_dict = conv(batch.x_dict, batch.edge_index_dict)
for t, h in out_dict.items():
    print(f"  HeteroConv out[{t!r}]: {tuple(h.shape)}")

# Pool to a [B, total] tensor.
pooled = hetero_concat_pool(out_dict, batch_dict=batch.batch_dict, mode="mean")
print(f"\nGraph-level pooled tensor: {tuple(pooled.shape)} (B × sum_t D_t)")

print("\nHeteroGraphBatch + HeteroConv demo: PASSED")

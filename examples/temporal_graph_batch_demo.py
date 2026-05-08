"""temporal_graph_batch_demo.py — TemporalGraphBatch with variable-length sequences.

Demonstrates:
1. Creating TemporalGraphSequence objects of different lengths.
2. Batching them with TemporalGraphBatch (variable-length, masked).
3. Iterating per snapshot with masks.

CPU-safe; experimental APIs (🧪 v0.2.5).
"""
import torch

from tgraphx import Graph, TemporalGraphSequence, TemporalGraphBatch

torch.manual_seed(0)

# Three sequences of lengths 4, 2, 3.
def make_seq(length, dim=8, n_nodes=4):
    return TemporalGraphSequence(
        graphs=[Graph(torch.randn(n_nodes, dim), None) for _ in range(length)],
        timestamps=[float(i) for i in range(length)],
    )

seqs = [make_seq(4), make_seq(2), make_seq(3)]
batch = TemporalGraphBatch(seqs)

print(f"Sequences: {batch.num_sequences}")
print(f"Lengths:   {batch.lengths}")
print(f"Max length:{batch.max_length}")
print(f"Variable:  {batch.is_variable_length}")

print(f"\nPer-snapshot iteration:")
for t, gb, mask in batch:
    active = mask.nonzero(as_tuple=True)[0].tolist()
    print(f"  t={t}: {gb.num_graphs} active sequences (indices {active}), "
          f"GraphBatch nodes={gb.num_nodes}, edges={gb.num_edges}")

ts = batch.timestamps_padded
print(f"\nTimestamps padded shape: {tuple(ts.shape)}")
print(f"Timestamps:\n{ts}")

print("\nTemporalGraphBatch demo: PASSED")

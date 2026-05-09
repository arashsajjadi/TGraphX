"""Demonstrate GraphSequenceEncoder and GraphRNNEdgeGenerator from tgraphx.mining.

Uses the existing sequence_models API correctly.
"""
import torch
import torch.nn.functional as F
from tgraphx.mining.sequence_models import (
    GraphSequenceEncoder,
    GraphRNNEdgeGenerator,
    bfs_sequence_encode,
    pad_sequences,
)
from tgraphx.generation.classical import FeatureAwareERGraph


def main():
    print("=== Graph LSTM Sequence Demo ===\n")

    torch.manual_seed(42)

    # Build a small set of graphs
    graphs = [FeatureAwareERGraph(n=10, p=0.3, node_feature_dim=8, seed=i) for i in range(4)]
    node_dim = 8
    hidden_dim = 32

    print(f"Dataset: {len(graphs)} graphs, node_feature_dim={node_dim}")

    # Encode each graph as a BFS sequence
    sequences = []
    for g in graphs:
        seq = bfs_sequence_encode(g.edge_index, g.num_nodes, g.node_features)
        sequences.append(seq)
    print(f"BFS sequences: lengths = {[s.shape[0] for s in sequences]}")

    # Pad sequences for batching
    padded, lengths = pad_sequences(sequences)  # [B, L_max, D]
    print(f"Padded batch: {list(padded.shape)}, lengths={lengths.tolist()}")

    # GraphSequenceEncoder
    encoder = GraphSequenceEncoder(
        input_dim=node_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        pooling="mean",
    )
    print(f"\nGraphSequenceEncoder: {sum(p.numel() for p in encoder.parameters())} parameters")

    emb = encoder(padded, lengths)
    print(f"Graph embeddings shape: {list(emb.shape)}")

    # GraphRNNEdgeGenerator
    max_nodes = 12
    gen_model = GraphRNNEdgeGenerator(
        max_nodes=max_nodes,
        hidden_dim=hidden_dim,
        embed_dim=16,
    )
    print(f"\nGraphRNNEdgeGenerator: {sum(p.numel() for p in gen_model.parameters())} parameters")

    # Train for a few steps using teacher forcing
    # Input: adjacency rows [B, T, max_nodes]; Target: next rows
    optimizer = torch.optim.Adam(gen_model.parameters(), lr=1e-3)
    losses = []
    g = graphs[0]
    n = min(g.num_nodes, max_nodes)

    # Build adjacency row sequences for teacher forcing
    adj = torch.zeros(n, max_nodes)
    if g.num_edges > 0:
        for s, d in zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()):
            if s < n and d < max_nodes:
                adj[s, d] = 1.0

    # Input: rows 0..n-2, Target: rows 1..n-1
    if n > 1:
        inp = adj[:-1].unsqueeze(0)   # [1, n-1, max_nodes]
        tgt = adj[1:].unsqueeze(0)    # [1, n-1, max_nodes]

        for step in range(5):
            optimizer.zero_grad()
            logits, _ = gen_model(inp)   # [1, n-1, max_nodes]
            loss = F.binary_cross_entropy_with_logits(logits, tgt)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            print(f"  Step {step+1}: loss={loss.item():.4f}")
    else:
        print("  Graph too small for training demo, skipping.")
        losses = [0.0]

    if len(losses) >= 2:
        print(f"\nLoss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Sample a new graph
    adj_gen = gen_model.generate(num_nodes=8, seed=42)
    n_edges = int(adj_gen.sum().item()) // 2
    print(f"\nGenerated graph: {8} nodes, ~{n_edges} edges")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

"""Graph sequence models: RNN/LSTM/GRU over graph traversal sequences.

These models treat graph mining tasks as sequential problems by encoding
BFS/DFS traversal orders, random-walk sequences, or temporal graph snapshots
as input to standard PyTorch recurrent modules.

Stability: Experimental (v0.4.3+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GraphSequenceEncoder",
    "GraphSequenceClassifier",
    "GraphRNNEdgeGenerator",
    "bfs_sequence_encode",
    "random_walk_sequence_encode",
    "pad_sequences",
]


def bfs_sequence_encode(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_features: Optional[torch.Tensor] = None,
    start: int = 0,
    directed: bool = False,
) -> torch.Tensor:
    """Encode graph as a BFS-ordered node feature sequence.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        node_features: Optional ``FloatTensor[N, D]``.  When ``None``,
            uses one-hot node IDs (capped to ``min(N, 64)`` dims).
        start: BFS start node.
        directed: Directed BFS when ``True``.

    Returns:
        ``FloatTensor[K, D]`` BFS-ordered feature sequence.
    """
    from .paths import bfs_order
    order = bfs_order(edge_index, start, num_nodes, directed=directed)
    if node_features is not None:
        return node_features[order].float()
    # One-hot encoding capped to 64 dims.
    D = min(num_nodes, 64)
    enc = torch.zeros(order.size(0), D, dtype=torch.float)
    for i, v in enumerate(order.tolist()):
        if v < D:
            enc[i, v] = 1.0
    return enc


def random_walk_sequence_encode(
    edge_index: torch.Tensor,
    num_nodes: int,
    walk_length: int,
    start: int = 0,
    node_features: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Encode graph as a random walk feature sequence.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        walk_length: Length of random walk.
        start: Starting node.
        node_features: Optional ``FloatTensor[N, D]``.
        seed: Optional RNG seed.

    Returns:
        ``FloatTensor[walk_length+1, D]`` sequence.
    """
    from .random_walk import random_walks
    starts = torch.tensor([start], dtype=torch.long)
    walks = random_walks(edge_index, starts, walk_length, num_nodes=num_nodes, seed=seed)
    walk_seq = walks[0]  # [walk_length+1]
    if node_features is not None:
        return node_features[walk_seq].float()
    D = min(num_nodes, 64)
    enc = torch.zeros(walk_seq.size(0), D, dtype=torch.float)
    for i, v in enumerate(walk_seq.tolist()):
        if 0 <= v < D:
            enc[i, v] = 1.0
    return enc


def pad_sequences(
    sequences: List[torch.Tensor],
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of variable-length sequences for batching.

    Args:
        sequences: List of ``FloatTensor[L_i, D]`` sequences.
        pad_value: Padding fill value.

    Returns:
        ``(padded, lengths)`` where ``padded`` is
        ``FloatTensor[B, L_max, D]`` and ``lengths`` is
        ``LongTensor[B]``.
    """
    if not sequences:
        return torch.zeros((0, 1, 1)), torch.zeros(0, dtype=torch.long)
    B = len(sequences)
    D = sequences[0].size(-1)
    L_max = max(s.size(0) for s in sequences)
    padded = torch.full((B, L_max, D), pad_value, dtype=torch.float)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, s in enumerate(sequences):
        L = s.size(0)
        padded[i, :L] = s.float()
        lengths[i] = L
    return padded, lengths


class GraphSequenceEncoder(nn.Module):
    """Encode a graph traversal sequence with an LSTM.

    Encodes a sequence of node feature vectors (from BFS/DFS/random walk)
    into a fixed-length graph embedding.

    Args:
        input_dim: Input feature dimension per step.
        hidden_dim: LSTM hidden dimension.
        num_layers: Number of LSTM layers.
        dropout: Dropout probability.
        bidirectional: Use bidirectional LSTM.
        pooling: ``"last"``, ``"mean"``, or ``"max"`` over sequence.

    Stability: Experimental.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        if pooling not in ("last", "mean", "max"):
            raise ValueError(f"pooling must be 'last', 'mean', or 'max'; got {pooling!r}")
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.pooling = pooling
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p.data)
            elif "bias" in name:
                nn.init.zeros_(p.data)

    def forward(
        self,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode padded sequence batch.

        Args:
            sequences: ``FloatTensor[B, L_max, D]``.
            lengths: Optional ``LongTensor[B]`` of actual lengths.

        Returns:
            ``FloatTensor[B, output_dim]``.
        """
        if lengths is not None and int(lengths.max().item()) > 0:
            packed = nn.utils.rnn.pack_padded_sequence(
                sequences, lengths.cpu(), batch_first=True, enforce_sorted=False,
            )
            out, (hn, _) = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, (hn, _) = self.lstm(sequences)

        if self.pooling == "last":
            if self.lstm.bidirectional:
                return torch.cat([hn[-2], hn[-1]], dim=1)
            return hn[-1]
        if self.pooling == "mean":
            return out.mean(dim=1)
        # max pooling.
        return out.max(dim=1).values


class GraphSequenceClassifier(nn.Module):
    """Graph classification via sequence encoding.

    Encodes a graph traversal sequence with an LSTM and classifies
    the resulting graph embedding.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: LSTM hidden dim.
        num_classes: Number of output classes.
        num_layers: LSTM layers.
        dropout: Dropout.
        pooling: Sequence pooling strategy.

    Stability: Experimental.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        self.encoder = GraphSequenceEncoder(
            input_dim, hidden_dim, num_layers, dropout, pooling=pooling,
        )
        self.head = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(
        self,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequences: ``FloatTensor[B, L_max, D]``.
            lengths: Optional ``LongTensor[B]``.

        Returns:
            ``FloatTensor[B, num_classes]`` raw logits.
        """
        emb = self.encoder(sequences, lengths)
        return self.head(emb)


class GraphRNNEdgeGenerator(nn.Module):
    """Simple RNN-based edge sequence generator for tiny graphs.

    Generates adjacency rows autoregressively: for each row i of the
    adjacency matrix (i=0,...,N-1), produces Bernoulli probabilities
    for edges to nodes j < i.

    This is inspired by GraphRNN (You et al., 2018) but simplified
    to a single GRU without the node-level RNN.

    Args:
        max_nodes: Maximum graph size.
        hidden_dim: GRU hidden dimension.
        embed_dim: Row-embedding dimension.

    Stability: Experimental.
    """

    def __init__(
        self,
        max_nodes: int,
        hidden_dim: int = 32,
        embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.max_nodes = int(max_nodes)
        self.hidden_dim = hidden_dim
        # Input: previous adjacency row (max_nodes-length binary vector).
        self.gru = nn.GRU(max_nodes, hidden_dim, batch_first=True)
        # Output: probability of edge to each previous node.
        self.head = nn.Linear(hidden_dim, max_nodes)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p.data)

    def forward(
        self,
        row_sequences: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forcing forward pass.

        Args:
            row_sequences: ``FloatTensor[B, T, max_nodes]`` adjacency rows.
            hidden: Optional initial hidden state ``[1, B, hidden_dim]``.

        Returns:
            ``(logits, hidden)`` where ``logits`` is
            ``FloatTensor[B, T, max_nodes]``.
        """
        out, h = self.gru(row_sequences, hidden)
        logits = self.head(out)
        return logits, h

    @torch.no_grad()
    def generate(
        self,
        num_nodes: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively generate an adjacency matrix.

        Args:
            num_nodes: Desired graph size (≤ max_nodes).
            seed: Optional RNG seed.

        Returns:
            ``BoolTensor[N, N]`` adjacency matrix (symmetric).
        """
        N = min(int(num_nodes), self.max_nodes)
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))
        self.eval()
        adj = torch.zeros(N, N, dtype=torch.bool)
        h = torch.zeros(1, 1, self.hidden_dim)
        row = torch.zeros(1, 1, self.max_nodes)  # [B=1, T=1, max_nodes]
        for i in range(1, N):
            logits, h = self(row, h)
            probs = torch.sigmoid(logits[0, 0, :i])  # [i] probs for nodes 0..i-1
            edges = torch.bernoulli(probs, generator=gen).bool()
            adj[i, :i] = edges
            adj[:i, i] = edges  # symmetric
            # Next row input = current adjacency row.
            row = torch.zeros(1, 1, self.max_nodes)
            row[0, 0, :i] = adj[i, :i].float()
        return adj

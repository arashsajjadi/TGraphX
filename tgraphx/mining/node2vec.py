"""Node2Vec and DeepWalk unsupervised graph embedding foundations.

These utilities generate biased random walks and train a skip-gram model
to produce node embeddings.  No external dependency is required.

Stability: Experimental (v0.4.4+).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "node2vec_walks",
    "deepwalk_walks",
    "generate_skipgram_pairs",
    "Node2VecEmbedding",
    "train_node2vec_step",
    "extract_node2vec_embeddings",
]


def _build_csr(edge_index: torch.Tensor, num_nodes: int) -> Tuple[list, list]:
    """Build adjacency lists for Node2Vec walk generation."""
    adj: list = [[] for _ in range(num_nodes)]
    if not edge_index.numel():
        return adj, adj
    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()
    for u, v in zip(src, dst):
        if u != v:
            adj[u].append(v)
    return adj


def node2vec_walks(
    edge_index: torch.Tensor,
    num_nodes: int,
    walk_length: int = 80,
    walks_per_node: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    seed: Optional[int] = None,
    directed: bool = False,
) -> torch.Tensor:
    """Node2Vec biased random walks.

    When ``p=1`` and ``q=1`` this reduces to DeepWalk (uniform walks).

    The return parameter ``p`` controls the likelihood of revisiting a node;
    the in-out parameter ``q < 1`` biases toward BFS-like exploration,
    ``q > 1`` toward DFS-like exploration.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        walk_length: Length of each walk (number of steps).
        walks_per_node: Number of walks starting from each node.
        p: Return parameter.
        q: In-out parameter.
        seed: Optional RNG seed (no global RNG pollution).
        directed: When ``True``, follows directed edges only.

    Returns:
        ``LongTensor[num_nodes * walks_per_node, walk_length + 1]``
        — row i is a walk of length ``walk_length + 1`` (including start).
        Dead-end nodes (no outgoing edges) repeat their current position.

    Notes:
        Biased walks (p≠1 or q≠1) require checking the previous node's
        neighbourhood, which is O(degree) per step.  For large dense
        graphs this may be slow; prefer p=q=1 for speed.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be [2, E]")
    if num_nodes <= 0 or walk_length <= 0 or walks_per_node <= 0:
        raise ValueError("num_nodes, walk_length, walks_per_node must be positive")

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    # Build adjacency lists (out-neighbours).
    adj: list = [[] for _ in range(num_nodes)]
    adj_set: list = [set() for _ in range(num_nodes)]
    if edge_index.numel():
        src_l = edge_index[0].cpu().tolist()
        dst_l = edge_index[1].cpu().tolist()
        for u, v in zip(src_l, dst_l):
            if u != v:
                adj[u].append(v)
                adj_set[u].add(v)
                if not directed:
                    adj[v].append(u)
                    adj_set[v].add(u)

    unbiased = (abs(p - 1.0) < 1e-9 and abs(q - 1.0) < 1e-9)
    total_walks = num_nodes * walks_per_node
    walks = torch.zeros(total_walks, walk_length + 1, dtype=torch.long)

    wi = 0
    for start in range(num_nodes):
        for _ in range(walks_per_node):
            walks[wi, 0] = start
            cur = start
            prev = -1
            for step in range(1, walk_length + 1):
                nbrs = adj[cur]
                if not nbrs:
                    walks[wi, step] = cur  # dead end
                    continue
                if unbiased or prev < 0:
                    idx = int(torch.randint(len(nbrs), (1,), generator=gen).item())
                    nxt = nbrs[idx]
                else:
                    # Biased selection using unnormalised probabilities.
                    prev_set = adj_set[prev]
                    probs = []
                    for v in nbrs:
                        if v == prev:
                            probs.append(1.0 / p)
                        elif v in prev_set:
                            probs.append(1.0)
                        else:
                            probs.append(1.0 / q)
                    prob_t = torch.tensor(probs, dtype=torch.float)
                    idx = int(torch.multinomial(prob_t, 1, generator=gen).item())
                    nxt = nbrs[idx]
                walks[wi, step] = nxt
                prev = cur
                cur = nxt
            wi += 1
    return walks


def deepwalk_walks(
    edge_index: torch.Tensor,
    num_nodes: int,
    walk_length: int = 40,
    walks_per_node: int = 10,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """DeepWalk uniform random walks (p=q=1 special case of Node2Vec).

    Args:
        edge_index: ``LongTensor[2, E]`` (treated as undirected).
        num_nodes: Node count.
        walk_length: Steps per walk.
        walks_per_node: Walks per node.
        seed: Optional RNG seed.

    Returns:
        ``LongTensor[N * walks_per_node, walk_length + 1]``.
    """
    return node2vec_walks(
        edge_index, num_nodes,
        walk_length=walk_length,
        walks_per_node=walks_per_node,
        p=1.0, q=1.0, seed=seed,
    )


def generate_skipgram_pairs(
    walks: torch.Tensor,
    window_size: int = 5,
    negative_ratio: int = 5,
    num_nodes: int = 0,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate (center, context, negative) skip-gram training pairs from walks.

    Args:
        walks: ``LongTensor[W, L]`` from :func:`node2vec_walks`.
        window_size: Context window size.
        negative_ratio: Number of negative samples per positive pair.
        num_nodes: Vocabulary size for negative sampling.
            When 0, inferred from ``walks.max() + 1``.
        seed: Optional RNG seed.

    Returns:
        ``(centers, contexts, negatives)`` — three ``LongTensor``s of
        shape ``[P]``, ``[P]``, ``[P * negative_ratio]``.
    """
    if num_nodes <= 0:
        num_nodes = int(walks.max().item()) + 1
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    centers: List[int] = []
    contexts: List[int] = []
    W, L = walks.shape
    walks_cpu = walks.cpu()
    for wi in range(W):
        walk = walks_cpu[wi].tolist()
        for ci in range(L):
            center = walk[ci]
            lo = max(0, ci - window_size)
            hi = min(L, ci + window_size + 1)
            for ctx_i in range(lo, hi):
                if ctx_i != ci:
                    centers.append(center)
                    contexts.append(walk[ctx_i])

    if not centers:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty, empty

    centers_t = torch.tensor(centers, dtype=torch.long)
    contexts_t = torch.tensor(contexts, dtype=torch.long)
    P = centers_t.size(0)
    negatives_t = torch.randint(num_nodes, (P * negative_ratio,), generator=gen)
    return centers_t, contexts_t, negatives_t


class Node2VecEmbedding(nn.Module):
    """Skip-gram embedding model for Node2Vec / DeepWalk.

    Trains node embeddings by maximising the log-likelihood of observing
    context nodes within a window of random walks.

    Args:
        num_nodes: Number of nodes (vocabulary size).
        embedding_dim: Embedding dimension.
        sparse: When ``True``, use sparse gradients (faster for large vocabs).

    Stability: Experimental.
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 64,
        sparse: bool = False,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.embedding_dim = int(embedding_dim)
        self.embeddings = nn.Embedding(num_nodes, embedding_dim, sparse=sparse)
        self.context_embeddings = nn.Embedding(num_nodes, embedding_dim, sparse=sparse)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embeddings.weight, -0.5 / self.embedding_dim,
                          0.5 / self.embedding_dim)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(
        self,
        centers: torch.Tensor,
        contexts: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Negative-sampling skip-gram loss.

        Args:
            centers: ``LongTensor[P]`` centre node IDs.
            contexts: ``LongTensor[P]`` positive context node IDs.
            negatives: ``LongTensor[P * K]`` negative sample node IDs.

        Returns:
            Scalar loss.
        """
        P = centers.size(0)
        K = negatives.size(0) // P

        # Center embeddings: [P, D]
        center_emb = self.embeddings(centers)
        # Positive context: [P, D]
        ctx_emb = self.context_embeddings(contexts)
        pos_score = (center_emb * ctx_emb).sum(dim=1)  # [P]
        pos_loss = -F.logsigmoid(pos_score).mean()

        # Negative: [P, K, D]
        neg_emb = self.context_embeddings(negatives).view(P, K, -1)
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)  # [P, K]
        neg_loss = -F.logsigmoid(-neg_score).mean()

        return pos_loss + neg_loss

    @torch.no_grad()
    def get_embeddings(self) -> torch.Tensor:
        """Return all node embeddings detached from the computation graph.

        Returns:
            ``FloatTensor[N, D]``.
        """
        return self.embeddings.weight.detach().cpu()


def train_node2vec_step(
    model: Node2VecEmbedding,
    optimizer: torch.optim.Optimizer,
    centers: torch.Tensor,
    contexts: torch.Tensor,
    negatives: torch.Tensor,
) -> float:
    """One training step for Node2VecEmbedding.

    Args:
        model: :class:`Node2VecEmbedding`.
        optimizer: PyTorch optimizer.
        centers: ``LongTensor[P]``.
        contexts: ``LongTensor[P]``.
        negatives: ``LongTensor[P * K]``.

    Returns:
        Float loss value.
    """
    model.train()
    optimizer.zero_grad()
    loss = model(centers, contexts, negatives)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


@torch.no_grad()
def extract_node2vec_embeddings(model: Node2VecEmbedding) -> torch.Tensor:
    """Extract trained node embeddings (alias for ``model.get_embeddings()``).

    Returns:
        ``FloatTensor[N, D]`` of L2-normalised embeddings.
    """
    emb = model.get_embeddings().float()
    norms = emb.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return emb / norms

"""Graph learning utilities: self-supervised, contrastive, and augmentation.

These are building blocks for graph representation learning workflows.
No mandatory heavy optional dependency is introduced.

Stability: Experimental (v0.4.3+).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    # Losses
    "contrastive_loss",
    "supervised_contrastive_loss",
    "triplet_loss",
    "bpr_loss",
    "reconstruction_loss",
    # Augmentations
    "drop_edges",
    "drop_nodes",
    "mask_node_features",
    "add_random_edges",
    "subgraph_sampling",
    # Self-supervised objectives
    "DGIObjective",
    "GraphCLObjective",
    # Utilities
    "create_negative_pairs",
    "create_positive_pairs_from_batch",
]


# ── Contrastive and metric losses ────────────────────────────────────────────


def contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """NT-Xent (normalised temperature-scaled cross-entropy) contrastive loss.

    Treats ``(z1[i], z2[i])`` as a positive pair and all other combinations
    as negatives within the batch.

    Args:
        z1: ``FloatTensor[B, D]`` — first view embeddings.
        z2: ``FloatTensor[B, D]`` — second view embeddings.
        temperature: Temperature scaling; smaller = sharper distribution.

    Returns:
        Scalar loss.
    """
    B = z1.size(0)
    if B < 2:
        raise ValueError("contrastive_loss requires batch size >= 2")
    z1_norm = F.normalize(z1, dim=1)
    z2_norm = F.normalize(z2, dim=1)
    # [2B, D]
    z = torch.cat([z1_norm, z2_norm], dim=0)
    sim = z @ z.t() / temperature  # [2B, 2B]
    # Mask self-similarities.
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)
    # Labels: positive of z1[i] is z2[i] at index B+i and vice versa.
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])
    return F.cross_entropy(sim, labels)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020).

    Positive pairs share the same class label; all other pairs are negative.

    Args:
        embeddings: ``FloatTensor[B, D]``.
        labels: ``LongTensor[B]`` class labels.
        temperature: Temperature.

    Returns:
        Scalar loss.
    """
    B = embeddings.size(0)
    z = F.normalize(embeddings, dim=1)
    sim = z @ z.t() / temperature  # [B, B]
    # Mask self-comparisons.
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(eye, -1e9)
    # Positive mask: same label, not self.
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye  # [B, B]
    if not pos_mask.any():
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    log_prob = F.log_softmax(sim, dim=1)  # [B, B]
    pos_sum = (log_prob * pos_mask.float()).sum(dim=1)  # [B]
    n_pos = pos_mask.sum(dim=1).float().clamp(min=1.0)
    return -(pos_sum / n_pos).mean()


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Triplet margin loss.

    Args:
        anchor: ``FloatTensor[B, D]``.
        positive: ``FloatTensor[B, D]``.
        negative: ``FloatTensor[B, D]``.
        margin: Margin.

    Returns:
        Scalar loss.
    """
    d_pos = (anchor - positive).pow(2).sum(dim=1)
    d_neg = (anchor - negative).pow(2).sum(dim=1)
    return F.relu(d_pos - d_neg + margin).mean()


def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    """Bayesian Personalised Ranking loss for recommendation/link prediction.

    Args:
        pos_scores: ``FloatTensor[B]`` scores for positive pairs.
        neg_scores: ``FloatTensor[B]`` scores for negative pairs.

    Returns:
        Scalar loss (to minimise).
    """
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def reconstruction_loss(
    x_original: torch.Tensor,
    x_reconstructed: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE reconstruction loss for autoencoders.

    Args:
        x_original: Original input tensor.
        x_reconstructed: Reconstructed tensor (same shape).
        reduction: ``"mean"`` or ``"sum"``.

    Returns:
        Scalar loss.
    """
    return F.mse_loss(x_reconstructed, x_original.float(), reduction=reduction)


# ── Graph augmentations ───────────────────────────────────────────────────────


def drop_edges(
    edge_index: torch.Tensor,
    p: float = 0.2,
    seed: Optional[int] = None,
    edge_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Randomly drop edges with probability ``p``.

    Args:
        edge_index: ``LongTensor[2, E]``.
        p: Drop probability.
        seed: Optional RNG seed.
        edge_weight: Optional ``FloatTensor[E]`` to drop in sync.

    Returns:
        ``(new_edge_index, new_edge_weight)`` tuple.
    """
    if p <= 0:
        return edge_index, edge_weight
    if p >= 1:
        ew = torch.zeros(0, dtype=edge_weight.dtype, device=edge_index.device) if edge_weight is not None else None
        return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device), ew
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    E = edge_index.size(1)
    keep = torch.rand(E, generator=gen) > p
    new_ei = edge_index[:, keep]
    new_ew = edge_weight[keep] if edge_weight is not None else None
    return new_ei, new_ew


def drop_nodes(
    edge_index: torch.Tensor,
    num_nodes: int,
    p: float = 0.1,
    seed: Optional[int] = None,
    node_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor], torch.Tensor]:
    """Randomly drop nodes and their incident edges.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        p: Node drop probability.
        seed: Optional RNG seed.
        node_features: Optional ``Tensor[N, *]`` node features.

    Returns:
        ``(new_edge_index, new_num_nodes, new_node_features, kept_node_ids)``
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    keep_mask = torch.rand(num_nodes, generator=gen) > p
    kept = keep_mask.nonzero(as_tuple=False).view(-1)  # [K]
    new_N = int(kept.size(0))
    if new_N == 0:
        empty_ei = torch.zeros((2, 0), dtype=torch.long)
        empty_nf = None if node_features is None else node_features[:0]
        return empty_ei, 0, empty_nf, kept
    # Remap node IDs.
    remap = torch.full((num_nodes,), -1, dtype=torch.long)
    remap[kept] = torch.arange(new_N, dtype=torch.long)
    if edge_index.numel():
        src, dst = edge_index[0], edge_index[1]
        src_new = remap[src]
        dst_new = remap[dst]
        valid = (src_new >= 0) & (dst_new >= 0)
        new_ei = torch.stack([src_new[valid], dst_new[valid]], dim=0)
    else:
        new_ei = torch.zeros((2, 0), dtype=torch.long)
    new_nf = node_features[kept] if node_features is not None else None
    return new_ei, new_N, new_nf, kept


def mask_node_features(
    node_features: torch.Tensor,
    p: float = 0.1,
    mask_value: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly mask node feature elements with ``mask_value``.

    Args:
        node_features: ``FloatTensor[N, D]``.
        p: Mask probability per element.
        mask_value: Fill value.
        seed: Optional RNG seed.

    Returns:
        ``(masked_features, mask_tensor)`` where ``mask_tensor`` is
        ``BoolTensor[N, D]`` (True = masked).
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    mask = torch.rand(node_features.shape, generator=gen) < p
    out = node_features.float().clone()
    out[mask] = mask_value
    return out, mask


def add_random_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_add: int,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Add random edges to the graph.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        num_add: Number of random edges to add.
        seed: Optional RNG seed.

    Returns:
        ``LongTensor[2, E+num_add]`` new edge_index (may contain duplicates).
    """
    if num_add <= 0 or num_nodes <= 1:
        return edge_index
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    new_src = torch.randint(num_nodes, (num_add,), generator=gen)
    new_dst = torch.randint(num_nodes, (num_add,), generator=gen)
    new_edges = torch.stack([new_src, new_dst], dim=0)
    return torch.cat([edge_index, new_edges], dim=1)


def subgraph_sampling(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_sample: int,
    seed: Optional[int] = None,
    node_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor], torch.Tensor]:
    """Sample a random induced subgraph.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        num_sample: Number of nodes to keep.
        seed: Optional RNG seed.
        node_features: Optional ``Tensor[N, *]``.

    Returns:
        ``(sub_edge_index, sub_num_nodes, sub_node_features, sampled_ids)``
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    num_sample = min(int(num_sample), num_nodes)
    sampled = torch.randperm(num_nodes, generator=gen)[:num_sample]
    remap = torch.full((num_nodes,), -1, dtype=torch.long)
    remap[sampled] = torch.arange(num_sample, dtype=torch.long)
    if edge_index.numel():
        src_new = remap[edge_index[0]]
        dst_new = remap[edge_index[1]]
        valid = (src_new >= 0) & (dst_new >= 0)
        new_ei = torch.stack([src_new[valid], dst_new[valid]], dim=0)
    else:
        new_ei = torch.zeros((2, 0), dtype=torch.long)
    new_nf = node_features[sampled] if node_features is not None else None
    return new_ei, num_sample, new_nf, sampled


# ── Self-supervised objectives ────────────────────────────────────────────────


class DGIObjective(nn.Module):
    """Deep Graph Infomax-style objective.

    Discriminates between local node embeddings paired with the correct
    global graph summary versus a corrupted (permuted) global summary.

    Args:
        embed_dim: Node embedding dimension.
        summary_dim: Graph summary dimension (often = embed_dim).

    Stability: Experimental.
    """

    def __init__(self, embed_dim: int, summary_dim: int) -> None:
        super().__init__()
        self.bilinear = nn.Bilinear(embed_dim, summary_dim, 1)

    def readout(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Mean pooling → global summary."""
        return torch.sigmoid(node_embeddings.mean(dim=0))

    def discriminate(
        self,
        node_emb: torch.Tensor,
        summary: torch.Tensor,
    ) -> torch.Tensor:
        """Discriminator scores for each node.

        Args:
            node_emb: ``FloatTensor[N, D]``.
            summary: ``FloatTensor[S]`` global graph summary.

        Returns:
            ``FloatTensor[N]`` logits.
        """
        s_expand = summary.unsqueeze(0).expand(node_emb.size(0), -1)
        return self.bilinear(node_emb, s_expand).squeeze(-1)

    def forward(
        self,
        pos_node_emb: torch.Tensor,
        neg_node_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DGI loss.

        Args:
            pos_node_emb: Real graph node embeddings ``[N, D]``.
            neg_node_emb: Corrupted graph node embeddings ``[N, D]``.

        Returns:
            Scalar BCE loss.
        """
        summary = self.readout(pos_node_emb)
        pos_logits = self.discriminate(pos_node_emb, summary)
        neg_logits = self.discriminate(neg_node_emb, summary)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits),
        ])
        return F.binary_cross_entropy_with_logits(logits, labels)


class GraphCLObjective(nn.Module):
    """GraphCL (Graph Contrastive Learning) objective.

    Applies two augmentations to the same graph and maximises agreement
    between the resulting graph-level embeddings.

    Args:
        project_dim: Projector output dimension.

    Stability: Experimental.
    """

    def __init__(self, project_dim: int) -> None:
        super().__init__()
        self.project_dim = project_dim

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """NT-Xent loss between two views.

        Args:
            z1: ``FloatTensor[B, D]`` first-view graph embeddings.
            z2: ``FloatTensor[B, D]`` second-view graph embeddings.
            temperature: Temperature.

        Returns:
            Scalar loss.
        """
        return contrastive_loss(z1, z2, temperature)


# ── Pair creation utilities ───────────────────────────────────────────────────


def create_negative_pairs(
    pos_edge_index: torch.Tensor,
    num_nodes: int,
    num_neg: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Create random negative pairs for link prediction training.

    Thin wrapper around negative_sampling for convenience.

    Args:
        pos_edge_index: ``LongTensor[2, E]`` positive edges.
        num_nodes: Node count.
        num_neg: Number of negative pairs (defaults to E).
        seed: Optional RNG seed.

    Returns:
        ``LongTensor[2, num_neg]``.
    """
    from tgraphx import negative_sampling
    return negative_sampling(pos_edge_index, num_nodes, num_neg, seed=seed)


def create_positive_pairs_from_batch(
    labels: torch.Tensor,
    max_pairs: int = 1024,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find positive (same-label) pairs for supervised contrastive learning.

    Args:
        labels: ``LongTensor[B]`` class labels.
        max_pairs: Cap on number of pairs returned.
        seed: Optional RNG seed.

    Returns:
        ``(idx_a, idx_b)`` — two ``LongTensor[P]`` index tensors
        with ``labels[idx_a] == labels[idx_b]`` and ``idx_a != idx_b``.
    """
    B = labels.size(0)
    pairs_a, pairs_b = [], []
    for i in range(B):
        for j in range(i + 1, B):
            if labels[i] == labels[j]:
                pairs_a.append(i)
                pairs_b.append(j)
    if not pairs_a:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty
    pa = torch.tensor(pairs_a, dtype=torch.long)
    pb = torch.tensor(pairs_b, dtype=torch.long)
    if pa.size(0) > max_pairs:
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))
        perm = torch.randperm(pa.size(0), generator=gen)[:max_pairs]
        pa = pa[perm]
        pb = pb[perm]
    return pa, pb

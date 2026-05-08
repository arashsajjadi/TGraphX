"""Neural graph mining models — trainable mining foundations.

This module provides compact, differentiable neural graph mining models
that complement the classical utilities in ``tgraphx.mining``.

Three model families are provided:

1. **PrototypeMembershipScorer** — GNN encoder that scores whether a
   query/candidate graph belongs to a class prototype.

2. **GraphAutoencoderAnomalyDetector** — graph auto-encoder that
   reconstructs node features; high reconstruction error = anomalous.

3. **GraphPatternClassifier** — graph-level classifier that learns to
   distinguish structural pattern families (e.g. path vs star vs
   triangle-rich graphs).

All models are:

- Pure PyTorch (no external GNN library required).
- Differentiable end-to-end.
- Compatible with ``tgraphx.Graph`` objects and raw tensors.
- CPU-first; CUDA optional.
- Free of hidden training, hidden downloads, or telemetry.

Stability: **Experimental** (v0.3.2+).  APIs may evolve in v0.3.4.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PrototypeMembershipScorer",
    "GraphAutoencoderAnomalyDetector",
    "GraphPatternClassifier",
    "create_synthetic_pattern_dataset",
    "train_prototype_membership_step",
    "train_anomaly_autoencoder_step",
    "train_graph_pattern_classifier_step",
]


# ── Internal helpers ─────────────────────────────────────────────────────────


def _mean_pool(
    node_feats: torch.Tensor,
    num_nodes: Optional[int] = None,
) -> torch.Tensor:
    """Simple mean pooling over all nodes → graph embedding [D]."""
    return node_feats.mean(dim=0)


def _message_pass_mean(
    node_feats: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """One-step mean-aggregation message passing without weight matrices.

    Used internally as a simple structural information aggregation step.
    """
    if edge_index.numel() == 0 or num_nodes == 0:
        return node_feats
    agg = torch.zeros_like(node_feats)
    cnt = torch.zeros(num_nodes, 1, dtype=node_feats.dtype, device=node_feats.device)
    src, dst = edge_index[0], edge_index[1]
    # Scatter messages: dst receives from src.
    src_feat = node_feats[src]  # [E, D]
    agg.scatter_add_(0, dst.unsqueeze(1).expand_as(src_feat), src_feat)
    cnt.scatter_add_(0, dst.unsqueeze(1), torch.ones(src.size(0), 1, dtype=node_feats.dtype, device=node_feats.device))
    cnt = cnt.clamp(min=1.0)
    return agg / cnt


class _GNNEncoder(nn.Module):
    """Compact 2-layer GNN encoder using existing LinearMessagePassing style."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.norms.append(nn.LayerNorm(dims[i + 1]))
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        for i, (lin, norm) in enumerate(zip(self.layers, self.norms)):
            x = _message_pass_mean(x, edge_index, num_nodes)
            x = lin(x)
            x = norm(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                if self.dropout > 0 and self.training:
                    x = F.dropout(x, p=self.dropout, training=True)
        return x


# ── 1. Prototype membership scorer ───────────────────────────────────────────


class PrototypeMembershipScorer(nn.Module):
    """Trainable prototype graph membership scorer.

    Given a **candidate graph** (support graph + query node) and the
    query node's index, scores the likelihood that the query belongs to
    the class represented by the support graph.

    Architecture:

        1. Encode all nodes with a shared 2-layer GNN.
        2. Produce a **graph embedding** by mean-pooling over *support*
           nodes (all nodes except the query).
        3. Produce a **query embedding** from the query node's GNN output.
        4. Score = MLP( [graph_emb ; query_emb ; |graph_emb - query_emb| ;
                         graph_emb * query_emb ] )

    This design is inspired by Siamese networks but uses a common
    encoder for both support and query, which allows the GNN to mix
    structural information.

    Args:
        in_dim: Input node feature dimension (after optional flattening
            for tensor node features).
        hidden_dim: Hidden dimension for the GNN encoder.
        out_dim: Encoder output dimension.
        num_gnn_layers: Number of GNN layers (default 2).
        dropout: Dropout probability.
        flatten_spatial: When ``True``, spatial/volumetric node features
            are flattened to a vector before encoding.  When ``False``,
            the caller must pass 2-D (``[N, D]``) node features.

    Forward:
        ``(node_features, edge_index, query_idx, num_nodes)``
        → ``FloatTensor[]`` (scalar logit, positive = belongs to class).

    Stability: Experimental.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
        num_gnn_layers: int = 2,
        dropout: float = 0.0,
        flatten_spatial: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.flatten_spatial = bool(flatten_spatial)
        self.encoder = _GNNEncoder(
            in_dim, hidden_dim, out_dim, num_gnn_layers, dropout
        )
        # Scoring MLP: input = concat of [g, q, |g-q|, g*q]
        self.scorer = nn.Sequential(
            nn.Linear(4 * out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _prep_features(self, node_features: torch.Tensor) -> torch.Tensor:
        if node_features.dim() == 2:
            return node_features.float()
        if self.flatten_spatial:
            return node_features.float().view(node_features.size(0), -1)
        raise ValueError(
            f"node_features has shape {tuple(node_features.shape)}; "
            f"expected [N, D].  Pass flatten_spatial=True to auto-flatten "
            f"spatial/volumetric features."
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        query_idx: int,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """Score the query graph.

        Args:
            node_features: ``FloatTensor[N, D]`` (or spatial if
                ``flatten_spatial=True``).
            edge_index: ``LongTensor[2, E]``.
            query_idx: Index of the query node in the graph.
            num_nodes: Optional; inferred when ``None``.

        Returns:
            Scalar logit (positive = query belongs to class).
        """
        x = self._prep_features(node_features)
        N = x.size(0)
        if num_nodes is None:
            num_nodes = N
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, E]")

        enc = self.encoder(x, edge_index, num_nodes)  # [N, out_dim]

        # Support embedding: mean over all nodes except the query.
        support_mask = torch.ones(N, dtype=torch.bool, device=enc.device)
        support_mask[query_idx] = False
        support_nodes = enc[support_mask]
        if support_nodes.size(0) == 0:
            # Edge case: only the query node.
            support_emb = torch.zeros(enc.size(1), device=enc.device, dtype=enc.dtype)
        else:
            support_emb = support_nodes.mean(dim=0)  # [out_dim]

        query_emb = enc[query_idx]  # [out_dim]

        combined = torch.cat([
            support_emb,
            query_emb,
            (support_emb - query_emb).abs(),
            support_emb * query_emb,
        ], dim=0)  # [4 * out_dim]

        return self.scorer(combined).squeeze(-1)

    def score_batch(
        self,
        candidates: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """Score a list of candidate graph dicts.

        Each dict must have keys ``node_features``, ``edge_index``,
        ``query_idx``.

        Returns:
            ``FloatTensor[len(candidates)]`` of logits.
        """
        logits = []
        for c in candidates:
            logit = self(
                c["node_features"], c["edge_index"], c["query_idx"],
            )
            logits.append(logit)
        return torch.stack(logits, dim=0)


# ── 2. Graph autoencoder anomaly detector ────────────────────────────────────


class GraphAutoencoderAnomalyDetector(nn.Module):
    """Node-feature reconstruction graph auto-encoder for anomaly detection.

    Trains on normal graphs to reconstruct node features.  At inference,
    nodes (or graphs) with high reconstruction error are flagged as
    anomalous.

    Architecture:

        Encoder: 2-layer GNN that maps ``[N, in_dim]`` → ``[N, latent_dim]``
        Decoder: 2-layer MLP that maps ``[N, latent_dim]`` → ``[N, in_dim]``
        Loss: MSE between input and reconstruction.

    Args:
        in_dim: Input (and output) node feature dimension.
        latent_dim: Latent embedding dimension.
        hidden_dim: Hidden dimension for encoder and decoder.
        num_gnn_layers: GNN encoder layers.
        dropout: Dropout for encoder.

    Stability: Experimental.
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        num_gnn_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = _GNNEncoder(in_dim, hidden_dim, latent_dim, num_gnn_layers, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            node_features: ``FloatTensor[N, in_dim]``.
            edge_index: ``LongTensor[2, E]``.
            num_nodes: Optional.

        Returns:
            Tuple ``(reconstruction, latent)`` where reconstruction has
            shape ``[N, in_dim]`` and latent has shape ``[N, latent_dim]``.
        """
        x = node_features.float()
        N = x.size(0)
        if num_nodes is None:
            num_nodes = N
        latent = self.encoder(x, edge_index, num_nodes)
        recon = self.decoder(latent)
        return recon, latent

    def reconstruction_loss(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """MSE reconstruction loss (scalar)."""
        recon, _ = self(node_features, edge_index, num_nodes)
        return F.mse_loss(recon, node_features.float())

    @torch.no_grad()
    def node_anomaly_scores(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """Per-node anomaly score = per-node mean squared reconstruction error.

        Higher score = more anomalous.  Detaches output from graph.

        Returns:
            ``FloatTensor[N]``.
        """
        self.eval()
        recon, _ = self(node_features, edge_index, num_nodes)
        scores = (recon - node_features.float()).pow(2).mean(dim=1)
        return scores.detach()

    @torch.no_grad()
    def graph_anomaly_score(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None,
    ) -> float:
        """Graph-level anomaly score = mean node anomaly score."""
        scores = self.node_anomaly_scores(node_features, edge_index, num_nodes)
        return float(scores.mean().item())


# ── 3. Graph pattern classifier ──────────────────────────────────────────────


class GraphPatternClassifier(nn.Module):
    """Trainable graph-level pattern classifier.

    Uses a GNN encoder + mean pooling + MLP head to classify graphs
    into structural pattern families (e.g. path/star/cycle/tree).

    Args:
        in_dim: Node feature dimension.
        hidden_dim: GNN hidden dimension.
        enc_dim: Encoder output dimension.
        num_classes: Number of graph pattern classes.
        num_gnn_layers: GNN encoder depth.
        dropout: Dropout probability.

    Forward:
        ``(node_features, edge_index, num_nodes)``
        → ``FloatTensor[num_classes]`` (raw logits).

    Stability: Experimental.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        enc_dim: int = 16,
        num_classes: int = 4,
        num_gnn_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = _GNNEncoder(in_dim, hidden_dim, enc_dim, num_gnn_layers, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.ReLU(),
            nn.Linear(enc_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: ``FloatTensor[N, in_dim]``.
            edge_index: ``LongTensor[2, E]``.
            num_nodes: Optional.

        Returns:
            ``FloatTensor[num_classes]`` of raw logits.
        """
        x = node_features.float()
        N = x.size(0)
        if num_nodes is None:
            num_nodes = N
        enc = self.encoder(x, edge_index, num_nodes)  # [N, enc_dim]
        graph_emb = enc.mean(dim=0)  # [enc_dim]
        return self.classifier(graph_emb)  # [num_classes]


# ── Synthetic dataset generators ─────────────────────────────────────────────


def create_synthetic_pattern_dataset(
    num_graphs_per_class: int = 50,
    num_nodes: int = 8,
    in_dim: int = 4,
    seed: int = 0,
    noise_std: float = 0.1,
) -> List[Dict[str, Any]]:
    """Create a synthetic graph pattern dataset with 4 classes.

    Pattern classes:

    0. **Path graph** — nodes connected in a chain.
    1. **Star graph** — one hub connected to all leaves.
    2. **Cycle graph** — nodes connected in a ring.
    3. **Complete graph** — all pairs connected.

    Each graph has random noise node features that are class-specific
    (different mean) to make the task learnable from both structure and
    features.

    Args:
        num_graphs_per_class: Number of graphs per pattern class.
        num_nodes: Number of nodes per graph.
        in_dim: Node feature dimension.
        seed: RNG seed.
        noise_std: Standard deviation of additive node feature noise.

    Returns:
        List of dicts with keys:
          - ``node_features``: ``FloatTensor[N, in_dim]``
          - ``edge_index``: ``LongTensor[2, E]``
          - ``num_nodes``: int
          - ``label``: int (0–3)
          - ``pattern``: str
    """
    torch.manual_seed(seed)
    N = int(num_nodes)
    patterns = ["path", "star", "cycle", "complete"]
    dataset = []

    def _path_edges(n: int) -> torch.Tensor:
        src = list(range(n - 1)) + list(range(1, n))
        dst = list(range(1, n)) + list(range(n - 1))
        return torch.tensor([src, dst], dtype=torch.long)

    def _star_edges(n: int) -> torch.Tensor:
        src = [0] * (n - 1) + list(range(1, n))
        dst = list(range(1, n)) + [0] * (n - 1)
        return torch.tensor([src, dst], dtype=torch.long)

    def _cycle_edges(n: int) -> torch.Tensor:
        src = list(range(n)) + list(range(n))
        dst = [(i + 1) % n for i in range(n)] + [(i - 1) % n for i in range(n)]
        return torch.tensor([src, dst], dtype=torch.long)

    def _complete_edges(n: int) -> torch.Tensor:
        src = [u for u in range(n) for v in range(n) if u != v]
        dst = [v for u in range(n) for v in range(n) if u != v]
        return torch.tensor([src, dst], dtype=torch.long)

    edge_fns = [_path_edges, _star_edges, _cycle_edges, _complete_edges]
    # Class-specific feature means.
    class_means = [
        torch.tensor([1.0, 0.0, 0.0, 0.0][:in_dim] + [0.0] * max(0, in_dim - 4)),
        torch.tensor([0.0, 1.0, 0.0, 0.0][:in_dim] + [0.0] * max(0, in_dim - 4)),
        torch.tensor([0.0, 0.0, 1.0, 0.0][:in_dim] + [0.0] * max(0, in_dim - 4)),
        torch.tensor([0.0, 0.0, 0.0, 1.0][:in_dim] + [0.0] * max(0, in_dim - 4)),
    ]

    for cls, (name, edge_fn, mean) in enumerate(zip(patterns, edge_fns, class_means)):
        for _ in range(num_graphs_per_class):
            ei = edge_fn(N)
            x = mean.unsqueeze(0).expand(N, -1).clone()
            x = x + noise_std * torch.randn(N, in_dim)
            dataset.append({
                "node_features": x,
                "edge_index": ei,
                "num_nodes": N,
                "label": cls,
                "pattern": name,
            })
    return dataset


# ── Training helpers ──────────────────────────────────────────────────────────


def train_prototype_membership_step(
    model: PrototypeMembershipScorer,
    optimizer: torch.optim.Optimizer,
    candidates: List[Dict[str, Any]],
    targets: torch.Tensor,
) -> float:
    """One training step for the prototype membership scorer.

    Args:
        model: :class:`PrototypeMembershipScorer`.
        optimizer: PyTorch optimizer.
        candidates: List of candidate graph dicts (``node_features``,
            ``edge_index``, ``query_idx``).
        targets: ``FloatTensor[B]`` of 0/1 targets (1 = true class).

    Returns:
        Float loss value.
    """
    model.train()
    optimizer.zero_grad()
    logits = model.score_batch(candidates)
    loss = F.binary_cross_entropy_with_logits(logits, targets.float())
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def train_anomaly_autoencoder_step(
    model: GraphAutoencoderAnomalyDetector,
    optimizer: torch.optim.Optimizer,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
) -> float:
    """One training step for the graph autoencoder.

    Args:
        model: :class:`GraphAutoencoderAnomalyDetector`.
        optimizer: PyTorch optimizer.
        node_features: ``FloatTensor[N, D]``.
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Optional.

    Returns:
        Float MSE loss.
    """
    model.train()
    optimizer.zero_grad()
    loss = model.reconstruction_loss(node_features, edge_index, num_nodes)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def train_graph_pattern_classifier_step(
    model: GraphPatternClassifier,
    optimizer: torch.optim.Optimizer,
    graphs: List[Dict[str, Any]],
    labels: torch.Tensor,
) -> float:
    """One training step for the graph pattern classifier.

    Args:
        model: :class:`GraphPatternClassifier`.
        optimizer: PyTorch optimizer.
        graphs: List of graph dicts (``node_features``, ``edge_index``,
            ``num_nodes``).
        labels: ``LongTensor[B]`` of class labels.

    Returns:
        Float cross-entropy loss.
    """
    model.train()
    optimizer.zero_grad()
    logits = torch.stack([
        model(g["node_features"], g["edge_index"], g["num_nodes"])
        for g in graphs
    ], dim=0)  # [B, C]
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())

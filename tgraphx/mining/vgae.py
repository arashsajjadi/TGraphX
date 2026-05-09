"""Graph Autoencoder (GAE) and Variational Graph Autoencoder (VGAE).

These models learn node embeddings by reconstructing the graph's adjacency
structure.  They are useful for unsupervised link prediction, graph
generation, and representation learning.

Reference: Kipf & Welling, 2016 — "Variational Graph Auto-Encoders".

Stability: Experimental (v0.5.0+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GraphAutoencoder",
    "VGAE",
    "DotProductDecoder",
    "MLPEdgeDecoder",
    "GCNEncoder",
    "train_gae_step",
    "evaluate_link_prediction",
]


# ── Encoders ─────────────────────────────────────────────────────────────────


class _MessagePassingMean(nn.Module):
    """Simple mean-aggregation GCN-style layer for the encoder."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        x = self.lin(x)
        if edge_index.numel() == 0:
            return F.relu(x)
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(x[src]), x[src])
        cnt = torch.zeros(num_nodes, 1, dtype=x.dtype, device=x.device)
        cnt.scatter_add_(0, dst.unsqueeze(1), torch.ones(src.size(0), 1, dtype=x.dtype, device=x.device))
        cnt = cnt.clamp(min=1)
        return F.relu(agg / cnt)


class GCNEncoder(nn.Module):
    """Two-layer GCN encoder for GAE/VGAE.

    Args:
        in_dim: Input node feature dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Output embedding dimension.

    Stability: Experimental.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 32) -> None:
        super().__init__()
        self.conv1 = _MessagePassingMean(in_dim, hidden_dim)
        self.conv2 = _MessagePassingMean(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode node features to embeddings.

        Args:
            x: ``FloatTensor[N, D]``.
            edge_index: ``LongTensor[2, E]``.
            num_nodes: Optional.

        Returns:
            ``FloatTensor[N, out_dim]``.
        """
        N = num_nodes if num_nodes is not None else x.size(0)
        h = self.conv1(x.float(), edge_index, N)
        h = self.conv2(h, edge_index, N)
        return h


# ── Decoders ─────────────────────────────────────────────────────────────────


class DotProductDecoder(nn.Module):
    """Dot-product edge decoder.  Score(u, v) = z_u · z_v.

    Stability: Experimental.
    """

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute edge scores.

        Args:
            z: ``FloatTensor[N, D]`` node embeddings.
            edge_index: ``LongTensor[2, E]`` edge pairs to score.

        Returns:
            ``FloatTensor[E]`` raw logits.
        """
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def full_adjacency_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Compute all-pairs logits ``z @ z.T``.

        Only safe for small graphs (≤ 2 000 nodes).

        Args:
            z: ``FloatTensor[N, D]``.

        Returns:
            ``FloatTensor[N, N]``.
        """
        if z.size(0) > 2_000:
            raise ValueError(
                "full_adjacency_logits: N > 2000.  Use DotProductDecoder.forward "
                "with explicit edge pairs for large graphs."
            )
        return z @ z.t()


class MLPEdgeDecoder(nn.Module):
    """MLP edge decoder.  Score(u, v) = MLP([z_u || z_v]).

    Args:
        in_dim: Node embedding dimension.
        hidden_dim: MLP hidden dimension.

    Stability: Experimental.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute edge scores.

        Args:
            z: ``FloatTensor[N, D]``.
            edge_index: ``LongTensor[2, E]``.

        Returns:
            ``FloatTensor[E]`` raw logits.
        """
        pair = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        return self.mlp(pair).squeeze(-1)


# ── Graph Autoencoder ─────────────────────────────────────────────────────────


class GraphAutoencoder(nn.Module):
    """Graph Autoencoder (GAE) for link prediction.

    Encodes node features into embeddings and reconstructs edges via a
    decoder.

    Args:
        encoder: A GNN encoder with signature
            ``forward(x, edge_index, num_nodes) -> FloatTensor[N, D]``.
        decoder: An edge decoder.  Default: :class:`DotProductDecoder`.

    Stability: Experimental.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder or DotProductDecoder()

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode node features.

        Returns:
            ``FloatTensor[N, D]`` embeddings.
        """
        return self.encoder(x, edge_index, num_nodes)

    def decode(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode scores for positive and negative edges.

        Returns:
            ``(pos_logits, neg_logits)`` — two ``FloatTensor[E]``.
        """
        pos_logits = self.decoder(z, pos_edge_index)
        neg_logits = self.decoder(z, neg_edge_index)
        return pos_logits, neg_logits

    def recon_loss(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy reconstruction loss.

        Args:
            z: ``FloatTensor[N, D]`` embeddings.
            pos_edge_index: ``LongTensor[2, E_pos]`` positive edges.
            neg_edge_index: ``LongTensor[2, E_neg]`` negative edges.

        Returns:
            Scalar loss.
        """
        pos_logits, neg_logits = self.decode(z, pos_edge_index, neg_edge_index)
        pos_loss = -F.logsigmoid(pos_logits).mean()
        neg_loss = -F.logsigmoid(-neg_logits).mean()
        return pos_loss + neg_loss

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode and compute reconstruction loss in one pass.

        Returns:
            Scalar loss.
        """
        z = self.encode(x, edge_index, num_nodes)
        return self.recon_loss(z, pos_edge_index, neg_edge_index)


# ── VGAE ─────────────────────────────────────────────────────────────────────


class _VGAEEncoder(nn.Module):
    """Dual-head encoder that outputs mu and log_sigma."""

    def __init__(self, base_encoder: nn.Module, out_dim: int) -> None:
        super().__init__()
        self.base = base_encoder
        # We need the output dimension of the base encoder.
        self._out_dim = out_dim
        self.mu_head = nn.Identity()   # base already produces mu
        # log_sigma head shares the same base up to the last layer.

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.base(x, edge_index, num_nodes)
        # For simplicity, log_sigma is a learned constant per dimension.
        return mu, self._log_sigma


class VGAE(nn.Module):
    """Variational Graph Auto-Encoder (Kipf & Welling, 2016).

    Learns a Gaussian posterior ``q(Z|X,A)`` via mean and log-variance
    encoders, then decodes via a dot-product (or custom) decoder.

    Args:
        mu_encoder: GNN encoder for the mean ``mu``.
        logstd_encoder: GNN encoder for ``log(sigma)``.  When ``None``,
            a separate :class:`GCNEncoder` with the same architecture is
            created.
        decoder: Edge decoder.  Default: :class:`DotProductDecoder`.

    Stability: Experimental.
    """

    def __init__(
        self,
        mu_encoder: nn.Module,
        logstd_encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.mu_encoder = mu_encoder
        # Build a twin log-sigma encoder.
        if logstd_encoder is not None:
            self.logstd_encoder = logstd_encoder
        elif isinstance(mu_encoder, GCNEncoder):
            self.logstd_encoder = GCNEncoder(
                mu_encoder.conv1.lin.in_features,
                mu_encoder.conv1.lin.out_features,
                mu_encoder.conv2.lin.out_features,
            )
        else:
            # Fallback: share the mu encoder (acceptable for tiny tasks).
            self.logstd_encoder = mu_encoder
        self.decoder = decoder or DotProductDecoder()
        self._mu: Optional[torch.Tensor] = None
        self._logstd: Optional[torch.Tensor] = None

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode via reparameterisation.

        Stores ``self._mu`` and ``self._logstd`` for KL computation.

        Returns:
            ``FloatTensor[N, D]`` sampled ``z``.
        """
        self._mu = self.mu_encoder(x, edge_index, num_nodes)
        self._logstd = self.logstd_encoder(x, edge_index, num_nodes).clamp(max=10.0)
        if self.training:
            eps = torch.randn_like(self._mu)
            return self._mu + eps * self._logstd.exp()
        return self._mu

    def kl_loss(self) -> torch.Tensor:
        """KL divergence: ``KL[q(Z|X,A) || p(Z)] = -0.5 * (1 + 2*logstd - mu² - exp(2*logstd))``.

        Returns:
            Scalar KL term (mean over nodes and dimensions).

        Raises:
            RuntimeError: If ``encode`` has not been called yet.
        """
        if self._mu is None or self._logstd is None:
            raise RuntimeError("Call encode() before kl_loss().")
        return -0.5 * (1 + 2 * self._logstd - self._mu.pow(2) - (2 * self._logstd).exp()).mean()

    def recon_loss(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy reconstruction loss."""
        pos_logits = self.decoder(z, pos_edge_index)
        neg_logits = self.decoder(z, neg_edge_index)
        pos_loss = -F.logsigmoid(pos_logits).mean()
        neg_loss = -F.logsigmoid(-neg_logits).mean()
        return pos_loss + neg_loss

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        num_nodes: Optional[int] = None,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Encode and compute ELBO loss = reconstruction + beta * KL.

        Args:
            x: Node features.
            edge_index: Graph edges.
            pos_edge_index: Positive training edges.
            neg_edge_index: Negative training edges.
            num_nodes: Optional.
            beta: KL weight (default 1.0; set to 0 for pure GAE).

        Returns:
            Scalar loss.
        """
        z = self.encode(x, edge_index, num_nodes)
        recon = self.recon_loss(z, pos_edge_index, neg_edge_index)
        kl = self.kl_loss() if beta > 0 else torch.tensor(0.0, device=x.device)
        return recon + beta * kl


# ── Training helpers ──────────────────────────────────────────────────────────


def train_gae_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
    beta: float = 1.0,
) -> float:
    """One training step for GAE or VGAE.

    Args:
        model: :class:`GraphAutoencoder` or :class:`VGAE`.
        optimizer: PyTorch optimizer.
        x: ``FloatTensor[N, D]`` node features.
        edge_index: ``LongTensor[2, E]`` graph edges.
        pos_edge_index: Positive training edges.
        neg_edge_index: Negative training edges.
        num_nodes: Optional.
        beta: KL weight for VGAE (ignored for GAE).

    Returns:
        Float loss value.
    """
    model.train()
    optimizer.zero_grad()
    if isinstance(model, VGAE):
        loss = model(x, edge_index, pos_edge_index, neg_edge_index, num_nodes, beta)
    else:
        loss = model(x, edge_index, pos_edge_index, neg_edge_index, num_nodes)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


@torch.no_grad()
def evaluate_link_prediction(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate link prediction metrics.

    Computes AUROC, AUPRC, and accuracy at threshold 0.5.

    Args:
        model: GAE or VGAE.
        x: Node features.
        edge_index: Observed graph edges.
        pos_edge_index: Positive test edges.
        neg_edge_index: Negative test edges.
        num_nodes: Optional.

    Returns:
        Dict with ``auroc``, ``auprc``, ``accuracy``.
    """
    model.eval()
    if isinstance(model, VGAE):
        z = model.encode(x, edge_index, num_nodes)
    else:
        z = model.encode(x, edge_index, num_nodes)

    pos_scores = torch.sigmoid(model.decoder(z, pos_edge_index))
    neg_scores = torch.sigmoid(model.decoder(z, neg_edge_index))

    scores = torch.cat([pos_scores, neg_scores]).cpu()
    labels = torch.cat([
        torch.ones(pos_scores.size(0)),
        torch.zeros(neg_scores.size(0)),
    ])

    # Sort descending for AUROC/AUPRC.
    sorted_idx = scores.argsort(descending=True)
    sorted_labels = labels[sorted_idx]
    sorted_scores = scores[sorted_idx]

    # AUROC via trapezoidal rule.
    n_pos = int(labels.sum().item())
    n_neg = int((1 - labels).sum().item())
    tp = sorted_labels.cumsum(0)
    fp = (1 - sorted_labels).cumsum(0)
    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)
    auroc = float(torch.trapezoid(tpr, fpr).abs().item())

    # AUPRC.
    precision = tp / (tp + fp).clamp(min=1)
    auprc = float(torch.trapezoid(precision, tpr).abs().item())

    # Accuracy at 0.5.
    preds = (scores >= 0.5).float()
    accuracy = float((preds == labels).float().mean().item())

    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "accuracy": round(accuracy, 4),
    }

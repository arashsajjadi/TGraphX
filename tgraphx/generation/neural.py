"""Neural graph generation models.

Extends existing VGAE (tgraphx.mining.vgae) with graph sampling capabilities,
plus autoregressive and transformer-based generators.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgraphx.mining.vgae import GCNEncoder
from .data_model import GeneratedGraph

__all__ = [
    "VGAEGraphGenerator",
    "AutoregressiveEdgeGenerator",
    "GraphTransformerGenerator",
]


class VGAEGraphGenerator(nn.Module):
    """Graph generator that samples from a VGAE latent space.

    Extends the existing VGAE (Kipf & Welling 2016) to sample new graphs:
        1. Sample z ~ N(0, I)^[n_nodes, latent_dim]
        2. Decode: A_logits = z z^T (dot product decoder)
        3. Threshold/sample to get edge_index

    Optional node feature decoder: z -> node_features.

    Loss (ELBO):
        L = -E[log p(A|z)] + KL(q(z|X,A) || p(z))
            where KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

    Args:
        encoder: GCN encoder (e.g. from tgraphx.mining.vgae.GCNEncoder).
        latent_dim: Latent space dimension.
        max_nodes: Maximum number of nodes in sampled graphs.
        directed: Whether to generate directed graphs.
        node_feature_dim: If set, adds an MLP decoder for node features.
    """

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        max_nodes: int = 50,
        directed: bool = False,
        node_feature_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
        self.directed = directed

        # Store encoder directly — we compute mu/logvar projections ourselves
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)

        # Optional node feature decoder
        self.node_feature_decoder: Optional[nn.Module] = None
        if node_feature_dim is not None:
            self.node_feature_decoder = nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 2),
                nn.ReLU(),
                nn.Linear(latent_dim * 2, node_feature_dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass: encode graph and compute adjacency logits.

        Args:
            x: Node features [N, F].
            edge_index: LongTensor [2, E].
            num_nodes: Number of nodes. Defaults to x.shape[0].

        Returns:
            (z, mu, logvar, adj_logits, feature_logits)
                z: [N, latent_dim]
                mu: [N, latent_dim]
                logvar: [N, latent_dim]
                adj_logits: [N, N]
                feature_logits: [N, node_feature_dim] or None
        """
        if x.dim() != 2:
            raise ValueError(
                f"VGAEGraphGenerator.forward expects x shape [N, F] but got {list(x.shape)}. "
                f"Pass ImageNodeEncoder-projected features."
            )
        N = x.shape[0]
        h = self.encoder(x, edge_index, N)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(max=10.0)
        # Reparameterization trick
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
        else:
            z = mu
        adj_logits = torch.mm(z, z.t())  # [N, N]

        feature_logits: Optional[torch.Tensor] = None
        if self.node_feature_decoder is not None:
            feature_logits = self.node_feature_decoder(z)

        return z, mu, logvar, adj_logits, feature_logits

    def reconstruction_loss(
        self,
        z: torch.Tensor,
        adj_target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """VGAE ELBO loss.

        L = BCE(sigma(z z^T), A) + KL(q(z|X,A) || N(0,I))
        KL = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))

        Args:
            z: Latent embeddings [N, latent_dim].
            adj_target: Target adjacency [N, N] (binary).
            mu: Mean [N, latent_dim].
            logvar: Log-variance [N, latent_dim].

        Returns:
            Scalar loss tensor.
        """
        adj_logits = torch.mm(z, z.t())
        recon_loss = F.binary_cross_entropy_with_logits(adj_logits, adj_target)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    def sample_graph(
        self,
        n_nodes: int,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
        threshold: float = 0.5,
    ) -> GeneratedGraph:
        """Sample a new graph from the prior.

        z ~ N(0, I), A = (sigmoid(z z^T) > threshold)

        Args:
            n_nodes: Number of nodes to sample.
            device: Target device.
            generator: Optional torch.Generator for reproducibility.
            threshold: Adjacency probability threshold.

        Returns:
            GeneratedGraph.
        """
        if n_nodes > self.max_nodes:
            raise ValueError(
                f"n_nodes={n_nodes} > max_nodes={self.max_nodes}"
            )
        with torch.no_grad():
            z = torch.randn(n_nodes, self.latent_dim, device=device, generator=generator)
            adj_logits = torch.mm(z, z.t())
            adj_probs = torch.sigmoid(adj_logits)

            if not self.directed:
                adj_probs = (adj_probs + adj_probs.t()) / 2.0

            adj_binary = (adj_probs > threshold).float()
            adj_binary.fill_diagonal_(0.0)

            edge_index = adj_binary.nonzero(as_tuple=False).t().contiguous()

            node_features: Optional[torch.Tensor] = None
            if self.node_feature_decoder is not None:
                node_features = self.node_feature_decoder(z)

        return GeneratedGraph(
            edge_index=edge_index.long(),
            num_nodes=n_nodes,
            directed=self.directed,
            node_features=node_features,
            metadata={"generator": "VGAEGraphGenerator", "n_nodes": n_nodes},
        )


class AutoregressiveEdgeGenerator(nn.Module):
    """Autoregressive RNN-based edge sequence generator.

    Predicts edges one by one in BFS order. The model outputs a binary
    decision at each step: add an edge or not.

    Loss:
        Cross-entropy over edge decisions:
        L = -sum_t [y_t log p_t + (1-y_t) log(1-p_t)]

    Args:
        num_nodes: Number of nodes (fixed graph size).
        hidden_dim: RNN hidden state dimension.
        rnn_type: 'gru' or 'lstm'.
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        rnn_type: str = "gru",
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type

        # Edge sequence length: n*(n-1)/2 for undirected
        self.seq_len = num_nodes * (num_nodes - 1) // 2
        input_dim = 2  # (edge_present, step_frac)

        if rnn_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"rnn_type must be 'gru' or 'lstm', got {rnn_type!r}")

        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, edge_sequences: torch.Tensor) -> torch.Tensor:
        """Forward pass with teacher forcing.

        Args:
            edge_sequences: BoolTensor [B, seq_len] — ground truth edge decisions.

        Returns:
            FloatTensor [B, seq_len] — logits for each edge decision.
        """
        B, S = edge_sequences.shape
        # Build input: (edge_decision, step_fraction)
        step_fracs = torch.arange(S, device=edge_sequences.device, dtype=torch.float) / max(S - 1, 1)
        step_fracs = step_fracs.unsqueeze(0).expand(B, -1)  # [B, S]

        inputs = torch.stack([
            edge_sequences.float(),
            step_fracs,
        ], dim=-1)  # [B, S, 2]

        out, _ = self.rnn(inputs)  # [B, S, hidden_dim]
        logits = self.output_head(out).squeeze(-1)  # [B, S]
        return logits

    def sample(
        self,
        n_nodes: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        temperature: float = 1.0,
    ) -> GeneratedGraph:
        """Sample a graph autoregressively.

        Args:
            n_nodes: Number of nodes (must equal self.num_nodes).
            generator: Optional torch.Generator.
            temperature: Sampling temperature.

        Returns:
            GeneratedGraph.
        """
        n = n_nodes if n_nodes is not None else self.num_nodes
        if n != self.num_nodes:
            raise ValueError(
                f"n_nodes={n} != self.num_nodes={self.num_nodes}. "
                f"Autoregressive generator is fixed to self.num_nodes."
            )

        seq_len = n * (n - 1) // 2
        with torch.no_grad():
            edge_decisions = torch.zeros(seq_len, dtype=torch.bool)
            hidden = None

            for t in range(seq_len):
                step_frac = t / max(seq_len - 1, 1)
                inp = torch.tensor(
                    [[float(edge_decisions[t - 1]) if t > 0 else 0.0, step_frac]],
                    dtype=torch.float,
                ).unsqueeze(0)  # [1, 1, 2]

                out, hidden = self.rnn(inp, hidden)
                logit = self.output_head(out.squeeze(0)).squeeze(-1)  # [1]
                prob = torch.sigmoid(logit / temperature)
                rand_val = torch.rand(1, generator=generator).item()
                edge_decisions[t] = rand_val < prob.item()

        # Build edge_index from upper-triangular decisions
        src_list, dst_list = [], []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if edge_decisions[idx]:
                    src_list.extend([i, j])
                    dst_list.extend([j, i])
                idx += 1

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return GeneratedGraph(
            edge_index=edge_index,
            num_nodes=n,
            directed=False,
            metadata={"generator": "AutoregressiveEdgeGenerator"},
        )


class GraphTransformerGenerator(nn.Module):
    """Transformer-based graph generator over action sequences.

    [Experimental] Uses a causal transformer to predict action tokens.
    Each token represents ADD_EDGE or STOP decisions.

    This model is experimental and intended for research. It does not
    guarantee convergence on small graphs without hyperparameter tuning.

    Args:
        max_nodes: Maximum number of nodes.
        hidden_dim: Transformer model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
    """

    def __init__(
        self,
        max_nodes: int = 20,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Vocab: 0=PAD, 1=ADD_EDGE, 2=STOP, 3..3+max_nodes*max_nodes = edge slots
        self.vocab_size = 3 + max_nodes * max_nodes
        self.embedding = nn.Embedding(self.vocab_size, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, self.vocab_size)

        # Sinusoidal positional encoding
        self._build_pos_enc(max_length=max_nodes * max_nodes + 10)

    def _build_pos_enc(self, max_length: int) -> None:
        pe = torch.zeros(max_length, self.hidden_dim)
        pos = torch.arange(max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2, dtype=torch.float) *
            (-math.log(10000.0) / self.hidden_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        if self.hidden_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pos_enc", pe)

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            action_tokens: LongTensor [B, T] of action token IDs.

        Returns:
            FloatTensor [B, T, vocab_size] — logits for next token.
        """
        B, T = action_tokens.shape
        emb = self.embedding(action_tokens)
        pos = self.pos_enc[:T].unsqueeze(0).expand(B, -1, -1)  # type: ignore
        x = emb + pos

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=action_tokens.device)
        out = self.transformer(x, mask=mask, is_causal=True)
        return self.output_head(out)  # [B, T, vocab_size]

    def sample(
        self,
        n_nodes: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        max_steps: Optional[int] = None,
    ) -> GeneratedGraph:
        """Sample a graph autoregressively using the transformer.

        [Experimental] Stops at STOP token or max_steps.

        Args:
            n_nodes: Number of nodes to generate for.
            generator: Optional torch.Generator.
            max_steps: Maximum generation steps.

        Returns:
            GeneratedGraph.
        """
        n = n_nodes if n_nodes is not None else self.max_nodes
        if n > self.max_nodes:
            raise ValueError(f"n_nodes={n} > max_nodes={self.max_nodes}")

        max_s = max_steps if max_steps is not None else n * (n - 1) // 2 + 5
        tokens = [1]  # Start with ADD_EDGE token

        src_list, dst_list = [], []
        edge_count = 0

        with torch.no_grad():
            for _ in range(max_s):
                inp = torch.tensor([tokens], dtype=torch.long)
                logits = self(inp)  # [1, T, vocab_size]
                next_logits = logits[0, -1]  # [vocab_size]
                next_token = int(torch.multinomial(
                    F.softmax(next_logits, dim=-1), 1, generator=generator
                ).item())
                tokens.append(next_token)

                if next_token == 2:  # STOP
                    break
                elif next_token >= 3:
                    edge_slot = next_token - 3
                    src = edge_slot // n
                    tgt = edge_slot % n
                    if src < n and tgt < n and src != tgt:
                        src_list.extend([src, tgt])
                        dst_list.extend([tgt, src])
                        edge_count += 1

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return GeneratedGraph(
            edge_index=edge_index,
            num_nodes=n,
            directed=False,
            metadata={"generator": "GraphTransformerGenerator", "experimental": True},
        )

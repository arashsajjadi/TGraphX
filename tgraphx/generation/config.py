"""Configuration dataclass for graph generation.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

__all__ = ["GraphGenerationConfig"]


@dataclass
class GraphGenerationConfig:
    """All hyperparameters for graph generation.

    Args:
        max_nodes: Maximum number of nodes per graph.
        max_edges: Maximum number of edges per graph.
        directed: Whether to generate directed graphs.
        node_feature_dim: Node feature dimensionality.
        edge_feature_dim: Edge feature dimensionality. None means no edge features.
        latent_dim: VGAE latent space dimension.
        num_layers: Number of GNN/RNN/transformer layers.
        hidden_dim: Hidden layer width.
        rnn_type: RNN type for autoregressive model ('gru' or 'lstm').
        num_heads: Number of attention heads for transformer.
        learning_rate: Optimizer learning rate.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        seed: Random seed.
        device: Device string ('cpu', 'cuda', etc.).
        threshold: Adjacency probability threshold for VGAE sampling.
        temperature: Sampling temperature for autoregressive models.
    """

    max_nodes: int = 50
    max_edges: int = 500
    directed: bool = False
    node_feature_dim: int = 16
    edge_feature_dim: Optional[int] = None
    latent_dim: int = 32
    num_layers: int = 2
    hidden_dim: int = 64
    rnn_type: str = "gru"
    num_heads: int = 4
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    seed: Optional[int] = None
    device: str = "cpu"
    threshold: float = 0.5
    temperature: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphGenerationConfig":
        """Parse from a JSON-safe dict. Uses json.loads internally for type safety.

        Args:
            d: Dict with config keys. Unknown keys go into ``extra``.

        Returns:
            GraphGenerationConfig instance.
        """
        # Re-parse via JSON to ensure no eval/exec usage
        text = json.dumps(d)
        parsed = json.loads(text)
        known = {
            "max_nodes", "max_edges", "directed", "node_feature_dim",
            "edge_feature_dim", "latent_dim", "num_layers", "hidden_dim",
            "rnn_type", "num_heads", "learning_rate", "batch_size", "epochs",
            "seed", "device", "threshold", "temperature",
        }
        kwargs: Dict[str, Any] = {k: v for k, v in parsed.items() if k in known}
        extra: Dict[str, Any] = {k: v for k, v in parsed.items() if k not in known}
        return cls(**kwargs, extra=extra)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict.

        Returns:
            Dict with all config values.
        """
        d = asdict(self)
        # Ensure JSON-serializable (no tensors)
        return json.loads(json.dumps(d, default=str))

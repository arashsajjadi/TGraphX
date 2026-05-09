"""RL configuration dataclasses.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

__all__ = ["RLTrainingConfig", "PolicyConfig", "RewardConfig", "ConstraintConfig"]


@dataclass
class RLTrainingConfig:
    """All hyperparameters for RL training.

    Args:
        algorithm: One of 'reinforce', 'a2c', 'dqn', 'double_dqn', 'ppo'.
        gamma: Discount factor.
        learning_rate: Optimizer learning rate.
        n_episodes: Number of training episodes.
        n_steps: Steps per rollout (for A2C/PPO).
        batch_size: Replay batch size (for DQN).
        replay_capacity: Replay buffer capacity (for DQN).
        target_update_freq: Target network update frequency (for DQN).
        entropy_coef: Entropy regularization coefficient.
        value_loss_coef: Value loss weight (for AC/PPO).
        grad_clip_norm: Gradient clipping norm.
        clip_eps: PPO clip epsilon.
        gae_lambda: GAE lambda for A2C/PPO.
        n_epochs: PPO epochs per rollout.
        eps_start: Initial epsilon (for DQN).
        eps_end: Final epsilon.
        eps_decay: Epsilon decay.
        seed: Random seed.
        device: Device.
    """

    algorithm: str = "reinforce"
    gamma: float = 0.99
    learning_rate: float = 1e-3
    n_episodes: int = 500
    n_steps: int = 64
    batch_size: int = 32
    replay_capacity: int = 10000
    target_update_freq: int = 100
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    grad_clip_norm: float = 1.0
    clip_eps: float = 0.2
    gae_lambda: float = 0.95
    n_epochs: int = 4
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 1000.0
    seed: Optional[int] = None
    device: str = "cpu"
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RLTrainingConfig":
        text = json.dumps(d)
        parsed = json.loads(text)
        known = {
            "algorithm", "gamma", "learning_rate", "n_episodes", "n_steps",
            "batch_size", "replay_capacity", "target_update_freq", "entropy_coef",
            "value_loss_coef", "grad_clip_norm", "clip_eps", "gae_lambda", "n_epochs",
            "eps_start", "eps_end", "eps_decay", "seed", "device",
        }
        kwargs = {k: v for k, v in parsed.items() if k in known}
        extra = {k: v for k, v in parsed.items() if k not in known}
        return cls(**kwargs, extra=extra)

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=str))


@dataclass
class PolicyConfig:
    """Policy network architecture configuration."""

    node_in_dim: int = 8
    edge_in_dim: int = 0
    hidden_dim: int = 64
    num_actions: int = 10
    num_gnn_layers: int = 2
    gnn_type: str = "mean"
    shared_encoder: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=str))


@dataclass
class RewardConfig:
    """Reward component weights."""

    step_penalty: float = -0.01
    reach_bonus: float = 10.0
    validity_bonus: float = 1.0
    constraint_penalty: float = -1.0
    completion_bonus: float = 20.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=str))


@dataclass
class ConstraintConfig:
    """Graph structure constraints for RL environments."""

    max_nodes: int = 50
    max_edges: int = 500
    no_self_loops: bool = True
    connected: bool = False
    acyclic: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=str))

"""Base graph RL environment.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from tgraphx.generation.data_model import GeneratedGraph

__all__ = ["GraphEnvConfig", "GraphEnv"]


@dataclass
class GraphEnvConfig:
    """Configuration for graph RL environments.

    Args:
        max_steps: Maximum steps per episode.
        reward_scale: Scalar multiplier for all rewards.
        seed: Random seed.
        directed: Whether the graph is directed.
        max_nodes: Maximum nodes.
        max_edges: Maximum edges.
        device: Device string.
    """

    max_steps: int = 100
    reward_scale: float = 1.0
    seed: Optional[int] = None
    directed: bool = False
    max_nodes: int = 50
    max_edges: int = 500
    device: str = "cpu"


class GraphEnv(abc.ABC):
    """Abstract base class for graph RL environments.

    State: partial or complete graph structure + step count.
    Action: discrete integer.
    Observation: dict with edge_index, node_features, edge_features, action_mask, step, done.

    Design contract:
        - reset() is deterministic with the same seed.
        - step() never silently modifies internal state — creates copies.
        - All tensors in observations are on self.device.
    """

    def __init__(self, config: GraphEnvConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self._step_count: int = 0
        self._done: bool = False
        self._rng: Optional[torch.Generator] = None
        if config.seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(config.seed)

    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the environment.

        Args:
            seed: Optional seed to override config.seed.

        Returns:
            Initial observation dict.
        """
        ...

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Take a step.

        Args:
            action: Integer action.

        Returns:
            (obs, reward, done, truncated, info)
        """
        ...

    @abc.abstractmethod
    def observe(self) -> Dict[str, Any]:
        """Return current observation dict.

        Keys: edge_index, node_features, edge_features, action_mask, step, done.
        """
        ...

    @abc.abstractmethod
    def valid_action_mask(self) -> torch.Tensor:
        """Return BoolTensor of valid actions.

        Returns:
            BoolTensor [num_actions].
        """
        ...

    @abc.abstractmethod
    def state_to_graph(self) -> GeneratedGraph:
        """Convert current state to a GeneratedGraph."""
        ...

    @property
    def num_nodes(self) -> int:
        raise NotImplementedError

    @property
    def num_edges(self) -> int:
        raise NotImplementedError

    @property
    def action_space(self) -> int:
        """Number of discrete actions."""
        raise NotImplementedError

    @property
    def observation_space(self) -> Dict[str, Any]:
        """Description of observation space."""
        return {}

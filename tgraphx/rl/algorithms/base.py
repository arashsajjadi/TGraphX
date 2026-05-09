"""Base RL agent.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import torch

__all__ = ["BaseAgent"]


class BaseAgent(abc.ABC):
    """Abstract base class for RL agents."""

    @abc.abstractmethod
    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        """Select an action given an observation.

        Args:
            obs: Observation dict.
            deterministic: If True, select the greedy action.
            generator: Optional torch.Generator.

        Returns:
            Integer action.
        """
        ...

    @abc.abstractmethod
    def update(self, batch: Any) -> Dict[str, float]:
        """Update policy from a batch.

        Args:
            batch: Batch of transitions.

        Returns:
            Dict of loss values.
        """
        ...

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return the agent's state."""
        ...

    @abc.abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load agent state."""
        ...

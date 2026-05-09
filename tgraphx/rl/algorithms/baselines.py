"""Random and greedy baseline policies.

Stability: Beta (v0.7.0+) — deterministic, no learning.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from .base import BaseAgent

__all__ = ["RandomPolicy", "GreedyPolicy"]


class RandomPolicy(BaseAgent):
    """Samples uniformly from valid actions (respects action mask).

    Args:
        n_actions: Total number of actions (fallback when mask absent).
    """

    def __init__(self, n_actions: int = 10) -> None:
        self.n_actions = n_actions

    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        """Select a random valid action.

        Args:
            obs: Observation dict (may contain 'action_mask').
            deterministic: Ignored for random policy (always random).
            generator: Optional torch.Generator for reproducibility.

        Returns:
            Integer action index.
        """
        mask = obs.get("action_mask")
        if mask is not None:
            mask = mask.bool()
            valid = mask.nonzero(as_tuple=False).squeeze(1)
            if valid.numel() > 0:
                idx = int(torch.randint(valid.numel(), (1,), generator=generator).item())
                return int(valid[idx].item())

        return int(torch.randint(self.n_actions, (1,), generator=generator).item())

    def update(self, batch: Any = None) -> Dict[str, float]:
        """No-op: random policy does not learn.

        Returns:
            Empty dict.
        """
        return {}

    def state_dict(self) -> Dict[str, Any]:
        return {"n_actions": self.n_actions}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.n_actions = state.get("n_actions", self.n_actions)


class GreedyPolicy(BaseAgent):
    """Selects action with highest scoring_fn(obs, action_idx) score.

    Args:
        scoring_fn: callable(obs, action_idx: int) -> float.
        n_actions: Number of possible actions.
    """

    def __init__(
        self,
        scoring_fn: Callable[[Dict[str, Any], int], float],
        n_actions: int,
    ) -> None:
        self.scoring_fn = scoring_fn
        self.n_actions = n_actions

    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        """Select the highest-scoring valid action.

        Args:
            obs: Observation dict (may contain 'action_mask').
            deterministic: Ignored (always greedy).
            generator: Ignored.

        Returns:
            Integer action index.
        """
        mask = obs.get("action_mask")
        best_action = 0
        best_score = float("-inf")

        for a in range(self.n_actions):
            if mask is not None:
                if a >= len(mask) or not bool(mask[a].item()):
                    continue
            score = self.scoring_fn(obs, a)
            if score > best_score:
                best_score = score
                best_action = a

        return best_action

    def update(self, batch: Any = None) -> Dict[str, float]:
        """No-op: greedy policy does not learn.

        Returns:
            Empty dict.
        """
        return {}

    def state_dict(self) -> Dict[str, Any]:
        return {"n_actions": self.n_actions}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.n_actions = state.get("n_actions", self.n_actions)

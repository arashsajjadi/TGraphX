"""Replay buffer for RL.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import copy
from collections import deque
from typing import Any, Dict, List, Optional

import torch

__all__ = ["ReplayBuffer", "RolloutBuffer"]


def _detach_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Detach all tensors in observation to avoid holding autograd graph."""
    result = {}
    for k, v in obs.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach().clone()
        elif v is None:
            result[k] = None
        else:
            result[k] = copy.copy(v)
    return result


class ReplayBuffer:
    """Experience replay buffer.

    Stores (obs_dict, action, reward, next_obs_dict, done) transitions.

    Args:
        capacity: Maximum number of transitions to store.
        device: Device to move tensors to when sampling.
    """

    def __init__(self, capacity: int, device: torch.device = torch.device("cpu")) -> None:
        self.capacity = capacity
        self.device = device
        self._buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        obs: Dict[str, Any],
        action: Any,
        reward: float,
        next_obs: Dict[str, Any],
        done: bool,
    ) -> None:
        """Add a transition to the buffer.

        Detaches all tensors to prevent holding the autograd graph.

        Args:
            obs: Observation dict.
            action: Action (int for discrete, Tensor for continuous).
            reward: Scalar reward.
            next_obs: Next observation dict.
            done: Whether episode ended.
        """
        # Normalize action: keep tensor as detached clone, convert scalars to int
        if isinstance(action, torch.Tensor):
            stored_action = action.detach().clone()
        else:
            try:
                stored_action = int(action)
            except (TypeError, ValueError):
                stored_action = action

        self._buffer.append((
            _detach_obs(obs),
            stored_action,
            float(reward),
            _detach_obs(next_obs),
            bool(done),
        ))

    def sample(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> List[Any]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.
            generator: Optional torch.Generator.

        Returns:
            List of (obs, action, reward, next_obs, done) tuples.
        """
        n = len(self._buffer)
        if n < batch_size:
            batch_size = n
        indices = torch.randperm(n, generator=generator)[:batch_size].tolist()
        buf_list = list(self._buffer)
        return [buf_list[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self._buffer) >= batch_size


class RolloutBuffer:
    """Rollout buffer for on-policy algorithms (PPO, A2C).

    Stores a complete rollout then computes returns and advantages using
    Generalized Advantage Estimation (GAE).

    GAE formula:
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        A_t = sum_l (gamma*lambda)^l * delta_{t+l}
        R_t = A_t + V(s_t)

    Args:
        capacity: Maximum rollout length.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
    """

    def __init__(
        self,
        capacity: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._obs: List[Any] = []
        self._actions: List[Any] = []
        self._rewards: List[float] = []
        self._dones: List[bool] = []
        self._values: List[float] = []
        self._log_probs: List[float] = []
        self._advantages: Optional[torch.Tensor] = None
        self._returns: Optional[torch.Tensor] = None

    def add(
        self,
        obs: Any,
        action: Any,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """Add a step to the buffer.

        Args:
            obs: Observation.
            action: Action taken.
            reward: Reward received.
            done: Whether episode ended.
            value: Value estimate V(s_t).
            log_prob: Log probability of action.
        """
        self._obs.append(obs)
        self._actions.append(action)
        self._rewards.append(float(reward))
        self._dones.append(bool(done))
        self._values.append(float(value))
        self._log_probs.append(float(log_prob))

    def compute_returns_and_advantages(self, last_value: float = 0.0) -> None:
        """Compute GAE advantages and discounted returns.

        Args:
            last_value: Value estimate for state after last step.
                        0.0 if last step was terminal.
        """
        T = len(self._rewards)
        advantages = torch.zeros(T)
        returns = torch.zeros(T)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(T)):
            mask = 1.0 - float(self._dones[t])
            delta = self._rewards[t] + self.gamma * next_value * mask - self._values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            next_value = self._values[t]

        values_t = torch.tensor(self._values, dtype=torch.float)
        returns = advantages + values_t
        self._advantages = advantages
        self._returns = returns

    def get_batches(
        self,
        mini_batch_size: int,
        generator: Optional[torch.Generator] = None,
    ):
        """Yield mini-batches of tensors.

        Args:
            mini_batch_size: Size of each mini-batch.
            generator: Optional RNG for shuffling.

        Yields:
            Dict with keys: actions, advantages, returns, values, log_probs.
        """
        T = len(self._actions)
        if T == 0:
            return

        indices = torch.randperm(T, generator=generator)
        actions_t = torch.tensor(
            [float(a) if not isinstance(a, (list, torch.Tensor)) else a for a in self._actions],
            dtype=torch.float,
        )
        log_probs_t = torch.tensor(self._log_probs, dtype=torch.float)
        values_t = torch.tensor(self._values, dtype=torch.float)

        adv = self._advantages if self._advantages is not None else torch.zeros(T)
        ret = self._returns if self._returns is not None else values_t

        start = 0
        while start < T:
            batch_idx = indices[start: start + mini_batch_size]
            yield {
                "actions": actions_t[batch_idx],
                "advantages": adv[batch_idx],
                "returns": ret[batch_idx],
                "values": values_t[batch_idx],
                "log_probs": log_probs_t[batch_idx],
                "indices": batch_idx,
            }
            start += mini_batch_size

    def clear(self) -> None:
        """Reset buffer to empty state."""
        self._obs = []
        self._actions = []
        self._rewards = []
        self._dones = []
        self._values = []
        self._log_probs = []
        self._advantages = None
        self._returns = None

    def __len__(self) -> int:
        return len(self._rewards)

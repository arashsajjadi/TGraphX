"""Exploration strategies for RL.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import math
from typing import Optional

import torch

__all__ = [
    "EpsilonGreedy",
    "LinearEpsilonDecay",
    "BoltzmannExploration",
    "UCBExploration",
    "EntropyRegularizer",
]


class EpsilonGreedy:
    """Exponential epsilon-greedy exploration.

    ε(t) = ε_end + (ε_start - ε_end) * exp(-t / decay)

    Args:
        eps_start: Initial epsilon.
        eps_end: Final epsilon.
        eps_decay: Decay constant.
    """

    def __init__(
        self,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 1000.0,
    ) -> None:
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def get_epsilon(self, step: int) -> float:
        """Get epsilon at step t."""
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-step / max(self.eps_decay, 1.0))

    def should_explore(
        self,
        step: int,
        generator: Optional[torch.Generator] = None,
    ) -> bool:
        """Return True if agent should explore (sample random action)."""
        eps = self.get_epsilon(step)
        return torch.rand(1, generator=generator).item() < eps


class LinearEpsilonDecay:
    """Linear epsilon decay.

    ε(t) = ε_start - t * (ε_start - ε_end) / n_steps  if t < n_steps
    ε(t) = ε_end                                         if t >= n_steps

    Args:
        eps_start: Initial epsilon.
        eps_end: Final epsilon.
        n_steps: Number of steps for linear decay.
    """

    def __init__(
        self,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        n_steps: int = 10000,
    ) -> None:
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.n_steps = n_steps

    def get_epsilon(self, step: int) -> float:
        """Get epsilon at step t."""
        if step >= self.n_steps:
            return self.eps_end
        frac = step / self.n_steps
        return self.eps_start - frac * (self.eps_start - self.eps_end)

    def should_explore(
        self,
        step: int,
        generator: Optional[torch.Generator] = None,
    ) -> bool:
        eps = self.get_epsilon(step)
        return torch.rand(1, generator=generator).item() < eps


class BoltzmannExploration:
    """Boltzmann (softmax) exploration.

    Selects action proportional to exp(Q(s,a) / temperature).

    Args:
        temperature: Softmax temperature (higher = more exploration).
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def sample(
        self,
        q_values: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        """Sample action using Boltzmann distribution.

        Args:
            q_values: FloatTensor [A].
            action_mask: Optional BoolTensor [A].
            generator: Optional torch.Generator.

        Returns:
            Integer action.
        """
        scaled = q_values / max(self.temperature, 1e-6)
        if action_mask is not None:
            scaled = scaled.masked_fill(~action_mask.bool(), -1e9)
        probs = torch.softmax(scaled, dim=-1).clamp(min=0)
        return int(torch.multinomial(probs, 1, generator=generator).item())


class UCBExploration:
    """UCB1 exploration for bandit-like settings.

    UCB score(a) = Q(a) + c * sqrt(ln(t) / n(a))

    Args:
        c: Exploration constant.
    """

    def __init__(self, c: float = 1.0) -> None:
        self.c = c

    def ucb_score(
        self,
        q_values: torch.Tensor,
        counts: torch.Tensor,
        total_steps: int,
    ) -> torch.Tensor:
        """Compute UCB scores.

        Args:
            q_values: FloatTensor [A] — estimated Q-values.
            counts: FloatTensor [A] — number of times each action was taken.
            total_steps: Total steps taken.

        Returns:
            FloatTensor [A] — UCB scores.
        """
        eps = 1e-9
        exploration_bonus = self.c * torch.sqrt(
            math.log(max(total_steps, 1)) / (counts + eps)
        )
        return q_values + exploration_bonus

    def select(
        self,
        q_values: torch.Tensor,
        counts: torch.Tensor,
        total_steps: int,
        action_mask: Optional[torch.Tensor] = None,
    ) -> int:
        """Select action using UCB.

        Args:
            q_values: FloatTensor [A].
            counts: FloatTensor [A].
            total_steps: Total steps.
            action_mask: Optional BoolTensor [A].

        Returns:
            Integer action.
        """
        scores = self.ucb_score(q_values, counts, total_steps)
        if action_mask is not None:
            scores = scores.masked_fill(~action_mask.bool(), -1e9)
        return int(scores.argmax().item())


class EntropyRegularizer:
    """Entropy regularization bonus for policy gradient methods.

    Entropy: H(π) = -Σ_a π(a|s) log π(a|s)
    Bonus: -coef * H(π) is subtracted from the loss (maximizes entropy).

    Args:
        coef: Entropy coefficient (higher = more exploration).
    """

    def __init__(self, coef: float = 0.01) -> None:
        self.coef = coef

    def bonus(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute entropy bonus (positive value to be added to reward/subtracted from loss).

        Args:
            logits: FloatTensor [A] — un-normalized logits.
            mask: Optional BoolTensor [A] — valid actions.

        Returns:
            Scalar entropy bonus.
        """
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        probs = torch.softmax(logits, dim=-1).clamp(min=1e-9)
        entropy = -(probs * torch.log(probs)).sum()
        return self.coef * entropy

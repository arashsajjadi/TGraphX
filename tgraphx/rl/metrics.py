"""RL metrics.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

__all__ = [
    "episodic_return_mean",
    "episodic_return_std",
    "success_rate",
    "episode_length_mean",
    "policy_entropy",
    "approximate_kl",
    "explained_variance",
    "gradient_norm",
    "action_validity_rate",
    "td_error_mean",
]


def episodic_return_mean(returns: List[float]) -> float:
    """Mean episodic return.

    Args:
        returns: List of per-episode total returns.

    Returns:
        Float.
    """
    if not returns:
        return 0.0
    return sum(returns) / len(returns)


def episodic_return_std(returns: List[float]) -> float:
    """Standard deviation of episodic returns.

    Args:
        returns: List of per-episode total returns.

    Returns:
        Float.
    """
    if len(returns) < 2:
        return 0.0
    t = torch.tensor(returns, dtype=torch.float)
    return float(t.std().item())


def success_rate(infos: List[Dict[str, Any]]) -> float:
    """Fraction of episodes where info['success'] == True.

    Args:
        infos: List of info dicts from env.step().

    Returns:
        Float in [0, 1].
    """
    if not infos:
        return 0.0
    successes = sum(1 for info in infos if info.get("success", False))
    return successes / len(infos)


def episode_length_mean(lengths: List[int]) -> float:
    """Mean episode length.

    Args:
        lengths: List of per-episode step counts.

    Returns:
        Float.
    """
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


def policy_entropy(
    logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """H(π) = -Σ_a π(a|s) log π(a|s) over valid actions.

    Args:
        logits: FloatTensor [A] or [B, A].
        mask: Optional BoolTensor [A] or [B, A].

    Returns:
        Float entropy.
    """
    if mask is not None:
        logits = logits.masked_fill(~mask.bool(), -1e9)
    probs = torch.softmax(logits, dim=-1).clamp(min=1e-9)
    entropy = -(probs * torch.log(probs)).sum(dim=-1)
    return float(entropy.mean().item())


def approximate_kl(
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
) -> float:
    """Approximate KL divergence: mean(old_logprob - new_logprob).

    This is a first-order approximation used in PPO.

    Args:
        old_logprobs: FloatTensor [T].
        new_logprobs: FloatTensor [T].

    Returns:
        Float.
    """
    return float((old_logprobs - new_logprobs).mean().item())


def explained_variance(
    values: torch.Tensor,
    returns: torch.Tensor,
) -> float:
    """Fraction of variance in returns explained by value predictions.

    explained_variance = 1 - Var(R - V) / Var(R)

    Args:
        values: Predicted values FloatTensor [T].
        returns: Actual returns FloatTensor [T].

    Returns:
        Float in (-inf, 1]. 1.0 = perfect prediction.
    """
    residual_var = (returns - values).var()
    return_var = returns.var()
    if return_var < 1e-8:
        return float("nan")
    return float((1 - residual_var / return_var).item())


def gradient_norm(model: torch.nn.Module) -> float:
    """Total L2 norm of all model gradients.

    Args:
        model: PyTorch nn.Module.

    Returns:
        Float norm (0.0 if no gradients).
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def action_validity_rate(infos: List[Dict[str, Any]]) -> float:
    """Fraction of steps where info['action_valid'] == True.

    Args:
        infos: List of info dicts from env.step().

    Returns:
        Float in [0, 1].
    """
    if not infos:
        return 0.0
    valid = sum(1 for info in infos if info.get("action_valid", True))
    return valid / len(infos)


def td_error_mean(td_errors: List[float]) -> float:
    """Mean |TD error|.

    Args:
        td_errors: List of TD errors.

    Returns:
        Float.
    """
    if not td_errors:
        return 0.0
    return sum(abs(e) for e in td_errors) / len(td_errors)

"""REINFORCE algorithm for graph RL.

Mathematics:
    Policy gradient: ∇J ≈ Σ_t ∇log π_θ(a_t|s_t) * G_t
    where G_t = Σ_{k≥t} γ^{k-t} r_k (discounted return)

    Entropy bonus: -entropy_coef * H(π)
    H(π) = -Σ_a π(a|s) log π(a|s)

    Optional baseline: subtract mean return E[G_t] from G_t.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import BaseAgent

__all__ = ["REINFORCEAgent"]


def _compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    """Compute discounted returns G_t = Σ_{k≥t} γ^{k-t} r_k."""
    G = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        G.insert(0, running)
    return torch.tensor(G, dtype=torch.float)


class REINFORCEAgent(BaseAgent):
    """REINFORCE (Monte Carlo policy gradient) agent.

    Args:
        policy: Policy network (e.g., GraphPolicyNetwork or GraphActorCriticNetwork).
        optimizer: PyTorch optimizer.
        gamma: Discount factor.
        entropy_coef: Entropy bonus coefficient.
        grad_clip_norm: Maximum gradient norm (None = no clipping).
        use_baseline: If True, subtract mean return as a baseline.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        grad_clip_norm: Optional[float] = 1.0,
        use_baseline: bool = True,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.grad_clip_norm = grad_clip_norm
        self.use_baseline = use_baseline

    def collect_episode(
        self,
        env: Any,
        generator: Optional[torch.Generator] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Collect a single episode trajectory.

        Args:
            env: A GraphEnv instance (must have reset/step/observe).
            generator: Optional torch.Generator.
            max_steps: Override max steps.

        Returns:
            Dict with 'log_probs', 'rewards', 'entropies', 'total_return'.
        """
        from tgraphx.rl.networks.policy import MaskedCategoricalPolicy

        obs = env.reset()
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        entropies: List[torch.Tensor] = []

        ms = max_steps or getattr(env.config, "max_steps", 100)

        for step in range(ms):
            node_features = obs.get("node_features")
            edge_index = obs.get("edge_index")
            action_mask = obs.get("action_mask")

            if node_features is None:
                node_features = torch.zeros(1, 1)
            if edge_index is None:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            self.policy.train()
            logits = self.policy(node_features, edge_index)
            logits = logits.squeeze(0)  # [num_actions] or [A]

            # Mask
            if action_mask is not None:
                m = action_mask.bool()
                if m.shape[0] != logits.shape[-1]:
                    # Trim mask to logits size or pad logits
                    m_size = min(m.shape[0], logits.shape[-1])
                    m = m[:m_size]
                    logits = logits[:m_size]
            else:
                m = torch.ones(logits.shape[-1], dtype=torch.bool)

            policy_dist = MaskedCategoricalPolicy(logits.unsqueeze(0), m.unsqueeze(0))
            action = int(policy_dist.sample(generator=generator).item())
            log_prob = policy_dist.log_prob(torch.tensor([action]))
            entropy = policy_dist.entropy()

            obs, reward, done, _, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                break

        return {
            "log_probs": log_probs,
            "rewards": rewards,
            "entropies": entropies,
            "total_return": sum(rewards),
        }

    def update(self, trajectory: Dict[str, Any]) -> Dict[str, float]:
        """Compute and apply REINFORCE gradient.

        Args:
            trajectory: Output from collect_episode.

        Returns:
            Dict with 'policy_loss', 'entropy_loss', 'total_loss'.
        """
        log_probs = trajectory["log_probs"]
        rewards = trajectory["rewards"]
        entropies = trajectory["entropies"]

        if not log_probs:
            return {"policy_loss": 0.0, "entropy_loss": 0.0, "total_loss": 0.0}

        returns = _compute_returns(rewards, self.gamma)

        if self.use_baseline:
            returns = returns - returns.mean()

        # Policy gradient loss: -sum(log_pi * G_t)
        policy_loss = torch.tensor(0.0, requires_grad=True)
        entropy_loss = torch.tensor(0.0, requires_grad=True)

        total_lp = sum(
            lp * G for lp, G in zip(log_probs, returns)
        )
        policy_loss = -total_lp / len(log_probs)

        if entropies:
            total_ent = sum(entropies)
            entropy_loss = -self.entropy_coef * total_ent / len(entropies)

        total_loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)

        self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "entropy_loss": float(entropy_loss.item()),
            "total_loss": float(total_loss.item()),
        }

    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        from tgraphx.rl.networks.policy import MaskedCategoricalPolicy

        node_features = obs.get("node_features", torch.zeros(1, 1))
        edge_index = obs.get("edge_index", torch.zeros((2, 0), dtype=torch.long))
        action_mask = obs.get("action_mask")

        with torch.no_grad():
            logits = self.policy(node_features, edge_index).squeeze(0)

        if action_mask is not None:
            m = action_mask.bool()
            m_size = min(m.shape[0], logits.shape[-1])
            m = m[:m_size]
            logits = logits[:m_size]
        else:
            m = torch.ones(logits.shape[-1], dtype=torch.bool)

        if deterministic:
            masked_logits = logits.clone()
            masked_logits[~m] = float("-inf")
            return int(masked_logits.argmax().item())

        policy_dist = MaskedCategoricalPolicy(logits.unsqueeze(0), m.unsqueeze(0))
        return int(policy_dist.sample(generator=generator).item())

    def state_dict(self) -> Dict[str, Any]:
        return {"policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])

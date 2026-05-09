"""Actor-Critic and A2C algorithms for graph RL.

A2C Mathematics:
    Advantage: A_t = Σ_l (γλ)^l δ_{t+l}
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)

    Policy loss: L_pi = -mean(log π(a_t|s_t) * A_t)
    Value loss: L_V = (V(s_t) - R_t)^2

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .base import BaseAgent

__all__ = ["ActorCriticAgent", "A2CAgent"]


def _compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float,
    gae_lambda: float,
    last_value: float = 0.0,
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (GAE).

    A_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
    δ_t = r_t + γ * V(s_{t+1}) * (1-done_t) - V(s_t)
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0

    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < len(values) else last_value
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_val * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae

    return advantages


class ActorCriticAgent(BaseAgent):
    """Online Actor-Critic agent.

    Args:
        actor_critic_net: GraphActorCriticNetwork.
        optimizer: PyTorch optimizer.
        gamma: Discount factor.
        entropy_coef: Entropy bonus coefficient.
        value_loss_coef: Value loss scaling.
        grad_clip_norm: Gradient clipping norm.
    """

    def __init__(
        self,
        actor_critic_net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        grad_clip_norm: Optional[float] = 1.0,
    ) -> None:
        self.net = actor_critic_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.grad_clip_norm = grad_clip_norm

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
            logits, _ = self.net(node_features, edge_index)
        logits = logits.squeeze(0)

        if action_mask is not None:
            m = action_mask.bool()
            m_size = min(m.shape[0], logits.shape[-1])
            m = m[:m_size]
            logits = logits[:m_size]
        else:
            m = torch.ones(logits.shape[-1], dtype=torch.bool)

        if deterministic:
            masked = logits.clone()
            masked[~m] = float("-inf")
            return int(masked.argmax().item())

        dist = MaskedCategoricalPolicy(logits.unsqueeze(0), m.unsqueeze(0))
        return int(dist.sample(generator=generator).item())

    def update(self, batch: Any) -> Dict[str, float]:
        """Single online update step.

        Args:
            batch: List of (obs, action, reward, next_obs, done) tuples.

        Returns:
            Loss dict.
        """
        from tgraphx.rl.networks.policy import MaskedCategoricalPolicy

        if not batch:
            return {}

        total_policy_loss = torch.tensor(0.0)
        total_value_loss = torch.tensor(0.0)
        total_entropy = torch.tensor(0.0)
        count = 0

        for obs, action, reward, next_obs, done in batch:
            node_features = obs.get("node_features", torch.zeros(1, 1))
            edge_index = obs.get("edge_index", torch.zeros((2, 0), dtype=torch.long))
            action_mask = obs.get("action_mask")

            next_node_feat = next_obs.get("node_features", torch.zeros(1, 1))
            next_edge_idx = next_obs.get("edge_index", torch.zeros((2, 0), dtype=torch.long))

            logits, value = self.net(node_features, edge_index)
            with torch.no_grad():
                _, next_value = self.net(next_node_feat, next_edge_idx)

            logits = logits.squeeze(0)
            value = value.squeeze()
            next_value = next_value.squeeze()

            # TD target
            td_target = reward + self.gamma * next_value.item() * (1.0 - float(done))
            advantage = td_target - value.item()

            # Mask
            if action_mask is not None:
                m = action_mask.bool()
                m_size = min(m.shape[0], logits.shape[-1])
                m = m[:m_size]
                logits = logits[:m_size]
            else:
                m = torch.ones(logits.shape[-1], dtype=torch.bool)

            dist = MaskedCategoricalPolicy(logits.unsqueeze(0), m.unsqueeze(0))
            log_prob = dist.log_prob(torch.tensor([action]))
            entropy = dist.entropy()

            policy_loss = -log_prob * advantage
            value_loss = F.mse_loss(value, torch.tensor(td_target))

            total_policy_loss = total_policy_loss + policy_loss
            total_value_loss = total_value_loss + value_loss
            total_entropy = total_entropy + entropy
            count += 1

        if count == 0:
            return {}

        loss = (total_policy_loss / count +
                self.value_loss_coef * total_value_loss / count -
                self.entropy_coef * total_entropy / count)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        return {
            "policy_loss": float((total_policy_loss / count).item()),
            "value_loss": float((total_value_loss / count).item()),
            "entropy": float((total_entropy / count).item()),
            "total_loss": float(loss.item()),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.net.load_state_dict(state["net"])
        self.optimizer.load_state_dict(state["optimizer"])


class A2CAgent(BaseAgent):
    """Advantage Actor-Critic with n-step returns and GAE.

    Args:
        actor_critic_net: GraphActorCriticNetwork.
        optimizer: PyTorch optimizer.
        gamma: Discount factor.
        n_steps: Steps per rollout.
        entropy_coef: Entropy bonus coefficient.
        value_loss_coef: Value loss scaling.
        gae_lambda: GAE lambda parameter.
        grad_clip_norm: Gradient clipping norm.
    """

    def __init__(
        self,
        actor_critic_net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        n_steps: int = 5,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        gae_lambda: float = 0.95,
        grad_clip_norm: Optional[float] = 1.0,
    ) -> None:
        self.net = actor_critic_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.gae_lambda = gae_lambda
        self.grad_clip_norm = grad_clip_norm

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
            logits, _ = self.net(node_features, edge_index)
        logits = logits.squeeze(0)

        if action_mask is not None:
            m = action_mask.bool()
            m_size = min(m.shape[0], logits.shape[-1])
            m = m[:m_size]
            logits = logits[:m_size]
        else:
            m = torch.ones(logits.shape[-1], dtype=torch.bool)

        if deterministic:
            masked = logits.clone()
            masked[~m] = float("-inf")
            return int(masked.argmax().item())

        dist = MaskedCategoricalPolicy(logits.unsqueeze(0), m.unsqueeze(0))
        return int(dist.sample(generator=generator).item())

    def update(self, rollout: Dict[str, Any]) -> Dict[str, float]:
        """Update from a rollout.

        Args:
            rollout: Dict with 'obs_list', 'actions', 'rewards', 'dones', 'values'.

        Returns:
            Loss dict.
        """
        from tgraphx.rl.networks.policy import MaskedCategoricalPolicy

        obs_list = rollout.get("obs_list", [])
        actions = rollout.get("actions", [])
        rewards = rollout.get("rewards", [])
        dones = rollout.get("dones", [])
        values = rollout.get("values", [])

        if not obs_list:
            return {}

        advantages = _compute_gae(rewards, values, dones, self.gamma, self.gae_lambda)
        returns = advantages + torch.tensor(values[:len(advantages)])

        total_policy_loss = torch.tensor(0.0)
        total_value_loss = torch.tensor(0.0)
        total_entropy = torch.tensor(0.0)

        for i, (obs, action) in enumerate(zip(obs_list, actions)):
            node_features = obs.get("node_features", torch.zeros(1, 1))
            edge_index = obs.get("edge_index", torch.zeros((2, 0), dtype=torch.long))
            action_mask = obs.get("action_mask")

            logits, value = self.net(node_features, edge_index)
            logits = logits.squeeze(0)
            value = value.squeeze()

            if action_mask is not None:
                m = action_mask.bool()
                m_size = min(m.shape[0], logits.shape[-1])
                m = m[:m_size]
                logits = logits[:m_size]
            else:
                m = torch.ones(logits.shape[-1], dtype=torch.bool)

            dist = MaskedCategoricalPolicy(logits.unsqueeze(0), m.unsqueeze(0))
            log_prob = dist.log_prob(torch.tensor([action]))
            entropy = dist.entropy()

            # Normalize advantages
            adv = advantages[i]
            if len(advantages) > 1:
                adv = (adv - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -log_prob * adv
            value_loss = F.mse_loss(value, returns[i])

            total_policy_loss = total_policy_loss + policy_loss
            total_value_loss = total_value_loss + value_loss
            total_entropy = total_entropy + entropy

        n = max(len(obs_list), 1)
        loss = (total_policy_loss / n +
                self.value_loss_coef * total_value_loss / n -
                self.entropy_coef * total_entropy / n)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        return {
            "policy_loss": float((total_policy_loss / n).item()),
            "value_loss": float((total_value_loss / n).item()),
            "entropy": float((total_entropy / n).item()),
            "total_loss": float(loss.item()),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.net.load_state_dict(state["net"])
        self.optimizer.load_state_dict(state["optimizer"])

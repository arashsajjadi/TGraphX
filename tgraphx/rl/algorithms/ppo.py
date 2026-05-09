"""Proximal Policy Optimization for graph RL.

PPO Mathematics:
    Clipped objective:
        L_clip = -mean(min(ρ_t * A_t, clip(ρ_t, 1-ε, 1+ε) * A_t))
        ρ_t = π_θ(a_t|s_t) / π_old(a_t|s_t)

    Value loss: L_V = (V(s_t) - R_t)^2
    Total: L = L_clip - entropy_coef * H(π) + value_coef * L_V

    Approximate KL: approx_kl = mean(old_logprob - new_logprob)
    Clip fraction: fraction of timesteps where ratio was clipped.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .base import BaseAgent
from .actor_critic import _compute_gae

__all__ = ["PPOAgent"]


class PPOAgent(BaseAgent):
    """PPO agent.

    Args:
        actor_critic: GraphActorCriticNetwork.
        optimizer: PyTorch optimizer.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_eps: Clip parameter epsilon.
        entropy_coef: Entropy regularization coefficient.
        value_loss_coef: Value loss weight.
        n_epochs: Number of optimization epochs per rollout.
        mini_batch_size: Mini-batch size.
        grad_clip_norm: Gradient clipping norm.
    """

    def __init__(
        self,
        actor_critic: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        n_epochs: int = 4,
        mini_batch_size: int = 16,
        grad_clip_norm: Optional[float] = 0.5,
    ) -> None:
        self.net = actor_critic
        self.optimizer = optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.grad_clip_norm = grad_clip_norm

    def collect_rollout(
        self,
        env: Any,
        n_steps: int,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Any]:
        """Collect n_steps of experience.

        Args:
            env: GraphEnv instance.
            n_steps: Number of steps to collect.
            generator: Optional torch.Generator.

        Returns:
            Rollout dict with obs_list, actions, rewards, dones, values, log_probs.
        """
        from tgraphx.rl.networks.policy import MaskedCategoricalPolicy

        obs = env.reset()
        obs_list, actions, rewards, dones, values, log_probs = [], [], [], [], [], []

        for _ in range(n_steps):
            node_features = obs.get("node_features", torch.zeros(1, 1))
            edge_index = obs.get("edge_index", torch.zeros((2, 0), dtype=torch.long))
            action_mask = obs.get("action_mask")

            with torch.no_grad():
                logits, value = self.net(node_features, edge_index)
            logits = logits.squeeze(0)
            value = value.squeeze().item()

            if action_mask is not None:
                m = action_mask.bool()
                m_size = min(m.shape[0], logits.shape[-1])
                m = m[:m_size]
                logits_m = logits[:m_size]
            else:
                m = torch.ones(logits.shape[-1], dtype=torch.bool)
                logits_m = logits

            dist = MaskedCategoricalPolicy(logits_m.unsqueeze(0), m.unsqueeze(0))
            action = int(dist.sample(generator=generator).item())
            log_prob = float(dist.log_prob(torch.tensor([action])).item())

            next_obs, reward, done, truncated, _ = env.step(action)

            obs_list.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            dones.append(done or truncated)
            values.append(float(value))
            log_probs.append(float(log_prob))

            obs = next_obs
            if done or truncated:
                obs = env.reset()

        return {
            "obs_list": obs_list,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "log_probs": log_probs,
        }

    def update(self, rollout: Dict[str, Any]) -> Dict[str, float]:
        """Update policy and value function from rollout.

        Args:
            rollout: Output from collect_rollout.

        Returns:
            Loss dict including approx_kl, clip_fraction.
        """
        from tgraphx.rl.networks.policy import MaskedCategoricalPolicy

        obs_list = rollout["obs_list"]
        actions = rollout["actions"]
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        values = rollout["values"]
        old_log_probs = rollout["log_probs"]

        if not obs_list:
            return {}

        # Compute GAE advantages
        advantages = _compute_gae(rewards, values, dones, self.gamma, self.gae_lambda)
        returns = advantages + torch.tensor(values[:len(advantages)], dtype=torch.float)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        total_entropy = 0.0
        count = 0

        for _ in range(self.n_epochs):
            for i in range(len(obs_list)):
                obs = obs_list[i]
                action = actions[i]
                old_lp = old_log_probs[i]
                adv = advantages[i]
                ret = returns[i]

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
                new_log_prob = dist.log_prob(torch.tensor([action]))
                entropy = dist.entropy()

                # Ratio ρ = π_new / π_old
                ratio = torch.exp(new_log_prob - old_lp)

                # Clipped objective
                clip_adv = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                policy_loss = -torch.min(ratio * adv, clip_adv)

                # Value loss
                value_loss = F.mse_loss(value, ret)

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = float((old_lp - new_log_prob.detach()).mean().item())
                    clip_frac = float((ratio.detach().abs() > 1 + self.clip_eps).float().mean().item())

                total_loss += float(loss.item())
                total_approx_kl += approx_kl
                total_clip_frac += clip_frac
                total_entropy += float(entropy.item())
                count += 1

        n = max(count, 1)
        return {
            "total_loss": total_loss / n,
            "approx_kl": total_approx_kl / n,
            "clip_fraction": total_clip_frac / n,
            "entropy": total_entropy / n,
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

    def state_dict(self) -> Dict[str, Any]:
        return {"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.net.load_state_dict(state["net"])
        self.optimizer.load_state_dict(state["optimizer"])

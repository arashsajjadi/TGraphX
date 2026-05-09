"""Continuous-action graph RL algorithms: DDPG, TD3, SAC.

These algorithms operate on continuous graph-action embeddings — a fixed-dim
float vector that a decoder maps to graph edits. They are NOT for discrete
graph actions; for those, use DQN/PPO.

Mathematics:

DDPG:
    Actor: a_t = mu_theta(s_t)
    Critic: Q_phi(s_t, a_t)
    Target: y = r + gamma Q_phi'(s_{t+1}, mu_theta'(s_{t+1}))
    Critic loss: L_Q = mean((Q(s,a) - y)^2)
    Actor loss: L_pi = -mean(Q(s, mu_theta(s)))
    Soft update: theta' <- tau*theta + (1-tau)*theta'

TD3 additions:
    Twin critics: Q_phi1, Q_phi2
    Target policy smoothing: a' = clip(mu_theta'(s') + clip(eps,-c,c), low, high)
    Target: y = r + gamma min(Q_phi1'(s',a'), Q_phi2'(s',a'))
    Delayed actor update every policy_delay critic steps

SAC:
    Stochastic actor: (a, log_pi) = actor(s)  [reparameterization trick]
    Twin critics
    Entropy-regularized target: y = r + gamma [min Q(s',a') - alpha log pi(a'|s')]
    Actor loss: mean(alpha log pi(a|s) - min Q(s,a))
    Alpha loss (auto entropy): -log_alpha * (log_pi + target_entropy).detach()

Stability: Experimental (v0.7.0+)
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAgent
from .replay_buffer import ReplayBuffer
from tgraphx.rl.networks.policy import _gnn_layer

__all__ = [
    "OUNoise",
    "GaussianNoise",
    "ContinuousGraphActor",
    "StochasticGraphActor",
    "ContinuousGraphCritic",
    "TwinContinuousGraphCritic",
    "soft_update",
    "GraphDDPGAgent",
    "GraphDelayedDDPGAgent",
    "GraphTD3Agent",
    "GraphSACAgent",
]


# ---------------------------------------------------------------------------
# Noise processes
# ---------------------------------------------------------------------------

class OUNoise:
    """Ornstein-Uhlenbeck action noise for DDPG.

    dX_t = theta*(mu - X_t)*dt + sigma*dW_t

    Args:
        action_dim: Dimensionality of the action vector.
        mu: Long-run mean.
        theta: Mean-reversion rate.
        sigma: Noise scale.
        seed: Optional RNG seed.
    """

    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self._state = torch.zeros(action_dim)
        self._gen: Optional[torch.Generator] = None
        if seed is not None:
            self._gen = torch.Generator()
            self._gen.manual_seed(seed)

    def reset(self) -> None:
        """Reset noise state to mu."""
        self._state = torch.full((self.action_dim,), self.mu)

    def sample(self) -> torch.Tensor:
        """Sample OU noise.

        Returns:
            FloatTensor [action_dim].
        """
        dx = self.theta * (self.mu - self._state) + self.sigma * torch.randn(
            self.action_dim, generator=self._gen
        )
        self._state = self._state + dx
        return self._state.clone()


class GaussianNoise:
    """Gaussian action noise for TD3/SAC.

    Args:
        action_dim: Dimensionality of action vector.
        sigma: Standard deviation.
        clip: Optional clipping value (|noise| <= clip).
    """

    def __init__(
        self,
        action_dim: int,
        sigma: float = 0.1,
        clip: Optional[float] = None,
    ) -> None:
        self.action_dim = action_dim
        self.sigma = sigma
        self.clip = clip

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample Gaussian noise.

        Returns:
            FloatTensor [action_dim].
        """
        noise = torch.randn(self.action_dim, generator=generator) * self.sigma
        if self.clip is not None:
            noise = noise.clamp(-self.clip, self.clip)
        return noise


# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------

class ContinuousGraphActor(nn.Module):
    """Deterministic actor for DDPG/TD3: state -> action in action_dim.

    Architecture: GNN encoder -> global mean pool -> MLP -> tanh-scaled action.

    Args:
        node_in_dim: Node feature input dimension.
        edge_in_dim: Edge feature dimension (currently unused in simple GNN).
        hidden_dim: Hidden layer dimension.
        action_dim: Output action dimension.
        action_scale: Multiplier for tanh output (default 1.0).
        action_bias: Bias added after scaling (default 0.0).
        num_gnn_layers: Number of GNN layers.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int = 0,
        hidden_dim: int = 64,
        action_dim: int = 8,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
        num_gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.action_bias = action_bias

        layers = []
        in_d = node_in_dim
        for _ in range(num_gnn_layers):
            layers.append(_gnn_layer(in_d, hidden_dim))
            in_d = hidden_dim
        self.gnn_layers = nn.ModuleList(layers)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: [N, node_in_dim].
            edge_index: [2, E].

        Returns:
            FloatTensor [1, action_dim] in range [-action_scale, action_scale].
        """
        if node_features.dim() != 2:
            raise ValueError(
                f"ContinuousGraphActor expects [N, F] node features but got {list(node_features.shape)}."
            )
        x = node_features
        n = x.shape[0]
        for layer in self.gnn_layers:
            x = layer(x, edge_index, n)
        pooled = x.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        raw = self.mlp(pooled)                 # [1, action_dim]
        return torch.tanh(raw) * self.action_scale + self.action_bias


class StochasticGraphActor(nn.Module):
    """Stochastic actor for SAC using reparameterization + tanh squashing.

    Architecture: GNN encoder -> global mean pool -> MLP -> mean/log_std -> sample.

    Log-prob correction for tanh squashing:
        log_prob -= sum(log(1 - tanh(x)^2 + 1e-6), dim=-1)

    Args:
        node_in_dim: Node feature input dimension.
        edge_in_dim: Edge feature dimension (unused in simple GNN).
        hidden_dim: Hidden layer dimension.
        action_dim: Output action dimension.
        action_scale: Multiplier for tanh output.
        action_bias: Bias added after scaling.
        num_gnn_layers: Number of GNN layers.
        log_std_min: Minimum log std (clamped).
        log_std_max: Maximum log std (clamped).
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int = 0,
        hidden_dim: int = 64,
        action_dim: int = 8,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
        num_gnn_layers: int = 2,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []
        in_d = node_in_dim
        for _ in range(num_gnn_layers):
            layers.append(_gnn_layer(in_d, hidden_dim))
            in_d = hidden_dim
        self.gnn_layers = nn.ModuleList(layers)

        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        for m in [self.shared_mlp, self.mean_head, self.log_std_head]:
            for layer in (m if isinstance(m, nn.Sequential) else [m]):
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            node_features: [N, node_in_dim].
            edge_index: [2, E].
            generator: Optional RNG generator.

        Returns:
            Tuple of:
                action: FloatTensor [1, action_dim] (tanh-squashed).
                log_prob: FloatTensor [1] (log probability with tanh correction).
        """
        if node_features.dim() != 2:
            raise ValueError(
                f"StochasticGraphActor expects [N, F] node features but got {list(node_features.shape)}."
            )
        x = node_features
        n = x.shape[0]
        for layer in self.gnn_layers:
            x = layer(x, edge_index, n)
        pooled = x.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        shared = self.shared_mlp(pooled)       # [1, hidden_dim]

        mean = self.mean_head(shared)          # [1, action_dim]
        log_std = self.log_std_head(shared)    # [1, action_dim]
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()

        # Reparameterization trick
        eps = torch.randn_like(mean, generator=generator) if generator is not None else torch.randn_like(mean)
        x_t = mean + eps * std                 # pre-squash

        # Tanh squash
        action_raw = torch.tanh(x_t)
        action = action_raw * self.action_scale + self.action_bias

        # Log prob with tanh correction
        # log pi(a|s) = log N(x_t; mean, std) - sum log(1 - tanh^2(x_t) + eps)
        log_prob_gaussian = -0.5 * ((x_t - mean) / (std + 1e-8)).pow(2) - log_std - 0.5 * 1.8378770664093455
        log_prob = log_prob_gaussian.sum(dim=-1)  # [1]
        # tanh squash correction
        log_prob -= torch.log(1.0 - action_raw.pow(2) + 1e-6).sum(dim=-1)  # [1]

        return action, log_prob


class ContinuousGraphCritic(nn.Module):
    """Q(s, a) critic: [state_repr, action] -> scalar.

    Architecture: GNN encoder -> global mean pool -> concat action -> MLP -> scalar.

    Args:
        node_in_dim: Node feature input dimension.
        edge_in_dim: Edge feature dimension (unused).
        action_dim: Action vector dimension.
        hidden_dim: Hidden layer dimension.
        num_gnn_layers: Number of GNN layers.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int = 0,
        action_dim: int = 8,
        hidden_dim: int = 64,
        num_gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim

        layers = []
        in_d = node_in_dim
        for _ in range(num_gnn_layers):
            layers.append(_gnn_layer(in_d, hidden_dim))
            in_d = hidden_dim
        self.gnn_layers = nn.ModuleList(layers)

        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.q_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: [N, node_in_dim].
            edge_index: [2, E].
            action: [1, action_dim] or [action_dim].

        Returns:
            FloatTensor [1, 1].
        """
        if node_features.dim() != 2:
            raise ValueError(
                f"ContinuousGraphCritic expects [N, F] node features but got {list(node_features.shape)}."
            )
        x = node_features
        n = x.shape[0]
        for layer in self.gnn_layers:
            x = layer(x, edge_index, n)
        pooled = x.mean(dim=0, keepdim=True)  # [1, hidden_dim]

        if action.dim() == 1:
            action = action.unsqueeze(0)       # [1, action_dim]
        sa = torch.cat([pooled, action], dim=-1)  # [1, hidden_dim + action_dim]
        return self.q_head(sa)                    # [1, 1]


class TwinContinuousGraphCritic(nn.Module):
    """Twin critics for TD3/SAC (two independent ContinuousGraphCritic).

    Args:
        node_in_dim: Node feature input dimension.
        edge_in_dim: Edge feature dimension.
        action_dim: Action vector dimension.
        hidden_dim: Hidden layer dimension.
        num_gnn_layers: Number of GNN layers.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int = 0,
        action_dim: int = 8,
        hidden_dim: int = 64,
        num_gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.q1 = ContinuousGraphCritic(
            node_in_dim, edge_in_dim, action_dim, hidden_dim, num_gnn_layers
        )
        self.q2 = ContinuousGraphCritic(
            node_in_dim, edge_in_dim, action_dim, hidden_dim, num_gnn_layers
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both Q values.

        Returns:
            (q1, q2): each FloatTensor [1, 1].
        """
        return self.q1(node_features, edge_index, action), self.q2(node_features, edge_index, action)

    def forward_min(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Return min(q1, q2).

        Returns:
            FloatTensor [1, 1].
        """
        q1, q2 = self.forward(node_features, edge_index, action)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# Soft update utility
# ---------------------------------------------------------------------------

def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Polyak averaging: theta_target <- tau*theta_source + (1-tau)*theta_target.

    Args:
        source: Source network.
        target: Target network (updated in-place).
        tau: Interpolation factor in [0, 1]. tau=1 copies source to target exactly.
    """
    with torch.no_grad():
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.copy_(tau * src_p.data + (1.0 - tau) * tgt_p.data)


# ---------------------------------------------------------------------------
# Helper to extract tensors from obs dict
# ---------------------------------------------------------------------------

def _get_obs_tensors(obs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    nf = obs.get("node_features", torch.zeros(1, 1))
    ei = obs.get("edge_index", torch.zeros((2, 0), dtype=torch.long))
    return nf, ei


# ---------------------------------------------------------------------------
# DDPG
# ---------------------------------------------------------------------------

class GraphDDPGAgent(BaseAgent):
    """Deep Deterministic Policy Gradient for continuous graph-action embeddings.

    Args:
        actor: ContinuousGraphActor (deterministic).
        critic: ContinuousGraphCritic.
        target_actor: Copy of actor for target computation.
        target_critic: Copy of critic for target computation.
        actor_optimizer: Optimizer for actor.
        critic_optimizer: Optimizer for critic.
        gamma: Discount factor.
        tau: Soft update coefficient.
        noise: OUNoise instance for exploration.
        replay_buffer: ReplayBuffer.
        batch_size: Batch size for updates.
    """

    def __init__(
        self,
        actor: ContinuousGraphActor,
        critic: ContinuousGraphCritic,
        target_actor: ContinuousGraphActor,
        target_critic: ContinuousGraphCritic,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise: Optional[OUNoise] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        batch_size: int = 64,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.buffer = replay_buffer if replay_buffer is not None else ReplayBuffer(10000)
        self.batch_size = batch_size

        # Ensure targets start equal to online networks
        self.target_actor.load_state_dict(actor.state_dict())
        self.target_critic.load_state_dict(critic.state_dict())
        self.target_actor.eval()
        self.target_critic.eval()

    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Select action.

        Args:
            obs: Observation dict.
            deterministic: If True, no exploration noise.
            generator: Unused (OUNoise has its own state).

        Returns:
            action tensor [action_dim] or numpy array.
        """
        nf, ei = _get_obs_tensors(obs)
        with torch.no_grad():
            action = self.actor(nf, ei).squeeze(0)  # [action_dim]

        if not deterministic and self.noise is not None:
            noise = self.noise.sample().to(action.device)
            action = action + noise
            # Clip to valid range
            action = action.clamp(
                -self.actor.action_scale, self.actor.action_scale
            )

        return action

    def update(self, batch: Optional[List] = None) -> Dict[str, float]:
        """Update critic and actor from replay buffer.

        Args:
            batch: Optional pre-sampled batch (list of (obs, action, reward, next_obs, done)).

        Returns:
            Dict with critic_loss, actor_loss, q_target_mean.
        """
        if batch is None:
            if not self.buffer.is_ready(self.batch_size):
                return {}
            batch = self.buffer.sample(self.batch_size)

        if not batch:
            return {}

        # Critic update
        critic_loss_total = torch.tensor(0.0)
        q_target_sum = 0.0

        for obs, action, reward, next_obs, done in batch:
            nf, ei = _get_obs_tensors(obs)
            next_nf, next_ei = _get_obs_tensors(next_obs)

            if isinstance(action, (int, float)):
                action_t = torch.tensor([action], dtype=torch.float)
            else:
                action_t = torch.as_tensor(action, dtype=torch.float)
            if action_t.dim() == 0:
                action_t = action_t.unsqueeze(0)

            with torch.no_grad():
                next_action = self.target_actor(next_nf, next_ei)  # [1, action_dim]
                q_next = self.target_critic(next_nf, next_ei, next_action)  # [1, 1]
                q_target = reward + self.gamma * q_next.squeeze() * (1.0 - float(done))
            q_target_sum += float(q_target.item())

            q_pred = self.critic(nf, ei, action_t)  # [1, 1]
            loss = F.mse_loss(q_pred.squeeze(), q_target.detach())
            critic_loss_total = critic_loss_total + loss

        critic_loss_total = critic_loss_total / len(batch)
        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss_total = torch.tensor(0.0)
        for obs, _, _, _, _ in batch:
            nf, ei = _get_obs_tensors(obs)
            action_pred = self.actor(nf, ei)         # [1, action_dim]
            q_val = self.critic(nf, ei, action_pred)  # [1, 1]
            actor_loss_total = actor_loss_total + (-q_val.squeeze())

        actor_loss_total = actor_loss_total / len(batch)
        self.actor_optimizer.zero_grad()
        actor_loss_total.backward()
        self.actor_optimizer.step()

        self.update_targets()

        return {
            "critic_loss": float(critic_loss_total.item()),
            "actor_loss": float(actor_loss_total.item()),
            "q_target_mean": q_target_sum / len(batch),
        }

    def update_targets(self) -> None:
        """Soft update both actor and critic target networks."""
        soft_update(self.actor, self.target_actor, self.tau)
        soft_update(self.critic, self.target_critic, self.tau)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_actor.load_state_dict(state["target_actor"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])


class GraphDelayedDDPGAgent(GraphDDPGAgent):
    """Delayed DDPG: actor updated every policy_delay critic steps.

    Extra arg:
        policy_delay: Actor update frequency (default 2).
    """

    def __init__(
        self,
        actor: ContinuousGraphActor,
        critic: ContinuousGraphCritic,
        target_actor: ContinuousGraphActor,
        target_critic: ContinuousGraphCritic,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise: Optional[OUNoise] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        batch_size: int = 64,
        policy_delay: int = 2,
    ) -> None:
        super().__init__(
            actor, critic, target_actor, target_critic,
            actor_optimizer, critic_optimizer,
            gamma=gamma, tau=tau, noise=noise,
            replay_buffer=replay_buffer, batch_size=batch_size,
        )
        self.policy_delay = policy_delay
        self._update_step = 0

    def update(self, batch: Optional[List] = None) -> Dict[str, float]:
        if batch is None:
            if not self.buffer.is_ready(self.batch_size):
                return {}
            batch = self.buffer.sample(self.batch_size)

        if not batch:
            return {}

        # Critic update
        critic_loss_total = torch.tensor(0.0)
        q_target_sum = 0.0

        for obs, action, reward, next_obs, done in batch:
            nf, ei = _get_obs_tensors(obs)
            next_nf, next_ei = _get_obs_tensors(next_obs)

            if isinstance(action, (int, float)):
                action_t = torch.tensor([action], dtype=torch.float)
            else:
                action_t = torch.as_tensor(action, dtype=torch.float)
            if action_t.dim() == 0:
                action_t = action_t.unsqueeze(0)

            with torch.no_grad():
                next_action = self.target_actor(next_nf, next_ei)
                q_next = self.target_critic(next_nf, next_ei, next_action)
                q_target = reward + self.gamma * q_next.squeeze() * (1.0 - float(done))
            q_target_sum += float(q_target.item())

            q_pred = self.critic(nf, ei, action_t)
            loss = F.mse_loss(q_pred.squeeze(), q_target.detach())
            critic_loss_total = critic_loss_total + loss

        critic_loss_total = critic_loss_total / len(batch)
        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        self.critic_optimizer.step()

        self._update_step += 1
        actor_loss_val = None

        if self._update_step % self.policy_delay == 0:
            actor_loss_total = torch.tensor(0.0)
            for obs, _, _, _, _ in batch:
                nf, ei = _get_obs_tensors(obs)
                action_pred = self.actor(nf, ei)
                q_val = self.critic(nf, ei, action_pred)
                actor_loss_total = actor_loss_total + (-q_val.squeeze())

            actor_loss_total = actor_loss_total / len(batch)
            self.actor_optimizer.zero_grad()
            actor_loss_total.backward()
            self.actor_optimizer.step()
            actor_loss_val = float(actor_loss_total.item())

            soft_update(self.actor, self.target_actor, self.tau)
            soft_update(self.critic, self.target_critic, self.tau)

        result: Dict[str, Any] = {
            "critic_loss": float(critic_loss_total.item()),
            "actor_loss": actor_loss_val,
            "q_target_mean": q_target_sum / len(batch),
        }
        return result


# ---------------------------------------------------------------------------
# TD3
# ---------------------------------------------------------------------------

class GraphTD3Agent(BaseAgent):
    """Twin Delayed DDPG (TD3).

    Args:
        actor: ContinuousGraphActor.
        twin_critic: TwinContinuousGraphCritic.
        target_actor: Copy of actor.
        target_twin_critic: Copy of twin_critic.
        actor_optimizer: Optimizer for actor.
        critic_optimizer: Optimizer for twin_critic.
        gamma: Discount factor.
        tau: Soft update coefficient.
        policy_delay: Actor update frequency.
        target_noise_std: Std dev of target policy smoothing noise.
        target_noise_clip: Clip value for target policy noise.
        action_low: Minimum action value.
        action_high: Maximum action value.
        replay_buffer: ReplayBuffer.
        batch_size: Batch size.
    """

    def __init__(
        self,
        actor: ContinuousGraphActor,
        twin_critic: TwinContinuousGraphCritic,
        target_actor: ContinuousGraphActor,
        target_twin_critic: TwinContinuousGraphCritic,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        target_noise_std: float = 0.2,
        target_noise_clip: float = 0.5,
        action_low: float = -1.0,
        action_high: float = 1.0,
        replay_buffer: Optional[ReplayBuffer] = None,
        batch_size: int = 64,
    ) -> None:
        self.actor = actor
        self.twin_critic = twin_critic
        self.target_actor = target_actor
        self.target_twin_critic = target_twin_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.action_low = action_low
        self.action_high = action_high
        self.buffer = replay_buffer if replay_buffer is not None else ReplayBuffer(10000)
        self.batch_size = batch_size
        self._update_step = 0

        self.target_actor.load_state_dict(actor.state_dict())
        self.target_twin_critic.load_state_dict(twin_critic.state_dict())
        self.target_actor.eval()
        self.target_twin_critic.eval()

    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Select action.

        Returns:
            FloatTensor [action_dim].
        """
        nf, ei = _get_obs_tensors(obs)
        with torch.no_grad():
            action = self.actor(nf, ei).squeeze(0)  # [action_dim]

        if not deterministic:
            noise = torch.randn_like(action, generator=generator) * self.target_noise_std
            action = (action + noise).clamp(self.action_low, self.action_high)

        return action

    def update(
        self,
        step: int,
        batch: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Update critics and optionally actor.

        Args:
            step: Current global step count.
            batch: Optional pre-sampled batch.

        Returns:
            Dict with critic1_loss, critic2_loss, actor_loss, q_target_mean, policy_delay.
        """
        if batch is None:
            if not self.buffer.is_ready(self.batch_size):
                return {}
            batch = self.buffer.sample(self.batch_size)

        if not batch:
            return {}

        # Critic update — BOTH critics get gradients
        c1_loss_total = torch.tensor(0.0)
        c2_loss_total = torch.tensor(0.0)
        q_target_sum = 0.0

        for obs, action, reward, next_obs, done in batch:
            nf, ei = _get_obs_tensors(obs)
            next_nf, next_ei = _get_obs_tensors(next_obs)

            if isinstance(action, (int, float)):
                action_t = torch.tensor([float(action)], dtype=torch.float)
            else:
                action_t = torch.as_tensor(action, dtype=torch.float)
            if action_t.dim() == 0:
                action_t = action_t.unsqueeze(0)

            with torch.no_grad():
                # Target policy with smoothing noise
                next_action = self.target_actor(next_nf, next_ei)  # [1, action_dim]
                noise = torch.randn_like(next_action) * self.target_noise_std
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_action = (next_action + noise).clamp(self.action_low, self.action_high)

                # Target: min of twin critics
                q1_next, q2_next = self.target_twin_critic(next_nf, next_ei, next_action)
                q_min_next = torch.min(q1_next, q2_next).squeeze()
                q_target = reward + self.gamma * q_min_next * (1.0 - float(done))

            q_target_sum += float(q_target.item())

            q1_pred, q2_pred = self.twin_critic(nf, ei, action_t)
            loss1 = F.mse_loss(q1_pred.squeeze(), q_target.detach())
            loss2 = F.mse_loss(q2_pred.squeeze(), q_target.detach())
            c1_loss_total = c1_loss_total + loss1
            c2_loss_total = c2_loss_total + loss2

        critic_loss = (c1_loss_total + c2_loss_total) / len(batch)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self._update_step += 1
        actor_loss_val: Optional[float] = None

        if self._update_step % self.policy_delay == 0:
            actor_loss_total = torch.tensor(0.0)
            for obs, _, _, _, _ in batch:
                nf, ei = _get_obs_tensors(obs)
                action_pred = self.actor(nf, ei)
                # Use only Q1 for actor gradient
                q1 = self.twin_critic.q1(nf, ei, action_pred)
                actor_loss_total = actor_loss_total + (-q1.squeeze())

            actor_loss_total = actor_loss_total / len(batch)
            self.actor_optimizer.zero_grad()
            actor_loss_total.backward()
            self.actor_optimizer.step()
            actor_loss_val = float(actor_loss_total.item())

            soft_update(self.actor, self.target_actor, self.tau)
            soft_update(self.twin_critic, self.target_twin_critic, self.tau)

        return {
            "critic1_loss": float(c1_loss_total.item() / len(batch)),
            "critic2_loss": float(c2_loss_total.item() / len(batch)),
            "actor_loss": actor_loss_val,
            "q_target_mean": q_target_sum / len(batch),
            "policy_delay": self.policy_delay,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "twin_critic": self.twin_critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_twin_critic": self.target_twin_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_step": self._update_step,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.twin_critic.load_state_dict(state["twin_critic"])
        self.target_actor.load_state_dict(state["target_actor"])
        self.target_twin_critic.load_state_dict(state["target_twin_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self._update_step = state.get("update_step", 0)


# ---------------------------------------------------------------------------
# SAC
# ---------------------------------------------------------------------------

class GraphSACAgent(BaseAgent):
    """Soft Actor-Critic.

    Args:
        actor: StochasticGraphActor.
        twin_critic: TwinContinuousGraphCritic.
        target_twin_critic: Copy of twin_critic.
        actor_optimizer: Optimizer for actor.
        critic_optimizer: Optimizer for twin_critic.
        gamma: Discount factor.
        tau: Soft update coefficient.
        alpha: Entropy regularization coefficient.
        auto_entropy: If True, auto-tune alpha.
        target_entropy: Target entropy (default: -action_dim).
        alpha_optimizer: Optimizer for log_alpha (required if auto_entropy=True).
        replay_buffer: ReplayBuffer.
        batch_size: Batch size.
    """

    def __init__(
        self,
        actor: StochasticGraphActor,
        twin_critic: TwinContinuousGraphCritic,
        target_twin_critic: TwinContinuousGraphCritic,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy: bool = True,
        target_entropy: Optional[float] = None,
        alpha_optimizer: Optional[torch.optim.Optimizer] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        batch_size: int = 64,
    ) -> None:
        self.actor = actor
        self.twin_critic = twin_critic
        self.target_twin_critic = target_twin_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.tau = tau
        self.auto_entropy = auto_entropy
        self.buffer = replay_buffer if replay_buffer is not None else ReplayBuffer(10000)
        self.batch_size = batch_size

        action_dim = actor.action_dim
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)

        if auto_entropy:
            self.log_alpha = torch.tensor(
                float(alpha) if alpha > 0 else 0.0, requires_grad=True
            ).log() if alpha > 0 else torch.tensor(0.0, requires_grad=True)
            # log_alpha = log(alpha) so alpha = exp(log_alpha)
            self.log_alpha = torch.tensor(
                float(torch.tensor(alpha).log().item()), requires_grad=True
            )
            if alpha_optimizer is None:
                self.alpha_optimizer: Optional[torch.optim.Optimizer] = torch.optim.Adam(
                    [self.log_alpha], lr=3e-4
                )
            else:
                self.alpha_optimizer = alpha_optimizer
        else:
            self.log_alpha = torch.tensor(float(torch.tensor(alpha).log().item()))
            self.alpha_optimizer = None

        self.target_twin_critic.load_state_dict(twin_critic.state_dict())
        self.target_twin_critic.eval()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Select action.

        Returns:
            FloatTensor [action_dim].
        """
        nf, ei = _get_obs_tensors(obs)
        if deterministic:
            with torch.no_grad():
                action, _ = self.actor(nf, ei, generator=generator)
        else:
            with torch.no_grad():
                action, _ = self.actor(nf, ei, generator=generator)
        return action.squeeze(0)

    def update(self, batch: Optional[List] = None) -> Dict[str, Any]:
        """Update all networks.

        Returns:
            Dict with: critic1_loss, critic2_loss, actor_loss, alpha_loss (if auto),
                       alpha, entropy, log_prob_mean, q_target_mean.
        """
        if batch is None:
            if not self.buffer.is_ready(self.batch_size):
                return {}
            batch = self.buffer.sample(self.batch_size)

        if not batch:
            return {}

        alpha_val = self.alpha.item()

        # Critic update
        c1_loss_total = torch.tensor(0.0)
        c2_loss_total = torch.tensor(0.0)
        q_target_sum = 0.0

        for obs, action, reward, next_obs, done in batch:
            nf, ei = _get_obs_tensors(obs)
            next_nf, next_ei = _get_obs_tensors(next_obs)

            if isinstance(action, (int, float)):
                action_t = torch.tensor([float(action)], dtype=torch.float)
            else:
                action_t = torch.as_tensor(action, dtype=torch.float)
            if action_t.dim() == 0:
                action_t = action_t.unsqueeze(0)

            with torch.no_grad():
                next_action, next_log_prob = self.actor(next_nf, next_ei)
                q1_next, q2_next = self.target_twin_critic(next_nf, next_ei, next_action)
                q_min_next = torch.min(q1_next, q2_next).squeeze()
                # Entropy-regularized target
                q_target = reward + self.gamma * (q_min_next - alpha_val * next_log_prob.squeeze()) * (1.0 - float(done))

            q_target_sum += float(q_target.item())

            q1_pred, q2_pred = self.twin_critic(nf, ei, action_t)
            loss1 = F.mse_loss(q1_pred.squeeze(), q_target.detach())
            loss2 = F.mse_loss(q2_pred.squeeze(), q_target.detach())
            c1_loss_total = c1_loss_total + loss1
            c2_loss_total = c2_loss_total + loss2

        critic_loss = (c1_loss_total + c2_loss_total) / len(batch)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss_total = torch.tensor(0.0)
        log_prob_sum = 0.0

        for obs, _, _, _, _ in batch:
            nf, ei = _get_obs_tensors(obs)
            action_pred, log_prob = self.actor(nf, ei)  # reparameterization
            q1, q2 = self.twin_critic(nf, ei, action_pred)
            q_min = torch.min(q1, q2).squeeze()
            # Actor loss: alpha * log_prob - min Q
            actor_loss_total = actor_loss_total + (self.alpha.detach() * log_prob.squeeze() - q_min)
            log_prob_sum += float(log_prob.item())

        actor_loss_total = actor_loss_total / len(batch)
        self.actor_optimizer.zero_grad()
        actor_loss_total.backward()
        self.actor_optimizer.step()

        log_prob_mean = log_prob_sum / len(batch)

        # Alpha (entropy) update
        alpha_loss_val: Optional[float] = None
        if self.auto_entropy and self.alpha_optimizer is not None:
            # Recompute log_probs for alpha update (detached from actor graph)
            log_probs_for_alpha = []
            with torch.no_grad():
                for obs, _, _, _, _ in batch:
                    nf, ei = _get_obs_tensors(obs)
                    _, lp = self.actor(nf, ei)
                    log_probs_for_alpha.append(lp.squeeze())
            lp_tensor = torch.stack(log_probs_for_alpha)  # [B]

            alpha_loss = -(self.log_alpha * (lp_tensor + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_val = float(alpha_loss.item())

        # Soft update target critic
        soft_update(self.twin_critic, self.target_twin_critic, self.tau)

        result: Dict[str, Any] = {
            "critic1_loss": float(c1_loss_total.item() / len(batch)),
            "critic2_loss": float(c2_loss_total.item() / len(batch)),
            "actor_loss": float(actor_loss_total.item()),
            "alpha": float(self.alpha.item()),
            "entropy": -log_prob_mean,
            "log_prob_mean": log_prob_mean,
            "q_target_mean": q_target_sum / len(batch),
        }
        if alpha_loss_val is not None:
            result["alpha_loss"] = alpha_loss_val

        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "twin_critic": self.twin_critic.state_dict(),
            "target_twin_critic": self.target_twin_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "alpha_optimizer": self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.twin_critic.load_state_dict(state["twin_critic"])
        self.target_twin_critic.load_state_dict(state["target_twin_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        with torch.no_grad():
            self.log_alpha.fill_(state["log_alpha"])
        if self.alpha_optimizer and state.get("alpha_optimizer"):
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])

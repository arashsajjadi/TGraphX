"""DQN and Double-DQN for graph RL.

DQN Target:
    y = r + γ * max_{a'} Q_target(s', a')  (masked by valid actions)

Double-DQN Target:
    a* = argmax_a Q_online(s', a)
    y  = r + γ * Q_target(s', a*)

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .base import BaseAgent
from .replay_buffer import ReplayBuffer
from tgraphx.rl.exploration.strategies import EpsilonGreedy

__all__ = ["DQNAgent", "DoubleDQNAgent"]

_MASK_FILL = -1e9


def _get_obs_tensors(obs: Dict[str, Any]) -> tuple:
    nf = obs.get("node_features", torch.zeros(1, 1))
    ei = obs.get("edge_index", torch.zeros((2, 0), dtype=torch.long))
    am = obs.get("action_mask")
    return nf, ei, am


class DQNAgent(BaseAgent):
    """Deep Q-Network agent.

    Args:
        q_network: Online Q-network.
        target_network: Target Q-network (a copy of q_network).
        optimizer: PyTorch optimizer.
        gamma: Discount factor.
        eps_start: Initial epsilon for epsilon-greedy.
        eps_end: Final epsilon.
        eps_decay: Exponential decay rate.
        target_update_freq: Steps between hard target updates.
        batch_size: Replay batch size.
        replay_buffer: ReplayBuffer instance.
    """

    def __init__(
        self,
        q_network: torch.nn.Module,
        target_network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.995,
        target_update_freq: int = 100,
        batch_size: int = 32,
        replay_buffer: Optional[ReplayBuffer] = None,
    ) -> None:
        self.q_net = q_network
        self.target_net = target_network
        # Ensure target net starts with same weights
        self.target_net.load_state_dict(q_network.state_dict())
        self.target_net.eval()

        self.optimizer = optimizer
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.buffer = replay_buffer if replay_buffer is not None else ReplayBuffer(10000)
        self.epsilon_schedule = EpsilonGreedy(eps_start, eps_end, eps_decay)
        self._step_count = 0

    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
        step: Optional[int] = None,
    ) -> int:
        s = step if step is not None else self._step_count

        if not deterministic and self.epsilon_schedule.should_explore(s, generator=generator):
            # Explore
            action_mask = obs.get("action_mask")
            if action_mask is not None:
                valid = action_mask.bool().nonzero(as_tuple=False).squeeze(1)
                if len(valid) > 0:
                    idx = int(torch.randint(len(valid), (1,), generator=generator).item())
                    return int(valid[idx].item())
            n_actions = self.q_net.num_actions if hasattr(self.q_net, "num_actions") else 10
            return int(torch.randint(n_actions, (1,), generator=generator).item())

        nf, ei, am = _get_obs_tensors(obs)
        with torch.no_grad():
            q_values = self.q_net(nf, ei).squeeze(0)  # [A]

        if am is not None:
            m = am.bool()
            m_size = min(m.shape[0], q_values.shape[-1])
            masked_q = q_values[:m_size].clone()
            masked_q[~m[:m_size]] = _MASK_FILL
        else:
            masked_q = q_values

        return int(masked_q.argmax().item())

    def update(self, batch: Optional[List] = None) -> Dict[str, float]:
        """Update Q-network from replay buffer.

        Args:
            batch: Optional pre-sampled batch. If None, samples from buffer.

        Returns:
            Loss dict.
        """
        if batch is None:
            if not self.buffer.is_ready(self.batch_size):
                return {}
            batch = self.buffer.sample(self.batch_size)

        if not batch:
            return {}

        total_loss = torch.tensor(0.0)

        for obs, action, reward, next_obs, done in batch:
            nf, ei, am = _get_obs_tensors(obs)
            next_nf, next_ei, next_am = _get_obs_tensors(next_obs)

            q_values = self.q_net(nf, ei).squeeze(0)  # [A]
            q_val = q_values[min(action, q_values.shape[0] - 1)]

            with torch.no_grad():
                next_q = self.target_net(next_nf, next_ei).squeeze(0)  # [A]
                if next_am is not None:
                    m = next_am.bool()
                    m_size = min(m.shape[0], next_q.shape[0])
                    next_q_masked = next_q[:m_size].clone()
                    next_q_masked[~m[:m_size]] = _MASK_FILL
                else:
                    next_q_masked = next_q
                target = reward + self.gamma * next_q_masked.max().item() * (1.0 - float(done))

            target_t = torch.tensor(target, dtype=torch.float)
            loss = F.mse_loss(q_val, target_t)
            total_loss = total_loss + loss

        total_loss = total_loss / len(batch)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.update_target_network()

        return {"q_loss": float(total_loss.item())}

    def update_target_network(self) -> None:
        """Hard copy online weights to target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def state_dict(self) -> Dict[str, Any]:
        return {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._step_count = state.get("step", 0)


class DoubleDQNAgent(DQNAgent):
    """Double DQN agent.

    a* = argmax_a Q_online(s', a)
    y  = r + γ * Q_target(s', a*)
    """

    def update(self, batch: Optional[List] = None) -> Dict[str, float]:
        if batch is None:
            if not self.buffer.is_ready(self.batch_size):
                return {}
            batch = self.buffer.sample(self.batch_size)

        if not batch:
            return {}

        total_loss = torch.tensor(0.0)

        for obs, action, reward, next_obs, done in batch:
            nf, ei, am = _get_obs_tensors(obs)
            next_nf, next_ei, next_am = _get_obs_tensors(next_obs)

            q_values = self.q_net(nf, ei).squeeze(0)
            q_val = q_values[min(action, q_values.shape[0] - 1)]

            with torch.no_grad():
                # Double DQN: select action with online, evaluate with target
                next_q_online = self.q_net(next_nf, next_ei).squeeze(0)
                if next_am is not None:
                    m = next_am.bool()
                    m_size = min(m.shape[0], next_q_online.shape[0])
                    next_q_online_m = next_q_online[:m_size].clone()
                    next_q_online_m[~m[:m_size]] = _MASK_FILL
                else:
                    next_q_online_m = next_q_online
                best_action = int(next_q_online_m.argmax().item())

                next_q_target = self.target_net(next_nf, next_ei).squeeze(0)
                q_target_val = next_q_target[min(best_action, next_q_target.shape[0] - 1)]
                target = reward + self.gamma * q_target_val.item() * (1.0 - float(done))

            target_t = torch.tensor(target, dtype=torch.float)
            loss = F.mse_loss(q_val, target_t)
            total_loss = total_loss + loss

        total_loss = total_loss / len(batch)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.update_target_network()

        return {"q_loss": float(total_loss.item())}

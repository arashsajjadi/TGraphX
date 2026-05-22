"""Trajectory buffer for PPO self-play.

Each entry corresponds to ONE player's turn.  The buffer stores everything
needed to compute advantages and run the PPO update.

Return computation uses alternating-perspective discounting:
    R[t] = r[t] - γ * R[t+1]
The minus sign captures the zero-sum flip between consecutive players' turns.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Transition:
    obs:        np.ndarray   # canonical observation [OBS_DIM]
    act_feats:  np.ndarray   # [N, ACT_DIM] for all legal actions at this step
    act_idx:    int           # index of chosen action in legal actions
    n_actions:  int           # number of legal actions
    log_prob:   float         # log π(a|s) at collection time
    value:      float         # V(s) at collection time
    reward:     float         # 0 except terminal step
    done:       bool


class RolloutBuffer:
    """Stores a batch of game trajectories and computes GAE returns."""

    def __init__(self, gamma: float = 0.995, gae_lambda: float = 0.95) -> None:
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self._data: List[Transition] = []

    def append(self, t: Transition) -> None:
        self._data.append(t)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def compute_returns_and_advantages(self) -> dict:
        """Compute discounted returns and GAE advantages.

        Consecutive turns alternate players in a zero-sum game, so the value
        of the next state from the *current* player's perspective is
        -V(next_state).  This sign flip is applied throughout.

        Returns a dict with numpy arrays ready for batched PPO update.
        """
        n = len(self._data)
        returns    = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)
        gae        = 0.0
        next_ret   = 0.0   # R[T+1] = 0 beyond terminal

        for i in reversed(range(n)):
            t = self._data[i]
            if t.done:
                # Terminal: next value = 0
                delta   = t.reward - t.value
                gae     = delta
                next_ret = t.reward
            else:
                # Opponent's value at t+1 is -V[t+1] from our perspective
                next_v   = -self._data[i + 1].value
                delta    = t.reward + self.gamma * next_v - t.value
                gae      = delta + self.gamma * self.gae_lambda * (-gae)
                next_ret = t.reward - self.gamma * next_ret

            returns[i]    = next_ret
            advantages[i] = gae

        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "obs":        np.stack([t.obs for t in self._data]),
            "act_feats":  [t.act_feats for t in self._data],   # variable N
            "act_idx":    np.array([t.act_idx for t in self._data], dtype=np.int64),
            "n_actions":  np.array([t.n_actions for t in self._data], dtype=np.int64),
            "log_probs":  np.array([t.log_prob for t in self._data], dtype=np.float32),
            "values":     np.array([t.value for t in self._data], dtype=np.float32),
            "returns":    returns,
            "advantages": advantages,
        }

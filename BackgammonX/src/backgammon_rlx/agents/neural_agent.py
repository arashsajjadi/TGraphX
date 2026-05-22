"""NeuralAgent — uses BackgammonPolicyValueNet to select actions.

Supports both greedy (argmax) and stochastic (sample) modes.
"""
from __future__ import annotations

import torch
import numpy as np
from typing import List, Optional, Tuple

from ..env.state import GameState, Turn
from ..env.movegen import get_legal_turns, canonicalize_state_for_player
from ..env.encoding import ObservationEncoder, ActionEncoder


class NeuralAgent:
    """Selects actions using a trained policy/value network.

    Parameters
    ----------
    model:       BackgammonPolicyValueNet (or compatible)
    obs_encoder: ObservationEncoder
    act_encoder: ActionEncoder
    device:      torch device string
    deterministic: if True use argmax, else sample from softmax
    temperature: softmax temperature (only for stochastic mode)
    """

    def __init__(
        self,
        model,
        obs_encoder: Optional[ObservationEncoder] = None,
        act_encoder: Optional[ActionEncoder] = None,
        device: str = "cpu",
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> None:
        self.model       = model
        self.obs_enc     = obs_encoder or ObservationEncoder()
        self.act_enc     = act_encoder or ActionEncoder()
        self.device      = torch.device(device)
        self.deterministic = deterministic
        self.temperature   = temperature
        self.model.eval()

    @torch.no_grad()
    def select_action(
        self,
        state: GameState,
        legal_turns: Optional[List[Turn]] = None,
    ) -> Turn:
        turns = legal_turns if legal_turns is not None else get_legal_turns(state)
        if not turns:
            from ..env.state import Turn
            return Turn()

        player = state.current_player
        canon  = canonicalize_state_for_player(state, player)

        obs = torch.tensor(self.obs_enc.encode(canon),
                           dtype=torch.float32, device=self.device).unsqueeze(0)

        act_feats = np.stack(
            [self.act_enc.encode(t, canon) for t in turns], axis=0
        )
        act_t = torch.tensor(act_feats, dtype=torch.float32,
                             device=self.device).unsqueeze(0)  # [1, N, A]

        logits, _ = self.model(obs, act_t)   # logits: [1, N]
        logits = logits.squeeze(0)           # [N]

        if self.deterministic:
            idx = int(logits.argmax().item())
        else:
            probs = torch.softmax(logits / self.temperature, dim=-1)
            idx   = int(torch.multinomial(probs, 1).item())

        return turns[idx]

    @torch.no_grad()
    def action_log_probs_and_value(
        self,
        state: GameState,
        legal_turns: Optional[List[Turn]] = None,
    ) -> Tuple[np.ndarray, float]:
        """Return (log_probs array over legal turns, value scalar)."""
        turns = legal_turns if legal_turns is not None else get_legal_turns(state)
        player = state.current_player
        canon  = canonicalize_state_for_player(state, player)

        obs = torch.tensor(self.obs_enc.encode(canon),
                           dtype=torch.float32, device=self.device).unsqueeze(0)
        act_feats = np.stack(
            [self.act_enc.encode(t, canon) for t in turns], axis=0
        )
        act_t = torch.tensor(act_feats, dtype=torch.float32,
                             device=self.device).unsqueeze(0)

        logits, value = self.model(obs, act_t)
        logits = logits.squeeze(0)
        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        v = float(value.squeeze().item())
        return log_probs, v

    def reset(self) -> None:
        pass

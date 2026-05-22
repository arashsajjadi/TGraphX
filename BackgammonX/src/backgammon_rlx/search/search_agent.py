"""SearchAgent wrapping expectimax search."""
from __future__ import annotations

from typing import List, Optional

import torch

from ..env.state import GameState, Turn
from ..env.movegen import get_legal_turns
from ..env.encoding import ObservationEncoder, ActionEncoder
from .expectimax import expectimax_action


class SearchAgent:
    """Agent that uses value-guided expectimax search.

    Parameters
    ----------
    model:          policy/value network
    search_depth:   1 = greedy leaf value, 2+ = expectimax
    n_dice_samples: dice outcome samples per node (36 = exhaustive for depth 2)
    deterministic:  if False, adds Dirichlet noise at root (exploration)
    """

    def __init__(
        self,
        model,
        obs_enc: Optional[ObservationEncoder] = None,
        act_enc: Optional[ActionEncoder] = None,
        device: str = "cpu",
        search_depth: int = 1,
        n_dice_samples: int = 36,
    ) -> None:
        self.model   = model
        self.obs_enc = obs_enc or ObservationEncoder()
        self.act_enc = act_enc or ActionEncoder()
        self.device  = torch.device(device)
        self.depth   = search_depth
        self.n_dice  = n_dice_samples
        self.model.eval()

    def select_action(
        self,
        state: GameState,
        legal_turns: Optional[List[Turn]] = None,
    ) -> Turn:
        return expectimax_action(
            model=self.model,
            state=state,
            depth=self.depth,
            obs_enc=self.obs_enc,
            act_enc=self.act_enc,
            device=self.device,
            n_dice_samples=self.n_dice,
        )

    def reset(self) -> None:
        pass

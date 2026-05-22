"""RandomLegalAgent — samples uniformly from legal turns."""
from __future__ import annotations

import random
from typing import List, Optional

from ..env.state import GameState, Turn
from ..env.movegen import get_legal_turns


class RandomLegalAgent:
    """Selects a uniformly random legal full-turn action."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, state: GameState,
                      legal_turns: Optional[List[Turn]] = None) -> Turn:
        turns = legal_turns if legal_turns is not None else get_legal_turns(state)
        return self._rng.choice(turns)

    def reset(self) -> None:
        pass

"""Money-game environment with doubling cube support.

Wraps BackgammonEnv and adds cube decision points:
  Before rolling dice: no_double | double
  After receiving double: take | pass

Cube actions are integrated as a separate action type.
The terminal reward is multiplied by the cube value.

game_mode:
  "checker_play" – no cube (default, identical to BackgammonEnv)
  "money_game"   – cube active, unlimited match
  "match_play"   – cube active, match score tracked

API:
  env = MoneyGameEnv(game_mode="money_game")
  obs, legal = env.reset()
  while not env.done():
      action = agent.act(obs, legal)
      obs, reward, done, info = env.step(action)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..env.env import BackgammonEnv
from ..env.state import GameState, Turn
from ..env.movegen import get_legal_turns
from ..env.encoding import ObservationEncoder, ActionEncoder
from ..env.rules import is_terminal, winner, score_value
from ..match.cube import CubeState
from ..match.match_state import MatchState


# ---------------------------------------------------------------------------
# Cube action types
# ---------------------------------------------------------------------------

class CubeAction(Enum):
    NO_DOUBLE = "no_double"
    DOUBLE    = "double"
    TAKE      = "take"
    PASS      = "pass"       # also called drop/resign


@dataclass(frozen=True)
class CubeDecision:
    action: CubeAction

    def __repr__(self) -> str:
        return f"Cube({self.action.value})"


# Either a checker-play Turn or a cube CubeDecision
AnyAction = Union[Turn, CubeDecision]


# ---------------------------------------------------------------------------
# Decision phase
# ---------------------------------------------------------------------------

class _Phase(Enum):
    ROLLING           = "rolling"        # about to roll (cube decision possible)
    MOVING            = "moving"         # dice rolled, choosing checker move
    AWAITING_TAKE     = "awaiting_take"  # opponent must take or pass


# ---------------------------------------------------------------------------
# MoneyGameEnv
# ---------------------------------------------------------------------------

class MoneyGameEnv:
    """Backgammon with doubling cube.

    State transitions:
      reset → ROLLING (player 0 goes first)
      ROLLING:   agent plays CubeDecision(DOUBLE) → AWAITING_TAKE
                 agent plays CubeDecision(NO_DOUBLE) → MOVING (dice rolled)
      AWAITING_TAKE: opponent plays TAKE → MOVING (dice rolled)
                     opponent plays PASS → terminal, loser pays cube × 1
      MOVING:    agent plays Turn → switch player → ROLLING (if not terminal)
    """

    def __init__(
        self,
        game_mode: str = "checker_play",
        obs_enc: Optional[ObservationEncoder] = None,
        act_enc: Optional[ActionEncoder] = None,
        match_length: int = 7,
        strict_invariants: bool = False,
    ) -> None:
        assert game_mode in ("checker_play", "money_game", "match_play")
        self.game_mode = game_mode
        self.obs_enc   = obs_enc or ObservationEncoder()
        self.act_enc   = act_enc or ActionEncoder()
        self._inner    = BackgammonEnv(self.obs_enc, self.act_enc, strict_invariants)
        self.cube      = CubeState()
        self.match     = MatchState(match_length=match_length) if game_mode == "match_play" else None
        self._phase    = _Phase.ROLLING
        self._done     = False
        self._reward   = 0.0
        self._info: Dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[AnyAction]]:
        self.cube  = CubeState()
        self._done = False
        self._phase = _Phase.ROLLING
        obs = self._inner.reset(seed=seed)
        return obs, self.legal_actions()

    def legal_actions(self) -> List[AnyAction]:
        """Return legal actions for the current phase."""
        if self._done:
            return []
        if self._phase == _Phase.ROLLING:
            actions: List[AnyAction] = [CubeDecision(CubeAction.NO_DOUBLE)]
            if self.game_mode != "checker_play":
                player = self._inner.current_player()
                if self.cube.can_double(player) and not self._is_crawford():
                    actions.append(CubeDecision(CubeAction.DOUBLE))
            return actions
        if self._phase == _Phase.AWAITING_TAKE:
            return [CubeDecision(CubeAction.TAKE), CubeDecision(CubeAction.PASS)]
        # MOVING
        return self._inner.legal_actions()

    def step(self, action: AnyAction) -> Tuple[np.ndarray, float, bool, Dict]:
        if self._done:
            raise RuntimeError("Game is over; call reset()")

        if isinstance(action, CubeDecision):
            return self._step_cube(action)
        return self._step_checker(action)

    def is_terminal(self) -> bool:
        return self._done

    def current_player(self) -> int:
        return self._inner.current_player()

    def current_observation(self) -> np.ndarray:
        return self._inner.current_observation()

    # ------------------------------------------------------------------
    # Internal step helpers
    # ------------------------------------------------------------------

    def _step_cube(self, decision: CubeDecision) -> Tuple[np.ndarray, float, bool, Dict]:
        ca = decision.action
        player = self._inner.current_player()
        reward = 0.0
        done   = False
        info: Dict = {}

        if ca == CubeAction.NO_DOUBLE:
            # Roll dice and move to MOVING phase
            self._inner._roll_dice()
            self._phase = _Phase.MOVING

        elif ca == CubeAction.DOUBLE:
            # Offer cube to opponent
            self.cube = self.cube.doubled_by(player)
            self._phase = _Phase.AWAITING_TAKE

        elif ca == CubeAction.TAKE:
            # Opponent accepts the double; roll dice for the doubler
            self._inner._roll_dice()
            self._phase = _Phase.MOVING

        elif ca == CubeAction.PASS:
            # Opponent drops — current player wins cube × 1 point
            winner_p = 1 - player   # the doubler wins
            reward = float(self.cube.value)
            done = True
            self._done = True
            info = {"winner": winner_p, "score": 1,
                    "cube_value": self.cube.value, "ended_by": "pass"}

        obs = self.current_observation()
        return obs, reward, done, info

    def _step_checker(self, action: Turn) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, done, info = self._inner.step(action)
        if done:
            # Scale reward by cube value
            scaled_reward = reward * self.cube.value
            info["cube_value"] = self.cube.value
            info["raw_reward"] = reward
            reward = scaled_reward
            self._done = True
        else:
            self._phase = _Phase.ROLLING
        return obs, reward, done, info

    def _is_crawford(self) -> bool:
        """True if this is the Crawford game (no doubling allowed)."""
        if self.match is None:
            return False
        return self.match.crawford_game


# ---------------------------------------------------------------------------
# Utility: encode cube phase as additional observation features
# ---------------------------------------------------------------------------

CUBE_OBS_EXTRA = 5   # cube_value_log, cube_owner_one_hot (3), is_crawford

def cube_features(env: MoneyGameEnv) -> np.ndarray:
    """Extra features for the cube state: [5] float32."""
    feat = np.zeros(CUBE_OBS_EXTRA, dtype=np.float32)
    import math
    feat[0] = math.log2(env.cube.value) / 4.0   # 0 to 1 for cube 1-16
    owner = env.cube.owner
    feat[1] = float(owner is None)               # centred
    feat[2] = float(owner == 0)                  # player 0 owns
    feat[3] = float(owner == 1)                  # player 1 owns
    feat[4] = float(env._is_crawford())
    return feat

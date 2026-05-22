"""BackgammonX — research-grade self-play RL for backgammon."""
__version__ = "0.1.0"

from .env import (
    BackgammonEnv,
    GameState, AtomicMove, Turn, BAR, OFF,
    get_legal_turns, apply_full_turn,
    OBS_DIM, ACT_DIM,
)

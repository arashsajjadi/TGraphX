"""External backgammon engine adapter interface.

Concrete implementations (e.g. GnuBackgammonAdapter) live in submodules.
The core package does NOT require any external engine to be installed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from ..env.state import GameState, Turn


class ExternalEngineError(RuntimeError):
    """Raised when the external engine is unavailable or returns an error."""


class ExternalEngineAgent(ABC):
    """Abstract base for external engine adapters.

    Subclasses must implement ``request_move`` and ``is_available``.
    If the engine is not installed, ``is_available()`` returns False and
    ``select_action`` raises ``ExternalEngineError`` with a clear message.
    """

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def request_move(
        self,
        state: GameState,
    ) -> Tuple[Turn, Optional[float]]:
        """Return (chosen_turn, equity_estimate_or_None)."""

    def select_action(
        self,
        state: GameState,
        legal_turns=None,
    ) -> Turn:
        if not self.is_available():
            raise ExternalEngineError(
                f"{self.__class__.__name__} is not available on this system. "
                "Install the engine and ensure it is on PATH."
            )
        turn, _ = self.request_move(state)
        return self._validate(state, turn, legal_turns)

    def _validate(self, state: GameState, turn: Turn, legal_turns=None) -> Turn:
        """Verify engine's move is legal; fall back to first legal move on failure."""
        from ..env.movegen import get_legal_turns
        from ..notation.move_notation import format_full_turn

        legal = legal_turns or get_legal_turns(state)
        legal_strs = {format_full_turn(t) for t in legal}
        if format_full_turn(turn) in legal_strs:
            return turn
        # Engine returned an illegal move — return first legal action
        import warnings
        warnings.warn(
            f"[ExternalEngineAgent] Engine returned illegal move "
            f"'{format_full_turn(turn)}'; falling back to first legal action."
        )
        return legal[0] if legal else Turn()

    def reset(self) -> None:
        pass

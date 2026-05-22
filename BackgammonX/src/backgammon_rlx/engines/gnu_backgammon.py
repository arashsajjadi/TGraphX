"""GNU Backgammon adapter (subprocess-based).

Requires ``gnubg`` on PATH with the ``--tty`` flag.
If not installed, ``is_available()`` returns False.

Position exchange uses GNU Backgammon's position-ID format
(or falls back to a JSON file-based protocol).
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Optional, Tuple

from ..env.state import GameState, Turn
from ..notation.position_io import state_to_dict
from .external_engine import ExternalEngineAgent, ExternalEngineError


class GnuBackgammonAdapter(ExternalEngineAgent):
    """Communicates with GNU Backgammon via subprocess pipe.

    Implementation note: full GNU Backgammon integration requires careful
    GnuBG scripting.  This class provides the interface and scaffolding;
    complete implementation is left as a future extension and depends on the
    installed GnuBG version.
    """

    def is_available(self) -> bool:
        return shutil.which("gnubg") is not None

    def request_move(
        self,
        state: GameState,
    ) -> Tuple[Turn, Optional[float]]:
        if not self.is_available():
            raise ExternalEngineError(
                "gnubg not found on PATH.  "
                "Install GNU Backgammon: https://www.gnu.org/software/gnubg/"
            )

        # Placeholder: real implementation would encode position to
        # GNU Backgammon's position-ID and pipe it to gnubg --tty.
        raise NotImplementedError(
            "GnuBackgammonAdapter.request_move is not yet fully implemented. "
            "Provide a position-to-GnuBG encoding and output parser."
        )

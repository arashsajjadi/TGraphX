"""Match Equity Table placeholder.

A proper MET maps (my_score, opp_score, match_length) → equity ∈ [0,1].
This stub returns 0.5 (neutral) until a real table is loaded.
"""
from __future__ import annotations

from typing import List


class MatchEquityTable:
    """Trivial MET stub.  Replace with a Zadeh or Snowie table for production."""

    def equity(self, my_score: int, opp_score: int,
               match_length: int) -> float:
        """Return winning probability for the player with *my_score* points."""
        needed_me  = match_length - my_score
        needed_opp = match_length - opp_score
        # Simple linear heuristic until a real table is loaded
        total = needed_me + needed_opp
        if total == 0:
            return 0.5
        return needed_opp / total

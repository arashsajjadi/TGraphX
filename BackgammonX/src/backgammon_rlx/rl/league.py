"""Checkpoint league and Elo rating system.

Maintains a pool of saved checkpoints and evaluates the current agent against
recent checkpoints and baseline agents.  Updates a running Elo table.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


K_FACTOR   = 32.0
INIT_ELO   = 1500.0


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def update_elo(elo_a: float, elo_b: float,
               score_a: float) -> tuple[float, float]:
    """Return updated (elo_a, elo_b) after a game where player-A scored *score_a*."""
    ea = expected_score(elo_a, elo_b)
    da = K_FACTOR * (score_a - ea)
    return elo_a + da, elo_b - da


class LeagueManager:
    """Manages checkpoint pool, Elo table, and match records."""

    def __init__(self, league_dir: Path) -> None:
        self.league_dir = league_dir
        self.league_dir.mkdir(parents=True, exist_ok=True)
        self._ratings: Dict[str, float] = {}
        self._matches: List[Dict] = []
        self._load()

    def _load(self) -> None:
        p = self.league_dir / "ratings.json"
        if p.exists():
            self._ratings = json.loads(p.read_text())
        m = self.league_dir / "matches.csv"
        if m.exists():
            lines = m.read_text().splitlines()[1:]  # skip header
            for line in lines:
                parts = line.split(",")
                if len(parts) >= 4:
                    self._matches.append({
                        "a": parts[0], "b": parts[1],
                        "wins_a": int(parts[2]), "wins_b": int(parts[3]),
                    })

    def _save(self) -> None:
        (self.league_dir / "ratings.json").write_text(
            json.dumps(self._ratings, indent=2))
        rows = ["agent_a,agent_b,wins_a,wins_b"]
        for m in self._matches:
            rows.append(f"{m['a']},{m['b']},{m['wins_a']},{m['wins_b']}")
        (self.league_dir / "matches.csv").write_text("\n".join(rows))

    def get_elo(self, name: str) -> float:
        return self._ratings.get(name, INIT_ELO)

    def record_match(self, name_a: str, name_b: str,
                     wins_a: int, wins_b: int, draws: int = 0) -> None:
        total = wins_a + wins_b + draws
        if total == 0:
            return
        score_a = (wins_a + 0.5 * draws) / total

        elo_a = self.get_elo(name_a)
        elo_b = self.get_elo(name_b)
        new_a, new_b = update_elo(elo_a, elo_b, score_a)
        self._ratings[name_a] = new_a
        self._ratings[name_b] = new_b

        self._matches.append({"a": name_a, "b": name_b,
                               "wins_a": wins_a, "wins_b": wins_b})
        self._save()

    def elo_table(self) -> List[tuple]:
        return sorted(self._ratings.items(), key=lambda x: -x[1])

    def recent_checkpoints(self, n: int = 5) -> List[Path]:
        ckpts = sorted(self.league_dir.parent.glob("checkpoints/ckpt_*.pt"))
        return ckpts[-n:]

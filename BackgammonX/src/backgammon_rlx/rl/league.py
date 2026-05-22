"""Checkpoint league, Elo rating, and opponent pool management.

Fully wired into SelfPlayTrainer via LeagueManager.

Pool management:
  - checkpoints saved periodically to runs/<run_id>/league/pool/
  - up to checkpoint_pool_size checkpoints retained
  - opponents sampled according to configured probabilities

Promotion gate:
  - checkpoint evaluated vs random and greedy agents
  - only promoted to pool if it passes min_win_rate thresholds
"""
from __future__ import annotations

import copy
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


K_FACTOR   = 32.0
INIT_ELO   = 1500.0


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def update_elo(elo_a: float, elo_b: float,
               score_a: float) -> Tuple[float, float]:
    ea = expected_score(elo_a, elo_b)
    da = K_FACTOR * (score_a - ea)
    return elo_a + da, elo_b - da


class LeagueManager:
    """Manages checkpoint pool, Elo ratings, and opponent sampling.

    Usage (inside SelfPlayTrainer):
        league = LeagueManager(run_dir / "league", cfg)
        # Periodically:
        league.maybe_add_checkpoint(model, games_played, eval_fn)
        opponent = league.sample_opponent()   # returns loaded model or None
    """

    def __init__(self, league_dir: Path, cfg: Dict[str, Any]) -> None:
        self.league_dir = Path(league_dir)
        self.pool_dir   = self.league_dir / "pool"
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.league_dir.mkdir(parents=True, exist_ok=True)

        league_cfg  = cfg.get("league", {})
        sampling    = league_cfg.get("opponent_sampling", {})
        promo       = league_cfg.get("promotion", {})

        self.pool_size       = league_cfg.get("checkpoint_pool_size", 10)
        self.prob_current    = sampling.get("current_policy_prob", 0.6)
        self.prob_recent     = sampling.get("recent_checkpoint_prob", 0.3)
        self.prob_older      = sampling.get("older_checkpoint_prob", 0.1)
        self.promo_games     = promo.get("eval_games", 100)
        self.min_wr_random   = promo.get("min_win_rate_vs_random", 0.90)
        self.min_wr_greedy   = promo.get("min_win_rate_vs_greedy", 0.60)

        self._pool:    List[Path]   = []
        self._ratings: Dict[str, float] = {}
        self._matches: List[Dict]   = []
        self._rng = random.Random(cfg.get("seed", 0))
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        p = self.league_dir / "ratings.json"
        if p.exists():
            try:
                self._ratings = json.loads(p.read_text())
            except Exception:
                pass
        m = self.league_dir / "matches.csv"
        if m.exists():
            lines = m.read_text().splitlines()[1:]
            for line in lines:
                parts = line.split(",")
                if len(parts) >= 4:
                    try:
                        self._matches.append({
                            "a": parts[0], "b": parts[1],
                            "wins_a": int(parts[2]), "wins_b": int(parts[3]),
                        })
                    except ValueError:
                        pass
        # Rebuild pool from pool_dir
        self._pool = sorted(self.pool_dir.glob("ckpt_*.pt"))

    def _save(self) -> None:
        (self.league_dir / "ratings.json").write_text(
            json.dumps(self._ratings, indent=2))
        rows = ["agent_a,agent_b,wins_a,wins_b"]
        for m in self._matches[-10000:]:   # cap to avoid huge files
            rows.append(f"{m['a']},{m['b']},{m['wins_a']},{m['wins_b']}")
        (self.league_dir / "matches.csv").write_text("\n".join(rows))

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

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

    def add_checkpoint(self, model: nn.Module, games: int) -> Path:
        """Save current model weights to the pool. Prune oldest if over limit."""
        ckpt_path = self.pool_dir / f"ckpt_{games:08d}.pt"
        torch.save({"model": model.state_dict(), "games": games}, ckpt_path)
        self._pool.append(ckpt_path)
        self._pool.sort()
        # Prune oldest
        while len(self._pool) > self.pool_size:
            old = self._pool.pop(0)
            old.unlink(missing_ok=True)
        name = ckpt_path.stem
        if name not in self._ratings:
            self._ratings[name] = INIT_ELO
        self._save()
        return ckpt_path

    def load_pool_model(self, model_template: nn.Module, ckpt_path: Path) -> nn.Module:
        """Load weights from pool into a copy of model_template."""
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = copy.deepcopy(model_template)
        m.load_state_dict(state["model"])
        m.eval()
        return m

    # ------------------------------------------------------------------
    # Opponent sampling
    # ------------------------------------------------------------------

    def sample_pool_checkpoint(self) -> Optional[Path]:
        """Sample a checkpoint from the pool according to configured probabilities."""
        if not self._pool:
            return None
        # Weighted: prefer recent checkpoints
        if len(self._pool) == 1:
            return self._pool[0]
        # Split into recent (last half) and older (first half)
        mid = len(self._pool) // 2
        recent = self._pool[mid:]
        older  = self._pool[:mid]

        r = self._rng.random()
        if r < self.prob_recent / (self.prob_recent + self.prob_older) and recent:
            return self._rng.choice(recent)
        elif older:
            return self._rng.choice(older)
        return self._rng.choice(self._pool)

    def use_self_play_this_round(self) -> bool:
        """True if this rollout round should use the current policy (vs league opp)."""
        return self._rng.random() < self.prob_current or not self._pool

    # ------------------------------------------------------------------
    # Promotion gate
    # ------------------------------------------------------------------

    def evaluate_for_promotion(
        self,
        model: nn.Module,
        obs_enc,
        act_enc,
        device: torch.device,
        n_games: int = None,
    ) -> Dict[str, float]:
        """Evaluate model against random and greedy baselines.

        Returns dict with win rates. Promotion is granted if thresholds pass.
        """
        from ..train.evaluate import run_evaluation
        from ..agents.neural_agent import NeuralAgent
        from ..agents.random_agent import RandomLegalAgent
        from ..agents.heuristic_agent import GreedyPipAgent

        n = n_games or self.promo_games
        agent = NeuralAgent(model, obs_enc, act_enc, device=str(device),
                            deterministic=True)

        res_random = run_evaluation(agent, RandomLegalAgent(), n, obs_enc, act_enc)
        res_greedy = run_evaluation(agent, GreedyPipAgent(), n, obs_enc, act_enc)

        return {
            "win_rate_vs_random": res_random["win_rate_a"],
            "win_rate_vs_greedy": res_greedy["win_rate_a"],
            "promoted": (res_random["win_rate_a"] >= self.min_wr_random and
                         res_greedy["win_rate_a"] >= self.min_wr_greedy),
        }

    # ------------------------------------------------------------------
    # Elo summary
    # ------------------------------------------------------------------

    def elo_table(self) -> List[Tuple[str, float]]:
        return sorted(self._ratings.items(), key=lambda x: -x[1])

    def pool_size_current(self) -> int:
        return len(self._pool)

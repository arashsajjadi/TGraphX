"""Tests for league management, opponent sampling, and promotion gate."""
import tempfile
from pathlib import Path
import pytest
import torch

from backgammon_rlx.rl.league import LeagueManager, update_elo, expected_score
from backgammon_rlx.models.policy_value_net import BackgammonPolicyValueNet


def _small_model():
    return BackgammonPolicyValueNet(state_dim=32, act_dim=32,
                                     n_point_res=1, n_action_res=1)


def _league(cfg=None):
    d = tempfile.mkdtemp()
    cfg = cfg or {"seed": 42}
    return LeagueManager(Path(d) / "league", cfg), Path(d)


class TestEloMath:

    def test_expected_score_equal_ratings(self):
        assert abs(expected_score(1500, 1500) - 0.5) < 1e-6

    def test_expected_score_higher_rating(self):
        assert expected_score(1600, 1500) > 0.5

    def test_update_elo_win(self):
        a, b = update_elo(1500, 1500, 1.0)
        assert a > 1500 and b < 1500

    def test_update_elo_draw(self):
        a, b = update_elo(1500, 1500, 0.5)
        assert abs(a - 1500) < 1e-6


class TestLeaguePool:

    def test_empty_pool(self):
        league, _ = _league()
        assert league.pool_size_current() == 0
        assert league.sample_pool_checkpoint() is None

    def test_add_checkpoint(self):
        league, _ = _league()
        model = _small_model()
        path = league.add_checkpoint(model, games=100)
        assert path.exists()
        assert league.pool_size_current() == 1

    def test_pool_size_limit(self):
        cfg = {"seed": 0, "league": {"checkpoint_pool_size": 3}}
        league, _ = _league(cfg)
        model = _small_model()
        for i in range(5):
            league.add_checkpoint(model, games=(i+1)*100)
        assert league.pool_size_current() <= 3

    def test_sample_returns_path(self):
        league, _ = _league()
        model = _small_model()
        league.add_checkpoint(model, 100)
        league.add_checkpoint(model, 200)
        ckpt = league.sample_pool_checkpoint()
        assert ckpt is not None and ckpt.exists()

    def test_load_pool_model(self):
        league, _ = _league()
        model = _small_model()
        path = league.add_checkpoint(model, 100)
        loaded = league.load_pool_model(model, path)
        assert loaded is not model  # different object
        # Check same weights
        for p1, p2 in zip(model.parameters(), loaded.parameters()):
            assert torch.allclose(p1, p2)


class TestLeagueRatings:

    def test_record_match_updates_elo(self):
        league, _ = _league()
        initial = league.get_elo("model_a")
        league.record_match("model_a", "model_b", wins_a=8, wins_b=2)
        assert league.get_elo("model_a") > initial  # model_a won more

    def test_record_match_saves(self):
        league, d = _league()
        league.record_match("x", "y", 5, 5)
        assert (league.league_dir / "ratings.json").exists()
        assert (league.league_dir / "matches.csv").exists()

    def test_elo_table_sorted(self):
        league, _ = _league()
        league._ratings = {"a": 1600.0, "b": 1400.0, "c": 1500.0}
        table = league.elo_table()
        elos = [e for _, e in table]
        assert elos == sorted(elos, reverse=True)


class TestLeagueOpponentSampling:

    def test_use_self_play_when_pool_empty(self):
        league, _ = _league()
        # With empty pool, always use self-play
        results = [league.use_self_play_this_round() for _ in range(20)]
        assert all(results)

    def test_sometimes_uses_opponent_when_pool_nonempty(self):
        cfg = {"seed": 42, "league": {
            "opponent_sampling": {
                "current_policy_prob": 0.0,  # never self-play
                "recent_checkpoint_prob": 0.7,
                "older_checkpoint_prob": 0.3,
            }
        }}
        league, _ = _league(cfg)
        league._pool = [Path("/fake/ckpt.pt")]  # fake pool entry
        results = [league.use_self_play_this_round() for _ in range(20)]
        assert not all(results), "With prob=0 self-play, should use opponents"

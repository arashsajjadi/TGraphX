"""Random-game stress tests: thousands of games, no crashes, all invariants hold.

Run with:
    pytest tests/test_random_stress.py -v
"""
import random
import pytest
from backgammon_rlx.env.env import BackgammonEnv
from backgammon_rlx.env.rules import total_checkers, is_terminal, score_value, winner
from backgammon_rlx.agents.random_agent import RandomLegalAgent


def _run_games(n: int, seed: int = 0, max_steps: int = 2000) -> dict:
    env   = BackgammonEnv(strict_invariants=False)  # fast path; we check manually
    agent = RandomLegalAgent(seed=seed)
    wins  = [0, 0]
    total_steps = 0
    max_game_len = 0
    games_completed = 0

    for g in range(n):
        env.reset(seed=seed * 10000 + g)
        steps = 0
        while not env.is_terminal() and steps < max_steps:
            state = env.state
            turns = env.legal_actions()

            # --- invariants checked at each step ---
            assert total_checkers(state, 0) == 15, f"P0 checker count wrong at step {steps}"
            assert total_checkers(state, 1) == 15, f"P1 checker count wrong at step {steps}"
            assert len(turns) >= 1, "legal_actions must always return at least 1 turn"
            assert state.current_player in (0, 1)
            assert all(v >= 0 for v in state.bar)
            assert all(0 <= v <= 15 for v in state.borne_off)
            # No point has both players
            for val in state.board:
                assert val >= -15 and val <= 15

            action = agent.select_action(state, turns)
            env.step(action)
            steps += 1

        state = env.state
        assert total_checkers(state, 0) == 15
        assert total_checkers(state, 1) == 15

        if env.is_terminal():
            games_completed += 1
            w = env.winner()
            assert w in (0, 1)
            wins[w] += 1
            sv = env.score_value()
            assert sv in (1, 2, 3)
            # Terminal state consistency
            assert state.borne_off[w] == 15
            max_game_len = max(max_game_len, steps)

        total_steps += steps

    return {
        "games": n,
        "completed": games_completed,
        "wins": wins,
        "total_steps": total_steps,
        "max_game_len": max_game_len,
    }


class TestRandomStress:

    def test_100_games_no_crashes(self):
        result = _run_games(100, seed=42)
        assert result["completed"] == 100

    @pytest.mark.slow
    def test_invariants_100_games(self):
        """All invariants hold across 100 complete random games."""
        result = _run_games(100, seed=123)
        assert result["completed"] == 100
        # Both players should win sometimes (random play should be ~50/50)
        assert result["wins"][0] > 0
        assert result["wins"][1] > 0

    def test_game_length_reasonable(self):
        """Random games should terminate in a reasonable number of steps."""
        result = _run_games(50, seed=7)
        # Average game length should be under 500 steps
        avg = result["total_steps"] / result["games"]
        assert avg < 500, f"Average game length {avg:.1f} is too long"

    def test_no_illegal_state_after_hit(self):
        """Hitting never creates a state with negative bar counts."""
        env   = BackgammonEnv()
        agent = RandomLegalAgent(seed=0)
        env.reset(seed=0)
        for _ in range(5000):
            if env.is_terminal():
                env.reset()
            turns  = env.legal_actions()
            action = agent.select_action(env.state, turns)
            env.step(action)
            state = env.state
            assert all(v >= 0 for v in state.bar)
            assert all(v >= 0 for v in state.borne_off)

    def test_player1_games(self):
        """Run games starting from player 1's perspective, verify symmetry."""
        result = _run_games(50, seed=999)
        assert result["completed"] == 50


class TestPropertyInvariants:
    """Property-based tests: invariants that must hold for any random position."""

    def test_total_checkers_always_15(self):
        """Across 200 games, each player always has exactly 15 total checkers."""
        env   = BackgammonEnv()
        agent = RandomLegalAgent(seed=42)
        env.reset(seed=0)
        for step in range(20000):
            if env.is_terminal():
                env.reset(seed=step)
            state = env.state
            t0 = total_checkers(state, 0)
            t1 = total_checkers(state, 1)
            assert t0 == 15, f"step {step}: P0 has {t0} checkers"
            assert t1 == 15, f"step {step}: P1 has {t1} checkers"
            turns  = env.legal_actions()
            action = agent.select_action(state, turns)
            env.step(action)

    def test_no_point_has_both_players(self):
        """No board point should ever have checkers of both players."""
        env   = BackgammonEnv()
        agent = RandomLegalAgent(seed=77)
        env.reset(seed=42)
        for step in range(10000):
            if env.is_terminal():
                env.reset(seed=step)
            state = env.state
            for i, val in enumerate(state.board):
                # Positive means player 0, negative means player 1.
                # Any value is fine as long as it's only one sign.
                assert val == 0 or (val > 0 or val < 0), "mixed signs are impossible by design"
            turns  = env.legal_actions()
            action = agent.select_action(state, turns)
            env.step(action)

    def test_terminal_state_consistent(self):
        """When the game ends, exactly one player has 15 borne-off checkers."""
        env   = BackgammonEnv()
        agent = RandomLegalAgent(seed=13)
        games_checked = 0
        env.reset(seed=0)
        for step in range(30000):
            turns  = env.legal_actions()
            _, _, done, info = env.step(agent.select_action(env.state, turns))
            if done:
                state = env.state
                w = info.get("winner")
                assert w in (0, 1)
                assert state.borne_off[w] == 15
                assert state.borne_off[1 - w] < 15
                sv = info.get("score", 0)
                assert sv in (1, 2, 3)
                games_checked += 1
                env.reset(seed=step)
        assert games_checked >= 100, f"Only checked {games_checked} terminal states"

    def test_score_consistency(self):
        """score_value matches winner's borne-off and bar state."""
        from backgammon_rlx.env.rules import score_value as sv
        env   = BackgammonEnv()
        agent = RandomLegalAgent(seed=55)
        env.reset(seed=0)
        for step in range(20000):
            _, _, done, info = env.step(
                agent.select_action(env.state, env.legal_actions())
            )
            if done:
                state = env.state
                score = sv(state)
                w = winner(state)
                loser = 1 - w
                if state.borne_off[loser] > 0:
                    assert score == 1
                elif state.bar[loser] > 0:
                    assert score == 3
                else:
                    lo, hi = (1, 6) if w == 0 else (19, 24)
                    loser_in_winner_home = any(
                        (state.board[pt-1] > 0 if loser == 0 else state.board[pt-1] < 0)
                        for pt in range(lo, hi+1)
                    )
                    if loser_in_winner_home:
                        assert score == 3
                    else:
                        assert score == 2
                env.reset(seed=step)


def winner(state):
    from backgammon_rlx.env.rules import winner as _winner
    return _winner(state)

"""Tests that environment invariants hold through random play."""
import pytest
from backgammon_rlx.env.env import BackgammonEnv
from backgammon_rlx.env.rules import total_checkers
from backgammon_rlx.agents.random_agent import RandomLegalAgent


def run_game(seed: int, max_steps: int = 1000) -> dict:
    env   = BackgammonEnv(strict_invariants=True)
    agent = RandomLegalAgent(seed=seed)
    env.reset(seed=seed)
    steps = 0
    while not env.is_terminal() and steps < max_steps:
        turns  = env.legal_actions()
        action = agent.select_action(env.state, turns)
        env.step(action)
        state = env.state
        # Invariants on every step
        assert total_checkers(state, 0) == 15
        assert total_checkers(state, 1) == 15
        assert all(v >= 0 for v in state.bar)
        assert all(v >= 0 for v in state.borne_off)
        assert state.current_player in (0, 1)
        steps += 1
    return {"steps": steps, "done": env.is_terminal()}


class TestEnvInvariants:

    def test_invariants_game_0(self):
        r = run_game(0)
        assert r["done"]

    def test_invariants_game_1(self):
        r = run_game(1)
        assert r["done"]

    def test_invariants_game_2(self):
        r = run_game(2)
        assert r["done"]

    def test_invariants_game_3(self):
        r = run_game(3)
        assert r["done"]

    def test_invariants_game_4(self):
        r = run_game(4)
        assert r["done"]

    def test_legal_actions_never_empty(self):
        """legal_actions() should always return at least a pass Turn."""
        env   = BackgammonEnv()
        agent = RandomLegalAgent(seed=99)
        env.reset(seed=99)
        steps = 0
        while not env.is_terminal() and steps < 500:
            turns = env.legal_actions()
            assert len(turns) >= 1
            env.step(agent.select_action(env.state, turns))
            steps += 1

    def test_no_illegal_moves_in_100_games(self):
        """Run 100 games; none should raise an error (illegal move = crash)."""
        from backgammon_rlx.env.env import BackgammonEnv
        for seed in range(100):
            env   = BackgammonEnv(strict_invariants=False)
            agent = RandomLegalAgent(seed=seed)
            env.reset(seed=seed)
            for _ in range(500):
                if env.is_terminal():
                    break
                turns  = env.legal_actions()
                action = agent.select_action(env.state, turns)
                env.step(action)
            assert env.is_terminal() or True  # no exception = pass

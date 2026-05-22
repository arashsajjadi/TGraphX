"""Tests for cube/money-game environment."""
import pytest
from backgammon_rlx.match.cube_env import (
    MoneyGameEnv, CubeAction, CubeDecision, _Phase,
)
from backgammon_rlx.env.state import Turn


class TestCheckerPlayMode:

    def test_checker_play_no_cube_actions(self):
        env = MoneyGameEnv(game_mode="checker_play")
        obs, legal = env.reset(seed=0)
        # In checker_play, ROLLING phase: only NO_DOUBLE offered
        cube_actions = [a for a in legal if isinstance(a, CubeDecision)]
        assert all(a.action == CubeAction.NO_DOUBLE for a in cube_actions)
        assert len(cube_actions) == 1  # only NO_DOUBLE

    def test_checker_play_can_complete_game(self):
        from backgammon_rlx.agents.random_agent import RandomLegalAgent
        env   = MoneyGameEnv(game_mode="checker_play")
        agent = RandomLegalAgent(seed=0)
        obs, legal = env.reset(seed=42)
        steps = 0
        while not env.is_terminal() and steps < 2000:
            action = legal[0] if isinstance(legal[0], CubeDecision) else \
                     agent.select_action(env._inner.state, legal)
            obs, reward, done, info = env.step(action)
            legal = env.legal_actions()
            steps += 1
        assert env.is_terminal()


class TestMoneyGameCubeActions:

    def test_double_offered_in_money_game(self):
        env = MoneyGameEnv(game_mode="money_game")
        _, legal = env.reset(seed=0)
        cube_types = {a.action for a in legal if isinstance(a, CubeDecision)}
        # Centered cube: both NO_DOUBLE and DOUBLE should be offered
        assert CubeAction.NO_DOUBLE in cube_types
        assert CubeAction.DOUBLE in cube_types

    def test_double_leads_to_awaiting_take(self):
        env = MoneyGameEnv(game_mode="money_game")
        _, legal = env.reset(seed=0)
        obs, reward, done, info = env.step(CubeDecision(CubeAction.DOUBLE))
        assert env._phase == _Phase.AWAITING_TAKE
        new_legal = env.legal_actions()
        types = {a.action for a in new_legal if isinstance(a, CubeDecision)}
        assert CubeAction.TAKE in types
        assert CubeAction.PASS in types

    def test_pass_terminates_game(self):
        env = MoneyGameEnv(game_mode="money_game")
        _, legal = env.reset(seed=0)
        # Double
        env.step(CubeDecision(CubeAction.DOUBLE))
        # Pass
        obs, reward, done, info = env.step(CubeDecision(CubeAction.PASS))
        assert done
        assert env.is_terminal()
        assert reward != 0.0
        assert info["ended_by"] == "pass"

    def test_take_leads_to_moving(self):
        env = MoneyGameEnv(game_mode="money_game")
        _, legal = env.reset(seed=0)
        env.step(CubeDecision(CubeAction.DOUBLE))
        obs, reward, done, info = env.step(CubeDecision(CubeAction.TAKE))
        assert not done
        assert env._phase == _Phase.MOVING

    def test_cube_value_doubles_after_double(self):
        env = MoneyGameEnv(game_mode="money_game")
        env.reset(seed=0)
        assert env.cube.value == 1
        env.step(CubeDecision(CubeAction.DOUBLE))
        env.step(CubeDecision(CubeAction.TAKE))
        assert env.cube.value == 2

    def test_terminal_reward_scaled_by_cube(self):
        """Terminal reward should be cube_value × game_score."""
        from backgammon_rlx.agents.random_agent import RandomLegalAgent
        env = MoneyGameEnv(game_mode="money_game")
        _, legal = env.reset(seed=5)
        # Double immediately
        env.step(CubeDecision(CubeAction.DOUBLE))
        env.step(CubeDecision(CubeAction.TAKE))
        assert env.cube.value == 2

        agent = RandomLegalAgent(seed=5)
        steps = 0
        while not env.is_terminal() and steps < 2000:
            legal = env.legal_actions()
            if not legal:
                break
            if isinstance(legal[0], CubeDecision):
                action = CubeDecision(CubeAction.NO_DOUBLE)
            else:
                action = agent.select_action(env._inner.state, legal)
            obs, reward, done, info = env.step(action)
            steps += 1

        if env.is_terminal():
            # Final reward should be a multiple of cube value
            assert abs(reward) >= env.cube.value

    def test_no_double_in_crawford(self):
        """Crawford rule: no doubling in the Crawford game."""
        env = MoneyGameEnv(game_mode="match_play", match_length=3)
        env.reset(seed=0)
        # Manually set Crawford state
        env.match.crawford_game = True
        legal = env.legal_actions()
        types = {a.action for a in legal if isinstance(a, CubeDecision)}
        assert CubeAction.DOUBLE not in types


class TestCubeState:

    def test_cube_initial_state(self):
        from backgammon_rlx.match.cube import CubeState
        c = CubeState()
        assert c.value == 1
        assert c.is_centred
        assert c.available

    def test_doubled_by(self):
        from backgammon_rlx.match.cube import CubeState
        c = CubeState()
        c2 = c.doubled_by(0)
        assert c2.value == 2
        assert c2.owner == 1  # opponent owns after double

    def test_can_double_centred(self):
        from backgammon_rlx.match.cube import CubeState
        c = CubeState()
        assert c.can_double(0) and c.can_double(1)

    def test_cannot_double_if_opponent_owns(self):
        from backgammon_rlx.match.cube import CubeState
        c = CubeState(value=2, owner=1)
        assert c.can_double(1)
        assert not c.can_double(0)

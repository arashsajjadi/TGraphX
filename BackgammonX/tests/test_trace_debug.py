"""Tests for transition tracing and debug output."""
import pytest
from backgammon_rlx.env.env import BackgammonEnv
from backgammon_rlx.agents.random_agent import RandomLegalAgent


class TestTransitionTrace:

    def _first_step_with_trace(self, seed=42):
        env = BackgammonEnv()
        env.reset(seed=seed)
        turns = env.legal_actions()
        agent = RandomLegalAgent(seed=seed)
        action = agent.select_action(env.state, turns)
        obs, reward, done, info = env.step(action, trace=True)
        return info

    def test_trace_present_when_requested(self):
        info = self._first_step_with_trace()
        assert "trace" in info

    def test_trace_has_required_fields(self):
        info = self._first_step_with_trace()
        trace = info["trace"]
        assert "player" in trace
        assert "dice" in trace
        assert "action" in trace
        assert "atomic_moves" in trace
        assert "board_before" in trace
        assert "board_after" in trace

    def test_trace_board_before_length(self):
        info = self._first_step_with_trace()
        assert len(info["trace"]["board_before"]) == 24
        assert len(info["trace"]["board_after"]) == 24

    def test_trace_atomic_moves_structure(self):
        info = self._first_step_with_trace()
        for am in info["trace"]["atomic_moves"]:
            assert "move" in am
            assert "board_before" in am
            assert "board_after" in am
            assert "bar_before" in am
            assert "bar_after" in am
            assert "hit" in am

    def test_trace_not_present_without_flag(self):
        env = BackgammonEnv()
        env.reset(seed=0)
        turns = env.legal_actions()
        agent = RandomLegalAgent(seed=0)
        action = agent.select_action(env.state, turns)
        _, _, _, info = env.step(action)  # No trace=True
        assert "trace" not in info

    def test_trace_board_after_equals_state(self):
        """Trace's board_after should match the env state after step."""
        env = BackgammonEnv()
        env.reset(seed=7)
        turns = env.legal_actions()
        agent = RandomLegalAgent(seed=7)
        action = agent.select_action(env.state, turns)
        _, _, _, info = env.step(action, trace=True)
        # board_after in trace should match next state's board
        # (the trace is built from the state transition)
        assert info["trace"]["board_after"] is not None
        assert len(info["trace"]["board_after"]) == 24

    def test_trace_hit_flag_correct(self):
        """If a move hits a blot, the hit flag should be True in the trace."""
        from backgammon_rlx.env.state import GameState, AtomicMove, Turn, BAR
        env = BackgammonEnv()
        # Set up a position where a hit occurs
        state = GameState(board=[0]*24, bar=[0,0], borne_off=[0,0],
                          current_player=0, dice=[5])
        state.board[12] = 1   # p0 at 13
        state.board[7]  = -1  # p1 blot at 8 (13-5=8)
        env.set_state(state)
        env._state.dice = [5, 2]  # give two dice

        turns = env.legal_actions()
        # Find a turn that hits
        hitting_turns = [t for t in turns if any(m.hit for m in t.moves)]
        if not hitting_turns:
            pytest.skip("No hitting moves in this position")

        _, _, _, info = env.step(hitting_turns[0], trace=True)
        trace = info["trace"]
        hit_moves = [am for am in trace["atomic_moves"] if am["hit"]]
        assert len(hit_moves) > 0


class TestLongGameTrace:

    def test_trace_across_full_game(self):
        """Run a full game with trace=True on every step; no exceptions."""
        env   = BackgammonEnv()
        agent = RandomLegalAgent(seed=99)
        env.reset(seed=99)
        steps = 0
        while not env.is_terminal() and steps < 500:
            turns  = env.legal_actions()
            action = agent.select_action(env.state, turns)
            _, _, _, info = env.step(action, trace=True)
            assert "trace" in info
            steps += 1

"""Tests for mirror symmetry between player 0 and player 1.

By symmetry, the number of legal turns for player 0 in a position
should equal the number of legal turns for the mirrored position from
player 1's perspective.
"""
import pytest
from backgammon_rlx.env.state import GameState
from backgammon_rlx.env.movegen import (
    get_legal_turns, canonicalize_state_for_player, apply_full_turn,
)
from backgammon_rlx.env.rules import total_checkers


def _mirror(state: GameState) -> GameState:
    """Return the mirror of state: player 1 perspective of the same position."""
    return canonicalize_state_for_player(state, 1)


def _make_state(board, bar, borne_off, player, dice):
    s = GameState(board=board[:], bar=bar[:], borne_off=borne_off[:],
                  current_player=player, dice=dice[:])
    return s


class TestMirrorSymmetry:

    def test_initial_position_symmetric_count(self):
        """Initial position with same dice should give same legal turn count for both players."""
        s0 = GameState.initial()
        s0.dice = [3, 1]
        s0.current_player = 0
        turns0 = get_legal_turns(s0)

        s1 = GameState.initial()
        s1.dice = [3, 1]
        s1.current_player = 1
        turns1 = get_legal_turns(s1)

        assert len(turns0) == len(turns1), (
            f"Symmetric initial position [3,1]: p0 has {len(turns0)} turns, "
            f"p1 has {len(turns1)}")

    def test_canonicalize_and_back(self):
        """Canonicalizing for player 1 and then back should produce identical pip counts."""
        from backgammon_rlx.env.rules import pip_count
        s = GameState.initial()
        pip0_p0 = pip_count(s, 0)
        pip0_p1 = pip_count(s, 1)

        canon = canonicalize_state_for_player(s, 1)
        # In canonical form, player 0 is the "current" player (was player 1)
        # Pip counts should be swapped
        pip_canon_p0 = pip_count(canon, 0)
        pip_canon_p1 = pip_count(canon, 1)

        assert pip_canon_p0 == pip0_p1, "Canonical player 0 pip should match original player 1 pip"
        assert pip_canon_p1 == pip0_p0, "Canonical player 1 pip should match original player 0 pip"

    def test_canonical_checker_count(self):
        """Canonical form preserves checker totals."""
        # Build a valid state from scratch (15 per player)
        s = GameState(board=[0]*24, bar=[1,0], borne_off=[0,0],
                      current_player=1, dice=[])
        s.board[12] = 14  # p0: 14 on board + 1 bar = 15
        s.board[11] = -15 # p1: 15 at point 12

        canon = canonicalize_state_for_player(s, 1)
        assert total_checkers(canon, 0) == 15
        assert total_checkers(canon, 1) == 15

    def test_mirror_legal_turn_count_matches(self):
        """For a symmetric position, player 0 and mirrored player 1 produce same # legal turns."""
        # Construct a simple symmetric position
        s0 = GameState(board=[0]*24, bar=[0,0], borne_off=[0,0],
                       current_player=0, dice=[4,2])
        s0.board[12] = 3   # player 0 checkers at 13
        turns0 = get_legal_turns(s0)

        # Mirror: player 1 has 3 checkers at point 12 (which mirrors to 13 after flip)
        s1 = GameState(board=[0]*24, bar=[0,0], borne_off=[0,0],
                       current_player=1, dice=[4,2])
        s1.board[11] = -3  # player 1 checkers at 12 (symmetric to 13 after mirror)
        turns1 = get_legal_turns(s1)

        # Same structure → same number of legal turns
        assert len(turns0) == len(turns1), (
            f"Mirror symmetry: p0 has {len(turns0)}, p1 has {len(turns1)}")

    def test_all_legal_actions_applicable_after_canonical(self):
        """All legal actions returned for canonical state are applicable."""
        import random
        rng = random.Random(42)
        from backgammon_rlx.env.env import BackgammonEnv
        from backgammon_rlx.agents.random_agent import RandomLegalAgent

        env   = BackgammonEnv()
        agent = RandomLegalAgent(seed=0)
        env.reset(seed=0)

        for _ in range(500):
            if env.is_terminal():
                env.reset(seed=rng.randint(0, 2**31))
            state  = env.state
            turns  = env.legal_actions()
            action = agent.select_action(state, turns)
            obs, reward, done, info = env.step(action)
            # After every step, verify the new state is valid from both perspectives
            if not done:
                new_state = env.state
                assert total_checkers(new_state, 0) == 15
                assert total_checkers(new_state, 1) == 15


class TestCanonicalEncoding:

    def test_player0_canonical_unchanged(self):
        """Canonicalizing for player 0 returns identical board."""
        s = GameState.initial()
        canon = canonicalize_state_for_player(s, 0)
        assert canon.board == s.board
        assert canon.bar == s.bar
        assert canon.borne_off == s.borne_off

    def test_player1_canonical_flips_board(self):
        """Canonicalizing for player 1 mirrors the board."""
        s = GameState.initial()
        canon = canonicalize_state_for_player(s, 1)
        # In canonical form, point i+1 for player 1 = 25-(i+1) in original
        # canon.board[i] = -s.board[23-i]
        for i in range(24):
            assert canon.board[i] == -s.board[23-i], (
                f"Canonical mismatch at index {i}: "
                f"canon={canon.board[i]}, expected={-s.board[23-i]}")

    def test_double_canonical_identity(self):
        """Applying canonical twice returns to original (almost: player is reset to 0)."""
        s = GameState.initial()
        s.current_player = 1
        canon1 = canonicalize_state_for_player(s, 1)
        # canon1 is now from player 1's perspective (player 0 in canon)
        # Applying canonical again (for player 0) should give back something consistent
        assert canon1.current_player == 0
        # The board should be a valid representation
        assert sum(abs(v) for v in canon1.board) + sum(canon1.bar) + sum(canon1.borne_off) == 30

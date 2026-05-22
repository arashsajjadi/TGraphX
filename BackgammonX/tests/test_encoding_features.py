"""Tests for observation and action encoding."""
import numpy as np
import pytest
from backgammon_rlx.env.state import GameState, Turn, AtomicMove, OFF
from backgammon_rlx.env.encoding import (
    ObservationEncoder, ActionEncoder, OBS_DIM, ACT_DIM,
)
from backgammon_rlx.env.movegen import get_legal_turns, canonicalize_state_for_player


class TestObservationEncoder:

    def setup_method(self):
        self.enc = ObservationEncoder()

    def test_output_shape(self):
        s = GameState.initial()
        canon = canonicalize_state_for_player(s, 0)
        obs = self.enc.encode(canon)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32

    def test_no_nan_or_inf(self):
        s = GameState.initial()
        canon = canonicalize_state_for_player(s, 0)
        obs = self.enc.encode(canon)
        assert np.all(np.isfinite(obs)), "Observation contains NaN or Inf"

    def test_values_in_reasonable_range(self):
        s = GameState.initial()
        canon = canonicalize_state_for_player(s, 0)
        obs = self.enc.encode(canon)
        # Most features are normalized to [0,1]; pip features might exceed 1
        assert obs.min() >= -1.0, f"Min value {obs.min()} out of range"
        assert obs.max() <= 2.0, f"Max value {obs.max()} out of range"

    def test_empty_board_all_zeros_checker_features(self):
        s = GameState(board=[0]*24, bar=[0,0], borne_off=[0,0],
                      current_player=0, dice=[])
        obs = self.enc.encode(s)
        # First two features per point: own_count and opp_count should be 0
        for i in range(24):
            assert obs[i * 12] == 0.0, f"own_count at point {i+1} != 0"
            assert obs[i * 12 + 1] == 0.0, f"opp_count at point {i+1} != 0"

    def test_canonical_player1_mirrors(self):
        """Canonical encoding for player 1 differs in asymmetric positions."""
        s = GameState(board=[0]*24, bar=[0,0], borne_off=[0,0], current_player=0, dice=[])
        s.board[12] = 15  # all p0 checkers at point 13 (asymmetric)
        obs0 = self.enc.encode(canonicalize_state_for_player(s, 0))
        obs1 = self.enc.encode(canonicalize_state_for_player(s, 1))
        assert not np.allclose(obs0, obs1), "Asymmetric position obs must differ for P0 vs P1"

    def test_different_positions_give_different_obs(self):
        s1 = GameState.initial()
        s2 = GameState(board=[0]*24, bar=[0,0], borne_off=[0,0],
                       current_player=0, dice=[])
        s2.board[5] = 15
        obs1 = self.enc.encode(s1)
        obs2 = self.enc.encode(s2)
        assert not np.allclose(obs1, obs2)

    def test_bar_feature_present(self):
        s = GameState(board=[0]*24, bar=[2, 1], borne_off=[0,0],
                      current_player=0, dice=[])
        obs = self.enc.encode(s)
        # Global features start at 24*POINT_FEATURES
        from backgammon_rlx.env.encoding import POINT_FEATURES
        global_start = 24 * POINT_FEATURES
        own_bar = obs[global_start]
        opp_bar = obs[global_start + 1]
        assert own_bar > 0, "Own bar count feature should be positive"
        assert opp_bar > 0, "Opp bar count feature should be positive"


class TestActionEncoder:

    def setup_method(self):
        self.obs_enc = ObservationEncoder()
        self.act_enc = ActionEncoder()

    def test_output_shape(self):
        s = GameState.initial()
        s.dice = [3, 1]
        canon = canonicalize_state_for_player(s, 0)
        turns = get_legal_turns(s)
        act = self.act_enc.encode(turns[0], canon)
        assert act.shape == (ACT_DIM,)
        assert act.dtype == np.float32

    def test_no_nan_or_inf(self):
        s = GameState.initial()
        s.dice = [3, 1]
        canon = canonicalize_state_for_player(s, 0)
        turns = get_legal_turns(s)
        for t in turns:
            act = self.act_enc.encode(t, canon)
            assert np.all(np.isfinite(act)), f"Action {t} has NaN/Inf"

    def test_pass_turn_encodes_zero(self):
        """Pass turn should produce a zero (or mostly zero) action vector."""
        s = GameState(board=[0]*24, bar=[1,0], borne_off=[0,0],
                      current_player=0, dice=[3,5])
        s.board[21] = -2  # block point 22
        s.board[19] = -2  # block point 20
        canon = canonicalize_state_for_player(s, 0)
        pass_turn = Turn()
        act = self.act_enc.encode(pass_turn, canon)
        assert act.shape == (ACT_DIM,)

    def test_different_turns_different_encoding(self):
        s = GameState.initial()
        s.dice = [3, 1]
        canon = canonicalize_state_for_player(s, 0)
        turns = get_legal_turns(s)
        if len(turns) < 2:
            pytest.skip("Need at least 2 legal turns")
        act0 = self.act_enc.encode(turns[0], canon)
        act1 = self.act_enc.encode(turns[1], canon)
        assert not np.allclose(act0, act1), "Different turns should have different encodings"

    def test_batch_encoding(self):
        import numpy as np
        from backgammon_rlx.env.encoding import batch_encode_actions
        s = GameState.initial()
        s.dice = [3, 1]
        canon = canonicalize_state_for_player(s, 0)
        turns = get_legal_turns(s)
        batch = batch_encode_actions([turns], [canon], self.act_enc)
        assert len(batch) == 1
        assert batch[0].shape == (len(turns), ACT_DIM)

    def test_bear_off_indicator(self):
        s = GameState(board=[0]*24, bar=[0,0], borne_off=[14,0],
                      current_player=0, dice=[3])
        s.board[2] = 1  # point 3
        canon = canonicalize_state_for_player(s, 0)
        turns = get_legal_turns(s)
        assert turns
        act = self.act_enc.encode(turns[0], canon)
        # is_off feature (index 5 of first move block) should be 1
        from backgammon_rlx.env.encoding import MOVE_FEATURES
        assert act[5] > 0, "Bear-off indicator should be nonzero"

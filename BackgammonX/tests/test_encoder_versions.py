"""Tests for v1 vs v2 encoder compatibility and correctness."""
import numpy as np
import pytest
from backgammon_rlx.env.state import GameState
from backgammon_rlx.env.encoding import (
    ObservationEncoder, ActionEncoder, make_encoders,
    OBS_DIM, ACT_DIM, OBS_DIM_V2, ACT_DIM_V2,
)
from backgammon_rlx.env.movegen import get_legal_turns, canonicalize_state_for_player


class TestEncoderVersions:

    def test_v1_dimensions(self):
        enc = ObservationEncoder(version="v1")
        assert enc.dim == OBS_DIM == 298
        s = GameState.initial()
        obs = enc.encode(canonicalize_state_for_player(s, 0))
        assert obs.shape == (OBS_DIM,)

    def test_v2_dimensions(self):
        enc = ObservationEncoder(version="v2")
        assert enc.dim == OBS_DIM_V2 == 320
        s = GameState.initial()
        obs = enc.encode(canonicalize_state_for_player(s, 0))
        assert obs.shape == (OBS_DIM_V2,)

    def test_v1_action_dimensions(self):
        enc = ActionEncoder(version="v1")
        assert enc.dim == ACT_DIM == 36

    def test_v2_action_dimensions(self):
        enc = ActionEncoder(version="v2")
        assert enc.dim == ACT_DIM_V2 == 52

    def test_v1_v2_first_298_match(self):
        """v2 first 298 dims are identical to v1 (point features + v1 globals)."""
        obs_v1 = ObservationEncoder("v1")
        obs_v2 = ObservationEncoder("v2")
        s = GameState.initial()
        canon = canonicalize_state_for_player(s, 0)
        e1 = obs_v1.encode(canon)
        e2 = obs_v2.encode(canon)
        np.testing.assert_allclose(e1, e2[:OBS_DIM], atol=1e-6,
                                   err_msg="v1 features must be prefix of v2")

    def test_v2_no_nan(self):
        enc = ObservationEncoder("v2")
        s = GameState.initial()
        obs = enc.encode(canonicalize_state_for_player(s, 0))
        assert np.all(np.isfinite(obs)), "v2 obs has NaN or Inf"

    def test_v2_action_no_nan(self):
        obs_enc = ObservationEncoder("v2")
        act_enc = ActionEncoder("v2")
        s = GameState.initial()
        s.dice = [3, 1]
        canon = canonicalize_state_for_player(s, 0)
        for t in get_legal_turns(s):
            act = act_enc.encode(t, canon)
            assert np.all(np.isfinite(act)), f"v2 action NaN for {t}"

    def test_make_encoders_factory(self):
        obs_enc, act_enc = make_encoders("v2")
        assert obs_enc.dim == OBS_DIM_V2
        assert act_enc.dim == ACT_DIM_V2

    def test_invalid_version_raises(self):
        with pytest.raises(AssertionError):
            ObservationEncoder("v3")

    def test_strategic_features_differ_across_positions(self):
        """v2 strategic features change between different board states."""
        enc = ObservationEncoder("v2")
        s1 = GameState.initial()
        s2 = GameState(board=[0]*24, bar=[0,0], borne_off=[14,0],
                       current_player=0, dice=[])
        s2.board[0] = 1
        o1 = enc.encode(s1)
        o2 = enc.encode(s2)
        # Strategic features (indices 288-319) should differ
        assert not np.allclose(o1[OBS_DIM:], o2[OBS_DIM:]), \
            "Strategic features should differ across positions"

    def test_v1_v2_action_size_differs(self):
        """v2 action dim is strictly larger than v1 (different per-move block size)."""
        enc_v1 = ActionEncoder("v1")
        enc_v2 = ActionEncoder("v2")
        assert enc_v2.dim > enc_v1.dim
        s = GameState.initial()
        s.dice = [3, 1]
        canon = canonicalize_state_for_player(s, 0)
        t = get_legal_turns(s)[0]
        a1 = enc_v1.encode(t, canon)
        a2 = enc_v2.encode(t, canon)
        assert a1.shape == (36,) and a2.shape == (52,)

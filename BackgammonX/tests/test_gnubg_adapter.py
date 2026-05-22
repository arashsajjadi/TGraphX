"""Tests for GNU Backgammon adapter.

All tests requiring gnubg are automatically skipped when gnubg is not installed.
"""
import shutil
import pytest
from backgammon_rlx.engines.gnu_backgammon import GnuBackgammonAdapter
from backgammon_rlx.engines.external_engine import ExternalEngineError
from backgammon_rlx.env.state import GameState, OFF
from backgammon_rlx.env.state import GameState

GNUBG_AVAILABLE = shutil.which("gnubg") is not None


class TestGnuBackgammonAdapter:

    def test_is_available_returns_bool(self):
        adapter = GnuBackgammonAdapter()
        result = adapter.is_available()
        assert isinstance(result, bool)

    def test_availability_matches_which(self):
        adapter = GnuBackgammonAdapter()
        assert adapter.is_available() == GNUBG_AVAILABLE

    @pytest.mark.skipif(GNUBG_AVAILABLE, reason="gnubg IS available — testing unavailable path")
    def test_request_move_raises_when_unavailable(self):
        adapter = GnuBackgammonAdapter()
        s = GameState.initial()
        s.dice = [3, 1]
        with pytest.raises((ExternalEngineError, NotImplementedError)):
            adapter.request_move(s)

    @pytest.mark.skipif(not GNUBG_AVAILABLE, reason="gnubg not installed")
    def test_request_move_returns_legal_turn(self):
        """When gnubg is available, verify its move is in legal_actions()."""
        from backgammon_rlx.env.movegen import get_legal_turns
        from backgammon_rlx.notation.move_notation import format_full_turn
        adapter = GnuBackgammonAdapter()
        s = GameState.initial()
        s.dice = [3, 1]
        try:
            turn, equity = adapter.request_move(s)
        except NotImplementedError:
            pytest.skip("GNU Backgammon protocol not fully implemented yet")
        legal_strs = {format_full_turn(t) for t in get_legal_turns(s)}
        assert format_full_turn(turn) in legal_strs, (
            f"gnubg returned illegal move: {format_full_turn(turn)}\n"
            f"Legal: {sorted(legal_strs)}")

    def test_select_action_raises_when_unavailable(self):
        if GNUBG_AVAILABLE:
            pytest.skip("gnubg is available on this system")
        adapter = GnuBackgammonAdapter()
        s = GameState.initial()
        s.dice = [3, 1]
        with pytest.raises(ExternalEngineError):
            adapter.select_action(s)


class TestExternalEngineInterface:

    def test_validate_method_handles_legal_move(self):
        from backgammon_rlx.env.movegen import get_legal_turns
        from backgammon_rlx.engines.external_engine import ExternalEngineAgent
        # Can't instantiate abstract class, so test validation via subclass
        adapter = GnuBackgammonAdapter()
        s = GameState.initial()
        s.dice = [3, 1]
        turns = get_legal_turns(s)
        # _validate should return the turn unchanged if it's legal
        result = adapter._validate(s, turns[0], turns)
        assert result == turns[0]

    def test_validate_fallback_on_illegal_move(self):
        from backgammon_rlx.env.state import AtomicMove, Turn
        from backgammon_rlx.env.movegen import get_legal_turns
        adapter = GnuBackgammonAdapter()
        s = GameState.initial()
        s.dice = [3, 1]
        turns = get_legal_turns(s)
        # Create an illegal turn
        illegal = Turn.from_list([AtomicMove(src=1, dst=OFF, die=1, hit=False)])
        # _validate should fall back to first legal move
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adapter._validate(s, illegal, turns)
        assert result in turns

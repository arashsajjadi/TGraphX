"""List all legal moves from the standard opening position with dice [3,1]."""
import sys
sys.path.insert(0, "src")

from backgammon_rlx.env.state import GameState
from backgammon_rlx.env.movegen import get_legal_turns
from backgammon_rlx.notation.move_notation import format_full_turn

s = GameState.initial()
s.dice = [3, 1]
turns = get_legal_turns(s)
print(f"Initial position, dice [3,1]: {len(turns)} legal turns")
for i, t in enumerate(turns, 1):
    print(f"  {i:2d}. {format_full_turn(t)}")

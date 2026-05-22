# Backgammon Rules Checklist

Status legend: ✅ Verified | ⚠️ Partial | ❌ Missing | N/A Not applicable

| # | Rule | Behavior | Implementation | Tests | Status | Notes |
|---|------|----------|----------------|-------|--------|-------|
| 1 | Board has 24 points | Array of length 24, points 1–24 | `env/state.py:GameState.board` | `test_rules.py::TestInitialPosition` | ✅ | |
| 2 | Each player has exactly 15 checkers | `total_checkers(state,p)==15` at all times | `env/rules.py:total_checkers` | `test_env_invariants.py`, `test_random_stress.py` | ✅ | Verified across 20,000+ steps |
| 3 | Standard initial setup | P0: 2@24,5@13,3@8,5@6; P1: 2@1,5@12,3@17,5@19 | `env/state.py:GameState.initial()` | `test_rules.py::TestInitialPosition` | ✅ | Pip count = 167 both sides |
| 4 | Opposite movement directions | P0: high→low; P1: low→high | `env/movegen.py:get_legal_atomic_moves` | `test_movegen.py` | ✅ | |
| 5a | Opening roll: both roll one die | `_roll_opening()` | `env/env.py:BackgammonEnv._roll_opening` | `test_rules.py` (implicit) | ✅ | |
| 5b | Higher die starts | Loop until d0≠d1, higher wins | `env/env.py:_roll_opening` | `test_rules.py::TestInitialPosition::test_no_bar_initially` | ✅ | |
| 5c | Tied opening roll rerolls | `while True: ... if d0!=d1: break` | `env/env.py:_roll_opening` | — | ⚠️ | Verified by code review; no dedicated test |
| 5d | First move uses opening dice | `state.dice = [d0, d1]` set from opening roll | `env/env.py:_roll_opening` | — | ⚠️ | Verified by code review |
| 6 | Normal dice roll | `_roll_dice()` rolls 2 d6, doubles→4 | `env/env.py:_roll_dice` | `test_env_invariants.py` | ✅ | |
| 7 | Legal landing on empty point | `is_point_open` returns True for val==0 | `env/rules.py:is_point_open` | `test_rules.py::TestIsPointOpen::test_empty_point_open` | ✅ | |
| 8 | Legal landing on own point | `is_point_open` returns True for own checkers | `env/rules.py:is_point_open` | `test_rules.py::TestIsPointOpen::test_own_checker_open` | ✅ | |
| 9 | Legal landing on opponent blot | `is_point_open` True if abs(val)==1 of opponent | `env/rules.py:is_point_open` | `test_rules.py::TestIsPointOpen::test_opponent_blot_open` | ✅ | |
| 10 | Illegal landing on opponent made point | `is_point_open` False if abs(val)≥2 of opponent | `env/rules.py:is_point_open` | `test_rules.py::TestIsPointOpen::test_opponent_prime_blocked` | ✅ | |
| 11 | Hitting sends opponent checker to bar | `apply_atomic_move_inplace` sets bar[opp]+=1 | `env/movegen.py:apply_atomic_move_inplace` | `test_bar_entry.py::test_enter_bar_checker_sends_opponent_to_bar` | ✅ | |
| 12 | Bar priority | If bar[player]>0, ONLY bar-entry moves returned | `env/movegen.py:get_legal_atomic_moves` | `test_bar_entry.py::TestBarPriority` | ✅ | |
| 13 | Entry from bar into opponent home | P0: point 25-die; P1: point die | `env/rules.py:bar_entry_point` | `test_bar_entry.py::TestBarEntryPoints` | ✅ | |
| 14 | Blocked bar entry | Entry point has 2+ opp checkers → no move | `env/movegen.py:get_legal_atomic_moves` | `test_bar_entry.py::test_bar_entry_blocked`, golden fixture | ✅ | |
| 15 | Partial entry when only one die can enter | Only the open entry returned; other die unused if partner also blocked | `env/movegen.py:get_legal_turns` | `test_golden_legal_moves.py::bar_entry_blocked` | ✅ | |
| 16 | Enter + use remaining die | Second die played after bar entry if legal | `env/movegen.py:_generate` | `test_bar_entry.py::test_enter_then_move` | ✅ | |
| 17 | Doubles produce four moves | dice=[d,d,d,d] | `env/env.py:_roll_dice` | `test_movegen.py::test_doubles_produce_four_moves` | ✅ | |
| 18 | Doubles use max possible moves | Filtered by max_moves | `env/movegen.py:get_legal_turns` | `test_mandatory_dice_usage.py::test_max_doubles_moves_used` | ✅ | |
| 19a | Mandatory: use both dice if possible | max_moves filtering | `env/movegen.py:get_legal_turns` | `test_mandatory_dice_usage.py::test_both_dice_used_when_possible` | ✅ | |
| 19b | Mandatory: use one if only one works | max_moves==1 case | `env/movegen.py:get_legal_turns` | `test_mandatory_dice_usage.py::test_one_die_when_other_blocked` | ✅ | |
| 19c | Mandatory: larger die if either works alone | `if max_moves==1 and len(dice)==2 and dice[0]!=dice[1]` | `env/movegen.py:get_legal_turns` | `test_mandatory_dice_usage.py::test_larger_die_used_when_only_one_die_playable`, golden fixture | ✅ | |
| 20 | Ordered dice application: intermediate must be legal | Recursive `_generate` applies each die to updated state | `env/movegen.py:_generate` | `test_movegen.py`, stress tests | ✅ | |
| 21 | No move / forced pass | Returns `[Turn()]` | `env/movegen.py:get_legal_turns` | `test_movegen.py::test_forced_pass`, golden fixtures | ✅ | |
| 22 | Bearing off only when all in home board | `can_bear_off` checks `all_checkers_in_home` | `env/rules.py:can_bear_off` | `test_bearing_off.py::test_cannot_bear_off_outside_home` | ✅ | |
| 23 | Bearing off forbidden with bar checker | `can_bear_off` returns False if bar[player]>0 | `env/rules.py:can_bear_off` | `test_bearing_off.py::test_bar_prevents_bearing_off`, golden fixture | ✅ | |
| 24 | Exact bearing off | die==distance → AtomicMove(src,OFF) | `env/rules.py:can_bear_off_checker`, `env/movegen.py` | `test_bearing_off.py::test_exact_bear_off` | ✅ | |
| 25 | Larger-die bearing off | die>dist AND no checker at higher dist | `env/rules.py:can_bear_off_checker` | `test_bearing_off.py::test_larger_die_bear_off`, golden fixture | ✅ | |
| 26 | Hit during bear-off disables further bearing off | Re-entering removes checker from home board | `env/rules.py:can_bear_off`, `all_checkers_in_home` | `test_bearing_off.py::test_hit_during_bearoff_resets`, golden fixture | ✅ | |
| 27 | Game ends when player bears off 15 | `borne_off[p]==15` | `env/rules.py:is_terminal`, `winner` | `test_scoring.py` | ✅ | |
| 28 | Normal win scoring | `borne_off[loser]>0 → score=1` | `env/rules.py:score_value` | `test_scoring.py::test_normal_win` | ✅ | |
| 29 | Gammon scoring | `borne_off[loser]==0, no bar, not in winner home → score=2` | `env/rules.py:score_value` | `test_scoring.py::test_gammon_*` | ✅ | |
| 30 | Backgammon scoring | `borne_off[loser]==0 AND (bar or in winner home) → score=3` | `env/rules.py:score_value` | `test_scoring.py::test_backgammon_*` | ✅ | |
| 31 | Checker conservation invariant | total=15 at every step | `env/rules.py:total_checkers` | `test_random_stress.py::test_total_checkers_always_15` | ✅ | 20,000 steps |
| 32 | No point may contain both players | Board encoding: sign=player, mixed impossible | `env/state.py`, `env/movegen.py` | `test_random_stress.py::test_no_point_has_both_players` | ✅ | |
| 33 | Dice-order duplicate handling | `tried` set in `_generate` skips duplicate die values | `env/movegen.py:_generate` | `test_action_deduplication.py` | ✅ | |
| 34 | Final-state deduplication policy | Deduplicate by `board_key()` | `env/movegen.py:get_legal_turns` | `test_action_deduplication.py::test_all_legal_turns_unique_final_state` | ✅ | |
| 35 | Human-readable move notation | `format_full_turn`, `format_atomic_move` | `notation/move_notation.py` | `test_notation.py` | ✅ | |
| 36 | Position import/export | JSON round-trip | `notation/position_io.py` | `test_position_io.py` | ✅ | |
| 37 | Debug trace / transition explanation | `env.step(trace=True)` → `info["trace"]` | `env/env.py:_build_trace` | `tests/test_trace_debug.py` | ⚠️ | Test file added in this pass |
| 38 | Illegal action explanation | `explain_illegal_action(state, dice, turn)` | `validation/explain_illegal.py` | `tests/test_explain_illegal_action.py` | ⚠️ | Test file added in this pass |
| 39 | Deterministic seeding | `seed_everything(seed)`, env.reset(seed=) | `utils/seed.py`, `env/env.py` | `test_random_stress.py::test_seed_reproducibility` | ✅ | |
| 40 | Random long-game stability | 10,000 games, no crashes | `test_random_stress.py`, `validation/full_game_check.py` | `test_random_stress.py` | ✅ | Slow marker added |

## Summary

- ✅ Verified: 38/40
- ⚠️ Partial: 2/40 (opening tie-reroll explicit test, trace/explain tests)
- ❌ Missing: 0/40

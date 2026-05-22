"""Shallow stochastic expectimax search for backgammon.

Implements value-guided expectimax over dice outcomes with configurable
depth.  At leaf nodes the neural value function is used.

Dice outcome enumeration is exact for depth 1 (36 outcomes); for deeper
search a subset of outcomes is sampled stochastically.

This module is intentionally lightweight.  Full MCTS / deeper search
can be layered on top of the ValueEstimator interface.
"""
from __future__ import annotations

import itertools
import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..env.state import GameState, Turn
from ..env.movegen import get_legal_turns, apply_full_turn
from ..env.rules import is_terminal
from ..env.encoding import ObservationEncoder, ActionEncoder
from ..env.movegen import canonicalize_state_for_player


# All 21 distinct unordered dice pairs (includes doubles)
ALL_DICE_ROLLS: List[Tuple[int, int]] = [
    (d1, d2) for d1 in range(1, 7) for d2 in range(d1, 7)
]
DICE_WEIGHTS: List[float] = [
    (1 if d1 == d2 else 2) / 36.0 for d1, d2 in ALL_DICE_ROLLS
]


@torch.no_grad()
def _leaf_value(
    model,
    state: GameState,
    player: int,
    obs_enc: ObservationEncoder,
    act_enc: ActionEncoder,
    device: torch.device,
    dummy_turns: Optional[List[Turn]] = None,
) -> float:
    """Evaluate *state* using the value head of *model*."""
    canon = canonicalize_state_for_player(state, player)
    obs   = torch.tensor(obs_enc.encode(canon),
                          dtype=torch.float32, device=device).unsqueeze(0)
    turns = dummy_turns or get_legal_turns(state)
    if not turns:
        turns_for_enc = [Turn()]
    else:
        turns_for_enc = [turns[0]]   # only need a single action to get value

    act_arr = np.stack([act_enc.encode(t, canon) for t in turns_for_enc], 0)
    act_t   = torch.tensor(act_arr, dtype=torch.float32,
                             device=device).unsqueeze(0)
    _, value, _aux = model(obs, act_t)
    return float(value.squeeze().item())


def expectimax_action(
    model,
    state:   GameState,
    depth:   int,
    obs_enc: ObservationEncoder,
    act_enc: ActionEncoder,
    device:  torch.device,
    n_dice_samples: int = 36,   # use all 21 exact, or sample subset
) -> Turn:
    """Select best action using shallow expectimax.

    Depth 1: evaluate legal actions by leaf value (no future dice).
    Depth 2: for each legal action, average value over all dice outcomes
             and take the best opposing action (one level of opponent).
    """
    player = state.current_player
    turns  = get_legal_turns(state)
    if not turns:
        return Turn()
    if len(turns) == 1:
        return turns[0]

    best_turn  = turns[0]
    best_score = float("-inf")

    for turn in turns:
        after = apply_full_turn(state, turn)
        if is_terminal(after):
            from ..env.rules import score_value
            score = float(score_value(after))
            return turn  # winning move found

        if depth <= 1:
            # Value from OUR perspective: opponent's value is negated
            v = -_leaf_value(model, after, 1 - player, obs_enc, act_enc, device)
        else:
            v = _expectimax_value(model, after, depth - 1,
                                   player, obs_enc, act_enc, device,
                                   n_dice_samples)

        if v > best_score:
            best_score = v
            best_turn  = turn

    return best_turn


def _expectimax_value(
    model, state: GameState, depth: int, original_player: int,
    obs_enc, act_enc, device, n_dice_samples: int,
) -> float:
    """Recursively compute expectimax value (from original_player's perspective)."""
    if is_terminal(state):
        from ..env.rules import score_value, winner
        w = winner(state)
        s = float(score_value(state))
        return s if w == original_player else -s

    current = state.current_player

    # Sample or use all dice outcomes
    if n_dice_samples >= 36:
        dice_pairs  = ALL_DICE_ROLLS
        weights     = DICE_WEIGHTS
    else:
        idx         = random.choices(range(len(ALL_DICE_ROLLS)),
                                     weights=DICE_WEIGHTS, k=n_dice_samples)
        dice_pairs  = [ALL_DICE_ROLLS[i] for i in idx]
        weights     = [1.0 / n_dice_samples] * n_dice_samples

    total = 0.0
    for (d1, d2), w in zip(dice_pairs, weights):
        dice_state = state.clone()
        dice_state.dice = [d1, d1, d1, d1] if d1 == d2 else [d1, d2]

        turns = get_legal_turns(dice_state)
        if not turns:
            # Pass turn
            next_state = apply_full_turn(dice_state, Turn())
            if depth <= 1:
                sign = 1 if next_state.current_player == original_player else -1
                v = sign * _leaf_value(model, next_state, next_state.current_player,
                                        obs_enc, act_enc, device)
            else:
                v = _expectimax_value(model, next_state, depth - 1,
                                       original_player, obs_enc, act_enc,
                                       device, n_dice_samples)
        else:
            # Max over legal moves
            best = float("-inf")
            for turn in turns:
                after = apply_full_turn(dice_state, turn)
                if is_terminal(after):
                    from ..env.rules import score_value, winner
                    sv = float(score_value(after))
                    mv = sv if winner(after) == original_player else -sv
                    best = max(best, mv)
                    break
                if depth <= 1:
                    sign = 1 if after.current_player != original_player else -1
                    cv = sign * _leaf_value(model, after, after.current_player,
                                             obs_enc, act_enc, device)
                else:
                    cv = _expectimax_value(model, after, depth - 1,
                                            original_player, obs_enc, act_enc,
                                            device, n_dice_samples)
                best = max(best, cv)
            v = best

        total += w * v

    return total

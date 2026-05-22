"""Self-play rollout collection.

Runs N games of self-play using the current policy, collecting transitions
for both players into a shared RolloutBuffer.

This is the synchronous single-process version.  A multiprocessing version
(SelfPlayWorker) is in self_play.py.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Optional

from ..env.state import GameState, Turn
from ..env.env import BackgammonEnv
from ..env.movegen import get_legal_turns, canonicalize_state_for_player
from ..env.encoding import ObservationEncoder, ActionEncoder
from ..env.rules import is_terminal
from .buffer import RolloutBuffer, Transition


@torch.no_grad()
def collect_rollouts(
    model,
    n_games: int,
    obs_enc: ObservationEncoder,
    act_enc: ActionEncoder,
    device: torch.device,
    buffer: RolloutBuffer,
    seed: Optional[int] = None,
    strict_invariants: bool = False,
) -> dict:
    """Play *n_games* self-play games and fill *buffer*.

    Both players share *model* (symmetric self-play).
    Returns summary statistics.
    """
    model.eval()
    total_steps = 0
    game_lengths: list = []
    scores: list = []

    rng = np.random.default_rng(seed)
    env = BackgammonEnv(obs_enc, act_enc,
                        strict_invariants=strict_invariants)

    for _ in range(n_games):
        obs = env.reset(seed=int(rng.integers(0, 2**31)))
        done = False
        game_trans: list = []

        while not done:
            state  = env.state
            player = state.current_player
            canon  = canonicalize_state_for_player(state, player)
            turns  = get_legal_turns(state)

            if not turns:
                turns = [Turn()]

            obs_t = torch.tensor(obs_enc.encode(canon),
                                 dtype=torch.float32, device=device).unsqueeze(0)
            n_acts = len(turns)
            act_arr = np.stack(
                [act_enc.encode(t, canon) for t in turns], axis=0
            )
            act_t = torch.tensor(act_arr, dtype=torch.float32,
                                  device=device).unsqueeze(0)   # [1, N, A]

            logits, value_t, _ = model(obs_t, act_t)
            logits  = logits.squeeze(0)               # [N]
            value_v = float(value_t.squeeze().item())

            log_probs = torch.log_softmax(logits, dim=-1)
            probs     = torch.exp(log_probs)
            act_idx   = int(torch.multinomial(probs, 1).item())
            log_prob  = float(log_probs[act_idx].item())

            chosen = turns[act_idx]
            next_obs, reward, done, info = env.step(chosen)

            trans = Transition(
                obs=obs_enc.encode(canon),
                act_feats=act_arr,
                act_idx=act_idx,
                n_actions=n_acts,
                log_prob=log_prob,
                value=value_v,
                reward=reward,
                done=done,
            )
            game_trans.append(trans)
            obs = next_obs
            total_steps += 1

        # Award the loser their penalty on their last transition
        if game_trans and info.get("winner") is not None:
            w     = info["winner"]
            score = info["score"]
            # Find the last transition where the loser was the current player
            # The loser's last move is the second-to-last transition if winner
            # acted last.  We iterate backwards to assign -score to the first
            # loser transition we find from the end.
            for i in reversed(range(len(game_trans) - 1)):
                # We cannot check current_player per transition; instead we use
                # the alternating structure: transitions alternate players.
                # The last transition was the winner's winning move (reward=+score).
                # The one before was the loser's last chance (assign -score).
                if game_trans[i].reward == 0.0:
                    game_trans[i] = Transition(
                        obs=game_trans[i].obs,
                        act_feats=game_trans[i].act_feats,
                        act_idx=game_trans[i].act_idx,
                        n_actions=game_trans[i].n_actions,
                        log_prob=game_trans[i].log_prob,
                        value=game_trans[i].value,
                        reward=-float(score),
                        done=True,  # treat as terminal for GAE
                    )
                    break

        for t in game_trans:
            buffer.append(t)

        game_lengths.append(info.get("game_length", len(game_trans)))
        scores.append(info.get("score", 0))

    return {
        "total_steps":  total_steps,
        "mean_length":  float(np.mean(game_lengths)),
        "mean_score":   float(np.mean(scores)),
    }

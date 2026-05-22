"""Multiprocess self-play rollout for RTX 5080 + i7-14700F.

Architecture
------------
  Worker processes (N)    ──req──>  GPU Inference Thread  ──resp──>  Workers
                          <──resp──                        <──req──
  Workers  ──game_data──>  Main process  (RolloutBuffer + PPO update)

Workers run on CPU (game simulation + encoding).
Inference thread runs in the main process with GPU access (no CUDA fork issue).
Communication uses mp.Queue (OS-level IPC, pickle serialization).

Concurrency notes
-----------------
- Workers are spawned with the 'fork' start method (Linux default), inheriting
  the parent's Python environment without CUDA contexts.
- The inference thread is a daemon thread in the main process; it reads the
  shared request queue and writes to per-worker response queues.
- Workers block on resp_queue.get() after each inference request.
- Main process collects completed game data from traj_queue.
"""
from __future__ import annotations

import multiprocessing as mp
import threading
import time
import queue
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..env.env import BackgammonEnv
from ..env.encoding import ObservationEncoder, ActionEncoder
from ..env.movegen import get_legal_turns, canonicalize_state_for_player
from ..env.state import Turn
from .buffer import RolloutBuffer, Transition


# ---------------------------------------------------------------------------
# Worker process function
# ---------------------------------------------------------------------------

def _worker_fn(
    worker_id:          int,
    n_games_per_round:  int,
    seed:               int,
    req_queue:          mp.Queue,     # shared: workers → inference thread
    resp_queue:         mp.Queue,     # per-worker: inference thread → this worker
    traj_queue:         mp.Queue,     # shared: workers → main process
    cmd_queue:          mp.Queue,     # per-worker: main → this worker
    strict_invariants:  bool = False,
) -> None:
    """Worker process body.  Runs games and communicates with inference thread."""
    # Recreate encoders (fresh in forked/spawned process)
    obs_enc = ObservationEncoder()
    act_enc = ActionEncoder()
    env     = BackgammonEnv(obs_enc, act_enc, strict_invariants=strict_invariants)
    rng     = np.random.default_rng(seed + worker_id * 97)

    while True:
        # Wait for a command from main
        try:
            cmd = cmd_queue.get(timeout=60)
        except Exception:
            break

        if cmd == "stop":
            break

        if cmd != "collect":
            continue

        for _ in range(n_games_per_round):
            game_obs = env.reset(seed=int(rng.integers(0, 2**31)))
            game_trans: List[Transition] = []

            while not env.is_terminal():
                state  = env.state
                player = state.current_player
                canon  = canonicalize_state_for_player(state, player)
                turns  = get_legal_turns(state)

                if not turns:
                    turns = [Turn()]

                obs_arr = obs_enc.encode(canon)
                act_arr = np.stack([act_enc.encode(t, canon) for t in turns], axis=0)
                n_acts  = len(turns)

                # Send inference request
                req_queue.put({
                    "worker_id": worker_id,
                    "obs":       obs_arr,
                    "act_feats": act_arr,
                    "n_actions": n_acts,
                })

                # Block until response arrives
                resp = resp_queue.get()
                act_idx  = resp["act_idx"]
                log_prob = resp["log_prob"]
                value    = resp["value"]

                chosen = turns[act_idx]
                _, reward, done, info = env.step(chosen)

                game_trans.append(Transition(
                    obs=obs_arr,
                    act_feats=act_arr,
                    act_idx=act_idx,
                    n_actions=n_acts,
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                    done=done,
                ))

            # Assign loser's terminal reward at their last transition
            if game_trans and info.get("winner") is not None:
                score = float(info.get("score", 1))
                for i in reversed(range(len(game_trans) - 1)):
                    if game_trans[i].reward == 0.0:
                        t = game_trans[i]
                        game_trans[i] = Transition(
                            obs=t.obs, act_feats=t.act_feats,
                            act_idx=t.act_idx, n_actions=t.n_actions,
                            log_prob=t.log_prob, value=t.value,
                            reward=-score, done=True,
                        )
                        break

            traj_queue.put({
                "transitions": game_trans,
                "info":        info,
            })

        # Signal round completion
        traj_queue.put({"done": True, "worker_id": worker_id})


# ---------------------------------------------------------------------------
# GPU inference thread (runs in main process)
# ---------------------------------------------------------------------------

class _InferenceThread(threading.Thread):
    """Daemon thread: batches worker requests → GPU forward → sends responses."""

    def __init__(
        self,
        model:            nn.Module,
        device:           torch.device,
        req_queue:        mp.Queue,
        resp_queues:      Dict[int, mp.Queue],
        batch_size:       int   = 256,
        max_wait_ms:      float = 5.0,
        use_amp:          bool  = True,
    ) -> None:
        super().__init__(daemon=True, name="InferenceThread")
        self.model       = model
        self.device      = device
        self.req_queue   = req_queue
        self.resp_queues = resp_queues
        self.batch_size  = batch_size
        self.max_wait    = max_wait_ms / 1000.0
        self.use_amp     = use_amp
        self._stop       = threading.Event()
        self.total_inferences = 0

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        self.model.eval()
        from ..env.encoding import OBS_DIM, ACT_DIM

        while not self._stop.is_set():
            batch: List[Dict] = []
            deadline = time.monotonic() + self.max_wait

            # Collect requests up to batch_size or timeout
            while len(batch) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self.req_queue.get(timeout=min(remaining, 0.002))
                    batch.append(item)
                except Exception:
                    break

            if not batch:
                continue

            B     = len(batch)
            max_n = max(r["n_actions"] for r in batch)

            obs_t  = torch.zeros(B, OBS_DIM, dtype=torch.float32, device=self.device)
            act_t  = torch.zeros(B, max_n, ACT_DIM, dtype=torch.float32, device=self.device)
            mask_t = torch.zeros(B, max_n, dtype=torch.bool, device=self.device)

            for k, req in enumerate(batch):
                obs_t[k]  = torch.from_numpy(req["obs"])
                n         = req["n_actions"]
                act_t[k, :n] = torch.from_numpy(req["act_feats"])
                mask_t[k, :n] = True

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=self.use_amp and self.device.type == "cuda"):
                    logits, values = self.model(obs_t, act_t, mask=mask_t)

            log_probs = torch.log_softmax(logits, dim=-1)
            probs     = log_probs.exp()

            for k, req in enumerate(batch):
                n      = req["n_actions"]
                p      = probs[k, :n]
                lp     = log_probs[k, :n]
                ai     = int(torch.multinomial(p, 1).item())
                lp_val = float(lp[ai].item())
                v      = float(values[k].item())

                self.resp_queues[req["worker_id"]].put({
                    "act_idx":  ai,
                    "log_prob": lp_val,
                    "value":    v,
                })

            self.total_inferences += B


# ---------------------------------------------------------------------------
# Multiprocess rollout collector
# ---------------------------------------------------------------------------

class MultiprocessRolloutCollector:
    """Parallel self-play: N workers + 1 GPU inference thread.

    Usage
    -----
    collector = MultiprocessRolloutCollector(model, n_workers=12, ...)
    collector.start()
    for update in range(total_updates):
        buffer, stats = collector.collect(n_games=128)
        metrics = ppo.update(buffer)
    collector.stop()
    """

    def __init__(
        self,
        model:             nn.Module,
        n_workers:         int,
        obs_enc:           ObservationEncoder,
        act_enc:           ActionEncoder,
        device:            torch.device,
        seed:              int   = 0,
        inference_batch_size:  int   = 256,
        inference_max_wait_ms: float = 5.0,
        use_amp:           bool  = True,
        strict_invariants: bool  = False,
        gamma:             float = 0.995,
        gae_lambda:        float = 0.95,
    ) -> None:
        self.model       = model
        self.n_workers   = n_workers
        self.obs_enc     = obs_enc
        self.act_enc     = act_enc
        self.device      = device
        self.seed        = seed
        self.use_amp     = use_amp
        self.strict_inv  = strict_invariants
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.inf_batch   = inference_batch_size
        self.inf_wait_ms = inference_max_wait_ms

        self._workers:   List[mp.Process] = []
        self._cmd_qs:    List[mp.Queue]   = []
        self._resp_qs:   List[mp.Queue]   = []
        self._req_q:     Optional[mp.Queue] = None
        self._traj_q:    Optional[mp.Queue] = None
        self._inf_thread: Optional[_InferenceThread] = None
        self._started    = False

    def start(self) -> None:
        """Spawn workers and start inference thread."""
        if self._started:
            return

        # Use 'spawn' to avoid inheriting CUDA context from parent process.
        # Workers are clean Python processes with no GPU memory.
        self._req_q  = mp.Queue(maxsize=self.n_workers * 32)
        self._traj_q = mp.Queue(maxsize=self.n_workers * 512)
        self._resp_qs  = [mp.Queue(maxsize=64) for _ in range(self.n_workers)]
        self._cmd_qs   = [mp.Queue(maxsize=4)  for _ in range(self.n_workers)]

        # Ensure model is on the target device
        self.model = self.model.to(self.device)

        # Start inference thread before forking workers
        self._inf_thread = _InferenceThread(
            model=self.model,
            device=self.device,
            req_queue=self._req_q,
            resp_queues={i: self._resp_qs[i] for i in range(self.n_workers)},
            batch_size=self.inf_batch,
            max_wait_ms=self.inf_wait_ms,
            use_amp=self.use_amp,
        )
        self._inf_thread.start()

        for i in range(self.n_workers):
            p = mp.Process(
                target=_worker_fn,
                args=(
                    i,
                    1,              # n_games_per_round (set per-collect call via cmd)
                    self.seed,
                    self._req_q,
                    self._resp_qs[i],
                    self._traj_q,
                    self._cmd_qs[i],
                    self.strict_inv,
                ),
                daemon=True,
                name=f"SelfPlayWorker-{i}",
            )
            p.start()
            self._workers.append(p)

        self._started = True

    def collect(self, n_games: int) -> Tuple[RolloutBuffer, Dict]:
        """Collect *n_games* games across workers.  Returns (buffer, stats)."""
        if not self._started:
            self.start()

        games_per_worker = max(1, n_games // self.n_workers)
        # Update workers' games-per-round via a new _worker_fn param isn't possible
        # post-spawn, so we send 'collect' N times and rely on 1 game/round.
        # For simplicity: each worker gets games_per_worker rounds of 1 game.
        # Send 'collect' to each worker games_per_worker times.
        for _ in range(games_per_worker):
            for i in range(self.n_workers):
                self._cmd_qs[i].put("collect")

        # Collect results
        buffer = RolloutBuffer(gamma=self.gamma, gae_lambda=self.gae_lambda)
        games_collected = 0
        game_lengths: List[int] = []
        scores: List[float] = []
        total_steps = 0
        workers_done = {i: 0 for i in range(self.n_workers)}

        target_dones = games_per_worker  # each worker sends 1 'done' per round

        while sum(workers_done.values()) < self.n_workers * target_dones:
            try:
                item = self._traj_q.get(timeout=60.0)
            except Exception:
                # Timeout — check if workers are alive
                alive = sum(1 for w in self._workers if w.is_alive())
                if alive == 0:
                    raise RuntimeError("All self-play workers died unexpectedly")
                continue

            if item.get("done"):
                workers_done[item["worker_id"]] += 1
                continue

            trans: List[Transition] = item["transitions"]
            info = item.get("info", {})
            for t in trans:
                buffer.append(t)
                total_steps += 1
            games_collected += 1
            game_lengths.append(info.get("game_length", len(trans)))
            scores.append(float(info.get("score", 0)))

        return buffer, {
            "games_collected": games_collected,
            "total_steps":     total_steps,
            "mean_length":     float(np.mean(game_lengths)) if game_lengths else 0.0,
            "mean_score":      float(np.mean(scores)) if scores else 0.0,
            "inf_total":       self._inf_thread.total_inferences,
        }

    def stop(self) -> None:
        """Cleanly shut down workers and inference thread."""
        for q in self._cmd_qs:
            try:
                q.put("stop", timeout=1.0)
            except Exception:
                pass
        for w in self._workers:
            w.join(timeout=5.0)
            if w.is_alive():
                w.terminate()
        if self._inf_thread:
            self._inf_thread.stop()
            self._inf_thread.join(timeout=3.0)
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

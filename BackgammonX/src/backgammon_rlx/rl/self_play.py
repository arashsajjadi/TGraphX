"""Symmetric self-play training loop.

Supports two rollout modes:
  "single"       – synchronous single-process (default for CPU/debug)
  "multiprocess" – N worker processes + GPU inference thread (for RTX 5080)

The mode is selected via config["rollout_mode"].
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import numpy as np

from ..env.encoding import ObservationEncoder, ActionEncoder
from ..rl.buffer import RolloutBuffer
from ..rl.rollout import collect_rollouts
from ..rl.ppo import PPOTrainer
from ..utils.checkpoint import save_checkpoint
from ..utils.logging import TrainingLogger


class SelfPlayTrainer:

    def __init__(
        self,
        model,
        optimizer:         torch.optim.Optimizer,
        scheduler:         Optional[object],
        config:            Dict[str, Any],
        device:            torch.device,
        run_dir:           Path,
        obs_enc:           ObservationEncoder,
        act_enc:           ActionEncoder,
    ) -> None:
        self.model      = model
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.cfg        = config
        self.device     = device
        self.run_dir    = run_dir
        self.obs_enc    = obs_enc
        self.act_enc    = act_enc

        self.ppo = PPOTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            clip_range=config.get("clip_range", 0.2),
            value_coef=config.get("value_coef", 0.5),
            entropy_coef=config.get("entropy_coef", 0.01),
            max_grad_norm=config.get("max_grad_norm", 1.0),
            n_epochs=config.get("update_epochs", 4),
            minibatch_size=config.get("minibatch_size", 512),
            use_amp=config.get("mixed_precision", True),
            target_kl=config.get("target_kl", None),
        )

        self.buffer = RolloutBuffer(
            gamma=config.get("gamma", 0.995),
            gae_lambda=config.get("gae_lambda", 0.95),
        )

        self.logger = TrainingLogger(run_dir / "logs",
                                     tb=config.get("tensorboard", False))
        self.global_step   = 0
        self.games_played  = 0
        self.update_count  = 0

        # Multiprocess collector (lazy-init)
        self._mp_collector = None

    def _get_mp_collector(self):
        if self._mp_collector is None:
            from ..rl.multiprocess_rollout import MultiprocessRolloutCollector
            self._mp_collector = MultiprocessRolloutCollector(
                model=self.model,
                n_workers=self.cfg.get("num_self_play_workers", 8),
                obs_enc=self.obs_enc,
                act_enc=self.act_enc,
                device=self.device,
                seed=self.cfg.get("seed", 0),
                inference_batch_size=self.cfg.get("inference_batch_size", 256),
                inference_max_wait_ms=self.cfg.get("inference_max_wait_ms", 5.0),
                use_amp=self.cfg.get("mixed_precision", True),
                strict_invariants=self.cfg.get("strict_invariants", False),
                gamma=self.cfg.get("gamma", 0.995),
                gae_lambda=self.cfg.get("gae_lambda", 0.95),
            )
            self._mp_collector.start()
        return self._mp_collector

    def train(self, total_games: int) -> None:
        rollout_mode     = self.cfg.get("rollout_mode", "single")
        games_per_update = self.cfg.get("rollout_games_per_update", 64)
        ckpt_interval    = self.cfg.get("checkpoint_interval", 1000)
        log_interval     = self.cfg.get("log_interval", 10)

        t0 = time.time()
        try:
            while self.games_played < total_games:
                # --- collect rollouts ---
                if rollout_mode == "multiprocess":
                    collector = self._get_mp_collector()
                    self.buffer, stats = collector.collect(games_per_update)
                else:
                    # single-process fallback
                    self.buffer.clear()
                    stats = collect_rollouts(
                        model=self.model,
                        n_games=games_per_update,
                        obs_enc=self.obs_enc,
                        act_enc=self.act_enc,
                        device=self.device,
                        buffer=self.buffer,
                        seed=self.games_played,
                    )

                n_games = stats.get("games_collected", games_per_update)
                self.games_played += n_games
                self.global_step  += stats.get("total_steps", len(self.buffer))

                # --- PPO update ---
                metrics = self.ppo.update(self.buffer)
                self.update_count += 1

                if self.scheduler is not None:
                    self.scheduler.step()

                # --- log ---
                if self.update_count % log_interval == 0:
                    elapsed = time.time() - t0
                    gps = self.games_played / elapsed
                    record = {
                        "update":       self.update_count,
                        "games":        self.games_played,
                        "steps":        self.global_step,
                        "mean_length":  stats.get("mean_length", 0),
                        "mean_score":   stats.get("mean_score", 0),
                        "gps":          gps,
                        "rollout_mode": rollout_mode,
                        **metrics,
                    }
                    self.logger.log(record)

                # --- checkpoint ---
                if self.games_played % ckpt_interval < n_games:
                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        step=self.global_step,
                        games=self.games_played,
                        config=self.cfg,
                        path=self.run_dir / "checkpoints" /
                             f"ckpt_{self.games_played:07d}.pt",
                        scaler=self.ppo.scaler,
                    )
        finally:
            if self._mp_collector is not None:
                self._mp_collector.stop()

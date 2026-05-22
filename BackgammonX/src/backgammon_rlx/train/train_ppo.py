"""Entry point: PPO self-play training.

    python -m backgammon_rlx.train.train_ppo --config configs/rtx5080.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml
import torch

from ..env.encoding import ObservationEncoder, ActionEncoder
from ..models.policy_value_net import BackgammonPolicyValueNet
from ..rl.self_play import SelfPlayTrainer
from ..utils.seed import seed_everything
from ..utils.device import get_device
from ..utils.checkpoint import load_checkpoint, save_checkpoint


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device) -> BackgammonPolicyValueNet:
    model = BackgammonPolicyValueNet(
        state_dim=cfg.get("state_dim", 256),
        act_dim=cfg.get("act_dim", 256),
        n_point_res=cfg.get("n_point_residual", 4),
        n_action_res=cfg.get("n_action_residual", 3),
    )
    if cfg.get("compile_model", False):
        try:
            model = torch.compile(model)
            print("[train] torch.compile enabled")
        except Exception as e:
            print(f"[train] torch.compile failed ({e}), skipping")
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="BackgammonX PPO training")
    parser.add_argument("--config",      required=True, help="YAML config file")
    parser.add_argument("--resume",      default=None,  help="Checkpoint to resume from")
    parser.add_argument("--total-games", type=int, default=None)
    parser.add_argument("--max-updates", type=int, default=None,
                        help="Stop after this many PPO updates (for smoke tests)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    seed   = cfg.get("seed", 42)
    seed_everything(seed)

    device = get_device(cfg.get("device", "auto"))
    print(f"[train] device={device}  seed={seed}")

    run_id  = cfg.get("run_id", f"run_{int(time.time())}")
    run_dir = Path(cfg.get("runs_dir", "runs")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.dump(cfg))
    print(f"[train] run_dir={run_dir}")

    obs_enc = ObservationEncoder()
    act_enc = ActionEncoder()

    model = build_model(cfg, device)
    print(f"[train] parameters={model.parameter_count():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 3e-4),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )

    scheduler = None
    if cfg.get("lr_schedule") == "cosine":
        total = args.total_games or cfg.get("total_games", 1_000_000)
        steps = total // cfg.get("rollout_games_per_update", 64)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=1e-6
        )

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, device=str(device))
        print(f"[train] resumed from {args.resume}  "
              f"step={ckpt['step']}  games={ckpt['games']}")

    # Save full run metadata
    from ..utils.run_metadata import save_run_metadata
    save_run_metadata(cfg, run_dir, model=model,
                      extra={"argv": " ".join(__import__("sys").argv)})
    print(f"[train] metadata saved to {run_dir}/metadata.json")

    trainer = SelfPlayTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=cfg,
        device=device,
        run_dir=run_dir,
        obs_enc=obs_enc,
        act_enc=act_enc,
    )

    total_games = args.total_games or cfg.get("total_games", 1_000_000)
    if args.max_updates is not None:
        # Override total_games to stop after max_updates PPO updates
        games_per_update = cfg.get("rollout_games_per_update", 64)
        total_games = min(total_games, args.max_updates * games_per_update)
        print(f"[train] --max-updates={args.max_updates} → capping at {total_games:,} games")
    else:
        print(f"[train] training for {total_games:,} games")

    try:
        trainer.train(total_games)
    finally:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=trainer.global_step,
            games=trainer.games_played,
            config=cfg,
            path=run_dir / "checkpoints" / "final.pt",
            scaler=trainer.ppo.scaler,
        )
        print(f"[train] saved final checkpoint to {run_dir}/checkpoints/final.pt")
        trainer.logger.close()


if __name__ == "__main__":
    main()

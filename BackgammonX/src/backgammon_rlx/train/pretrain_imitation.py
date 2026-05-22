"""Imitation-learning pretraining from expert datasets.

Expert dataset format (JSONL, one position per line):
{
  "board": [...],
  "bar": [0,0],
  "borne_off": [0,0],
  "current_player": 0,
  "dice": [3,5],
  "expert_action_idx": 2,          # index in legal_actions() ordering
  "expert_probs": [...],            # optional: full distribution over legal actions
  "expert_value": 0.62             # optional: equity estimate [-1,1]
}

    python -m backgammon_rlx.train.pretrain_imitation \
        --config configs/imitation.yaml \
        --dataset data/expert_positions.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ..env.state import GameState
from ..env.movegen import get_legal_turns, canonicalize_state_for_player
from ..env.encoding import ObservationEncoder, ActionEncoder
from ..models.policy_value_net import BackgammonPolicyValueNet
from ..utils.seed import seed_everything
from ..utils.device import get_device
from ..utils.checkpoint import save_checkpoint


def load_dataset(path: str) -> List[Dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_batch(
    samples: List[Dict],
    obs_enc: ObservationEncoder,
    act_enc: ActionEncoder,
    device:  torch.device,
):
    obs_list, act_list, ai_list, val_list, prob_list = [], [], [], [], []
    valid = []
    for s in samples:
        state = GameState(
            board=s["board"], bar=s["bar"], borne_off=s["borne_off"],
            current_player=s["current_player"], dice=s["dice"]
        )
        turns = get_legal_turns(state)
        if not turns:
            continue
        player = state.current_player
        canon  = canonicalize_state_for_player(state, player)
        obs    = obs_enc.encode(canon)
        acts   = np.stack([act_enc.encode(t, canon) for t in turns], 0)

        ai = s.get("expert_action_idx", 0)
        if ai >= len(turns):
            continue

        obs_list.append(obs)
        act_list.append(acts)
        ai_list.append(ai)
        val_list.append(s.get("expert_value", None))
        prob_list.append(s.get("expert_probs", None))
        valid.append(True)

    if not obs_list:
        return None

    max_n = max(a.shape[0] for a in act_list)
    act_dim = act_list[0].shape[1]
    B = len(obs_list)

    obs_t  = torch.tensor(np.stack(obs_list),
                           dtype=torch.float32, device=device)
    act_t  = torch.zeros(B, max_n, act_dim, dtype=torch.float32, device=device)
    mask_t = torch.zeros(B, max_n, dtype=torch.bool, device=device)
    ai_t   = torch.tensor(ai_list, dtype=torch.long, device=device)

    for k, acts in enumerate(act_list):
        n = acts.shape[0]
        act_t[k, :n]   = torch.tensor(acts, dtype=torch.float32, device=device)
        mask_t[k, :n]  = True

    val_t  = None
    if any(v is not None for v in val_list):
        vals   = [v if v is not None else 0.0 for v in val_list]
        val_t  = torch.tensor(vals, dtype=torch.float32, device=device)

    return obs_t, act_t, mask_t, ai_t, val_t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out",     default="runs/pretrain")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("seed", 42))
    device  = get_device(cfg.get("device", "auto"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_enc = ObservationEncoder()
    act_enc = ActionEncoder()

    model = BackgammonPolicyValueNet(
        state_dim=cfg.get("state_dim", 256),
        act_dim=cfg.get("act_dim", 256),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )

    dataset   = load_dataset(args.dataset)
    n_epochs  = cfg.get("n_epochs", 10)
    batch_sz  = cfg.get("batch_size", 256)
    val_coef  = cfg.get("value_coef", 0.5)

    print(f"[pretrain] {len(dataset)} positions, {n_epochs} epochs")

    for epoch in range(n_epochs):
        random.shuffle(dataset)
        total_loss = n_batches = 0

        for start in range(0, len(dataset), batch_sz):
            batch = dataset[start: start + batch_sz]
            result = build_batch(batch, obs_enc, act_enc, device)
            if result is None:
                continue
            obs_t, act_t, mask_t, ai_t, val_t = result

            logits, values, _ = model(obs_t, act_t, mask=mask_t)
            pi_loss = F.cross_entropy(logits, ai_t)
            loss    = pi_loss
            if val_t is not None:
                loss = loss + val_coef * F.mse_loss(values, val_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg = total_loss / max(n_batches, 1)
        print(f"  epoch {epoch+1}/{n_epochs}  loss={avg:.4f}")

    save_checkpoint(model, optimizer, step=0, games=0,
                    config=cfg, path=out_dir / "pretrain.pt")
    print(f"[pretrain] saved to {out_dir}/pretrain.pt")


if __name__ == "__main__":
    main()

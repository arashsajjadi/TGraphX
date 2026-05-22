"""PPO update step.

Processes a filled RolloutBuffer:
1. Compute discounted returns + GAE advantages.
2. Run *n_epochs* passes over minibatches.
3. For each minibatch:
   - Recompute log π(a|s) and V(s) using the current model.
   - Clipped policy loss.
   - Value loss (clipped or plain MSE).
   - Entropy bonus.
4. Gradient clip + optimizer step.

Because each step has a variable-length action set, minibatching pads
action tensors to the maximum N in the batch.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from typing import Dict, Optional

from .buffer import RolloutBuffer


class PPOTrainer:

    def __init__(
        self,
        model:            nn.Module,
        optimizer:        torch.optim.Optimizer,
        device:           torch.device,
        clip_range:       float = 0.2,
        value_coef:       float = 0.5,
        entropy_coef:     float = 0.01,
        max_grad_norm:    float = 1.0,
        n_epochs:         int   = 4,
        minibatch_size:   int   = 512,
        use_amp:          bool  = True,
        clip_value_loss:  bool  = True,
        target_kl:        Optional[float] = None,
    ) -> None:
        self.model           = model
        self.optimizer       = optimizer
        self.device          = device
        self.clip_range      = clip_range
        self.value_coef      = value_coef
        self.entropy_coef    = entropy_coef
        self.max_grad_norm   = max_grad_norm
        self.n_epochs        = n_epochs
        self.minibatch_size  = minibatch_size
        self.clip_value_loss = clip_value_loss
        self.target_kl       = target_kl
        self.scaler          = GradScaler("cuda", enabled=use_amp)
        self.use_amp         = use_amp

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Run PPO update over *buffer*.  Returns training metrics."""
        data = buffer.compute_returns_and_advantages()
        self.model.train()

        obs        = torch.tensor(data["obs"],        device=self.device)
        act_idx    = torch.tensor(data["act_idx"],    device=self.device)
        old_lp     = torch.tensor(data["log_probs"],  device=self.device)
        returns    = torch.tensor(data["returns"],    device=self.device)
        advantages = torch.tensor(data["advantages"], device=self.device)
        old_values = torch.tensor(data["values"],     device=self.device)
        act_feats_list = data["act_feats"]   # list of [N_i, ACT_DIM] arrays
        n_actions  = data["n_actions"]

        N = len(obs)
        metrics: dict = {
            "policy_loss": 0.0,
            "value_loss":  0.0,
            "entropy":     0.0,
            "approx_kl":   0.0,
            "clip_frac":   0.0,
            "n_updates":   0,
        }

        kl_early_stop = False
        for _epoch in range(self.n_epochs):
            if kl_early_stop:
                break
            idx = torch.randperm(N)
            for start in range(0, N, self.minibatch_size):
                mb_idx = idx[start: start + self.minibatch_size]
                if len(mb_idx) == 0:
                    continue

                # --- build padded action tensor for this minibatch ---
                max_n = int(n_actions[mb_idx.cpu().numpy()].max())
                mb_act = torch.zeros(
                    len(mb_idx), max_n,
                    act_feats_list[0].shape[-1],
                    dtype=torch.float32, device=self.device,
                )
                mb_mask = torch.zeros(len(mb_idx), max_n,
                                      dtype=torch.bool, device=self.device)
                for k, bi in enumerate(mb_idx.tolist()):
                    ni  = int(n_actions[bi])
                    arr = torch.tensor(act_feats_list[bi],
                                       dtype=torch.float32, device=self.device)
                    mb_act[k, :ni] = arr
                    mb_mask[k, :ni] = True

                mb_obs    = obs[mb_idx]
                mb_ai     = act_idx[mb_idx]
                mb_old_lp = old_lp[mb_idx]
                mb_ret    = returns[mb_idx]
                mb_adv    = advantages[mb_idx]
                mb_old_v  = old_values[mb_idx]

                with autocast("cuda", enabled=self.use_amp):
                    logits, values = self.model(mb_obs, mb_act, mask=mb_mask)
                    # log probs for chosen actions
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs     = log_probs.exp()
                    chosen_lp = log_probs[torch.arange(len(mb_idx)), mb_ai]

                    # entropy: replace -inf (masked) log probs with 0 so 0*0=0, not NaN
                    safe_lp   = log_probs.nan_to_num(0.0)
                    entropy   = -(probs * safe_lp).sum(dim=-1).mean()

                    # PPO clipped policy loss
                    ratio     = torch.exp(chosen_lp - mb_old_lp)
                    pg_loss1  = -mb_adv * ratio
                    pg_loss2  = -mb_adv * ratio.clamp(
                        1.0 - self.clip_range, 1.0 + self.clip_range
                    )
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    if self.clip_value_loss:
                        v_clipped = mb_old_v + (values - mb_old_v).clamp(
                            -self.clip_range, self.clip_range
                        )
                        vf_loss = torch.max(
                            F.mse_loss(values, mb_ret),
                            F.mse_loss(v_clipped, mb_ret),
                        )
                    else:
                        vf_loss = F.mse_loss(values, mb_ret)

                    loss = (policy_loss
                            + self.value_coef * vf_loss
                            - self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                          self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                with torch.no_grad():
                    approx_kl = (mb_old_lp - chosen_lp).mean().item()
                    clip_frac  = ((ratio - 1.0).abs() > self.clip_range
                                  ).float().mean().item()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"]  += vf_loss.item()
                metrics["entropy"]     += entropy.item()
                metrics["approx_kl"]   += approx_kl
                metrics["clip_frac"]   += clip_frac
                metrics["n_updates"]   += 1

                # KL early stopping
                if self.target_kl is not None and approx_kl > self.target_kl:
                    kl_early_stop = True
                    break

        n = max(metrics["n_updates"], 1)
        return {k: v / n if k != "n_updates" else v
                for k, v in metrics.items()}

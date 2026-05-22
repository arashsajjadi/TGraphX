"""BackgammonPolicyValueNet — the main neural network.

Architecture
------------
1. Observation path
   - Reshape flat obs into [24, POINT_FEAT] + [GLOBAL_FEAT]
   - PointEncoder  → state_point_emb  [B, state_dim]
   - Global MLP    → state_glob_emb   [B, state_dim]
   - Fusion MLP    → state_emb        [B, state_dim]

2. Action path
   - Action features [B, N, act_feat_dim]
   - ActionMLP       → act_emb [B, N, act_dim]

3. Policy head
   - Broadcast state_emb over N legal actions
   - cat(state_emb, act_emb, state*act) → score per action [B, N]

4. Value head
   - MLP(state_emb) → scalar [B]

Forward signature:
    logits, value = model(obs, act_feats, mask=None)
    obs:       [B, OBS_DIM]
    act_feats: [B, N, ACT_DIM]   (N = number of legal actions; padded)
    mask:      [B, N] bool, True = valid action
    Returns:
      logits: [B, N]   (pre-softmax; masked positions = -1e9)
      value:  [B]
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..env.encoding import OBS_DIM, ACT_DIM, POINT_FEATURES, GLOBAL_FEATURES
from .layers import ResidualMLP, MLP, PointEncoder, _init_orthogonal


class BackgammonPolicyValueNet(nn.Module):

    def __init__(
        self,
        state_dim:    int = 256,
        act_dim:      int = 256,
        n_point_res:  int = 4,
        n_action_res: int = 3,
        n_head_layers: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.act_dim   = act_dim

        # ---- observation branch ----
        self.point_enc  = PointEncoder(POINT_FEATURES, state_dim, n_point_res)
        self.global_mlp = MLP(GLOBAL_FEATURES, state_dim, state_dim, n_layers=2)
        self.state_fuse = MLP(state_dim * 2, state_dim * 2, state_dim, n_layers=2)

        # ---- action branch ----
        self.act_proj = nn.Linear(ACT_DIM, act_dim)
        self.act_res  = nn.ModuleList(
            [ResidualMLP(act_dim) for _ in range(n_action_res)]
        )

        # ---- policy head ----
        #   input = cat(s, a, s*a) → 3 * state_dim (assumes act_dim == state_dim)
        policy_in = state_dim + act_dim + min(state_dim, act_dim)
        self.policy_head = MLP(policy_in, state_dim, 1, n_layers=n_head_layers)

        # ---- value head ----
        self.value_head = MLP(state_dim, state_dim // 2, 1, n_layers=n_head_layers)

        # ---- init ----
        self.apply(_init_orthogonal)
        # Scale output layers down for stability
        nn.init.orthogonal_(self.policy_head.net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.net[-1].weight,  gain=0.01)

    # ------------------------------------------------------------------
    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        point_feat  = obs[:, : 24 * POINT_FEATURES].view(B, 24, POINT_FEATURES)
        global_feat = obs[:, 24 * POINT_FEATURES:]

        pt_emb   = self.point_enc(point_feat)       # [B, D]
        gl_emb   = self.global_mlp(global_feat)     # [B, D]
        state_in = torch.cat([pt_emb, gl_emb], dim=-1)
        return self.state_fuse(state_in)             # [B, D]

    def _encode_actions(self, act_feats: torch.Tensor) -> torch.Tensor:
        # act_feats: [B, N, ACT_DIM]
        h = self.act_proj(act_feats)                # [B, N, act_dim]
        for blk in self.act_res:
            h = blk(h)
        return h                                    # [B, N, act_dim]

    # ------------------------------------------------------------------
    def forward(
        self,
        obs:       torch.Tensor,          # [B, OBS_DIM]
        act_feats: torch.Tensor,          # [B, N, ACT_DIM]
        mask:      Optional[torch.Tensor] = None,  # [B, N] bool
    ):
        state_emb = self._encode_obs(obs)            # [B, D]
        act_emb   = self._encode_actions(act_feats)  # [B, N, ad]

        N = act_emb.shape[1]
        s_exp = state_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, D]

        # Elementwise product for interaction
        interact = s_exp * act_emb                         # [B, N, min(D,ad)]
        policy_in = torch.cat([s_exp, act_emb, interact], dim=-1)  # [B, N, 3D]

        logits = self.policy_head(policy_in).squeeze(-1)   # [B, N]

        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))

        value = self.value_head(state_emb).squeeze(-1)     # [B]
        return logits, value

    # ------------------------------------------------------------------
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

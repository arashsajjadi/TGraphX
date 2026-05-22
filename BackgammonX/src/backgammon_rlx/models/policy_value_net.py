"""BackgammonPolicyValueNet — configurable policy/value network.

Core architecture (always active):
  obs → PointEncoder (residual MLP or Transformer) + GlobalMLP → state_emb [D]
  action_feats → ActionMLP → act_emb [D]
  policy_score = MLP(cat(state, action, state*action)) → logit per action
  value = MLP(state_emb) → scalar

Optional extensions (configured at construction time):
  use_transformer    - Transformer over 24 board points instead of residual MLP
  use_auxiliary_heads - win/gammon/backgammon probability + pip-count heads

Forward signature:
    logits, value, aux = model(obs, act_feats, mask=None)
    aux is None when use_auxiliary_heads=False; otherwise dict with keys:
      win_prob, gammon_prob, backgammon_prob, pip_count_pred
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..env.encoding import POINT_FEATURES
from .layers import ResidualMLP, MLP, PointEncoder, _init_orthogonal


# ---------------------------------------------------------------------------
# Transformer-based point encoder
# ---------------------------------------------------------------------------

class TransformerPointEncoder(nn.Module):
    """Encodes [B, 24, point_feat_dim] using a Transformer.

    Positional embedding is learned.
    Output: mean-pool over 24 positions → [B, d_model]
    """

    def __init__(
        self,
        point_feat_dim: int,
        d_model: int,
        n_layers: int = 2,
        n_heads: int  = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.proj     = nn.Linear(point_feat_dim, d_model)
        self.pos_emb  = nn.Parameter(torch.zeros(24, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm is more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 24, P]
        h = self.proj(x) + self.pos_emb.unsqueeze(0)  # [B, 24, D]
        h = self.transformer(h)                         # [B, 24, D]
        h = self.norm(h)
        return h.mean(dim=1)                            # [B, D]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class BackgammonPolicyValueNet(nn.Module):
    """Configurable policy/value network for backgammon.

    Parameters
    ----------
    state_dim:          Hidden dimension for state representation
    act_dim:            Hidden dimension for action representation
    n_point_res:        Residual blocks in PointEncoder (ignored if Transformer used)
    n_action_res:       Residual blocks in ActionEncoder
    n_head_layers:      Depth of policy/value MLP heads
    use_transformer:    Use Transformer encoder instead of residual MLP
    transformer_layers: Transformer encoder depth
    transformer_heads:  Attention heads
    dropout:            Dropout rate (applied in Transformer)
    use_auxiliary_heads: Add win/gammon/backgammon + pip-count heads
    obs_dim:            Total obs dimension (inferred from encoder if not set)
    act_feat_dim:       Action feature dimension (inferred from encoder if not set)
    """

    def __init__(
        self,
        state_dim:           int   = 256,
        act_dim:             int   = 256,
        n_point_res:         int   = 4,
        n_action_res:        int   = 3,
        n_head_layers:       int   = 2,
        use_transformer:     bool  = False,
        transformer_layers:  int   = 2,
        transformer_heads:   int   = 4,
        dropout:             float = 0.0,
        use_auxiliary_heads: bool  = False,
        obs_dim:             Optional[int] = None,
        act_feat_dim:        Optional[int] = None,
    ) -> None:
        super().__init__()
        self.state_dim           = state_dim
        self.act_dim             = act_dim
        self.use_transformer     = use_transformer
        self.use_auxiliary_heads = use_auxiliary_heads

        # Resolve dimensions from encoding module if not provided
        if obs_dim is None:
            from ..env.encoding import OBS_DIM
            obs_dim = OBS_DIM
        if act_feat_dim is None:
            from ..env.encoding import ACT_DIM
            act_feat_dim = ACT_DIM

        global_feat_dim = obs_dim - 24 * POINT_FEATURES  # whatever is left after per-point

        # ---- observation branch ----
        if use_transformer:
            self.point_enc = TransformerPointEncoder(
                POINT_FEATURES, state_dim,
                n_layers=transformer_layers,
                n_heads=transformer_heads,
                dropout=dropout,
            )
        else:
            self.point_enc = PointEncoder(POINT_FEATURES, state_dim, n_point_res)

        self.global_mlp = MLP(global_feat_dim, state_dim, state_dim, n_layers=2)
        self.state_fuse = MLP(state_dim * 2, state_dim * 2, state_dim, n_layers=2)

        # ---- action branch ----
        self.act_proj = nn.Linear(act_feat_dim, act_dim)
        self.act_res  = nn.ModuleList(
            [ResidualMLP(act_dim) for _ in range(n_action_res)]
        )

        # ---- policy head (dueling-inspired: state baseline + action advantage) ----
        policy_in          = state_dim + act_dim + min(state_dim, act_dim)
        self.policy_head   = MLP(policy_in, state_dim, 1, n_layers=n_head_layers)
        self.state_baseline = MLP(state_dim, state_dim // 2, 1, n_layers=1)

        # ---- value head ----
        self.value_head = MLP(state_dim, state_dim // 2, 1, n_layers=n_head_layers)

        # ---- auxiliary heads ----
        if use_auxiliary_heads:
            self.win_head            = MLP(state_dim, state_dim // 4, 1, n_layers=1)
            self.gammon_head         = MLP(state_dim, state_dim // 4, 1, n_layers=1)
            self.backgammon_head     = MLP(state_dim, state_dim // 4, 1, n_layers=1)
            self.pip_count_head      = MLP(state_dim, state_dim // 4, 1, n_layers=1)

        # ---- init ----
        self.apply(_init_orthogonal)
        nn.init.orthogonal_(self.policy_head.net[-1].weight,   gain=0.01)
        nn.init.orthogonal_(self.value_head.net[-1].weight,    gain=0.01)
        nn.init.orthogonal_(self.state_baseline.net[-1].weight, gain=0.01)

        self._obs_dim      = obs_dim
        self._act_feat_dim = act_feat_dim

    # ------------------------------------------------------------------
    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        point_feat  = obs[:, : 24 * POINT_FEATURES].view(B, 24, POINT_FEATURES)
        global_feat = obs[:, 24 * POINT_FEATURES:]

        pt_emb   = self.point_enc(point_feat)
        gl_emb   = self.global_mlp(global_feat)
        return self.state_fuse(torch.cat([pt_emb, gl_emb], dim=-1))

    def _encode_actions(self, act_feats: torch.Tensor) -> torch.Tensor:
        h = self.act_proj(act_feats)
        for blk in self.act_res:
            h = blk(h)
        return h

    # ------------------------------------------------------------------
    def forward(
        self,
        obs:       torch.Tensor,
        act_feats: torch.Tensor,
        mask:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Returns
        -------
        logits : [B, N]
        value  : [B]
        aux    : dict or None  (win/gammon/backgammon probs, pip count)
        """
        state_emb = self._encode_obs(obs)          # [B, D]
        act_emb   = self._encode_actions(act_feats) # [B, N, ad]

        N     = act_emb.shape[1]
        s_exp = state_emb.unsqueeze(1).expand(-1, N, -1)

        interact  = s_exp * act_emb
        policy_in = torch.cat([s_exp, act_emb, interact], dim=-1)

        # Dueling: advantage per action + state baseline
        adv      = self.policy_head(policy_in).squeeze(-1)   # [B, N]
        baseline = self.state_baseline(state_emb)             # [B, 1]
        logits   = adv + baseline

        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))

        value = self.value_head(state_emb).squeeze(-1)  # [B]

        aux = None
        if self.use_auxiliary_heads:
            aux = {
                "win_prob":        torch.sigmoid(self.win_head(state_emb).squeeze(-1)),
                "gammon_prob":     torch.sigmoid(self.gammon_head(state_emb).squeeze(-1)),
                "backgammon_prob": torch.sigmoid(self.backgammon_head(state_emb).squeeze(-1)),
                "pip_count_pred":  self.pip_count_head(state_emb).squeeze(-1),
            }

        return logits, value, aux

    # ------------------------------------------------------------------
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self) -> str:
        lines = [
            f"BackgammonPolicyValueNet",
            f"  state_dim={self.state_dim}  act_dim={self.act_dim}",
            f"  point_encoder={'Transformer' if self.use_transformer else 'ResidualMLP'}",
            f"  auxiliary_heads={self.use_auxiliary_heads}",
            f"  obs_dim={self._obs_dim}  act_feat_dim={self._act_feat_dim}",
            f"  parameters={self.parameter_count():,}",
        ]
        return "\n".join(lines)

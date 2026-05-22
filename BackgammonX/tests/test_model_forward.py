"""Tests for neural network forward pass correctness."""
import pytest
import torch
import numpy as np
from backgammon_rlx.models.policy_value_net import BackgammonPolicyValueNet
from backgammon_rlx.env.encoding import OBS_DIM, ACT_DIM
from backgammon_rlx.env.state import GameState
from backgammon_rlx.env.encoding import ObservationEncoder, ActionEncoder
from backgammon_rlx.env.movegen import get_legal_turns, canonicalize_state_for_player


def small_model():
    return BackgammonPolicyValueNet(
        state_dim=64, act_dim=64, n_point_res=2, n_action_res=1
    )


class TestModelForward:

    def test_basic_forward_cpu(self):
        model = small_model()
        obs = torch.randn(2, OBS_DIM)
        act = torch.randn(2, 5, ACT_DIM)
        logits, values, _ = model(obs, act)
        assert logits.shape == (2, 5)
        assert values.shape == (2,)

    def test_no_nan_in_output(self):
        model = small_model()
        obs = torch.randn(4, OBS_DIM)
        act = torch.randn(4, 10, ACT_DIM)
        logits, values, _ = model(obs, act)
        assert torch.all(torch.isfinite(logits)), "NaN/Inf in logits"
        assert torch.all(torch.isfinite(values)), "NaN/Inf in values"

    def test_mask_prevents_selection(self):
        model = small_model()
        model.eval()
        obs = torch.randn(1, OBS_DIM)
        act = torch.randn(1, 5, ACT_DIM)
        mask = torch.ones(1, 5, dtype=torch.bool)
        mask[0, 3] = False  # mask out action 3

        logits, _val, _ = model(obs, act, mask=mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        assert probs[0, 3].item() < 1e-6, "Masked action should have ~0 probability"

    def test_variable_action_counts(self):
        """Model must handle different N per sample via padding + masking."""
        model = small_model()
        model.eval()
        B = 3
        max_n = 8
        obs = torch.randn(B, OBS_DIM)
        act = torch.randn(B, max_n, ACT_DIM)
        mask = torch.zeros(B, max_n, dtype=torch.bool)
        mask[0, :3] = True  # 3 actions
        mask[1, :6] = True  # 6 actions
        mask[2, :8] = True  # 8 actions

        logits, values, _ = model(obs, act, mask=mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # Masked positions should have ~0 probability
        assert probs[0, 3:].max().item() < 1e-6
        assert probs[1, 6:].max().item() < 1e-6
        # Valid positions should have positive probability
        assert probs[0, :3].min().item() > 0

    def test_entropy_not_nan(self):
        """Entropy must not be NaN even with masked actions."""
        model = small_model()
        model.eval()
        obs = torch.randn(2, OBS_DIM)
        act = torch.randn(2, 5, ACT_DIM)
        mask = torch.ones(2, 5, dtype=torch.bool)
        mask[0, 3:] = False  # 3 valid for sample 0, 5 for sample 1

        logits, _val, _ = model(obs, act, mask=mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        safe_lp = log_probs.nan_to_num(0.0)
        entropy = -(probs * safe_lp).sum(dim=-1).mean()
        assert torch.isfinite(entropy), f"Entropy is {entropy}"

    def test_parameter_count(self):
        model = small_model()
        n = model.parameter_count()
        assert n > 1000, f"Model too small: {n} params"
        assert n < 1_000_000, f"Small model shouldn't have >1M params"

    def test_real_game_position(self):
        """Forward pass with actual game position encoding."""
        model = small_model()
        model.eval()
        obs_enc = ObservationEncoder()
        act_enc = ActionEncoder()

        s = GameState.initial()
        s.dice = [3, 1]
        canon = canonicalize_state_for_player(s, 0)
        turns = get_legal_turns(s)

        obs = torch.tensor(obs_enc.encode(canon),
                           dtype=torch.float32).unsqueeze(0)
        act_arr = np.stack([act_enc.encode(t, canon) for t in turns], 0)
        act = torch.tensor(act_arr, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits, values, _ = model(obs, act)

        assert logits.shape == (1, len(turns))
        assert values.shape == (1,)
        assert torch.all(torch.isfinite(logits))
        probs = torch.softmax(logits, dim=-1)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_amp_forward_pass(self):
        """AMP forward pass must not produce NaN."""
        model = small_model().cuda()
        model.eval()
        obs = torch.randn(4, OBS_DIM, device="cuda")
        act = torch.randn(4, 8, ACT_DIM, device="cuda")
        mask = torch.ones(4, 8, dtype=torch.bool, device="cuda")
        mask[:, 5:] = False

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                logits, values, _ = model(obs, act, mask=mask)

        log_probs = torch.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        safe_lp = log_probs.nan_to_num(0.0)
        entropy = -(probs * safe_lp).sum(dim=-1).mean()
        assert torch.isfinite(entropy), f"AMP entropy is NaN: {entropy}"

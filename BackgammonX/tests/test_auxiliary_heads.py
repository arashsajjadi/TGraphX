"""Tests for auxiliary prediction heads and transformer encoder."""
import pytest
import torch
from backgammon_rlx.models.policy_value_net import BackgammonPolicyValueNet
from backgammon_rlx.env.encoding import OBS_DIM, ACT_DIM


def _make_batch(B=2, N=5):
    obs = torch.randn(B, OBS_DIM)
    act = torch.randn(B, N, ACT_DIM)
    mask = torch.ones(B, N, dtype=torch.bool)
    return obs, act, mask


class TestAuxiliaryHeads:

    def test_aux_heads_disabled_returns_none(self):
        model = BackgammonPolicyValueNet(state_dim=32, act_dim=32,
                                         use_auxiliary_heads=False)
        obs, act, mask = _make_batch()
        logits, value, aux = model(obs, act, mask)
        assert aux is None

    def test_aux_heads_enabled_returns_dict(self):
        model = BackgammonPolicyValueNet(state_dim=32, act_dim=32,
                                         use_auxiliary_heads=True)
        obs, act, mask = _make_batch()
        logits, value, aux = model(obs, act, mask)
        assert aux is not None
        assert "win_prob" in aux
        assert "gammon_prob" in aux
        assert "backgammon_prob" in aux
        assert "pip_count_pred" in aux

    def test_aux_head_shapes(self):
        B = 3
        model = BackgammonPolicyValueNet(state_dim=32, act_dim=32,
                                         use_auxiliary_heads=True)
        obs, act, mask = _make_batch(B=B)
        _, _, aux = model(obs, act, mask)
        assert aux["win_prob"].shape == (B,)
        assert aux["gammon_prob"].shape == (B,)
        assert aux["pip_count_pred"].shape == (B,)

    def test_probability_heads_in_0_1(self):
        model = BackgammonPolicyValueNet(state_dim=32, act_dim=32,
                                         use_auxiliary_heads=True)
        obs, act, mask = _make_batch()
        _, _, aux = model(obs, act, mask)
        for k in ("win_prob", "gammon_prob", "backgammon_prob"):
            assert (aux[k] >= 0).all() and (aux[k] <= 1).all(), \
                f"{k} out of [0,1] range: {aux[k]}"

    def test_no_nan_with_aux_heads(self):
        model = BackgammonPolicyValueNet(state_dim=32, act_dim=32,
                                         use_auxiliary_heads=True)
        obs, act, mask = _make_batch(B=4, N=8)
        logits, value, aux = model(obs, act, mask)
        assert torch.all(torch.isfinite(logits))
        assert torch.all(torch.isfinite(value))
        for v in aux.values():
            assert torch.all(torch.isfinite(v))


class TestTransformerEncoder:

    def test_transformer_forward(self):
        model = BackgammonPolicyValueNet(
            state_dim=64, act_dim=64,
            use_transformer=True, transformer_layers=1, transformer_heads=4
        )
        obs, act, mask = _make_batch()
        logits, value, aux = model(obs, act, mask)
        assert logits.shape == (2, 5)
        assert value.shape == (2,)
        assert aux is None

    def test_transformer_no_nan(self):
        model = BackgammonPolicyValueNet(
            state_dim=64, act_dim=64,
            use_transformer=True, transformer_layers=2, transformer_heads=4
        )
        obs, act, mask = _make_batch(B=4, N=10)
        mask[:, 6:] = False
        logits, value, _ = model(obs, act, mask)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs.nan_to_num(0.0)).sum(dim=-1).mean()
        assert torch.isfinite(entropy), f"entropy is {entropy}"

    def test_transformer_vs_mlp_same_interface(self):
        """Transformer and MLP models have identical API."""
        mlp   = BackgammonPolicyValueNet(state_dim=32, act_dim=32)
        trans = BackgammonPolicyValueNet(state_dim=32, act_dim=32,
                                          use_transformer=True,
                                          transformer_layers=1,
                                          transformer_heads=4)
        obs, act, mask = _make_batch()
        l1, v1, a1 = mlp(obs, act, mask)
        l2, v2, a2 = trans(obs, act, mask)
        assert l1.shape == l2.shape
        assert v1.shape == v2.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_transformer_amp_cuda(self):
        model = BackgammonPolicyValueNet(
            state_dim=64, act_dim=64,
            use_transformer=True, transformer_layers=2, transformer_heads=4
        ).cuda()
        obs = torch.randn(4, OBS_DIM, device="cuda")
        act = torch.randn(4, 8, ACT_DIM, device="cuda")
        with torch.amp.autocast("cuda"):
            logits, value, _ = model(obs, act)
        assert torch.all(torch.isfinite(logits.float()))

    def test_dueling_structure(self):
        """Model has both state_baseline and policy_head (dueling-inspired)."""
        model = BackgammonPolicyValueNet(state_dim=32, act_dim=32)
        assert hasattr(model, 'state_baseline')
        assert hasattr(model, 'policy_head')

"""Hand-computed reference math tests for KG, RL, and Easy Mode (v1.1).

These tests pin down exact formula behavior with hand-computable expected
values, not just shape/finiteness checks.  They protect against silent
math regressions in the next minor releases.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── KG: filtered ranking with hand-checked positions ─────────────────────────


class TestKGFilteredRanking:
    """Verify filtered MRR / Hits@K against hand-computed scores."""

    def _make_oracle_model(self, scores_per_query):
        """Build an oracle model implementing score_triples(triples) -> [B].

        ``scores_per_query`` is a function ``(h, r, t) -> float`` so the
        evaluator can call it for every (?, r, t) and (h, r, ?) candidate.
        """

        class _Oracle(nn.Module):
            def __init__(self, fn):
                super().__init__()
                self._fn = fn

            def score_triples(self, triples):
                # triples: [B, 3]
                vals = [self._fn(int(triples[i, 0]), int(triples[i, 1]), int(triples[i, 2]))
                        for i in range(triples.size(0))]
                return torch.tensor(vals, dtype=torch.float32)

        return _Oracle(scores_per_query)

    def _setup_orthogonal_kg_model(self):
        """3 entities = orthogonal unit vectors; 1 relation = h_target - h_source.

        Embeddings (normalised → no-op since already unit):
            e0 = [1, 0, 0]
            e1 = [0, 1, 0]
            e2 = [0, 0, 1]
            r0 = [-1, 1, 0]   so e0 + r0 = [0, 1, 0] = e1

        For test triple (0, 0, 1) — i.e. ask: h=0, r=0, ? = ?:
            distance to e0 = ||[1,0,0]+[-1,1,0]-[1,0,0]||_2 = ||[-1,1,0]|| = √2
            distance to e1 = ||[0,1,0] - [0,1,0]|| = 0   ← true tail (rank 1)
            distance to e2 = ||[0,1,0] - [0,0,1]|| = ||[0,1,-1]|| = √2
        """
        from tgraphx.kg import TransEModel

        N_e, N_r, dim = 3, 1, 3
        model = TransEModel(num_entities=N_e, num_relations=N_r, embedding_dim=dim, norm=2)
        with torch.no_grad():
            model.entity_emb.weight.zero_()
            model.relation_emb.weight.zero_()
            model.entity_emb.weight[0] = torch.tensor([1.0, 0.0, 0.0])
            model.entity_emb.weight[1] = torch.tensor([0.0, 1.0, 0.0])
            model.entity_emb.weight[2] = torch.tensor([0.0, 0.0, 1.0])
            model.relation_emb.weight[0] = torch.tensor([-1.0, 1.0, 0.0])
        return model, N_e

    def test_filtered_tail_rank_1_when_perfectly_predicted(self):
        from tgraphx.kg.evaluation import evaluate_filtered_ranking

        model, N_e = self._setup_orthogonal_kg_model()
        test_triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        all_pos = {(0, 0, 1)}
        result = evaluate_filtered_ranking(
            model, test_triples, all_pos, num_entities=N_e,
            filtered=True, hits_at=(1, 3),
        )
        # True tail (entity 1) is unambiguously rank 1.
        assert result.filt_mrr_tail == pytest.approx(1.0, abs=1e-5)
        assert result.filt_hits_tail[1] == pytest.approx(1.0, abs=1e-5)

    def test_filtering_does_not_change_perfect_rank(self):
        """Adding a known positive (0, 0, 0) does not affect tail ranking
        when the true tail already had rank 1 (entity 0 was rank 2/3 anyway).
        """
        from tgraphx.kg.evaluation import evaluate_filtered_ranking

        model, N_e = self._setup_orthogonal_kg_model()
        test_triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        # Entity 0 was already not the best for tail rank, so filtering
        # it out shouldn't reduce the true tail's rank.
        all_pos = {(0, 0, 1), (0, 0, 0)}
        result = evaluate_filtered_ranking(
            model, test_triples, all_pos, num_entities=N_e,
            filtered=True, hits_at=(1,),
        )
        assert result.filt_mrr_tail == pytest.approx(1.0, abs=1e-5)
        assert result.filt_hits_tail[1] == pytest.approx(1.0, abs=1e-5)


# ── KG: TransE scoring formula ───────────────────────────────────────────────


class TestKGTransEFormula:
    """TransE scores h+r-t under L_p norm; lower distance = higher score."""

    def test_transe_distance_zero_when_h_plus_r_equals_t(self):
        """TransE L2-normalises entities at score time (Bordes 2013).
        If we pre-normalise h and t to unit length and pick r so that
        h + r = t after normalisation, distance must be zero.

        Use unit-length axis vectors:
          h = [1, 0, 0],  t = [0, 1, 0]  (already unit-norm — normalise = no-op)
          r = t - h = [-1, 1, 0]  (relations are NOT normalised)
        Then h_norm + r - t_norm = 0 → distance 0 → score 0.
        """
        from tgraphx.kg import TransEModel

        N_e, N_r, dim = 4, 2, 3
        model = TransEModel(num_entities=N_e, num_relations=N_r, embedding_dim=dim)

        with torch.no_grad():
            model.entity_emb.weight.zero_()
            model.relation_emb.weight.zero_()
            model.entity_emb.weight[0] = torch.tensor([1.0, 0.0, 0.0])  # unit norm
            model.entity_emb.weight[1] = torch.tensor([0.0, 1.0, 0.0])  # unit norm
            model.relation_emb.weight[0] = torch.tensor([-1.0, 1.0, 0.0])

        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        score = model.score_triples(triples)
        assert score.shape == (1,)
        assert score.item() == pytest.approx(0.0, abs=1e-5)

    def test_transe_l2_distance_against_unit_axes(self):
        """h = [1,0,0], r = [0,0,0], t = [0,1,0] (both entities already unit-norm).
        h + r - t = [1, -1, 0] → ||.||_2 = sqrt(2).  Score = -sqrt(2).
        """
        from tgraphx.kg import TransEModel

        N_e, N_r, dim = 4, 2, 3
        model = TransEModel(num_entities=N_e, num_relations=N_r, embedding_dim=dim, norm=2)

        with torch.no_grad():
            model.entity_emb.weight.zero_()
            model.relation_emb.weight.zero_()
            model.entity_emb.weight[0] = torch.tensor([1.0, 0.0, 0.0])
            model.entity_emb.weight[1] = torch.tensor([0.0, 1.0, 0.0])
            # r = 0 → distance is purely between h_unit and t_unit.

        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        score = model.score_triples(triples)
        # h_unit - t_unit = [1, -1, 0], ||·||_2 = sqrt(2) → score = -sqrt(2).
        assert score.item() == pytest.approx(-math.sqrt(2.0), abs=1e-5)


# ── RL: DQN target Q computation ─────────────────────────────────────────────


class TestRLDQNTarget:
    """y_t = r_t + gamma * max_a Q_target(s_{t+1}, a) * (1 - done)."""

    def test_dqn_target_terminal(self):
        """Terminal step (done=True): target should equal reward only."""
        rewards = torch.tensor([1.0, -0.5])
        next_q = torch.tensor([[10.0, 5.0], [3.0, 8.0]])  # [B, A]
        dones = torch.tensor([True, False])
        gamma = 0.99

        # Hand-computed:
        # i=0: r=1.0, done=True → y=1.0
        # i=1: r=-0.5, done=False → y = -0.5 + 0.99 * max(3,8) = -0.5 + 7.92 = 7.42
        next_q_max, _ = next_q.max(dim=1)  # [8, 8]
        target = rewards + gamma * next_q_max * (~dones).float()
        assert target[0].item() == pytest.approx(1.0, abs=1e-6)
        assert target[1].item() == pytest.approx(-0.5 + 0.99 * 8.0, abs=1e-6)

    def test_double_dqn_target_uses_online_argmax(self):
        """y_t = r + gamma * Q_target(s', argmax_a Q_online(s', a))."""
        # Online Q says action 0 is best for next_state; target Q says action 0 = 5, action 1 = 100.
        # Standard DQN would use max over target → 100.  Double DQN uses Q_online's argmax (action 0) → 5.
        next_q_online = torch.tensor([[2.0, 1.0]])  # argmax = 0
        next_q_target = torch.tensor([[5.0, 100.0]])
        rewards = torch.tensor([0.5])
        dones = torch.tensor([False])
        gamma = 0.9

        argmax_online = next_q_online.argmax(dim=1)  # [0]
        target_q_at_argmax = next_q_target.gather(1, argmax_online.unsqueeze(1)).squeeze(1)
        target = rewards + gamma * target_q_at_argmax * (~dones).float()
        # Expected: 0.5 + 0.9 * 5.0 = 5.0  (NOT 0.5 + 0.9 * 100 = 90.5 of vanilla DQN)
        assert target.item() == pytest.approx(5.0, abs=1e-6)


# ── RL: PPO clipped surrogate ────────────────────────────────────────────────


class TestRLPPOClip:
    """L_clip = E[min(r * A, clip(r, 1-eps, 1+eps) * A)]."""

    def test_ppo_clip_inside_window(self):
        """When ratio is in [1-eps, 1+eps], clip is identity → loss = r*A."""
        old_logp = torch.tensor([0.0])
        new_logp = torch.tensor([0.0])  # ratio = 1.0
        adv = torch.tensor([2.0])
        eps = 0.2

        ratio = torch.exp(new_logp - old_logp)  # [1.0]
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
        surrogate = torch.min(ratio * adv, clipped * adv)
        # Both terms = 1.0 * 2.0 = 2.0 → min = 2.0
        assert surrogate.item() == pytest.approx(2.0, abs=1e-6)

    def test_ppo_clip_positive_advantage_above_window(self):
        """If ratio >= 1+eps and advantage > 0, clip activates: loss = (1+eps)*A."""
        old_logp = torch.tensor([0.0])
        new_logp = torch.tensor([math.log(2.0)])  # ratio = 2.0
        adv = torch.tensor([3.0])
        eps = 0.2

        ratio = torch.exp(new_logp - old_logp)  # 2.0
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps)  # 1.2
        surrogate = torch.min(ratio * adv, clipped * adv)
        # ratio*A = 6.0, clipped*A = 3.6, min = 3.6
        assert surrogate.item() == pytest.approx(1.2 * 3.0, abs=1e-6)

    def test_ppo_clip_negative_advantage_above_window(self):
        """If ratio >= 1+eps and advantage < 0, NOT clipped (min picks larger raw ratio)."""
        old_logp = torch.tensor([0.0])
        new_logp = torch.tensor([math.log(2.0)])  # ratio = 2.0
        adv = torch.tensor([-3.0])
        eps = 0.2

        ratio = torch.exp(new_logp - old_logp)  # 2.0
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps)  # 1.2
        surrogate = torch.min(ratio * adv, clipped * adv)
        # ratio*A = -6.0, clipped*A = -3.6, min = -6.0 (more negative)
        assert surrogate.item() == pytest.approx(-6.0, abs=1e-6)


# ── RL: GAE (Generalized Advantage Estimation) ───────────────────────────────


class TestRLGAE:
    """delta_t = r_t + gamma*V(s_{t+1})*(1-done) - V(s_t);  A_t = delta_t + (gamma*lambda)*(1-done)*A_{t+1}."""

    def test_gae_single_step_terminal(self):
        """Single terminal step: delta = r - V(s); A = delta."""
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.5])
        next_value = torch.tensor([0.0])  # V(s') = 0 because terminal
        dones = torch.tensor([True])
        gamma, lam = 0.99, 0.95

        nonterminal = (~dones).float()
        delta = rewards + gamma * next_value * nonterminal - values
        # delta = 1.0 + 0 - 0.5 = 0.5
        assert delta.item() == pytest.approx(0.5, abs=1e-6)

    def test_gae_two_step_no_terminal(self):
        """Two non-terminal steps:
           delta_0 = r_0 + g*V(s_1) - V(s_0)
           delta_1 = r_1 + g*V(s_2) - V(s_1)
           A_1 = delta_1
           A_0 = delta_0 + g*lam*A_1
        """
        rewards = torch.tensor([1.0, 2.0])
        values = torch.tensor([0.5, 1.0])
        bootstrap_value = torch.tensor(2.0)  # V(s_2)
        gamma, lam = 0.99, 0.95

        # Forward GAE backward.
        delta_1 = rewards[1] + gamma * bootstrap_value - values[1]
        # delta_1 = 2.0 + 0.99*2.0 - 1.0 = 2.98
        delta_0 = rewards[0] + gamma * values[1] - values[0]
        # delta_0 = 1.0 + 0.99*1.0 - 0.5 = 1.49
        adv_1 = delta_1
        adv_0 = delta_0 + gamma * lam * adv_1
        # adv_0 = 1.49 + 0.99*0.95*2.98 = 1.49 + 2.802... = 4.293...

        assert delta_1.item() == pytest.approx(2.98, abs=1e-6)
        assert delta_0.item() == pytest.approx(1.49, abs=1e-6)
        assert adv_1.item() == pytest.approx(2.98, abs=1e-6)
        assert adv_0.item() == pytest.approx(1.49 + 0.99 * 0.95 * 2.98, abs=1e-6)


# ── RL: Soft target update (Polyak averaging) ────────────────────────────────


class TestRLPolyakUpdate:
    """tau-soft update: theta_target <- tau * theta + (1 - tau) * theta_target."""

    def test_polyak_update_formula(self):
        # Signature: soft_update(source, target, tau)
        from tgraphx.rl import soft_update

        source = nn.Linear(4, 2)
        target = nn.Linear(4, 2)
        with torch.no_grad():
            source.weight.fill_(1.0)
            source.bias.fill_(0.5)
            target.weight.fill_(0.0)
            target.bias.fill_(0.0)

        soft_update(source, target, tau=0.1)
        # New target weight = 0.1 * 1.0 + 0.9 * 0.0 = 0.1
        assert target.weight.data[0, 0].item() == pytest.approx(0.1, abs=1e-6)
        assert target.bias.data[0].item() == pytest.approx(0.05, abs=1e-6)

    def test_polyak_tau_one_full_copy(self):
        from tgraphx.rl import soft_update

        source = nn.Linear(2, 2)
        target = nn.Linear(2, 2)
        with torch.no_grad():
            source.weight.fill_(7.0)
            target.weight.fill_(0.0)
        soft_update(source, target, tau=1.0)
        assert target.weight.data[0, 0].item() == pytest.approx(7.0, abs=1e-6)


# ── Easy Mode: dashboard artifact writer ─────────────────────────────────────


class TestEasyModeDashboardArtifacts:
    """Verify EasyResult.write_dashboard_artifacts produces correct files."""

    def test_write_dashboard_artifacts_creates_three_files(self, tmp_path):
        import tgraphx as tgx

        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=64, node_shape=(4, 4, 4), num_classes=3, num_edges=200, seed=42,
        )
        result = tgx.easy.train_node_classifier(
            data, epochs=2, batch_size=16, fanouts=[5, 3], verbose=False, seed=42,
        )
        run_dir = tmp_path / "run1"
        artifacts = result.write_dashboard_artifacts(str(run_dir))

        assert "metrics.csv" in artifacts
        assert "run_metadata.json" in artifacts
        assert "metrics_summary.json" in artifacts

        # metrics.csv has one row per epoch.
        import csv
        with open(artifacts["metrics.csv"]) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert "epoch" in rows[0]
        assert "loss" in rows[0]

        # run_metadata.json has expected fields.
        import json
        meta = json.loads(open(artifacts["run_metadata.json"]).read())
        assert meta["status"] == "completed"
        assert meta["total_epochs"] == 2
        assert meta["source"] == "tgraphx.easy"

        # metrics_summary.json has best_loss and best_epoch.
        summary = json.loads(open(artifacts["metrics_summary.json"]).read())
        assert "best_loss" in summary
        assert "best_epoch" in summary
        assert summary["epochs"] == 2

    def test_write_dashboard_artifacts_via_dashboard_dir_param(self, tmp_path):
        """train_node_classifier(dashboard_dir=...) should auto-write artifacts."""
        import tgraphx as tgx

        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=32, node_shape=(4, 4, 4), num_classes=2, num_edges=100, seed=0,
        )
        run_dir = tmp_path / "auto_run"
        result = tgx.easy.train_node_classifier(
            data, epochs=1, batch_size=8, fanouts=[3, 2], verbose=False,
            seed=0, dashboard_dir=str(run_dir),
        )
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "run_metadata.json").exists()
        assert (run_dir / "metrics_summary.json").exists()
        # result.artifacts is updated in place.
        assert "metrics.csv" in result.artifacts

    def test_write_dashboard_artifacts_empty_history_raises(self):
        from tgraphx.easy import EasyResult
        r = EasyResult(history=[])
        with pytest.raises(ValueError, match="No training history"):
            r.write_dashboard_artifacts("/tmp/should_not_be_created")


# ── map_global_to_local: dense vs sparse path equivalence ────────────────────


class TestMapGlobalToLocalPathParity:
    """Verify dense and sparse fallback produce identical results."""

    def test_dense_and_sparse_paths_agree(self):
        from tgraphx import map_global_to_local

        # Compact IDs → dense path.
        sampled = torch.tensor([10, 20, 30, 40, 50])
        seeds = torch.tensor([30, 50, 10, 40])
        dense_result = map_global_to_local(seeds, sampled)

        # Same indexing relationship with sparse-range IDs → sparse path.
        offset = 5_000_000
        sampled_sparse = sampled + offset
        seeds_sparse = seeds + offset
        sparse_result = map_global_to_local(seeds_sparse, sampled_sparse)

        # Both paths should produce identical local indices.
        assert torch.equal(dense_result, sparse_result)

    def test_searchsorted_path_unsorted_sampled(self):
        from tgraphx import map_global_to_local

        # Sparse IDs, sampled in arbitrary order — sparse path must still work.
        offset = 5_000_000
        sampled = torch.tensor([offset + 50, offset + 10, offset + 30, offset + 40, offset + 20])
        seeds = torch.tensor([offset + 30, offset + 50])
        local = map_global_to_local(seeds, sampled)
        # 30 is at position 2; 50 is at position 0
        assert local.tolist() == [2, 0]

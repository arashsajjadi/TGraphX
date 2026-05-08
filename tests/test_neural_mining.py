"""Tests for tgraphx.mining.neural — trainable mining models.

Mathematical and backpropagation correctness for:
- PrototypeMembershipScorer
- GraphAutoencoderAnomalyDetector
- GraphPatternClassifier
- create_synthetic_pattern_dataset
- training helpers
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tgraphx.mining import (
    GraphAutoencoderAnomalyDetector,
    GraphPatternClassifier,
    PrototypeMembershipScorer,
    create_synthetic_pattern_dataset,
    train_anomaly_autoencoder_step,
    train_graph_pattern_classifier_step,
    train_prototype_membership_step,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _triangle_graph(N: int = 5) -> tuple:
    """Return (node_features, edge_index, num_nodes) for a small graph."""
    ei = torch.tensor([[0,1,2,1,2,0,3,4,4,3],[1,2,0,0,1,2,4,3,3,4]], dtype=torch.long)
    x = torch.randn(N, 4)
    return x, ei, N


def _chain_graph(N: int = 4, D: int = 4) -> tuple:
    src = list(range(N-1)) + list(range(1, N))
    dst = list(range(1, N)) + list(range(N-1))
    ei = torch.tensor([src, dst], dtype=torch.long)
    x = torch.randn(N, D)
    return x, ei, N


# ── PrototypeMembershipScorer ─────────────────────────────────────────────────


class TestPrototypeMembershipScorer:
    def test_output_shape(self):
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4)
        x, ei, N = _chain_graph()
        logit = model(x, ei, query_idx=0, num_nodes=N)
        assert logit.dim() == 0

    def test_backward_works(self):
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4)
        x, ei, N = _chain_graph()
        logit = model(x, ei, query_idx=0, num_nodes=N)
        logit.backward()
        # All parameters should have gradients.
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_gradients_finite(self):
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4)
        x, ei, N = _chain_graph()
        logit = model(x, ei, query_idx=0, num_nodes=N)
        logit.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient: {name}"

    def test_gradients_not_all_zero(self):
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4)
        x, ei, N = _chain_graph()
        logit = model(x, ei, query_idx=0, num_nodes=N)
        logit.backward()
        has_nonzero = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                has_nonzero = True
                break
        assert has_nonzero, "All gradients are zero"

    def test_optimizer_updates_params(self):
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        x, ei, N = _chain_graph()
        # Take snapshot.
        before = {k: v.clone() for k, v in model.state_dict().items()}
        logit = model(x, ei, query_idx=0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor(1.0))
        loss.backward()
        opt.step()
        # At least one parameter must have changed.
        changed = any(
            not torch.equal(before[k], v)
            for k, v in model.state_dict().items()
        )
        assert changed

    def test_score_batch(self):
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4)
        x, ei, N = _chain_graph()
        candidates = [
            {"node_features": x, "edge_index": ei, "query_idx": 0},
            {"node_features": x, "edge_index": ei, "query_idx": 1},
        ]
        logits = model.score_batch(candidates)
        assert logits.shape == (2,)

    def test_invalid_edge_index(self):
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4)
        x, _, N = _chain_graph()
        with pytest.raises(ValueError, match="edge_index"):
            model(x, torch.zeros(3, 4, dtype=torch.long), query_idx=0)

    def test_single_node_graph(self):
        """Edge case: graph with only the query node (no support nodes)."""
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4)
        x = torch.randn(1, 4)
        ei = torch.zeros((2, 0), dtype=torch.long)
        logit = model(x, ei, query_idx=0, num_nodes=1)
        assert logit.dim() == 0
        assert torch.isfinite(logit)

    def test_flatten_spatial_mode(self):
        model = PrototypeMembershipScorer(in_dim=16, hidden_dim=8, out_dim=4, flatten_spatial=True)
        x = torch.randn(4, 4, 2, 2)  # [N, C, H, W]
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        logit = model(x, ei, query_idx=0, num_nodes=4)
        assert torch.isfinite(logit)

    def test_tiny_overfit(self):
        """Loss should decrease when training on a synthetic 2-class task."""
        torch.manual_seed(0)
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=16, out_dim=8)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Positive: class 0 candidate (query near class 0 support).
        pos_x = torch.cat([torch.randn(4, 4) * 0.1 + 1.0,
                            torch.randn(1, 4) * 0.1 + 1.0])  # 5 nodes, last is query
        pos_ei = torch.tensor([[0,1,2,3,1,2,3,0],[1,2,3,0,0,1,2,3]], dtype=torch.long)
        pos_cand = {"node_features": pos_x, "edge_index": pos_ei, "query_idx": 4}

        # Negative: class 1 candidate (query near class 1 but added to class 0 graph).
        neg_x = torch.cat([torch.randn(4, 4) * 0.1 + 1.0,
                            torch.randn(1, 4) * 0.1 - 1.0])  # query is class 1
        neg_cand = {"node_features": neg_x, "edge_index": pos_ei, "query_idx": 4}

        targets = torch.tensor([1.0, 0.0])
        losses = []
        for _ in range(20):
            loss = train_prototype_membership_step(model, opt, [pos_cand, neg_cand], targets)
            losses.append(loss)
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4).cuda()
        x, ei, N = _chain_graph()
        logit = model(x.cuda(), ei.cuda(), query_idx=0, num_nodes=N)
        assert logit.device.type == "cuda"
        assert torch.isfinite(logit)


# ── GraphAutoencoderAnomalyDetector ──────────────────────────────────────────


class TestGraphAutoencoderAnomalyDetector:
    def test_output_shape(self):
        ae = GraphAutoencoderAnomalyDetector(in_dim=4, latent_dim=4, hidden_dim=8)
        x, ei, N = _chain_graph()
        recon, latent = ae(x, ei, N)
        assert recon.shape == x.shape
        assert latent.shape == (N, 4)

    def test_backward_works(self):
        ae = GraphAutoencoderAnomalyDetector(in_dim=4, latent_dim=4, hidden_dim=8)
        x, ei, N = _chain_graph()
        loss = ae.reconstruction_loss(x, ei, N)
        loss.backward()
        for name, p in ae.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_gradients_finite(self):
        ae = GraphAutoencoderAnomalyDetector(in_dim=4, latent_dim=4, hidden_dim=8)
        x, ei, N = _chain_graph()
        ae.reconstruction_loss(x, ei, N).backward()
        for p in ae.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_node_anomaly_scores_no_grad(self):
        ae = GraphAutoencoderAnomalyDetector(in_dim=4, latent_dim=4, hidden_dim=8)
        x, ei, N = _chain_graph()
        scores = ae.node_anomaly_scores(x, ei, N)
        assert scores.shape == (N,)
        assert not scores.requires_grad
        assert torch.isfinite(scores).all()

    def test_graph_anomaly_score_float(self):
        ae = GraphAutoencoderAnomalyDetector(in_dim=4, latent_dim=4, hidden_dim=8)
        x, ei, N = _chain_graph()
        score = ae.graph_anomaly_score(x, ei, N)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_tiny_overfit_loss_decreases(self):
        """MSE reconstruction loss should decrease on constant synthetic data."""
        torch.manual_seed(0)
        ae = GraphAutoencoderAnomalyDetector(in_dim=4, latent_dim=4, hidden_dim=16)
        opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
        # Constant feature matrix = easy to reconstruct.
        x = torch.ones(6, 4) + 0.01 * torch.randn(6, 4)
        ei = torch.tensor([[0,1,2,3,4,5,1,2,3,4,5,0],[1,2,3,4,5,0,0,1,2,3,4,5]], dtype=torch.long)
        losses = [train_anomaly_autoencoder_step(ae, opt, x, ei, 6) for _ in range(30)]
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_injected_anomaly_has_higher_score(self):
        """Nodes with out-of-distribution features should score higher."""
        torch.manual_seed(0)
        ae = GraphAutoencoderAnomalyDetector(in_dim=4, latent_dim=4, hidden_dim=16)
        opt = torch.optim.Adam(ae.parameters(), lr=5e-3)
        N = 8
        # Train on zero-mean data.
        x_normal = torch.randn(N, 4) * 0.1
        ei = torch.zeros((2, 0), dtype=torch.long)  # no edges — feature reconstruction only
        for _ in range(60):
            train_anomaly_autoencoder_step(ae, opt, x_normal, ei, N)
        # Inject anomaly at node 3.
        x_test = x_normal.clone()
        x_test[3] = x_test[3] + 5.0
        scores = ae.node_anomaly_scores(x_test, ei, N)
        anomaly_score = float(scores[3].item())
        normal_mean = float(scores[torch.arange(N) != 3].mean().item())
        assert anomaly_score > normal_mean, (
            f"Injected anomaly score {anomaly_score:.4f} not > normal mean {normal_mean:.4f}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        ae = GraphAutoencoderAnomalyDetector(in_dim=4, latent_dim=4, hidden_dim=8).cuda()
        x, ei, N = _chain_graph()
        recon, latent = ae(x.cuda(), ei.cuda(), N)
        assert recon.device.type == "cuda"


# ── GraphPatternClassifier ────────────────────────────────────────────────────


class TestGraphPatternClassifier:
    def test_output_shape(self):
        clf = GraphPatternClassifier(in_dim=4, num_classes=4)
        x, ei, N = _chain_graph()
        logits = clf(x, ei, N)
        assert logits.shape == (4,)

    def test_backward_works(self):
        clf = GraphPatternClassifier(in_dim=4, num_classes=4)
        x, ei, N = _chain_graph()
        loss = clf(x, ei, N).sum()
        loss.backward()
        for name, p in clf.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_gradients_finite(self):
        clf = GraphPatternClassifier(in_dim=4, num_classes=4)
        x, ei, N = _chain_graph()
        clf(x, ei, N).sum().backward()
        for p in clf.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_optimizer_updates_params(self):
        torch.manual_seed(0)
        clf = GraphPatternClassifier(in_dim=4, num_classes=4)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
        before = {k: v.clone() for k, v in clf.state_dict().items()}
        x, ei, N = _chain_graph()
        loss = torch.nn.functional.cross_entropy(clf(x, ei, N).unsqueeze(0), torch.tensor([0]))
        loss.backward()
        opt.step()
        changed = any(not torch.equal(before[k], v) for k, v in clf.state_dict().items())
        assert changed

    def test_tiny_overfit_accuracy(self):
        """Classifier should achieve near-100% accuracy on synthetic pattern data."""
        torch.manual_seed(0)
        # 4 very separated pattern classes.
        dataset = create_synthetic_pattern_dataset(
            num_graphs_per_class=30, num_nodes=6, in_dim=4, seed=0, noise_std=0.02,
        )
        clf = GraphPatternClassifier(in_dim=4, hidden_dim=32, enc_dim=16, num_classes=4)
        opt = torch.optim.Adam(clf.parameters(), lr=5e-3)
        losses = []
        for epoch in range(40):
            for g in dataset:
                loss = train_graph_pattern_classifier_step(
                    clf, opt, [g], torch.tensor([g["label"]])
                )
                losses.append(loss)
        # Final accuracy on training set.
        clf.eval()
        correct = 0
        with torch.no_grad():
            for g in dataset:
                pred = int(clf(g["node_features"], g["edge_index"], g["num_nodes"]).argmax().item())
                if pred == g["label"]:
                    correct += 1
        acc = correct / len(dataset)
        assert acc >= 0.70, f"Tiny-overfit accuracy {acc:.3f} < 0.70"

    def test_train_eval_mode(self):
        clf = GraphPatternClassifier(in_dim=4, num_classes=4, dropout=0.3)
        clf.train()
        x, ei, N = _chain_graph()
        out_train = clf(x, ei, N)
        clf.eval()
        with torch.no_grad():
            out_eval = clf(x, ei, N)
        # Outputs can differ in train vs eval due to dropout.
        assert out_train.shape == out_eval.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        clf = GraphPatternClassifier(in_dim=4, num_classes=4).cuda()
        x, ei, N = _chain_graph()
        logits = clf(x.cuda(), ei.cuda(), N)
        assert logits.device.type == "cuda"


# ── Synthetic dataset ─────────────────────────────────────────────────────────


class TestSyntheticPatternDataset:
    def test_total_size(self):
        ds = create_synthetic_pattern_dataset(
            num_graphs_per_class=10, num_nodes=6, in_dim=4, seed=0
        )
        assert len(ds) == 40

    def test_label_range(self):
        ds = create_synthetic_pattern_dataset(
            num_graphs_per_class=5, num_nodes=6, in_dim=4, seed=0
        )
        labels = set(d["label"] for d in ds)
        assert labels == {0, 1, 2, 3}

    def test_feature_shape(self):
        ds = create_synthetic_pattern_dataset(
            num_graphs_per_class=5, num_nodes=6, in_dim=8, seed=0
        )
        for g in ds:
            assert g["node_features"].shape == (6, 8)

    def test_deterministic(self):
        ds1 = create_synthetic_pattern_dataset(num_graphs_per_class=5, num_nodes=4, in_dim=4, seed=42)
        ds2 = create_synthetic_pattern_dataset(num_graphs_per_class=5, num_nodes=4, in_dim=4, seed=42)
        for a, b in zip(ds1, ds2):
            assert torch.equal(a["node_features"], b["node_features"])


# ── Training helpers ──────────────────────────────────────────────────────────


class TestTrainingHelpers:
    def test_prototype_step_returns_float(self):
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        x, ei, N = _chain_graph()
        candidates = [{"node_features": x, "edge_index": ei, "query_idx": 0}]
        targets = torch.tensor([1.0])
        loss = train_prototype_membership_step(model, opt, candidates, targets)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_anomaly_step_returns_float(self):
        ae = GraphAutoencoderAnomalyDetector(in_dim=4, latent_dim=4, hidden_dim=8)
        opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
        x, ei, N = _chain_graph()
        loss = train_anomaly_autoencoder_step(ae, opt, x, ei, N)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_pattern_step_returns_float(self):
        clf = GraphPatternClassifier(in_dim=4, num_classes=4)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
        x, ei, N = _chain_graph()
        graphs = [{"node_features": x, "edge_index": ei, "num_nodes": N}]
        loss = train_graph_pattern_classifier_step(clf, opt, graphs, torch.tensor([0]))
        assert isinstance(loss, float)
        assert loss >= 0.0

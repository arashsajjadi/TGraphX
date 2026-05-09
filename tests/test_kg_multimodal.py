"""Tests for multimodal tensor-aware KG support.

Covers:
- Data model: entity_types, entity_feature_masks, to(device), detach_for_report
- Projectors: vector, image, user, text, relation, triple
- MultimodalEntityFusion: all three fusion modes
- MultimodalKGModel: score sensitivity, backprop, learning
- Score sensitivity: changing features changes scores
- No silent flattening of image tensors
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from tgraphx.kg import (
    KnowledgeGraph,
    MultimodalKGModel,
    VectorEntityProjector,
    ImageEntityProjector,
    TextEntityProjector,
    UserEntityProjector,
    RelationFeatureProjector,
    TripleFeatureProjector,
    MultimodalEntityFusion,
)
from tgraphx.kg import generate_synthetic_kg


# ── Helpers ──────────────────────────────────────────────────────────────────


def _multimodal_kg(N_e: int = 12, N_r: int = 3, seed: int = 0) -> KnowledgeGraph:
    """KG with 3 entity types: image (0), user (1), text (2), 4 each."""
    torch.manual_seed(seed)
    n_per_type = N_e // 3
    et = torch.tensor([0] * n_per_type + [1] * n_per_type + [2] * (N_e - 2 * n_per_type))
    im_mask = et == 0
    us_mask = et == 1
    tx_mask = et == 2
    triples = torch.tensor([[0, 0, 4], [4, 1, 8], [0, 2, 8], [4, 0, 8]], dtype=torch.long)
    return KnowledgeGraph(
        triples, num_entities=N_e, num_relations=N_r,
        entity_types=et,
        entity_feature_masks={"image": im_mask, "user": us_mask, "text": tx_mask},
        entity_features={
            "image": torch.randn(N_e, 3, 8, 8),
            "user": torch.randn(N_e, 16),
            "text": torch.randn(N_e, 8),
        },
        relation_features={"r": torch.randn(N_r, 4)},
        entity_type_to_id={"image": 0, "user": 1, "text": 2},
    )


# ── Data model tests ─────────────────────────────────────────────────────────


class TestMultimodalDataModel:

    def test_entity_types_preserved(self):
        kg = _multimodal_kg()
        assert kg.entity_types is not None
        assert kg.entity_types.shape == (12,)
        assert int((kg.entity_types == 0).sum()) == 4
        assert int((kg.entity_types == 1).sum()) == 4
        assert int((kg.entity_types == 2).sum()) == 4

    def test_entity_feature_masks_preserved(self):
        kg = _multimodal_kg()
        assert "image" in kg.entity_feature_masks
        assert int(kg.entity_feature_masks["image"].sum()) == 4
        assert int(kg.entity_feature_masks["user"].sum()) == 4

    def test_image_tensor_not_flattened(self):
        kg = _multimodal_kg()
        assert kg.entity_features["image"].shape == (12, 3, 8, 8)

    def test_text_embedding_shape_preserved(self):
        kg = _multimodal_kg()
        assert kg.entity_features["text"].shape == (12, 8)

    def test_user_vector_shape_preserved(self):
        kg = _multimodal_kg()
        assert kg.entity_features["user"].shape == (12, 16)

    def test_entity_type_counts(self):
        kg = _multimodal_kg()
        counts = kg.entity_type_counts()
        assert counts.get("image", 0) == 4
        assert counts.get("user", 0) == 4
        assert counts.get("text", 0) == 4

    def test_entity_type_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="entity_types"):
            KnowledgeGraph(
                torch.zeros(2, 3, dtype=torch.long),
                num_entities=3, num_relations=1,
                entity_types=torch.zeros(5, dtype=torch.long),  # wrong length
            )

    def test_mask_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="entity_feature_masks"):
            KnowledgeGraph(
                torch.zeros(2, 3, dtype=torch.long),
                num_entities=3, num_relations=1,
                entity_feature_masks={"x": torch.zeros(5, dtype=torch.bool)},
            )

    def test_to_device_moves_all_fields(self):
        kg = _multimodal_kg()
        kg.to("cpu")
        assert kg.entity_types.device.type == "cpu"
        for v in kg.entity_feature_masks.values():
            assert v.device.type == "cpu"
        for v in kg.entity_features.values():
            assert v.device.type == "cpu"

    def test_detach_for_report_no_autograd(self):
        kg = _multimodal_kg()
        kg.entity_features["user"] = torch.randn(12, 16, requires_grad=True)
        kg2 = kg.detach_for_report()
        assert not kg2.entity_features["user"].requires_grad
        assert not kg2.entity_types.requires_grad

    def test_summary_json_safe(self):
        import json
        kg = _multimodal_kg()
        s = kg.summary()
        json.dumps(s)
        assert s["has_entity_types"] is True
        assert "entity_type_counts" in s
        assert "entity_feature_masks" in s

    def test_summary_mask_coverage_correct(self):
        kg = _multimodal_kg()
        s = kg.summary()
        assert s["entity_feature_masks"]["image"]["coverage"] == 4
        assert s["entity_feature_masks"]["image"]["total"] == 12

    def test_clone_preserves_multimodal_fields(self):
        kg = _multimodal_kg()
        kg2 = kg.clone()
        assert kg2.entity_types is not None
        assert torch.equal(kg2.entity_types, kg.entity_types)
        assert "image" in kg2.entity_feature_masks


# ── Projector tests ───────────────────────────────────────────────────────────


class TestProjectors:

    def test_vector_projector_shape(self):
        proj = VectorEntityProjector(16, 32)
        x = torch.randn(10, 16)
        out = proj(x)
        assert out.shape == (10, 32)

    def test_image_projector_shape_not_flattened(self):
        proj = ImageEntityProjector(3, 32)
        img = torch.randn(10, 3, 8, 8)
        out = proj(img)
        assert out.shape == (10, 32)

    def test_image_projector_wrong_dim_raises(self):
        proj = ImageEntityProjector(3, 32)
        with pytest.raises(ValueError, match="NOT silently flattened"):
            proj(torch.randn(10, 192))  # flattened image — should raise

    def test_vector_projector_wrong_dim_raises(self):
        proj = VectorEntityProjector(4, 8)
        with pytest.raises(ValueError, match="2-D"):
            proj(torch.randn(5))  # 1-D

    def test_relation_projector(self):
        proj = RelationFeatureProjector(4, 8)
        out = proj(torch.randn(3, 4))
        assert out.shape == (3, 8)

    def test_triple_projector(self):
        proj = TripleFeatureProjector(6, 8)
        out = proj(torch.randn(10, 6))
        assert out.shape == (10, 8)

    def test_projectors_differentiable(self):
        for proj_cls, feat in [
            (lambda: VectorEntityProjector(4, 8), torch.randn(5, 4, requires_grad=True)),
            (lambda: ImageEntityProjector(3, 8), torch.randn(5, 3, 4, 4, requires_grad=True)),
        ]:
            p = proj_cls()
            out = p(feat)
            out.sum().backward()
            assert feat.grad is not None
            assert torch.isfinite(feat.grad).all()


# ── Fusion tests ──────────────────────────────────────────────────────────────


class TestMultimodalEntityFusion:

    def _fusion(self, mode: str, N: int = 8, D: int = 8):
        projectors = {
            "image": ImageEntityProjector(3, D),
            "user": VectorEntityProjector(4, D),
        }
        return MultimodalEntityFusion(projectors, D, N, fusion_mode=mode)

    def test_add_mode_shape(self):
        f = self._fusion("add")
        feats = {"image": torch.randn(8, 3, 4, 4), "user": torch.randn(8, 4)}
        out = f(feats)
        assert out.shape == (8, 8)

    def test_gated_mode_shape(self):
        f = self._fusion("gated")
        feats = {"image": torch.randn(8, 3, 4, 4), "user": torch.randn(8, 4)}
        out = f(feats)
        assert out.shape == (8, 8)

    def test_concat_project_mode_shape(self):
        f = self._fusion("concat_project")
        feats = {"image": torch.randn(8, 3, 4, 4), "user": torch.randn(8, 4)}
        out = f(feats)
        assert out.shape == (8, 8)

    def test_mask_zeros_absent_entities(self):
        """Entities with mask=False should have zero contribution from that modality."""
        D, N = 4, 6
        proj = VectorEntityProjector(4, D, activation=False)
        # Only entities 0-2 have user features (mask).
        mask = torch.tensor([True, True, True, False, False, False])
        fusion = MultimodalEntityFusion({"user": proj}, D, N, fusion_mode="add",
                                        add_learnable_bias=False)
        feats = {"user": torch.ones(N, 4)}
        out = fusion(feats, {"user": mask})
        # Entities 3,4,5 should be zero.
        assert torch.equal(out[3:], torch.zeros(3, D))
        # Entities 0,1,2 should be nonzero.
        assert (out[:3].abs() > 0).any()

    def test_gated_mode_backward(self):
        f = self._fusion("gated")
        feats = {"image": torch.randn(8, 3, 4, 4), "user": torch.randn(8, 4)}
        out = f(feats)
        out.sum().backward()
        for name, param in f.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.isfinite(param.grad).all()


# ── MultimodalKGModel tests ───────────────────────────────────────────────────


class TestMultimodalKGModel:

    def _model(self, kg: KnowledgeGraph, D: int = 16) -> MultimodalKGModel:
        return MultimodalKGModel(
            kg.num_entities, kg.num_relations, D,
            projectors={
                "image": ImageEntityProjector(3, D),
                "user": VectorEntityProjector(16, D),
                "text": TextEntityProjector(8, D),
            },
            fusion_mode="gated",
        )

    def test_score_shape(self):
        kg = _multimodal_kg()
        model = self._model(kg)
        scores = model.score_from_kg(kg, kg.triples)
        assert scores.shape == (kg.num_triples,)

    def test_scores_finite(self):
        kg = _multimodal_kg()
        model = self._model(kg)
        scores = model.score_from_kg(kg, kg.triples)
        assert torch.isfinite(scores).all()

    def test_image_feature_affects_score(self):
        """Changing image feature must change the score."""
        kg = _multimodal_kg()
        model = self._model(kg)
        s1 = model.score_from_kg(kg, kg.triples[:1]).detach()
        kg2 = kg.clone()
        kg2.entity_features["image"] = torch.randn_like(kg.entity_features["image"]) * 10
        s2 = model.score_from_kg(kg2, kg.triples[:1]).detach()
        assert not torch.allclose(s1, s2), "Image feature change did not affect score"

    def test_user_feature_affects_score(self):
        kg = _multimodal_kg()
        model = self._model(kg)
        s1 = model.score_from_kg(kg, kg.triples[:1]).detach()
        kg2 = kg.clone()
        kg2.entity_features["user"] = torch.randn_like(kg.entity_features["user"]) * 10
        s2 = model.score_from_kg(kg2, kg.triples[:1]).detach()
        assert not torch.allclose(s1, s2)

    def test_text_feature_affects_score(self):
        # Use triple [4, 1, 8]: tail is entity 8 (text type, text_mask=True).
        kg = _multimodal_kg()
        model = self._model(kg)
        text_triple = kg.triples[1:2]  # [4, 1, 8] — entity 8 is text type
        s1 = model.score_from_kg(kg, text_triple).detach()
        kg2 = kg.clone()
        kg2.entity_features["text"] = torch.randn_like(kg.entity_features["text"]) * 10
        s2 = model.score_from_kg(kg2, text_triple).detach()
        assert not torch.allclose(s1, s2), "Text feature change did not affect score"

    def test_backprop_gradients_finite(self):
        kg = _multimodal_kg()
        model = self._model(kg)
        scores = model.score_from_kg(kg, kg.triples)
        scores.sum().backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"{name} has non-finite grad"

    def test_image_projector_gradients_nonzero(self):
        kg = _multimodal_kg()
        model = self._model(kg)
        scores = model.score_from_kg(kg, kg.triples)
        scores.sum().backward()
        img_proj = model.fusion.projectors["image"]
        assert img_proj.proj.weight.grad is not None
        assert img_proj.proj.weight.grad.abs().sum().item() > 0

    def test_user_projector_gradients_nonzero(self):
        kg = _multimodal_kg()
        model = self._model(kg)
        scores = model.score_from_kg(kg, kg.triples)
        scores.sum().backward()
        user_proj = model.fusion.projectors["user"]
        assert user_proj.proj.weight.grad is not None
        assert user_proj.proj.weight.grad.abs().sum().item() > 0

    def test_text_projector_gradients_nonzero(self):
        kg = _multimodal_kg()
        model = self._model(kg)
        scores = model.score_from_kg(kg, kg.triples)
        scores.sum().backward()
        text_proj = model.fusion.projectors["text"]
        assert text_proj.proj.weight.grad is not None
        assert text_proj.proj.weight.grad.abs().sum().item() > 0

    def test_optimizer_step_changes_projector_params(self):
        kg = _multimodal_kg()
        model = self._model(kg)
        img_w_before = model.fusion.projectors["image"].proj.weight.clone().detach()
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        opt.zero_grad()
        scores = model.score_from_kg(kg, kg.triples)
        scores.sum().backward()
        opt.step()
        img_w_after = model.fusion.projectors["image"].proj.weight.clone().detach()
        assert not torch.equal(img_w_before, img_w_after)

    def test_learning_loss_decreases(self):
        """Loss must decrease after multiple optimizer steps on tiny multimodal KG."""
        torch.manual_seed(0)
        kg = _multimodal_kg(N_e=12, N_r=3)
        model = self._model(kg, D=8)
        from tgraphx.kg import UniformNegativeSampler, SoftplusKGLoss
        sampler = UniformNegativeSampler(kg.num_entities, 2)
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        gen = torch.Generator().manual_seed(0)
        loss_fn = SoftplusKGLoss()
        losses = []
        for _ in range(30):
            opt.zero_grad()
            neg = sampler.sample(kg.triples, generator=gen).view(-1, 3)
            pos_scores = model.score_from_kg(kg, kg.triples)
            neg_scores = model.score_from_kg(kg, neg)
            loss = loss_fn(pos_scores, neg_scores)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().item()))
        avg_first = sum(losses[:5]) / 5
        avg_last = sum(losses[-5:]) / 5
        assert avg_last < avg_first, \
            f"Loss did not decrease: {avg_first:.4f} -> {avg_last:.4f}"

    def test_no_train_valid_leakage(self):
        kg = _multimodal_kg()
        tr, va, te = kg.train_valid_test_split(0.7, 0.15, 0.15, seed=0)
        tr_set = set(map(tuple, tr.triples.tolist()))
        va_set = set(map(tuple, va.triples.tolist()))
        assert not (tr_set & va_set)

    def test_cuda_smoke(self):
        """CUDA smoke — skip cleanly if unavailable."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        kg = _multimodal_kg()
        kg.to("cuda")
        model = self._model(kg)
        model = model.cuda()
        scores = model.score_from_kg(kg, kg.triples.cuda())
        assert scores.shape == (kg.num_triples,)
        assert torch.isfinite(scores).all()

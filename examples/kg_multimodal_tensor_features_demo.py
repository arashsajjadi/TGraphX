"""Multimodal tensor-aware KG demonstration.

Shows a KG with three entity types:
  - Image entities    (image-like [N, C, H, W] features)
  - User entities     (vector profile features)
  - Text entities     (pre-computed text embedding features)

Triples:
  (image, viewedBy, user)
  (user, wrote, text)
  (text, describes, image)

The MultimodalKGModel learns entity embeddings from typed features and
scores triples via DistMult.

Usage:
    python examples/kg_multimodal_tensor_features_demo.py
    python examples/kg_multimodal_tensor_features_demo.py --epochs 50 --dim 32
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from tgraphx.kg import (
    KnowledgeGraph,
    MultimodalKGModel,
    ImageEntityProjector,
    VectorEntityProjector,
    UniformNegativeSampler,
    SoftplusKGLoss,
    FilteredNegativeSampler,
)
from tgraphx.kg.reports import write_kg_summary, write_kg_training_report


def _build_multimodal_kg(
    n_images: int = 10,
    n_users: int = 8,
    n_texts: int = 8,
    seed: int = 0,
) -> KnowledgeGraph:
    """Build a synthetic image/user/text KG."""
    torch.manual_seed(seed)
    N_e = n_images + n_users + n_texts
    N_r = 3  # viewedBy, wrote, describes

    # Entity type IDs.
    et = torch.tensor([0] * n_images + [1] * n_users + [2] * n_texts)

    # Index offsets.
    u_off = n_images
    t_off = n_images + n_users

    # Triples: (image→viewedBy→user), (user→wrote→text), (text→describes→image).
    triples = []
    for i in range(n_images):
        u = u_off + (i % n_users)
        triples.append((i, 0, u))          # image viewedBy user
    for u in range(n_users):
        t = t_off + (u % n_texts)
        triples.append((u_off + u, 1, t))  # user wrote text
    for t in range(n_texts):
        i = t % n_images
        triples.append((t_off + t, 2, i))  # text describes image
    triples_t = torch.tensor(triples, dtype=torch.long)

    # Modality masks.
    img_mask = et == 0
    usr_mask = et == 1
    txt_mask = et == 2

    # Typed entity features.
    img_feat = torch.randn(N_e, 3, 8, 8)    # image features [N_e, C, H, W]
    usr_feat = torch.randn(N_e, 16)          # user profile vector
    txt_feat = torch.randn(N_e, 8)           # text embedding

    return KnowledgeGraph(
        triples_t,
        num_entities=N_e,
        num_relations=N_r,
        entity_types=et,
        entity_feature_masks={"image": img_mask, "user": usr_mask, "text": txt_mask},
        entity_features={"image": img_feat, "user": usr_feat, "text": txt_feat},
        entity_type_to_id={"image": 0, "user": 1, "text": 2},
        relation_to_id={"viewedBy": 0, "wrote": 1, "describes": 2},
        metadata={"dataset": "multimodal_demo", "seed": seed},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal tensor KG demo")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-run-dir", default="logs/kg_multimodal_demo")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print("=" * 60)
    print("Multimodal Tensor-Aware Knowledge Graph Demo")
    print("=" * 60)

    kg = _build_multimodal_kg(seed=args.seed)

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\nKG: {kg}")
    print(f"Entity type counts: {kg.entity_type_counts()}")
    print(f"Relations: {kg.relation_to_id}")
    print(f"Feature shapes:")
    for name, feat in kg.entity_features.items():
        mask_cov = int(kg.entity_feature_masks.get(name, torch.ones(kg.num_entities, dtype=torch.bool)).sum())
        print(f"  '{name}': shape={list(feat.shape)}, coverage={mask_cov}/{kg.num_entities}")
    print(f"  (image tensors kept as [N, C, H, W] — NOT flattened)")

    # ── Build model ────────────────────────────────────────────────────────
    D = args.dim
    model = MultimodalKGModel(
        kg.num_entities, kg.num_relations, D,
        projectors={
            "image": ImageEntityProjector(3, D),
            "user": VectorEntityProjector(16, D),
            "text": VectorEntityProjector(8, D),
        },
        fusion_mode="gated",
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: MultimodalKGModel (gated fusion, D={D}, {n_params} params)")

    # ── Training ──────────────────────────────────────────────────────────
    sampler = FilteredNegativeSampler(
        kg.num_entities, 2, positive_set=kg.positive_triple_set(),
        base_sampler=UniformNegativeSampler(kg.num_entities, 1),
    )
    loss_fn = SoftplusKGLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    losses = []
    print(f"\nTraining {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        opt.zero_grad()
        neg = sampler.sample(kg.triples, generator=gen).view(-1, 3)
        pos_scores = model.score_from_kg(kg, kg.triples)
        neg_scores = model.score_from_kg(kg, neg)
        loss = loss_fn(pos_scores, neg_scores)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))

    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss:   {losses[-1]:.4f}")
    print(f"Loss decreased: {losses[-1] < losses[0]}")

    # ── Score check ───────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        pos_s = model.score_from_kg(kg, kg.triples[:4])
        neg_sample = sampler.sample(kg.triples[:4], generator=gen).view(-1, 3)
        neg_s = model.score_from_kg(kg, neg_sample)
    print(f"\nPositive scores (mean): {pos_s.mean():.4f}")
    print(f"Negative scores (mean): {neg_s.mean():.4f}")
    print(f"Positive > Negative: {pos_s.mean() > neg_s.mean()}")

    # ── Gradient check ────────────────────────────────────────────────────
    print("\nProjector gradient status (from last training step):")
    for name, proj in model.fusion.projectors.items():
        g = proj.proj.weight.grad
        gstatus = f"grad_norm={g.norm().item():.4f}" if g is not None else "no_grad"
        print(f"  {name:8s}: {gstatus}")

    # ── Dashboard artifact ────────────────────────────────────────────────
    os.makedirs(args.output_run_dir, exist_ok=True)
    summary_path = os.path.join(args.output_run_dir, "kg_summary.json")
    write_kg_summary(summary_path, kg.summary())
    training_path = os.path.join(args.output_run_dir, "kg_training_report.json")
    write_kg_training_report(training_path, {
        "model": "MultimodalKGModel", "fusion_mode": "gated",
        "embedding_dim": D, "num_params": n_params,
        "loss_history": losses,
        "final_loss": losses[-1], "loss_decreased": losses[-1] < losses[0],
        "entity_type_counts": kg.entity_type_counts(),
        "modality_features": {k: list(v.shape) for k, v in kg.entity_features.items()},
    })
    print(f"\nArtifacts written to: {args.output_run_dir}")
    print(f"  {summary_path}")
    print(f"  {training_path}")
    print("\nDemo complete.")


if __name__ == "__main__":
    main()

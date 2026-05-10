"""KG benchmark quickstart with filtered MRR / Hits@K.

Trains a small KG embedding model (default: TransE) on a deterministic
synthetic KG and reports filtered MRR/Hits@K against held-out test triples.

No network required.  No PyKEEN dependency.

Usage::

    python tutorials/kg_benchmark_quickstart.py
    python tutorials/kg_benchmark_quickstart.py --model RESCAL --epochs 30
"""
from __future__ import annotations

import argparse
import time

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="TransE",
                   choices=["TransE", "DistMult", "RESCAL"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def _build_synthetic_kg(seed: int):
    """Deterministic synthetic KG with 50 entities, 4 relations, 200 triples."""
    torch.manual_seed(seed)
    N_e, N_r, N_t = 50, 4, 200
    heads = torch.randint(0, N_e, (N_t,))
    rels = torch.randint(0, N_r, (N_t,))
    tails = torch.randint(0, N_e, (N_t,))
    # Drop self-loops for cleanliness.
    keep = heads != tails
    heads, rels, tails = heads[keep], rels[keep], tails[keep]
    return heads, rels, tails, N_e, N_r


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    from tgraphx.kg import (
        KnowledgeGraph,
        TransEModel, DistMultModel, RESCALModel,
        evaluate_filtered_ranking,
    )

    heads, rels, tails, N_e, N_r = _build_synthetic_kg(args.seed)
    triples = torch.stack([heads, rels, tails], dim=1)

    # 80/10/10 train/val/test split.
    perm = torch.randperm(triples.size(0))
    n_train = int(0.8 * triples.size(0))
    n_val = int(0.1 * triples.size(0))
    train = triples[perm[:n_train]].to(device)
    val = triples[perm[n_train:n_train + n_val]].to(device)
    test = triples[perm[n_train + n_val:]].to(device)
    print(f"KG: {N_e} entities, {N_r} relations, "
          f"{train.size(0)} train / {val.size(0)} val / {test.size(0)} test triples")

    if args.model == "TransE":
        model = TransEModel(N_e, N_r, embedding_dim=32).to(device)
    elif args.model == "DistMult":
        model = DistMultModel(N_e, N_r, embedding_dim=32).to(device)
    else:
        model = RESCALModel(N_e, N_r, embedding_dim=16).to(device)
    print(f"Model: {args.model}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    # ---- Margin training with random tail-corruption negatives ---------
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # Negatives: corrupt tails uniformly at random.
        neg_tails = torch.randint(0, N_e, (train.size(0),), device=device)
        neg = train.clone()
        neg[:, 2] = neg_tails

        pos_score = model.score_triples(train)
        neg_score = model.score_triples(neg)
        loss = (1.0 + neg_score - pos_score).clamp(min=0.0).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % max(1, args.epochs // 5) == 0 or epoch == 1:
            print(f"  epoch {epoch:>3d}/{args.epochs}  loss={loss.item():.4f}")

    train_time = time.time() - t0

    # ---- Filtered ranking on the test set ------------------------------
    all_pos = set(map(tuple, triples.tolist()))
    eval_t0 = time.time()
    result = evaluate_filtered_ranking(
        model, test, all_pos, num_entities=N_e,
        filtered=True, hits_at=(1, 3, 10),
    )
    eval_time = time.time() - eval_t0

    print(f"\nFiltered MRR : {result.filt_mrr:.4f}")
    print(f"Filtered MR  : {result.filt_mr:.2f}")
    for k, v in sorted(result.filt_hits.items()):
        print(f"Filtered H@{k:<2d}: {v:.4f}")
    print(f"\nTraining time:   {train_time:.1f}s")
    print(f"Evaluation time: {eval_time:.1f}s")
    print(f"Tutorial PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

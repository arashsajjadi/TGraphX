"""Smoke test: notebook 34 — MovieLens user–item KG recommendation.

Uses a synthetic multi-relational KG fallback by default. Pass
--no-download=False to attempt MovieLens 100K download.

Usage:
    python examples/advanced_real_datasets/movielens_kg_smoke.py
    python examples/advanced_real_datasets/movielens_kg_smoke.py --fast
    python examples/advanced_real_datasets/movielens_kg_smoke.py --fast --no-download
"""
from __future__ import annotations

import argparse
import math
import random
import time

import torch


def main(fast: bool = True, no_download: bool = True) -> None:
    from tgraphx.reproducibility import set_seed
    from tgraphx import KnowledgeGraph, KGTrainer, KGTrainingConfig, count_parameters
    from tgraphx.kg import TransEModel, KGEvaluator
    from tgraphx.kg.hpo import run_kg_hpo

    SEED = 42
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    MAX_USERS, MAX_MOVIES, NUM_GENRES, NUM_OCC = 30, 60, 5, 3
    MOVIE_OFFSET = MAX_USERS
    GENRE_OFFSET = MOVIE_OFFSET + MAX_MOVIES
    OCC_OFFSET = GENRE_OFFSET + NUM_GENRES
    NUM_ENTITIES = OCC_OFFSET + NUM_OCC
    NUM_RELATIONS = 4

    USING_REAL = False

    if not no_download:
        try:
            import csv, pathlib, urllib.request, zipfile
            ML_DIR = pathlib.Path("/tmp/ml-100k")
            if not ML_DIR.exists():
                url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
                urllib.request.urlretrieve(url, "/tmp/ml-100k.zip")
                with zipfile.ZipFile("/tmp/ml-100k.zip") as z:
                    z.extractall("/tmp/")
            triples_real = []
            with open(ML_DIR / "u.data") as f:
                for row in csv.reader(f, delimiter="\t"):
                    uid, mid, rating = int(row[0]), int(row[1]), int(row[2])
                    if uid <= MAX_USERS and mid <= MAX_MOVIES:
                        rel = 0 if rating >= 4 else 1
                        triples_real.append([uid - 1, rel, mid - 1 + MOVIE_OFFSET])
            if triples_real:
                triples_tensor = torch.tensor(triples_real, dtype=torch.long)
                USING_REAL = True
                print(f"[ML] Real ML-100K: {len(triples_real)} triples")
        except Exception as exc:
            print(f"[ML] MovieLens unavailable ({exc}), using synthetic.")

    if not USING_REAL:
        random.seed(SEED)
        rng = torch.Generator().manual_seed(SEED)
        triples = []
        for uid in range(MAX_USERS):
            for _ in range(5):
                mid = random.randint(0, MAX_MOVIES - 1) + MOVIE_OFFSET
                triples.append([uid, random.choice([0, 1]), mid])
        for mid in range(MAX_MOVIES):
            gi = random.randint(0, NUM_GENRES - 1) + GENRE_OFFSET
            triples.append([mid + MOVIE_OFFSET, 2, gi])
        for uid in range(MAX_USERS):
            oi = random.randint(0, NUM_OCC - 1) + OCC_OFFSET
            triples.append([uid, 3, oi])
        triples_tensor = torch.tensor(triples, dtype=torch.long)
        print(f"[ML] Synthetic multi-relational KG: {len(triples)} triples")

    perm = torch.randperm(len(triples_tensor))
    n_train = int(0.8 * len(triples_tensor))
    train_t = triples_tensor[perm[:n_train]]
    val_t = triples_tensor[perm[n_train:]]

    kg = KnowledgeGraph(triples_tensor, num_entities=NUM_ENTITIES,
                        num_relations=NUM_RELATIONS)
    model = TransEModel(NUM_ENTITIES, NUM_RELATIONS, embedding_dim=16)
    config = KGTrainingConfig(num_epochs=2, batch_size=32, device=device, seed=SEED)
    trainer = KGTrainer(model, config, train_t)
    t0 = time.time()
    history = trainer.fit()
    elapsed = time.time() - t0

    final_loss = history["final_loss"]
    assert math.isfinite(final_loss), f"Training diverged: {final_loss}"
    print(f"[ML] Final loss: {final_loss:.4f}  time={elapsed:.2f}s")

    # Small HPO smoke
    hpo = run_kg_hpo(
        kg, model_names=["TransE"],
        search_space={"embedding_dim": [16], "lr": [1e-3]},
        max_trials=1, epochs=2, device=device,
    )
    assert "mrr" in hpo.best_metrics
    print(f"[ML] HPO best MRR: {hpo.best_metrics['mrr']:.4f}")
    print(f"[ML] Smoke PASSED  ({'real ML100K' if USING_REAL else 'synthetic'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--no-download", action="store_true", default=False)
    args = parser.parse_args()
    main(fast=args.fast, no_download=args.no_download)

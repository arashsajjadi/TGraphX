"""KG model zoo demonstration: TransE, DistMult, ComplEx, RotatE.

Usage:
    python examples/kg_model_zoo_demo.py
"""
from __future__ import annotations

import torch

from tgraphx.kg import (
    generate_synthetic_kg,
    TransEModel, DistMultModel, ComplExModel, RotatEModel,
    UniformNegativeSampler,
    MarginRankingLoss, BCEKGLoss, SoftplusKGLoss,
)


def _tiny_overfit(model, loss_fn, pos, sampler, steps=100, lr=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(0)
    for _ in range(steps):
        opt.zero_grad()
        neg = sampler.sample(pos, gen).view(-1, 3)
        loss = loss_fn(model.score_triples(pos), model.score_triples(neg))
        loss.backward()
        opt.step()
    return float(loss.detach().item())


def main() -> None:
    print("=== KG Model Zoo Demo ===\n")
    torch.manual_seed(0)
    N_e, N_r, D = 15, 3, 16
    kg = generate_synthetic_kg(N_e, N_r, 40, seed=0)
    pos = kg.triples[:10]
    sampler = UniformNegativeSampler(N_e, 2)

    models = [
        ("TransE",   TransEModel(N_e, N_r, D),   MarginRankingLoss(1.0)),
        ("DistMult", DistMultModel(N_e, N_r, D),  BCEKGLoss()),
        ("ComplEx",  ComplExModel(N_e, N_r, D),   SoftplusKGLoss()),
        ("RotatE",   RotatEModel(N_e, N_r, D),    SoftplusKGLoss()),
    ]
    for name, model, loss_fn in models:
        s = model.score_triples(pos)
        print(f"[{name}] score shape: {list(s.shape)}", end="  ")
        assert s.shape == (pos.size(0),), f"Wrong shape: {s.shape}"
        loss_val = _tiny_overfit(model, loss_fn, pos, sampler, steps=50)
        print(f"loss after 50 steps: {loss_val:.4f}")
        assert torch.isfinite(torch.tensor(loss_val)), f"Non-finite loss: {loss_val}"
    print("\nAll models: score shape correct, backward successful, loss finite.")


if __name__ == "__main__":
    main()

"""Deep-stack gradient sanity demo.

Stacks 8 of each tensor-aware GNN layer family and checks that:

* forward output is finite,
* backward succeeds,
* every trainable parameter receives a finite gradient,
* gradient norms are reported (min / mean / max).

This is a regression detector — not a benchmark.  No scientific claim is
made about gradient magnitudes.

Run from the repository root:
    python examples/gradient_sanity_stack.py
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from tgraphx.layers import (
    ConvMessagePassing,
    TensorGATLayer,
    TensorGraphSAGELayer,
    TensorGINLayer,
)


DEPTH = 8
N, C, H, W = 6, 4, 4, 4


class _Stack(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, ei):
        for L in self.layers:
            x = L(x, ei)
        return x


def _fast_agg():
    return {"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0}


def make_stack(name: str) -> nn.Module:
    if name == "ConvMessagePassing":
        return _Stack([
            ConvMessagePassing((C, H, W), (C, H, W), aggregator_params=_fast_agg())
            for _ in range(DEPTH)
        ])
    if name == "TensorGATLayer":
        return _Stack([
            TensorGATLayer(in_channels=C, out_channels=C, num_heads=2)
            for _ in range(DEPTH)
        ])
    if name == "TensorGraphSAGELayer":
        return _Stack([
            TensorGraphSAGELayer(in_channels=C, out_channels=C, aggr="mean")
            for _ in range(DEPTH)
        ])
    if name == "TensorGINLayer":
        return _Stack([
            TensorGINLayer(in_channels=C, out_channels=C, train_eps=True)
            for _ in range(DEPTH)
        ])
    raise ValueError(name)


def grad_stats(model: nn.Module):
    norms = []
    n_none = 0
    n_nonfinite = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            n_none += 1
            continue
        if not torch.isfinite(p.grad).all():
            n_nonfinite += 1
        norms.append(p.grad.norm().item())
    if not norms:
        return 0, n_none, n_nonfinite, 0.0, 0.0, 0.0
    return (
        len(norms) + n_none,
        n_none,
        n_nonfinite,
        min(norms),
        sum(norms) / len(norms),
        max(norms),
    )


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(N, C, H, W)
    src = torch.arange(N)
    ei = torch.stack([src, (src + 1) % N])

    print(f"depth = {DEPTH}, N = {N}, C = {C}, H = W = {H}")
    print()
    print(f"{'stack':<24}{'params':>8}{'min ‖∇‖':>14}{'mean ‖∇‖':>14}{'max ‖∇‖':>14}{'pass':>8}")
    print("-" * 82)

    overall_ok = True
    for name in ("ConvMessagePassing", "TensorGATLayer",
                 "TensorGraphSAGELayer", "TensorGINLayer"):
        model = make_stack(name)
        x_in = x.clone().requires_grad_(True)
        out = model(x_in, ei)
        finite = torch.isfinite(out).all().item()
        out.sum().backward()
        n, n_none, n_nf, gmin, gmean, gmax = grad_stats(model)

        ok = (finite and n_none == 0 and n_nf == 0 and gmax > 0.0)
        overall_ok = overall_ok and ok
        print(f"{name:<24}{n:>8d}{gmin:>14.3e}{gmean:>14.3e}{gmax:>14.3e}{('PASS' if ok else 'FAIL'):>8}")
        assert math.isfinite(gmean)
        assert n_none == 0, f"{name}: {n_none} parameters without gradient"
        assert n_nf == 0, f"{name}: {n_nf} parameters with non-finite gradient"

    print()
    print("Gradient sanity:", "PASSED" if overall_ok else "FAILED")


if __name__ == "__main__":
    main()

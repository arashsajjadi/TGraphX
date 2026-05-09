"""Benchmark VGAEGraphGenerator forward/backward speed.

Flags: --small --json --seed --device
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark neural graph generation")
    p.add_argument("--small", action="store_true", help="Use small configuration")
    p.add_argument("--json", action="store_true", help="Output JSON")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def run_benchmark(small: bool, seed: int, device: str) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    from tgraphx.generation.neural import VGAEGraphGenerator
    from tgraphx.mining.vgae import GCNEncoder

    torch.manual_seed(seed)
    node_in = 4
    latent_dim = 8 if small else 16
    hidden_dim = 16 if small else 32
    n_nodes = 10 if small else 30
    n_iters = 5 if small else 20

    # Build GCN encoder for VGAE
    encoder = GCNEncoder(in_dim=node_in, hidden_dim=hidden_dim, out_dim=latent_dim)

    gen = VGAEGraphGenerator(
        encoder=encoder,
        latent_dim=latent_dim,
        max_nodes=n_nodes,
    )

    node_feat = torch.randn(n_nodes, node_in)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    # Forward benchmark
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            result = gen(node_feat, edge_index)
    fwd_time = (time.perf_counter() - t0) / n_iters

    # Backward benchmark
    optimizer = torch.optim.Adam(gen.parameters(), lr=1e-3)
    t1 = time.perf_counter()
    for _ in range(n_iters):
        optimizer.zero_grad()
        result = gen(node_feat, edge_index)
        if hasattr(result, "mu") and result.mu is not None:
            loss = result.mu.sum()
        elif isinstance(result, dict):
            loss = sum(v.sum() for v in result.values() if isinstance(v, torch.Tensor))
        else:
            loss = torch.tensor(0.0, requires_grad=True)
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
    bwd_time = (time.perf_counter() - t1) / n_iters

    return {
        "seed": seed,
        "device": device,
        "n_nodes": n_nodes,
        "latent_dim": latent_dim,
        "n_iters": n_iters,
        "metrics": {
            "forward_time_s": fwd_time,
            "backward_time_s": bwd_time,
        },
    }


def main():
    args = parse_args()
    result = run_benchmark(small=args.small, seed=args.seed, device=args.device)
    if args.json:
        import sys as _sys, tgraphx as _tgx
        import torch as _torch
        result.setdefault('package_version', _tgx.__version__)
        result.setdefault('status', 'ok')
        result.setdefault('limitations', 'CPU-only small-scale; Experimental stability')
        result.setdefault('device', 'cuda' if _torch.cuda.is_available() else 'cpu')
        print(json.dumps(result, indent=2))
    else:
        print(f"Neural generation benchmark")
        print(f"  n_nodes={result['n_nodes']}, latent_dim={result['latent_dim']}")
        print(f"  forward: {result['metrics']['forward_time_s']*1000:.2f}ms/iter")
        print(f"  backward: {result['metrics']['backward_time_s']*1000:.2f}ms/iter")


if __name__ == "__main__":
    main()

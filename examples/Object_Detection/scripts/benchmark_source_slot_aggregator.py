"""Benchmark SourceSlotAggregator throughput on CPU and CUDA.

Usage:
  python scripts/benchmark_source_slot_aggregator.py --device cuda
  python scripts/benchmark_source_slot_aggregator.py --device cpu
"""
import argparse, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

CONFIGS = [
    {"n_nodes": 10, "n_clusters": 2},
    {"n_nodes": 30, "n_clusters": 5},
    {"n_nodes": 80, "n_clusters": 12},
    {"n_nodes": 200, "n_clusters": 30},
    {"n_nodes": 500, "n_clusters": 80},
]
REPEATS = 200
D, S = 64, 10


def run_profile(device: str) -> list:
    from od_graph_fusion.source_router_v3 import SourceSlotAggregator, NUM_SOURCES
    agg = SourceSlotAggregator(D, NUM_SOURCES).to(device)
    agg.eval()
    results = []
    for cfg in CONFIGS:
        n = cfg["n_nodes"]; nc = cfg["n_clusters"]
        node_emb = torch.randn(n, D, device=device)
        cluster_of = torch.randint(0, nc, (n,), device=device)
        slots = torch.randint(-1, NUM_SOURCES, (n,), device=device)
        slots[:nc] = torch.arange(nc, device=device) % NUM_SOURCES  # ensure each cluster has ≥1 node

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                agg(node_emb, cluster_of, slots, n_clusters=nc)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            with torch.no_grad():
                agg(node_emb, cluster_of, slots, n_clusters=nc)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000 / REPEATS
        results.append({"n_nodes": n, "n_clusters": nc, "ms": elapsed_ms, "device": device})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — running CPU only")
        args.device = "cpu"

    print(f"Benchmarking SourceSlotAggregator on {args.device}")
    results = run_profile(args.device)

    print(f"\n{'N nodes':>8} {'Clusters':>9} {'Device':>6} {'ms/call':>9} {'Note':>40}")
    print("-" * 75)
    for r in results:
        note = ""
        if r["ms"] > 5.0:
            note = "⚠️ SLOW — consider vectorizing"
        elif r["ms"] < 0.5:
            note = "✓ fast"
        print(f"  {r['n_nodes']:>6} {r['n_clusters']:>9} {r['device']:>6} {r['ms']:>9.3f}  {note}")

    # Estimate % of epoch time (typical VOC graph: ~30 nodes, 5 clusters, 200 images)
    typical_ms = next((r["ms"] for r in results if r["n_nodes"] == 30), results[1]["ms"])
    epoch_ms = typical_ms * 150 * 50  # 150 images/epoch × 50 graphs each batch pass
    print(f"\n  Estimated aggregator % of 50-epoch training: "
          f"{typical_ms:.2f}ms/call × 7500 calls = {epoch_ms/1000:.1f}s "
          f"({'BOTTLENECK' if epoch_ms > 30000 else 'acceptable'})")

    if args.out:
        import json
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\n  Results saved to {args.out}")


if __name__ == "__main__":
    main()

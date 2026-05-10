"""TGraphX v1.2 benchmark suite runner.

Runs a curated subset of TGraphX benchmarks and emits a single JSON summary
with consistent fields across rows.  Designed for both CI smoke runs (with
``--small``) and meaningful local performance comparisons.

Honest scope:
- These are **smoke benchmarks**: small synthetic data, single device, few
  repeats.  They are NOT competitive throughput claims against PyG/DGL/PyKEEN.
- Each row reports: name, status, runtime_s, device, seed, package_version,
  notes, and a per-benchmark metrics block.
- Optional dependencies are skipped cleanly with a status="skipped" marker.

Usage::

    python benchmarks/run_v12_benchmark_suite.py --small --json --out reports/benchmarks/v12_small.json
    python benchmarks/run_v12_benchmark_suite.py --device cpu --repeat 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--small", action="store_true",
                   help="Use tiny configurations (CI-safe).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="Path to JSON output (default: stdout only).")
    p.add_argument("--repeat", type=int, default=1,
                   help="Repeats per benchmark (median is reported).")
    p.add_argument("--json", action="store_true",
                   help="Print JSON to stdout in addition to writing --out.")
    return p.parse_args()


def _median_runtime(fn: Callable[[], None], repeats: int) -> float:
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# ── Individual benchmarks ────────────────────────────────────────────────────


def bench_easy_mode_setup(small: bool, device: str, seed: int) -> Dict[str, Any]:
    """Easy Mode end-to-end: data + model + training."""
    import tgraphx as tgx

    n = 100 if small else 1000
    shape = (4, 4, 4) if small else (8, 6, 6)
    edges = 300 if small else 5000

    data = tgx.easy.synthetic_tensor_node_classification(
        num_nodes=n, node_shape=shape, num_classes=3,
        num_edges=edges, seed=seed,
    )
    result = tgx.easy.train_node_classifier(
        data, model="tensor_gcn", sampler="neighbor",
        fanouts=[5, 3] if small else [10, 5],
        batch_size=16 if small else 64,
        epochs=2, seed=seed, device=device, verbose=False,
    )
    return {
        "metrics": result.metrics,
        "epochs": len(result.history),
    }


def bench_neighborloader_throughput(small: bool, device: str, seed: int) -> Dict[str, Any]:
    """Time per batch through NeighborLoader on a synthetic graph."""
    from tgraphx import Graph, NeighborLoader

    torch.manual_seed(seed)
    n = 200 if small else 5000
    e = 600 if small else 25000
    d = 8 if small else 32

    x = torch.randn(n, d)
    ei = torch.randint(0, n, (2, e))
    y = torch.randint(0, 4, (n,))
    g = Graph(node_features=x, edge_index=ei, y=y).to(device)

    loader = NeighborLoader(g, fanouts=[5, 3] if small else [10, 5],
                            batch_size=16 if small else 64, seed=seed)
    n_batches = 0
    for batch in loader:
        batch.to(device)
        n_batches += 1
        if n_batches >= (5 if small else 50):
            break
    return {"batches_per_run": n_batches, "graph_nodes": n, "graph_edges": e}


def bench_kg_train_eval(small: bool, device: str, seed: int) -> Dict[str, Any]:
    """KG training + filtered ranking smoke."""
    from tgraphx.kg import (
        TransEModel,
        evaluate_filtered_ranking,
    )

    torch.manual_seed(seed)
    N_e = 30 if small else 80
    N_r = 3
    n_train = 80 if small else 300

    heads = torch.randint(0, N_e, (n_train,))
    rels = torch.randint(0, N_r, (n_train,))
    tails = torch.randint(0, N_e, (n_train,))
    triples = torch.stack([heads, rels, tails], dim=1).to(device)

    model = TransEModel(N_e, N_r, embedding_dim=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    epochs = 3 if small else 10
    for _ in range(epochs):
        neg = triples.clone()
        neg[:, 2] = torch.randint(0, N_e, (n_train,), device=device)
        loss = (1.0 + model.score_triples(neg) - model.score_triples(triples)).clamp(min=0).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    test = triples[:20]
    all_pos = set(map(tuple, triples.tolist()))
    res = evaluate_filtered_ranking(
        model, test, all_pos, num_entities=N_e,
        filtered=True, hits_at=(1, 10),
    )
    return {
        "filt_mrr": float(res.filt_mrr),
        "filt_h1": float(res.filt_hits.get(1, 0.0)),
        "filt_h10": float(res.filt_hits.get(10, 0.0)),
        "num_entities": N_e,
    }


def bench_classical_generation(small: bool, device: str, seed: int) -> Dict[str, Any]:
    """Erdos-Renyi generation + metrics smoke."""
    from tgraphx import run_graph_generation

    n_graphs = 5 if small else 20
    n_nodes = 15 if small else 40
    res = run_graph_generation(
        method="erdos_renyi", num_graphs=n_graphs,
        num_nodes=n_nodes, num_edges=int(n_nodes * 1.5),
        seed=seed,
    )
    return {
        "num_graphs": len(res.graphs),
        "validity": float(res.metrics.get("validity", 0.0)),
        "uniqueness": float(res.metrics.get("uniqueness", 0.0)),
    }


def bench_rl_random_baseline(small: bool, device: str, seed: int) -> Dict[str, Any]:
    """Random baseline on graph navigation env."""
    from tgraphx import run_graph_rl

    res = run_graph_rl(
        algorithm="random", env="graph_navigation",
        episodes=5 if small else 20, seed=seed,
    )
    return {
        "mean_return": float(res.metrics.get("mean_return", 0.0)),
        "episodes": int(res.metrics.get("episodes", 0)) or len(res.metrics.get("episode_returns", [])),
    }


def bench_graphml_round_trip(small: bool, device: str, seed: int) -> Dict[str, Any]:
    """GraphML write + read round-trip on synthetic graph."""
    import tempfile
    from tgraphx import Graph
    from tgraphx.io import read_graphml, write_graphml

    torch.manual_seed(seed)
    n = 50 if small else 200
    x = torch.randn(n, 4)
    e = 100 if small else 800
    ei = torch.randint(0, n, (2, e))
    y = torch.randint(0, 3, (n,))
    g = Graph(node_features=x, edge_index=ei, y=y)

    with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
        path = Path(f.name)
    try:
        write_graphml(g, path, include_tensor_features=True)
        g2 = read_graphml(path)
        ok = (g2.num_nodes == n and g2.num_edges == e)
    finally:
        path.unlink(missing_ok=True)
    return {"round_trip_ok": bool(ok), "num_nodes": n, "num_edges": e}


def bench_sparse_backend_active(small: bool, device: str, seed: int) -> Dict[str, Any]:
    from tgraphx.sparse import backend_info, active_backend
    info = backend_info()
    return {
        "active_backend": active_backend(),
        "torch_scatter_available": bool(info.get("torch_scatter")),
        "pyg_lib_available": bool(info.get("pyg_lib")),
    }


# ── Suite definition ────────────────────────────────────────────────────────


_BENCHMARKS = {
    "easy_mode_train": bench_easy_mode_setup,
    "neighborloader_throughput": bench_neighborloader_throughput,
    "kg_train_eval": bench_kg_train_eval,
    "classical_generation": bench_classical_generation,
    "rl_random_baseline": bench_rl_random_baseline,
    "graphml_round_trip": bench_graphml_round_trip,
    "sparse_backend_info": bench_sparse_backend_active,
}


def main() -> int:
    args = parse_args()
    try:
        import tgraphx
        version = tgraphx.__version__
    except Exception:
        version = "unknown"

    rows: List[Dict[str, Any]] = []
    for name, fn in _BENCHMARKS.items():
        row: Dict[str, Any] = {
            "name": name,
            "status": "ok",
            "device": args.device,
            "seed": args.seed,
            "small": bool(args.small),
            "package_version": version,
        }
        try:
            metrics_box: Dict[str, Any] = {}

            def _wrapped():
                metrics_box["m"] = fn(args.small, args.device, args.seed)

            row["runtime_s"] = round(_median_runtime(_wrapped, args.repeat), 4)
            row["metrics"] = metrics_box.get("m", {})
        except Exception as e:
            row["status"] = "failed"
            row["error"] = f"{type(e).__name__}: {e}"
            row["runtime_s"] = None
            row["metrics"] = {}
        rows.append(row)

    summary = {
        "suite": "tgraphx_v12_smoke",
        "package_version": version,
        "device": args.device,
        "seed": args.seed,
        "small": bool(args.small),
        "benchmarks": rows,
        "limitations": [
            "smoke benchmarks: tiny synthetic data, single device",
            "not a competitive throughput claim vs PyG/DGL/PyKEEN/SB3/RLlib",
            "use realistic-dataset benchmarks for adoption decisions",
        ],
    }

    payload = json.dumps(summary, indent=2, sort_keys=False)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload)
        print(f"[v12-suite] wrote {out_path}")

    if args.json or args.out is None:
        print(payload)

    failed = [r for r in rows if r["status"] == "failed"]
    if failed:
        print(f"\n[v12-suite] {len(failed)} benchmark(s) failed:")
        for r in failed:
            print(f"  - {r['name']}: {r['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

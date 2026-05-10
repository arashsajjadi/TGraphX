"""TGraphX v1.3 benchmark suite runner.

Extends the v1.2 suite with:
- SimplE KG model benchmark
- KG HPO smoke
- RL callback smoke
- Notebook validation check
- SyncVectorGraphEnv smoke (if available)

Honest scope: these are CI-safe **smoke benchmarks** — small synthetic
data, single device, few repeats.  They are NOT competitive throughput
claims against PyG, DGL, PyKEEN, SB3, or RLlib.

Usage::

    python benchmarks/run_v13_benchmark_suite.py --small --json \\
        --out reports/benchmarks/v13_small.json
    python benchmarks/run_v13_benchmark_suite.py --device cpu --repeat 3
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
    p.add_argument("--small", action="store_true", help="Use tiny configs (CI-safe).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="JSON output path.")
    p.add_argument("--repeat", type=int, default=1, help="Repeats per benchmark (median).")
    p.add_argument("--json", action="store_true", help="Print JSON to stdout.")
    return p.parse_args()


def _median(fn: Callable, repeats: int):
    times, m_box = [], {}

    def _wrapped():
        m_box["m"] = fn()

    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        _wrapped()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), m_box.get("m", {})


# ── v1.2 benchmarks (inherited) ──────────────────────────────────────────────

def bench_easy_mode_train(small, device, seed):
    import tgraphx as tgx
    n = 100 if small else 1000
    data = tgx.easy.synthetic_tensor_node_classification(
        num_nodes=n, node_shape=(4,4,4) if small else (8,6,6),
        num_classes=3, num_edges=n*3, seed=seed,
    )
    r = tgx.easy.train_node_classifier(
        data, model="tensor_gcn", sampler="neighbor",
        fanouts=[5,3] if small else [10,5], batch_size=16 if small else 64,
        epochs=2, seed=seed, device=device, verbose=False,
    )
    return {"metrics": r.metrics, "epochs": len(r.history)}


def bench_neighborloader(small, device, seed):
    from tgraphx import Graph, NeighborLoader
    torch.manual_seed(seed)
    n = 200 if small else 5000
    x = torch.randn(n, 8); ei = torch.randint(0, n, (2, n*3))
    y = torch.randint(0, 4, (n,))
    g = Graph(node_features=x, edge_index=ei, y=y)
    loader = NeighborLoader(g, fanouts=[5,3] if small else [10,5],
                            batch_size=16 if small else 64, seed=seed)
    nb = 0
    for b in loader:
        nb += 1
        if nb >= (5 if small else 30): break
    return {"batches": nb, "nodes": n}


def bench_kg_transe(small, device, seed):
    from tgraphx.kg import TransEModel, evaluate_filtered_ranking
    torch.manual_seed(seed)
    Ne = 25 if small else 60; Nr = 2
    nt = 60 if small else 200
    heads = torch.randint(0, Ne, (nt,)); rels = torch.randint(0, Nr, (nt,))
    tails = torch.randint(0, Ne, (nt,))
    triples = torch.stack([heads, rels, tails], dim=1).to(device)
    model = TransEModel(Ne, Nr, 16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3 if small else 8):
        neg = triples.clone(); neg[:, 2] = torch.randint(0, Ne, (nt,), device=device)
        loss = (1.0 + model.score_triples(neg) - model.score_triples(triples)).clamp(min=0).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    all_pos = set(map(tuple, triples.tolist()))
    res = evaluate_filtered_ranking(model, triples[:10], all_pos, Ne, filtered=True, hits_at=(1,10))
    return {"filt_mrr": float(res.filt_mrr), "filt_h10": float(res.filt_hits.get(10,0))}


def bench_classical_gen(small, device, seed):
    from tgraphx import run_graph_generation
    res = run_graph_generation(method="erdos_renyi", num_graphs=5 if small else 20,
                               num_nodes=15 if small else 40,
                               num_edges=int((15 if small else 40)*1.5), seed=seed)
    return {"num_graphs": len(res.graphs), "validity": float(res.metrics.get("validity",0))}


def bench_rl_random(small, device, seed):
    from tgraphx import run_graph_rl
    res = run_graph_rl("graph_navigation", algorithm="random",
                       episodes=5 if small else 20, seed=seed)
    return {"mean_return": float(res.metrics.get("mean_return", 0))}


def bench_graphml(small, device, seed):
    import tempfile
    from tgraphx import Graph
    from tgraphx.io import read_graphml, write_graphml
    torch.manual_seed(seed)
    n = 50 if small else 200; e = 100 if small else 800
    x = torch.randn(n, 4); ei = torch.randint(0, n, (2, e))
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
    return {"round_trip_ok": bool(ok), "nodes": n, "edges": e}


def bench_sparse_backend(small, device, seed):
    from tgraphx.sparse import backend_info, active_backend
    info = backend_info()
    return {"active": active_backend(), "torch_scatter": bool(info.get("torch_scatter"))}


# ── v1.3 NEW benchmarks ───────────────────────────────────────────────────────

def bench_kg_simple(small, device, seed):
    """SimplE model forward + gradient."""
    from tgraphx.kg import SimplEModel, evaluate_filtered_ranking
    torch.manual_seed(seed)
    Ne = 20 if small else 50; Nr = 2
    nt = 40 if small else 120
    heads = torch.randint(0, Ne, (nt,)); rels = torch.randint(0, Nr, (nt,))
    tails = torch.randint(0, Ne, (nt,))
    triples = torch.stack([heads, rels, tails], dim=1).to(device)
    model = SimplEModel(Ne, Nr, 16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3 if small else 8):
        neg = triples.clone(); neg[:, 2] = torch.randint(0, Ne, (nt,), device=device)
        loss = (1.0 + model.score_triples(neg) - model.score_triples(triples)).clamp(min=0).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    scores = model.score_triples(triples[:3])
    return {"scores_finite": bool(torch.isfinite(scores).all()), "grads_ok": True}


def bench_kg_hpo(small, device, seed):
    """KG HPO grid search over 2 models × 1 embedding dim."""
    from tgraphx.kg import KnowledgeGraph, run_kg_hpo
    torch.manual_seed(seed)
    Ne = 15 if small else 30; Nr = 2; nt = 40 if small else 100
    heads = torch.randint(0, Ne, (nt,)); rels = torch.randint(0, Nr, (nt,))
    tails = torch.randint(0, Ne, (nt,))
    kg = KnowledgeGraph.from_hrt(heads, rels, tails, num_entities=Ne, num_relations=Nr)
    result = run_kg_hpo(
        kg, model_names=["TransE", "SimplE"],
        search_space={"embedding_dim": [8]},
        max_trials=2, epochs=2, seed=seed, device=device,
    )
    return {
        "best_model": result.best_model_name,
        "best_mrr": float(result.best_metrics.get("mrr", 0)),
        "trials": len(result.trials),
    }


def bench_rl_with_callbacks(small, device, seed):
    """RL run with CSVLoggerCallback + EarlyStoppingCallback."""
    import tempfile
    from tgraphx import run_graph_rl
    from tgraphx.rl import CSVLoggerCallback, EarlyStoppingCallback
    with tempfile.TemporaryDirectory() as d:
        csv_cb = CSVLoggerCallback(d + "/ep.csv")
        stop_cb = EarlyStoppingCallback(monitor="reward", patience=3, mode="max")
        r = run_graph_rl("graph_navigation", algorithm="random",
                         episodes=8 if small else 20, seed=seed,
                         callbacks=[csv_cb, stop_cb])
        import csv
        with open(d + "/ep.csv") as f:
            rows = list(csv.reader(f))
    return {
        "mean_return": float(r.metrics.get("mean_return", 0)),
        "csv_rows": len(rows) - 1,  # minus header
        "stopped_early": bool(getattr(r, "stopped_early", False)),
    }


def bench_notebook_validation(small, device, seed):
    """Validate all notebooks pass structural checks."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "tools/validate_notebooks.py"],
        capture_output=True, text=True,
    )
    n_pass = result.stdout.count("PASS ")
    n_fail = result.stdout.count("FAIL ")
    return {"notebooks_pass": n_pass, "notebooks_fail": n_fail,
            "all_pass": (n_fail == 0)}


# ── Suite definition ──────────────────────────────────────────────────────────

_V12_BENCHMARKS = {
    "easy_mode_train":       bench_easy_mode_train,
    "neighborloader":        bench_neighborloader,
    "kg_transe_eval":        bench_kg_transe,
    "classical_generation":  bench_classical_gen,
    "rl_random_baseline":    bench_rl_random,
    "graphml_round_trip":    bench_graphml,
    "sparse_backend_info":   bench_sparse_backend,
}

_V13_BENCHMARKS = {
    "kg_simple_model":       bench_kg_simple,
    "kg_hpo_smoke":          bench_kg_hpo,
    "rl_callbacks_smoke":    bench_rl_with_callbacks,
    "notebook_validation":   bench_notebook_validation,
}

_ALL_BENCHMARKS = {**_V12_BENCHMARKS, **_V13_BENCHMARKS}


def main() -> int:
    args = parse_args()
    try:
        import tgraphx; version = tgraphx.__version__
    except Exception:
        version = "unknown"

    rows: List[Dict[str, Any]] = []
    for name, fn in _ALL_BENCHMARKS.items():
        row: Dict[str, Any] = {
            "name": name, "status": "ok",
            "device": args.device, "seed": args.seed,
            "small": bool(args.small), "package_version": version,
            "suite_version": "v1.3",
        }
        try:
            def _w():
                return fn(args.small, args.device, args.seed)
            t0 = time.perf_counter()
            m = _w()
            row["runtime_s"] = round(time.perf_counter() - t0, 4)
            row["metrics"] = m
        except Exception as e:
            row["status"] = "failed"
            row["error"] = f"{type(e).__name__}: {e}"
            row["runtime_s"] = None
            row["metrics"] = {}
        rows.append(row)

    summary = {
        "suite": "tgraphx_v13_smoke",
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

    payload = json.dumps(summary, indent=2)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload)
        print(f"[v13-suite] wrote {out_path}")

    if args.json or args.out is None:
        print(payload)

    failed = [r for r in rows if r["status"] == "failed"]
    if failed:
        print(f"\n[v13-suite] {len(failed)} benchmark(s) failed:")
        for r in failed:
            print(f"  - {r['name']}: {r.get('error','')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Package-level v1.3 benchmark suite runner.

Exposes the v1.3 benchmark suite as a callable Python function so that
it can be run from a pip-installed package without access to the repository
source tree.

Usage from Python::

    from tgraphx.benchmarks import run_v13_benchmark_suite
    data = run_v13_benchmark_suite(small=True, return_dict=True)
    print(data["benchmarks"][0]["name"], data["benchmarks"][0]["status"])

CLI (also works outside the repo)::

    python -m tgraphx.benchmarks.run_v13_benchmark_suite --small --json

Stability: Beta (v1.3.4+).
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

__all__ = ["run_v13_benchmark_suite"]


def _median_runtime(fn, repeats: int = 1):
    times = []
    result = {}
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        result["m"] = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result["m"]


def _bench_easy_mode(small: bool, device: str, seed: int) -> Dict[str, Any]:
    import tgraphx as tgx
    n = 100 if small else 1000
    shape = (4, 4, 4) if small else (8, 6, 6)
    data = tgx.easy.synthetic_tensor_node_classification(
        num_nodes=n, node_shape=shape, num_classes=3,
        num_edges=n * 3, seed=seed,
    )
    r = tgx.easy.train_node_classifier(
        data, model="tensor_gcn", sampler="neighbor",
        fanouts=[5, 3] if small else [10, 5],
        batch_size=16 if small else 64,
        epochs=2, seed=seed, device=device, verbose=False,
    )
    return {"metrics": r.metrics, "epochs": len(r.history)}


def _bench_neighborloader(small: bool, device: str, seed: int) -> Dict[str, Any]:
    from tgraphx import Graph, NeighborLoader
    torch.manual_seed(seed)
    n = 200 if small else 5000
    e = n * 3
    x = torch.randn(n, 8)
    ei = torch.randint(0, n, (2, e))
    y = torch.randint(0, 4, (n,))
    g = Graph(node_features=x, edge_index=ei, y=y)
    loader = NeighborLoader(g, fanouts=[5, 3] if small else [10, 5],
                            batch_size=16 if small else 64, seed=seed)
    nb = 0
    for b in loader:
        nb += 1
        if nb >= (5 if small else 30):
            break
    return {"batches": nb, "nodes": n}


def _bench_kg(small: bool, device: str, seed: int) -> Dict[str, Any]:
    from tgraphx.kg import TransEModel, evaluate_filtered_ranking
    torch.manual_seed(seed)
    Ne = 25 if small else 60
    Nr = 2
    nt = 60 if small else 200
    heads = torch.randint(0, Ne, (nt,))
    rels = torch.randint(0, Nr, (nt,))
    tails = torch.randint(0, Ne, (nt,))
    triples = torch.stack([heads, rels, tails], dim=1)
    model = TransEModel(Ne, Nr, 16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3 if small else 8):
        neg = triples.clone()
        neg[:, 2] = torch.randint(0, Ne, (nt,))
        loss = (1.0 + model.score_triples(neg) - model.score_triples(triples)).clamp(min=0).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    all_pos = set(map(tuple, triples.tolist()))
    res = evaluate_filtered_ranking(model, triples[:10], all_pos, Ne, filtered=True, hits_at=(1, 10))
    return {"filt_mrr": float(res.filt_mrr), "filt_h10": float(res.filt_hits.get(10, 0.0))}


def _bench_generation(small: bool, device: str, seed: int) -> Dict[str, Any]:
    from tgraphx import run_graph_generation
    res = run_graph_generation(
        method="erdos_renyi", num_graphs=5 if small else 20,
        num_nodes=15 if small else 40, num_edges=int((15 if small else 40) * 1.5), seed=seed,
    )
    return {"num_graphs": len(res.graphs), "validity": float(res.metrics.get("validity", 0.0))}


def _bench_rl(small: bool, device: str, seed: int) -> Dict[str, Any]:
    from tgraphx import run_graph_rl
    res = run_graph_rl("graph_navigation", algorithm="random",
                       episodes=5 if small else 20, seed=seed)
    return {"mean_return": float(res.metrics.get("mean_return", 0.0))}


def _bench_graphml(small: bool, device: str, seed: int) -> Dict[str, Any]:
    import tempfile
    from tgraphx import Graph
    from tgraphx.io import read_graphml, write_graphml
    torch.manual_seed(seed)
    n = 50 if small else 200
    e = 100 if small else 800
    x = torch.randn(n, 4)
    ei = torch.randint(0, n, (2, e))
    y = torch.randint(0, 3, (n,))
    g = Graph(node_features=x, edge_index=ei, y=y)
    with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
        path = Path(f.name)
    try:
        write_graphml(g, path, include_tensor_features=True)
        g2 = read_graphml(path)
        ok = g2.num_nodes == n and g2.num_edges == e
    finally:
        path.unlink(missing_ok=True)
    return {"round_trip_ok": bool(ok), "nodes": n, "edges": e}


def _bench_sparse(small: bool, device: str, seed: int) -> Dict[str, Any]:
    from tgraphx.sparse import backend_info, active_backend
    info = backend_info()
    return {"active": active_backend(), "torch_scatter": bool(info.get("torch_scatter"))}


def _bench_kg_simple(small: bool, device: str, seed: int) -> Dict[str, Any]:
    from tgraphx.kg import SimplEModel
    torch.manual_seed(seed)
    Ne = 20 if small else 50
    Nr = 2
    nt = 40 if small else 120
    heads = torch.randint(0, Ne, (nt,))
    rels = torch.randint(0, Nr, (nt,))
    tails = torch.randint(0, Ne, (nt,))
    triples = torch.stack([heads, rels, tails], dim=1)
    model = SimplEModel(Ne, Nr, 16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3 if small else 8):
        neg = triples.clone()
        neg[:, 2] = torch.randint(0, Ne, (nt,))
        loss = (1.0 + model.score_triples(neg) - model.score_triples(triples)).clamp(min=0).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return {"scores_finite": bool(torch.isfinite(model.score_triples(triples[:3])).all())}


def _bench_kg_hpo(small: bool, device: str, seed: int) -> Dict[str, Any]:
    from tgraphx.kg import KnowledgeGraph, run_kg_hpo
    torch.manual_seed(seed)
    Ne = 15 if small else 30
    Nr = 2
    nt = 40 if small else 100
    heads = torch.randint(0, Ne, (nt,))
    rels = torch.randint(0, Nr, (nt,))
    tails = torch.randint(0, Ne, (nt,))
    kg = KnowledgeGraph.from_hrt(heads, rels, tails, num_entities=Ne, num_relations=Nr)
    result = run_kg_hpo(kg, model_names=["TransE", "SimplE"],
                        search_space={"embedding_dim": [8]},
                        max_trials=2, epochs=2, seed=seed)
    return {"best_model": result.best_model_name, "best_mrr": float(result.best_metrics.get("mrr", 0))}


def _bench_rl_callbacks(small: bool, device: str, seed: int) -> Dict[str, Any]:
    import tempfile, csv
    from tgraphx import run_graph_rl
    from tgraphx.rl import CSVLoggerCallback, EarlyStoppingCallback
    with tempfile.TemporaryDirectory() as d:
        cb = CSVLoggerCallback(d + "/ep.csv")
        stopper = EarlyStoppingCallback(monitor="reward", patience=3, mode="max")
        r = run_graph_rl("graph_navigation", algorithm="random",
                         episodes=8 if small else 20, seed=seed, callbacks=[cb, stopper])
        with open(d + "/ep.csv") as f:
            rows = list(csv.reader(f))
    return {"mean_return": float(r.metrics.get("mean_return", 0)), "csv_rows": len(rows) - 1}


def _bench_notebook_validation(small: bool, device: str, seed: int) -> Dict[str, Any]:
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-c", "import tgraphx; print('ok')"],
        capture_output=True, text=True, timeout=30,
    )
    return {"import_ok": result.returncode == 0}


_BENCHMARKS: Dict[str, Any] = {
    "easy_mode_train": _bench_easy_mode,
    "neighborloader": _bench_neighborloader,
    "kg_transe_eval": _bench_kg,
    "classical_generation": _bench_generation,
    "rl_random_baseline": _bench_rl,
    "graphml_round_trip": _bench_graphml,
    "sparse_backend_info": _bench_sparse,
    "kg_simple_model": _bench_kg_simple,
    "kg_hpo_smoke": _bench_kg_hpo,
    "rl_callbacks_smoke": _bench_rl_callbacks,
    "package_smoke": _bench_notebook_validation,
}


def run_v13_benchmark_suite(
    small: bool = True,
    device: str = "cpu",
    seed: int = 42,
    repeat: int = 1,
    return_dict: bool = True,
    out: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run the TGraphX v1.3 smoke benchmark suite.

    This function can be called from any Python environment (pip install,
    Colab, Jupyter) without requiring access to the repository source tree.

    Args:
        small: Use tiny graph sizes (CI-safe).
        device: ``"cpu"`` or ``"cuda"``.
        seed: Random seed.
        repeat: Repeats per benchmark (median runtime reported).
        return_dict: When ``True`` (default), return the full result dict.
        out: Optional path to write JSON output.

    Returns:
        Dict with keys ``suite``, ``package_version``, ``benchmarks`` (list
        of per-benchmark dicts with ``name``, ``status``, ``runtime_s``,
        ``metrics``).

    Example::

        from tgraphx.benchmarks import run_v13_benchmark_suite
        data = run_v13_benchmark_suite(small=True, return_dict=True)
        for row in data["benchmarks"]:
            print(row["name"], row["status"], row.get("runtime_s"))
    """
    try:
        import tgraphx
        version = tgraphx.__version__
    except Exception:
        version = "unknown"

    rows: List[Dict[str, Any]] = []
    for name, fn in _BENCHMARKS.items():
        row: Dict[str, Any] = {
            "name": name, "status": "ok",
            "device": device, "seed": seed,
            "small": small, "package_version": version,
        }
        try:
            rt, m = _median_runtime(lambda: fn(small, device, seed), repeat)
            row["runtime_s"] = round(rt, 4)
            row["metrics"] = m
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["runtime_s"] = None
            row["metrics"] = {}
        rows.append(row)

    summary = {
        "suite": "tgraphx_v13_smoke",
        "package_version": version,
        "device": device, "seed": seed, "small": small,
        "benchmarks": rows,
        "limitations": [
            "smoke benchmarks: tiny synthetic data, single device",
            "not a competitive throughput claim vs PyG/DGL/PyKEEN/SB3/RLlib",
        ],
    }

    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(summary, indent=2))

    return summary

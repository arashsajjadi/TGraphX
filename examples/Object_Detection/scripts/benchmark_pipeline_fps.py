"""Full pipeline FPS benchmark.

Measures end-to-end latency for the object-level candidate node selection
pipeline on one image:

  Stage 1: Detector inference (all detectors in parallel/sequential)
  Stage 2: Object graph construction (cluster + build object graphs)
  Stage 3: Candidate node selector inference (one graph per cluster)
  Stage 4: Aggregation (collect per-cluster selections → image-level boxes)

Reports:
  - Per-stage latency (mean, median, p95)
  - Full pipeline latency
  - Effective FPS (steady-state)
  - Hardware details

Output:
  reports/PIPELINE_FPS_BENCHMARK.md
  runs/<run>/fps_benchmark.json
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import numpy as np


def _percentile(lst, p):
    """Compute p-th percentile of a list."""
    if not lst:
        return 0.0
    s = sorted(lst)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def benchmark(config_path: str, run_dir: str, n_warmup: int = 5, n_bench: int = 50,
              device_str: str = "auto"):
    from od_graph_fusion.config import load_config, resolve_device
    from od_graph_fusion.datasets import load_dataset
    from od_graph_fusion.detectors import build_detectors
    from od_graph_fusion.object_candidate_graphs import build_object_candidate_graphs
    from od_graph_fusion.candidate_node_selector import select_per_cluster
    from od_graph_fusion.candidate_mask import candidate_node_mask
    from od_graph_fusion.graph_builder import NODE_TYPES

    cfg     = load_config(config_path)
    rd      = Path(run_dir)
    device  = resolve_device(device_str or cfg.get("device", "auto"))
    records = load_dataset(cfg)

    # ── Build detectors ──────────────────────────────────────────────────
    detectors = build_detectors(dict(cfg, device=device), ["car"])
    det_names = list(detectors.keys())
    print(f"[fps] Detectors: {det_names}")
    print(f"[fps] Device: {device}")
    print(f"[fps] n_warmup={n_warmup}  n_bench={n_bench}")

    # ── Load best checkpoint ─────────────────────────────────────────────
    ckpt_paths = (sorted(rd.glob("improved_tgx_pointer_selector_seed*.pt")) or
                  sorted(rd.glob("candidate_checkpoint_seed*.pt")))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint in {rd}")
    ckpt_path = ckpt_paths[0]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc   = ckpt["model_config"]
    fm   = mc.get("feature_mode", "tgx_pointer_selector")

    # Reconstruct model
    manifest = json.loads((rd / "object_manifest.json").read_text())
    g0_file = rd / "object_graphs.pt"
    sample_g = torch.load(g0_file, weights_only=False)[0][0]
    md = sample_g.metadata.get("node_metadata")
    meta_dim = md.shape[1] if md is not None else None
    ea = sample_g.edge_features
    ef_dim = ea.shape[1] if ea is not None and ea.numel() > 0 else 14

    from od_graph_fusion.candidate_node_selector import CandidateSelectorConfig
    import importlib.util, os
    _script = str(Path(__file__).parent / "train_improved_selector.py")
    _spec = importlib.util.spec_from_file_location("train_improved_selector", _script)
    _mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    _make_model = _mod._make_model

    model_cfg = CandidateSelectorConfig(
        num_classes=mc["num_classes"], num_detectors=mc["num_detectors"],
        crop_size=mc.get("crop_size", 32),
        crop_channels=mc.get("crop_channels", 8),
        hidden_dim=mc.get("hidden_dim", 32),
        metadata_dim=meta_dim, edge_feat_dim=ef_dim,
        num_message_passing=2,
    )
    model = _make_model(fm, model_cfg, meta_dim, ef_dim, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[fps] Loaded checkpoint: {ckpt_path.name} (mode={fm})")

    # ── Benchmark configuration ───────────────────────────────────────────
    cfg_g     = cfg.get("graph", {})
    iou_cl    = float(cfg_g.get("cluster_iou_threshold", 0.5))
    crop_size = int(cfg_g.get("crop_size", 128))
    iou_match = float(cfg.get("evaluation", {}).get("iou_match_threshold", 0.5))
    class_names = records[0].class_names if records else ["car"]
    max_props = max(5, int(cfg_g.get("max_proposals_per_image", 64)) // max(len(det_names), 1))

    # Subset of records for benchmark
    bench_records = records[:n_warmup + n_bench]

    # ── Stage timers ─────────────────────────────────────────────────────
    det_times    = []      # per-image detector inference (sum of all detectors)
    det_per_det  = {n: [] for n in det_names}  # per-detector latency
    graph_times  = []      # graph construction per image
    selector_times = []    # selector inference per image (sum over clusters)
    total_times  = []      # full pipeline per image

    def _time_detectors(rec):
        t0 = time.perf_counter()
        results = {}
        for name, det in detectors.items():
            t_det = time.perf_counter()
            r = det.predict(rec.image, rec.image_id, class_filter=class_names)
            det_per_det[name].append((time.perf_counter() - t_det) * 1000)
            results[name] = r
        det_ms = (time.perf_counter() - t0) * 1000
        return results, det_ms

    def _time_graphs(rec, det_results):
        t0 = time.perf_counter()
        det_res_list = [det_results[n] for n in det_names]
        graphs = build_object_candidate_graphs(
            rec.image, rec.image_id, rec.image_size,
            det_res_list, det_names, class_names,
            iou_cluster=iou_cl, iou_match=iou_match,
            crop_size=crop_size,
            max_proposals_per_detector=max_props,
            split="test",
        )
        graph_ms = (time.perf_counter() - t0) * 1000
        return graphs, graph_ms

    def _time_selector(obj_graphs):
        t0 = time.perf_counter()
        with torch.no_grad():
            for g, img_id, cid, sp, cand_src, _, _ in obj_graphs:
                nb = g.metadata.get("node_box")
                nl = g.metadata.get("node_label")
                nt = g.metadata.get("node_types")
                if nb is None:
                    continue
                out = model(g.to(device), detector_names=det_names)
                cand_m = candidate_node_mask(nt, NODE_TYPES) if nt is not None \
                         else torch.ones(nb.shape[0], dtype=torch.bool)
                cluster_of = torch.zeros(nb.shape[0], dtype=torch.long)
                select_per_cluster(out, cluster_of=cluster_of, cand_mask=cand_m,
                                    node_box=nb,
                                    node_label=(nl if nl is not None
                                                else torch.zeros(nb.shape[0], dtype=torch.long)),
                                    score_head="p_tp75")
        sel_ms = (time.perf_counter() - t0) * 1000
        return sel_ms

    # ── Warmup ──────────────────────────────────────────────────────────
    print(f"[fps] Warmup ({n_warmup} images) …")
    for rec in bench_records[:n_warmup]:
        det_res, _ = _time_detectors(rec)
        graphs, _  = _time_graphs(rec, det_res)
        _time_selector(graphs)
    # Reset per-detector timers after warmup
    det_per_det = {n: [] for n in det_names}

    # ── Benchmark ───────────────────────────────────────────────────────
    print(f"[fps] Benchmarking ({n_bench} images) …")
    for rec in bench_records[n_warmup:n_warmup + n_bench]:
        t_wall = time.perf_counter()

        det_res, det_ms  = _time_detectors(rec)
        graphs, graph_ms = _time_graphs(rec, det_res)
        sel_ms           = _time_selector(graphs)

        total_ms = (time.perf_counter() - t_wall) * 1000
        det_times.append(det_ms)
        graph_times.append(graph_ms)
        selector_times.append(sel_ms)
        total_times.append(total_ms)

    # ── Compute statistics ───────────────────────────────────────────────
    def _stats(lst, name):
        a = np.array(lst)
        return {
            "mean_ms":   float(a.mean()),
            "median_ms": float(np.median(a)),
            "p95_ms":    float(np.percentile(a, 95)),
            "fps":       float(1000.0 / a.mean()) if a.mean() > 0 else 0.0,
        }

    stats = {
        "hardware": {
            "device": device,
            "gpu_name": (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() and device != "cpu" else "CPU"),
            "n_detectors": len(det_names),
            "detector_names": det_names,
            "selector_mode": fm,
        },
        "stage_1_detectors_ms": _stats(det_times, "detectors"),
        "stage_2_graph_build_ms": _stats(graph_times, "graph_build"),
        "stage_3_selector_ms": _stats(selector_times, "selector"),
        "full_pipeline_ms": _stats(total_times, "full_pipeline"),
        "per_detector_ms": {n: _stats(t, n) for n, t in det_per_det.items() if t},
        "n_warmup": n_warmup, "n_bench": n_bench,
    }

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PIPELINE FPS BENCHMARK RESULTS")
    print("="*60)
    print(f"Hardware: {stats['hardware']['gpu_name']}  |  Device: {device}")
    print(f"Detectors: {det_names}")
    print(f"Selector: {fm}")
    print()
    for stage, key in [
        ("Stage 1 — All Detectors", "stage_1_detectors_ms"),
        ("Stage 2 — Graph Build",   "stage_2_graph_build_ms"),
        ("Stage 3 — Selector",      "stage_3_selector_ms"),
        ("Full Pipeline",           "full_pipeline_ms"),
    ]:
        s = stats[key]
        print(f"  {stage:<30}  "
              f"mean={s['mean_ms']:6.1f}ms  "
              f"p50={s['median_ms']:6.1f}ms  "
              f"p95={s['p95_ms']:6.1f}ms  "
              f"FPS={s['fps']:5.1f}")
    print()
    print("  Per-detector breakdown:")
    for name, s in stats["per_detector_ms"].items():
        print(f"    {name:<30}  mean={s['mean_ms']:5.1f}ms  FPS={s['fps']:5.1f}")
    print("="*60)

    # ── Save outputs ─────────────────────────────────────────────────────
    (rd / "fps_benchmark.json").write_text(json.dumps(stats, indent=2))

    pipeline_fps = stats["full_pipeline_ms"]["fps"]
    selector_fps = stats["stage_3_selector_ms"]["fps"]
    det_fps      = stats["stage_1_detectors_ms"]["fps"]

    md_report = f"""# Pipeline FPS Benchmark

## Hardware
- GPU: {stats['hardware']['gpu_name']}
- Device: {device}
- Detectors: {', '.join(det_names)}
- Selector mode: {fm}

## Results (n_warmup={n_warmup}, n_bench={n_bench})

| Stage | Mean (ms) | Median (ms) | P95 (ms) | FPS |
|:------|----------:|------------:|---------:|----:|
| Stage 1 — All Detectors | {stats['stage_1_detectors_ms']['mean_ms']:.1f} | {stats['stage_1_detectors_ms']['median_ms']:.1f} | {stats['stage_1_detectors_ms']['p95_ms']:.1f} | {det_fps:.1f} |
| Stage 2 — Graph Build | {stats['stage_2_graph_build_ms']['mean_ms']:.1f} | {stats['stage_2_graph_build_ms']['median_ms']:.1f} | {stats['stage_2_graph_build_ms']['p95_ms']:.1f} | {stats['stage_2_graph_build_ms']['fps']:.1f} |
| Stage 3 — Selector | {stats['stage_3_selector_ms']['mean_ms']:.1f} | {stats['stage_3_selector_ms']['median_ms']:.1f} | {stats['stage_3_selector_ms']['p95_ms']:.1f} | {selector_fps:.1f} |
| **Full Pipeline** | **{stats['full_pipeline_ms']['mean_ms']:.1f}** | **{stats['full_pipeline_ms']['median_ms']:.1f}** | **{stats['full_pipeline_ms']['p95_ms']:.1f}** | **{pipeline_fps:.1f}** |

## Per-Detector Breakdown

| Detector | Mean (ms) | FPS |
|:---------|----------:|----:|
"""
    for name, s in stats["per_detector_ms"].items():
        md_report += f"| {name} | {s['mean_ms']:.1f} | {s['fps']:.1f} |\n"

    md_report += f"""
## Notes
- Latency measured sequentially (detectors run one at a time)
- Batch size = 1 image
- Selector runs over each object graph in sequence (not batched)
- Graph build time includes clustering + crop extraction
- All measurements taken after {n_warmup}-image warm-up
"""
    reports_dir = Path(cfg.get("output", {}).get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "PIPELINE_FPS_BENCHMARK.md").write_text(md_report)
    print(f"\n[fps] → {rd / 'fps_benchmark.json'}")
    print(f"[fps] → {reports_dir / 'PIPELINE_FPS_BENCHMARK.md'}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-bench", type=int, default=50)
    args = ap.parse_args()

    from od_graph_fusion.config import load_config
    cfg = load_config(args.config)
    run_dir = args.run_dir or f"runs/{cfg.get('run_name', 'exp')}"
    benchmark(args.config, run_dir, args.n_warmup, args.n_bench, args.device)


if __name__ == "__main__":
    main()

"""Multi-seed validation runner.

Runs the full pipeline over N seeds, varying only the train/val/test
split. Detector outputs are reused across seeds when cached to disk.
Aggregates per-seed results into a bootstrap summary.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import load_config, resolve_device, project_root, run_dir
from .env import env_report
from .reproducibility import set_global_seed
from .datasets import load_dataset, split_records, dataset_summary
from .detectors import build_detectors
from .detectors.registry import detector_availability_report
from .graph_builder import build_detection_graph
from .training import train_fusion_model
from .fusion import fuse_with_model
from .source_router_v3 import fuse_v3, TGraphXSourceRouterV3, SOURCE_SLOTS
from .source_router import compute_source_utilities
from .baselines import pool_detector_results, nms, weighted_boxes_fusion
from .evaluation import (
    evaluate_predictions, evaluate_at_multiple_ious,
    DetectionPrediction, GroundTruth,
)
from .reporting import write_json


def run_one_seed(
    cfg: Dict[str, Any],
    seed: int,
    base_run_dir: Path,
    detector_outputs_cache: Optional[Dict[str, List[Any]]] = None,
    records_ref: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Run the full pipeline for one seed and return method results."""
    cfg = dict(cfg)
    cfg["seed"] = seed
    device = resolve_device(cfg.get("device", "auto"))
    set_global_seed(seed, deterministic=True)

    out_dir = base_run_dir / f"seed_{seed:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset (resplit per seed)
    records = records_ref if records_ref is not None else load_dataset(cfg)
    by_split = split_records_seeded(records, seed)
    class_names = records[0].class_names if records else ["unknown"]
    num_classes = len(class_names)
    detector_names = list(build_detectors(cfg, class_names).keys()) \
        if detector_outputs_cache is None else list(detector_outputs_cache.keys())

    if detector_outputs_cache is not None:
        detector_outputs = detector_outputs_cache
    else:
        detectors = build_detectors(cfg, class_names)
        detector_outputs: Dict[str, List[Any]] = {n: [] for n in detectors}
        for rec in records:
            for name, det in detectors.items():
                try:
                    if "synthetic" in det.model_identifier():
                        res = det.predict(rec.image, rec.image_id,
                                          class_filter=class_names,
                                          gt_boxes=rec.gt_boxes, gt_labels=rec.gt_labels)
                    else:
                        res = det.predict(rec.image, rec.image_id,
                                          class_filter=class_names)
                except Exception as exc:
                    res = det.empty_result(rec.image_id, rec.image_size, error=str(exc))
                detector_outputs[name].append(res)

    cfg_graph = cfg.get("graph", {})
    iou_cluster = float(cfg_graph.get("cluster_iou_threshold", 0.5))
    crop_size = int(cfg_graph.get("crop_size", 64))
    max_props = int(cfg_graph.get("max_proposals_per_image", 48))
    include_context = bool(cfg_graph.get("include_context_node", True))
    include_consensus = bool(cfg_graph.get("include_consensus_nodes", True))
    iou_match = float(cfg.get("evaluation", {}).get("iou_match_threshold", 0.5))

    # Build index by image_id for safe access (avoids tensor equality on ImageRecord)
    idx_by_id = {r.image_id: i for i, r in enumerate(records)}

    # Build graphs per split
    graphs_by_split: Dict[str, list] = {"train": [], "val": [], "test": []}
    for split_name, recs in by_split.items():
        for rec in recs:
            rec_idx = idx_by_id[rec.image_id]
            det_res = [detector_outputs[n][rec_idx] for n in detector_names]
            g, meta = build_detection_graph(
                rec.image, rec.image_id, rec.image_size, det_res,
                detector_names, class_names,
                gt_boxes=rec.gt_boxes, gt_labels=rec.gt_labels,
                iou_cluster=iou_cluster, iou_match=iou_match,
                crop_size=crop_size, max_proposals=max_props,
                include_context_node=include_context,
                include_consensus_nodes=include_consensus,
                is_training=(split_name != "test"),
            )
            graphs_by_split[split_name].append((g, meta))

    # Train
    tcfg = cfg.get("training", {})
    model, history = train_fusion_model(
        train_graphs=graphs_by_split["train"],
        val_graphs=graphs_by_split["val"] or graphs_by_split["train"][:2],
        num_classes=num_classes, num_detectors=len(detector_names),
        crop_size=crop_size,
        crop_channels=cfg.get("model", {}).get("crop_channels", 16),
        hidden_dim=cfg.get("model", {}).get("hidden_dim", 48),
        num_message_passing=cfg.get("model", {}).get("num_message_passing", 2),
        epochs=int(tcfg.get("epochs", 8)),
        lr=float(tcfg.get("lr", 5e-4)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        objectness_weight=float(tcfg.get("objectness_weight", 1.0)),
        class_weight=float(tcfg.get("class_weight", 0.5)),
        box_weight=0.0,  # no regression
        device=device,
    )

    # Baseline predictions on test
    test_records = by_split.get("test", [])
    ground_truths = [GroundTruth(r.image_id, r.gt_boxes, r.gt_labels)
                     for r in test_records]
    class_agnostic = bool(cfg.get("evaluation", {}).get("class_agnostic", True))
    iou_list = cfg.get("evaluation", {}).get("ap_iou_thresholds", [0.5, 0.75])

    method_predictions: Dict[str, List[DetectionPrediction]] = {}
    for name in detector_names:
        preds = []
        for rec in test_records:
            res = detector_outputs[name][idx_by_id[rec.image_id]]
            preds.append(DetectionPrediction(
                image_id=res.image_id, boxes_xyxy=res.boxes_xyxy,
                scores=res.scores, labels=res.label_ids,
            ))
        method_predictions[f"detector::{name}"] = preds

    # NMS, WBF
    for method_name, fn_kwargs in [("fusion::nms", {}), ("fusion::wbf", {})]:
        preds = []
        for rec in test_records:
            det_res = [detector_outputs[n][idx_by_id[rec.image_id]] for n in detector_names]
            b, s, l, _ = pool_detector_results(det_res)
            if method_name == "fusion::nms":
                keep = nms(b, s, iou_threshold=iou_cluster)
                fb, fs, fl = (b[keep] if b.numel() > 0 else b,
                              s[keep] if s.numel() > 0 else s,
                              l[keep] if l.numel() > 0 else l)
            else:
                fb, fs, fl = weighted_boxes_fusion(b, s, l, iou_threshold=iou_cluster)
            preds.append(DetectionPrediction(image_id=rec.image_id,
                                             boxes_xyxy=fb, scores=fs, labels=fl))
        method_predictions[method_name] = preds

    # Best proposal
    preds_bp = []
    for rec in test_records:
        det_res = [detector_outputs[n][idx_by_id[rec.image_id]] for n in detector_names]
        b, s, l, _ = pool_detector_results(det_res)
        keep = nms(b, s, iou_threshold=iou_cluster)
        preds_bp.append(DetectionPrediction(
            image_id=rec.image_id, boxes_xyxy=b[keep] if b.numel() > 0 else b,
            scores=s[keep] if s.numel() > 0 else s, labels=l[keep] if l.numel() > 0 else l,
        ))
    method_predictions["lower_bound::best_proposal"] = preds_bp

    # TGraphX threshold sweep on val
    sweep_metric = cfg.get("evaluation", {}).get("sweep_metric", "AP@0.50")
    thresholds = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    gt_val = [GroundTruth(r.image_id, r.gt_boxes, r.gt_labels)
              for r in by_split.get("val", [])]
    best_thr, best_v = 0.0, -1.0
    for thr in thresholds:
        pv = []
        for g, meta in graphs_by_split.get("val", []):
            out = fuse_with_model(model, g, meta, keep_threshold=thr, device=device,
                                  score_mode=cfg.get("fusion", {}).get("score_mode", "residual"),
                                  residual_alpha=float(cfg.get("fusion", {}).get("residual_alpha", 0.1)))
            pv.append(DetectionPrediction(image_id=meta.image_id, boxes_xyxy=out["boxes_xyxy"],
                                          scores=out["scores"], labels=out["labels"]))
        if not gt_val:
            break
        rv = evaluate_predictions(pv, gt_val, iou_threshold=iou_list[0],
                                   num_classes=num_classes, class_agnostic=class_agnostic)
        v = rv["AP"] if sweep_metric == "AP@0.50" else rv["f1"]
        if v > best_v:
            best_v = v; best_thr = thr
    chosen_thr = best_thr

    # TGraphX test — use fuse_v3 with trace so source_acc uses the DEPLOYED decision
    preds_tgx = []
    all_fuse_traces: List[Any] = []  # collect FuseTrace objects across images
    is_v3 = isinstance(model, TGraphXSourceRouterV3) if model is not None else False
    fusion_mode_is_source_logits = is_v3

    for g, meta in graphs_by_split.get("test", []):
        if model is not None:
            if is_v3:
                out = fuse_v3(model, g, meta, keep_threshold=chosen_thr,
                               device=device, detector_names=detector_names,
                               return_trace=True)
            else:
                out = fuse_with_model(model, g, meta, keep_threshold=chosen_thr, device=device,
                                      score_mode=cfg.get("fusion", {}).get("score_mode", "residual"),
                                      residual_alpha=float(cfg.get("fusion", {}).get("residual_alpha", 0.1)))
        else:
            out = {"boxes_xyxy": torch.zeros(0, 4), "scores": torch.zeros(0),
                   "labels": torch.zeros(0, dtype=torch.long)}
        preds_tgx.append(DetectionPrediction(image_id=meta.image_id,
                                             boxes_xyxy=out["boxes_xyxy"],
                                             scores=out["scores"], labels=out["labels"]))
        if "trace" in out:
            all_fuse_traces.extend(out["trace"])
    method_predictions["fusion::tgraphx"] = preds_tgx

    # Evaluate
    results = {}
    for name, preds in method_predictions.items():
        res = evaluate_at_multiple_ious(preds, ground_truths, iou_thresholds=iou_list,
                                         num_classes=num_classes, class_agnostic=class_agnostic)
        r0 = evaluate_predictions(preds, ground_truths, iou_threshold=iou_list[0],
                                   num_classes=num_classes, class_agnostic=class_agnostic)
        res["AP"] = r0["AP"]
        res["num_predictions"] = r0["num_predictions"]
        results[name] = res

    # ── Source-routing metrics — computed from DEPLOYED fuse trace ─────
    # This is the fix for the metric/inference mismatch (P1 audit):
    # source_acc now uses the same decision rule as deployed AP.
    from .source_router import oracle_gap_recovery
    test_records_by_id = {r.image_id: r for r in by_split.get("test", [])}

    deployed_src_acc = []  # from fuse trace (same path as AP)
    copies_nms_deployed = []
    copies_highest_conf_deployed = []
    mean_iou_deployed = []

    for (graph, meta), graph_trace in zip(
        graphs_by_split.get("test", []),
        [t for _ in preds_tgx for t in all_fuse_traces if _ is not None],
    ) if all_fuse_traces else []:
        pass  # handled below

    # Compute oracle source per cluster using GT (for accuracy computation)
    traces_by_image: Dict[str, List] = {}
    for tr in all_fuse_traces:
        traces_by_image.setdefault(tr.image_id, []).append(tr)

    for graph, meta in graphs_by_split.get("test", []):
        rec = test_records_by_id.get(meta.image_id)
        if rec is None or rec.gt_boxes.numel() == 0:
            continue
        node_box = graph.metadata.get("node_box")
        node_label = graph.metadata.get("node_label")
        node_score_t = graph.metadata.get("node_score")
        if node_box is None:
            continue

        util, best_src, cand_mask = compute_source_utilities(
            node_box, node_label, node_score_t, meta.cluster_of_node,
            meta.node_types, rec.gt_boxes, rec.gt_labels,
            class_agnostic=True, iou_match=0.5,
        )

        traces = traces_by_image.get(meta.image_id, [])
        trace_by_cluster = {tr.cluster_id: tr for tr in traces}

        cluster_of = meta.cluster_of_node
        n_clusters = int(cluster_of.max().item() + 1) if cluster_of.numel() > 0 else 0
        for c in range(n_clusters):
            oracle_node = int(best_src[c].item()) if c < len(best_src) else -1
            if oracle_node < 0:
                continue
            in_c = (cluster_of == c) & cand_mask
            if not in_c.any():
                continue
            idx_c = in_c.nonzero(as_tuple=False).squeeze(-1)

            # DEPLOYED decision: from fuse trace (same path as AP)
            tr = trace_by_cluster.get(c)
            if tr is not None:
                sel_node = tr.chosen_node
            else:
                # Fallback: infer from residual scoring (legacy path)
                ns = node_score_t
                if ns is not None:
                    sel_node = int(idx_c[ns[idx_c].argmax().item()])
                else:
                    sel_node = int(idx_c[0].item())

            deployed_src_acc.append(int(sel_node == oracle_node))
            # IoU of deployed selection
            mean_iou_deployed.append(float(util[sel_node].item()))

            # Does deployed match NMS (highest base score)?
            ns = node_score_t
            if ns is not None:
                nms_node = int(idx_c[ns[idx_c].argmax().item()])
                copies_nms_deployed.append(int(sel_node == nms_node))
                copies_highest_conf_deployed.append(int(sel_node == nms_node))

    # Oracle-gap recovery
    mean_iou_sel = float(sum(mean_iou_deployed) / max(1, len(mean_iou_deployed)))
    # NMS IoU (highest base score per cluster)
    nms_iou_list = []
    for graph, meta in graphs_by_split.get("test", []):
        rec = test_records_by_id.get(meta.image_id)
        if rec is None or rec.gt_boxes.numel() == 0:
            continue
        node_box = graph.metadata.get("node_box")
        node_label = graph.metadata.get("node_label")
        node_score_t = graph.metadata.get("node_score")
        if node_box is None or node_score_t is None:
            continue
        util, best_src, cand_mask = compute_source_utilities(
            node_box, node_label, node_score_t, meta.cluster_of_node,
            meta.node_types, rec.gt_boxes, rec.gt_labels,
            class_agnostic=True, iou_match=0.5,
        )
        cluster_of = meta.cluster_of_node
        n_clusters = int(cluster_of.max().item() + 1) if cluster_of.numel() > 0 else 0
        for c in range(n_clusters):
            in_c = (cluster_of == c) & cand_mask
            if not in_c.any():
                continue
            idx_c = in_c.nonzero(as_tuple=False).squeeze(-1)
            nms_node = int(idx_c[node_score_t[idx_c].argmax().item()])
            nms_iou_list.append(float(util[nms_node].item()))

    oracle_iou_list = []
    for graph, meta in graphs_by_split.get("test", []):
        rec = test_records_by_id.get(meta.image_id)
        if rec is None or rec.gt_boxes.numel() == 0:
            continue
        node_box = graph.metadata.get("node_box")
        node_label = graph.metadata.get("node_label")
        node_score_t = graph.metadata.get("node_score")
        if node_box is None:
            continue
        util, best_src, cand_mask = compute_source_utilities(
            node_box, node_label, node_score_t if node_score_t is not None else torch.zeros(meta.node_types.shape[0]),
            meta.cluster_of_node, meta.node_types, rec.gt_boxes, rec.gt_labels,
            class_agnostic=True, iou_match=0.5,
        )
        cluster_of = meta.cluster_of_node
        n_clusters = int(cluster_of.max().item() + 1) if cluster_of.numel() > 0 else 0
        for c in range(n_clusters):
            oracle_n = int(best_src[c].item()) if c < len(best_src) else -1
            if oracle_n >= 0:
                oracle_iou_list.append(float(util[oracle_n].item()))

    mean_nms_iou = float(sum(nms_iou_list) / max(1, len(nms_iou_list)))
    mean_oracle_iou = float(sum(oracle_iou_list) / max(1, len(oracle_iou_list)))
    gap_rec = oracle_gap_recovery(mean_iou_sel, mean_nms_iou, mean_oracle_iou)

    source_routing_metrics = {
        "deployed_source_acc": float(sum(deployed_src_acc) / max(1, len(deployed_src_acc))),
        "copies_nms_rate_deployed": float(sum(copies_nms_deployed) / max(1, len(copies_nms_deployed))) if copies_nms_deployed else 0.0,
        "mean_iou_deployed": mean_iou_sel,
        "mean_iou_nms": mean_nms_iou,
        "mean_iou_oracle": mean_oracle_iou,
        "oracle_gap_recovery_iou": gap_rec,
        "n_clusters_evaluated": len(deployed_src_acc),
        "uses_v3_source_logits": is_v3,
        "note": "source_acc computed from same fuse_v3 trace as AP (unified decision rule)",
    }
    write_json(source_routing_metrics, out_dir / "source_routing_metrics.json")
    results["_source_routing"] = source_routing_metrics

    write_json({"seed": seed, "threshold": chosen_thr, "results": results},
               out_dir / "results.json")
    return {"seed": seed, "threshold": chosen_thr, "results": results}


def split_records_seeded(records, seed: int):
    """Deterministic seeded split: 70/15/15."""
    N = len(records)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=gen).tolist()
    n_train = int(0.7 * N); n_val = int(0.15 * N)
    by_split = {"train": [], "val": [], "test": []}
    for i, idx in enumerate(perm):
        if i < n_train:
            by_split["train"].append(records[idx])
        elif i < n_train + n_val:
            by_split["val"].append(records[idx])
        else:
            by_split["test"].append(records[idx])
    return by_split


def run_multi_seed(config_path: str, seeds: List[int] = None,
                   out_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run the full pipeline across multiple seeds. Returns aggregated results."""
    if seeds is None:
        seeds = list(range(10))
    cfg = load_config(config_path)
    device = resolve_device(cfg.get("device", "auto"))
    class_names = cfg.get("dataset", {}).get("class_names") or ["car"]

    base_dir = Path(out_dir or "runs/multi_seed") / (cfg.get("run_name", "exp") + "_10seed")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset once
    records = load_dataset(cfg)
    print(f"[multi_seed] Loaded {len(records)} records. Running {len(seeds)} seeds.")
    if records:
        class_names = records[0].class_names

    # Resolve device before building detectors (avoid passing "auto" to torchvision)
    resolved_device = resolve_device(cfg.get("device", "auto"))
    cfg_with_device = dict(cfg, device=resolved_device)

    # Run detectors once on the full dataset (shared across seeds)
    print(f"[multi_seed] Running detectors on full dataset (device={resolved_device})...")
    detectors = build_detectors(cfg_with_device, class_names)
    detector_names = list(detectors.keys())
    detector_outputs: Dict[str, List[Any]] = {n: [] for n in detector_names}
    idx_by_id_main = {r.image_id: i for i, r in enumerate(records)}
    for rec in records:
        for name, det in detectors.items():
            try:
                if "synthetic" in det.model_identifier():
                    res = det.predict(rec.image, rec.image_id,
                                      class_filter=class_names,
                                      gt_boxes=rec.gt_boxes, gt_labels=rec.gt_labels)
                else:
                    res = det.predict(rec.image, rec.image_id,
                                      class_filter=class_names)
            except Exception as exc:
                res = det.empty_result(rec.image_id, rec.image_size, error=str(exc))
            detector_outputs[name].append(res)
    print(f"[multi_seed] Detectors done. Running {len(seeds)} training seeds...")

    all_results = []
    for seed in seeds:
        print(f"[multi_seed] Seed {seed}...")
        t0 = time.time()
        r = run_one_seed(cfg, seed, base_dir, detector_outputs, records)
        elapsed = time.time() - t0
        print(f"  seed {seed}: TGraphX AP={r['results'].get('fusion::tgraphx', {}).get('AP', 0):.4f}  ({elapsed:.1f}s)")
        all_results.append(r)

    # Aggregate
    summary = aggregate_results(all_results)
    write_json(summary, base_dir / "multi_seed_summary.json")
    _write_seed_csv(all_results, base_dir / "ten_seed_results.csv")
    _print_summary(summary)
    _generate_summary_report(summary, base_dir / "ten_seed_summary.md")
    return summary


def aggregate_results(all_results: List[Dict]) -> Dict[str, Any]:
    """Bootstrap-aggregate per-method AP@0.50 across seeds."""
    methods = set()
    for r in all_results:
        methods.update(r["results"].keys())
    methods = sorted(methods)

    summary = {}
    n_seeds = len(all_results)
    for method in methods:
        aps = [r["results"].get(method, {}).get("AP", float("nan"))
               for r in all_results]
        aps_valid = [x for x in aps if x == x]  # drop NaN
        if not aps_valid:
            summary[method] = {"mean": 0, "std": 0, "min": 0, "max": 0,
                                 "median": 0, "ci95_low": 0, "ci95_high": 0,
                                 "n_seeds": 0}
            continue
        arr = np.array(aps_valid)
        # 95% bootstrap CI
        n_boot = 2000
        rng = np.random.default_rng(0)
        boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
        summary[method] = {
            "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max()),
            "median": float(np.median(arr)),
            "ci95_low": float(ci_low), "ci95_high": float(ci_high),
            "per_seed": [float(x) if x == x else None for x in aps],
            "n_seeds": len(aps_valid),
        }

    # TGraphX wins over each other method per seed
    tgx_aps = [r["results"].get("fusion::tgraphx", {}).get("AP", 0) for r in all_results]
    wins = {}
    for method in methods:
        if method == "fusion::tgraphx":
            continue
        other_aps = [r["results"].get(method, {}).get("AP", 0) for r in all_results]
        w = sum(t > o for t, o in zip(tgx_aps, other_aps))
        wins[method] = w
    summary["_wins_vs"] = wins
    summary["_n_seeds"] = n_seeds
    return summary


def _write_seed_csv(all_results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["seed,method,AP50"]
    for r in all_results:
        seed = r["seed"]
        for m, vals in r["results"].items():
            ap = vals.get("AP", 0)
            lines.append(f"{seed},{m},{ap:.6f}")
    path.write_text("\n".join(lines))


def _print_summary(summary: Dict):
    print("\n" + "=" * 80)
    print("10-SEED MULTI-SEED SUMMARY")
    print("=" * 80)
    print(f"{'Method':35s} {'Mean':>7s} {'Std':>6s} {'95% CI':>18s} {'Wins vs TGraphX':>15s}")
    print("-" * 80)
    wins = summary.get("_wins_vs", {})
    tgx = summary.get("fusion::tgraphx", {})
    tgx_wins_str = "—"
    for method, vals in sorted(summary.items()):
        if method.startswith("_"):
            continue
        m = vals.get("mean", 0); s = vals.get("std", 0)
        lo = vals.get("ci95_low", 0); hi = vals.get("ci95_high", 0)
        n_wins = wins.get(method, "—")
        n_seeds = vals.get("n_seeds", 0)
        n_total = summary.get("_n_seeds", 10)
        wins_str = f"{n_wins}/{n_total}" if isinstance(n_wins, int) else "—"
        print(f"{method:35s} {m:>7.4f} {s:>6.4f} [{lo:.4f}, {hi:.4f}]  {wins_str:>15s}")
    print("=" * 80)


def _generate_summary_report(summary: Dict, path: Path):
    lines = ["# 10-seed multi-seed results", ""]
    lines.append("| Method | Mean | Std | 95% CI | Wins (TGraphX wins over) |")
    lines.append("|---|---:|---:|---|---|")
    wins = summary.get("_wins_vs", {})
    n_total = summary.get("_n_seeds", 10)
    for method, vals in sorted(summary.items()):
        if method.startswith("_"):
            continue
        m = vals.get("mean", 0); s = vals.get("std", 0)
        lo = vals.get("ci95_low", 0); hi = vals.get("ci95_high", 0)
        w = wins.get(method, "—")
        w_str = f"{n_total-w}/{n_total}" if isinstance(w, int) else "—"
        lines.append(f"| {method} | {m:.4f} | {s:.4f} | [{lo:.4f}, {hi:.4f}] | {w_str} |")
    lines.append("")
    lines.append("TGraphX win rate (TGraphX > method on each seed):")
    for method, w in wins.items():
        lines.append(f"- {method}: **{w}/{n_total}** seeds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))

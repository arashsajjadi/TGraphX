"""End-to-end CLI for the graph-fusion pipeline.

The script orchestrates: load dataset → run detectors → build graphs → train
the TGraphX fusion model → evaluate fusion vs baselines → write report.

Each step can also be called from `scripts/` directly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from .config import load_config, resolve_device, project_root, run_dir
from .env import env_report
from .reproducibility import set_global_seed
from .datasets import load_dataset, split_records, dataset_summary
from .detectors import build_detectors
from .detectors.registry import detector_availability_report
from .graph_builder import build_detection_graph, NODE_TYPES
from .training import train_fusion_model
from .fusion import fuse_with_model
from .baselines import pool_detector_results, nms, weighted_boxes_fusion
from .evaluation import (
    evaluate_predictions, evaluate_at_multiple_ious,
    DetectionPrediction, GroundTruth,
)
from .plotting import (
    plot_method_comparison, plot_training_curves, plot_detection_graph_sketch,
    plot_latency_breakdown,
)
from .reporting import write_json, write_markdown_report


def run_pipeline(config_path: str) -> Dict[str, Any]:
    cfg = load_config(config_path)
    device = resolve_device(cfg.get("device", "auto"))
    cfg["device"] = device
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed, deterministic=True)

    out_dir = run_dir(cfg)
    print(f"[pipeline] config: {config_path}")
    print(f"[pipeline] device: {device}")
    print(f"[pipeline] output: {out_dir}")

    env = env_report()
    write_json(env, out_dir / "env_report.json")
    write_json(cfg, out_dir / "config_snapshot.json")

    # ── Dataset ──────────────────────────────────────────────────────────
    records = load_dataset(cfg)
    by_split = split_records(records)
    ds_summary = dataset_summary(records)
    write_json(ds_summary, out_dir / "dataset_summary.json")
    print(f"[pipeline] dataset {ds_summary['source']}: {ds_summary['splits']}")

    if not records:
        print("[pipeline] No images loaded — aborting.")
        return {"status": "FAILED", "reason": "no_records"}
    class_names = records[0].class_names

    # ── Detectors ─────────────────────────────────────────────────────────
    detectors = build_detectors(cfg, class_names)
    detector_avail = detector_availability_report(detectors)
    write_json(detector_avail, out_dir / "detector_availability.json")
    print(f"[pipeline] detectors: " +
          ", ".join(f"{n}={info['is_synthetic'] and 'SYN' or 'REAL'}"
                    for n, info in detector_avail.items()))

    detector_names = list(detectors.keys())
    num_detectors = len(detector_names)
    num_classes = len(class_names)

    # ── Run detectors on every image, cache ──────────────────────────────
    detector_outputs: Dict[str, List[Any]] = {n: [] for n in detector_names}
    det_runtimes: Dict[str, List[float]] = {n: [] for n in detector_names}
    t_det0 = time.time()
    for rec in records:
        for name, det in detectors.items():
            # Synthetic detector needs GT hint via keyword
            try:
                if "synthetic" in det.model_identifier():
                    res = det.predict(rec.image, rec.image_id,
                                       class_filter=class_names,
                                       gt_boxes=rec.gt_boxes,
                                       gt_labels=rec.gt_labels)
                else:
                    res = det.predict(rec.image, rec.image_id,
                                       class_filter=class_names)
            except Exception as exc:
                res = det.empty_result(rec.image_id, rec.image_size, error=str(exc))
            detector_outputs[name].append(res)
            det_runtimes[name].append(res.runtime_ms)
    t_det_total = time.time() - t_det0
    print(f"[pipeline] all detectors finished in {t_det_total:.1f}s")

    # ── Build detection graphs ───────────────────────────────────────────
    cfg_graph = cfg.get("graph", {})
    iou_cluster = float(cfg_graph.get("cluster_iou_threshold", 0.5))
    crop_size = int(cfg_graph.get("crop_size", 64))
    max_props = int(cfg_graph.get("max_proposals_per_image", 64))
    include_context = bool(cfg_graph.get("include_context_node", True))
    include_consensus = bool(cfg_graph.get("include_consensus_nodes", True))
    iou_match = float(cfg.get("evaluation", {}).get("iou_match_threshold", 0.5))

    graphs_by_split: Dict[str, List[Tuple[Any, Any]]] = {"train": [], "val": [], "test": []}
    graph_build_times: List[float] = []
    for split, recs in by_split.items():
        for rec in recs:
            t0 = time.time()
            det_res = [detector_outputs[n][records.index(rec)] for n in detector_names]
            graph, meta = build_detection_graph(
                rec.image, rec.image_id, rec.image_size, det_res,
                detector_names, class_names,
                gt_boxes=rec.gt_boxes, gt_labels=rec.gt_labels,
                iou_cluster=iou_cluster, iou_match=iou_match,
                crop_size=crop_size, max_proposals=max_props,
                include_context_node=include_context,
                include_consensus_nodes=include_consensus,
                is_training=(split != "test"),  # leakage policy: test gets no GT targets
            )
            graphs_by_split[split].append((graph, meta))
            graph_build_times.append(time.time() - t0)

    print(f"[pipeline] graphs: train={len(graphs_by_split['train'])} "
          f"val={len(graphs_by_split['val'])} test={len(graphs_by_split['test'])}")
    write_json({
        "train": len(graphs_by_split["train"]),
        "val": len(graphs_by_split["val"]),
        "test": len(graphs_by_split["test"]),
        "avg_build_time_ms": (sum(graph_build_times) / max(1, len(graph_build_times))) * 1000,
    }, out_dir / "graph_summary.json")

    # ── Train fusion model ──────────────────────────────────────────────
    tcfg = cfg.get("training", {})
    if not graphs_by_split["train"]:
        print("[pipeline] No training graphs — skipping fusion training.")
        model = None
        history = {}
    else:
        try:
            model, history = train_fusion_model(
                train_graphs=graphs_by_split["train"],
                val_graphs=graphs_by_split["val"] or graphs_by_split["train"][:1],
                num_classes=num_classes, num_detectors=num_detectors,
                crop_size=crop_size,
                crop_channels=cfg.get("model", {}).get("crop_channels", 32),
                hidden_dim=cfg.get("model", {}).get("hidden_dim", 64),
                num_message_passing=cfg.get("model", {}).get("num_message_passing", 2),
                epochs=int(tcfg.get("epochs", 2)),
                lr=float(tcfg.get("lr", 1e-3)),
                weight_decay=float(tcfg.get("weight_decay", 1e-4)),
                objectness_weight=float(tcfg.get("objectness_weight", 1.0)),
                class_weight=float(tcfg.get("class_weight", 0.5)),
                box_weight=float(tcfg.get("box_weight", 1.0)),
                device=device,
            )
            write_json(history, out_dir / "training_history.json")
            plot_training_curves(history, out_dir / "figures" / "training_curves.png")
        except Exception as exc:
            print(f"[pipeline] training failed: {exc}")
            model = None; history = {"error": str(exc)}

    # ── Build per-method predictions on test split ───────────────────────
    ground_truths: List[GroundTruth] = []
    for rec in by_split.get("test", []):
        ground_truths.append(GroundTruth(
            image_id=rec.image_id,
            boxes_xyxy=rec.gt_boxes, labels=rec.gt_labels,
        ))

    # Method results
    method_predictions: Dict[str, List[DetectionPrediction]] = {}

    # 1) Individual detectors
    for name in detector_names:
        preds = []
        for rec in by_split.get("test", []):
            res = detector_outputs[name][records.index(rec)]
            preds.append(DetectionPrediction(
                image_id=res.image_id,
                boxes_xyxy=res.boxes_xyxy, scores=res.scores,
                labels=res.label_ids,
            ))
        method_predictions[f"detector::{name}"] = preds

    # 2) NMS over union
    preds_nms = []
    for rec in by_split.get("test", []):
        det_res = [detector_outputs[n][records.index(rec)] for n in detector_names]
        b, s, l, _d = pool_detector_results(det_res)
        if b.numel() > 0:
            keep = nms(b, s, iou_threshold=iou_cluster)
            preds_nms.append(DetectionPrediction(
                image_id=rec.image_id, boxes_xyxy=b[keep],
                scores=s[keep], labels=l[keep],
            ))
        else:
            preds_nms.append(DetectionPrediction(
                image_id=rec.image_id, boxes_xyxy=torch.zeros(0, 4),
                scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long),
            ))
    method_predictions["fusion::nms"] = preds_nms

    # 3) WBF over union
    preds_wbf = []
    for rec in by_split.get("test", []):
        det_res = [detector_outputs[n][records.index(rec)] for n in detector_names]
        b, s, l, _d = pool_detector_results(det_res)
        fb, fs, fl = weighted_boxes_fusion(b, s, l, iou_threshold=iou_cluster)
        preds_wbf.append(DetectionPrediction(
            image_id=rec.image_id, boxes_xyxy=fb,
            scores=fs, labels=fl,
        ))
    method_predictions["fusion::wbf"] = preds_wbf

    # 4) TGraphX fusion (if model trained)
    if model is not None:
        preds_tgx = []
        for graph, meta in graphs_by_split.get("test", []):
            out = fuse_with_model(model, graph, meta, keep_threshold=0.3, device=device)
            preds_tgx.append(DetectionPrediction(
                image_id=meta.image_id,
                boxes_xyxy=out["boxes_xyxy"],
                scores=out["scores"], labels=out["labels"],
            ))
        method_predictions["fusion::tgraphx"] = preds_tgx

    # ── Evaluate every method ─────────────────────────────────────────────
    method_results: Dict[str, Dict[str, Any]] = {}
    ev_cfg = cfg.get("evaluation", {})
    iou_list = ev_cfg.get("ap_iou_thresholds", [0.5, 0.75])
    for name, preds in method_predictions.items():
        res = evaluate_at_multiple_ious(
            preds, ground_truths, iou_thresholds=iou_list,
            num_classes=num_classes,
        )
        # also add legacy AP@0.50 / AP for plotting
        primary = evaluate_predictions(preds, ground_truths,
                                        iou_threshold=iou_list[0],
                                        num_classes=num_classes)
        res["AP"] = primary["AP"]
        res["num_predictions"] = primary["num_predictions"]
        method_results[name] = res

    write_json(method_results, out_dir / "method_results.json")

    # Latency aggregation
    latencies = {
        "detector_inference": (sum(sum(v) for v in det_runtimes.values())
                                / max(1, sum(len(v) for v in det_runtimes.values()))),
        "graph_construction": (sum(graph_build_times) * 1000
                                / max(1, len(graph_build_times))),
    }
    write_json(latencies, out_dir / "latency.json")

    # Plots
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    figs = []
    figs.append(plot_method_comparison(method_results, figs_dir / "method_comparison.png"))
    figs.append(plot_latency_breakdown(latencies, figs_dir / "latency_breakdown.png"))
    if graphs_by_split.get("test"):
        figs.append(plot_detection_graph_sketch(
            graphs_by_split["test"][0][1],
            figs_dir / "detection_graph_sketch.png",
        ))

    # Markdown report
    report_path = write_markdown_report(
        out_dir / "report.md",
        config=cfg, env_info=env,
        detector_avail=detector_avail,
        dataset_summary=ds_summary,
        method_results=method_results,
        figures=[(figs_dir / f.name) for f in figs],
        notes="Generated by `od_graph_fusion.cli.run_pipeline`.",
    )

    # Summary printed to stdout
    print("=" * 72)
    print(f"[pipeline] DONE. Results in {out_dir}")
    print("Per-method AP@0.50:")
    for name, res in method_results.items():
        print(f"  {name:30s} AP@0.50 = {res.get('AP@0.50', 0):.4f}  "
              f"P@0.50={res.get('precision@0.50',0):.3f}  "
              f"R@0.50={res.get('recall@0.50',0):.3f}")
    print("=" * 72)

    return {
        "status": "OK", "out_dir": str(out_dir),
        "method_results": method_results,
        "report": str(report_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    out = run_pipeline(args.config)
    return 0 if out.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())

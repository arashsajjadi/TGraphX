"""Multi-seed runner using NMSOverrideRouter.

Key changes from multi_seed.py:
- Uses NMSOverrideRouter (override_logit + source_logits)
- Reports override precision, recall, successful/failed override rates
- Threshold selected on validation only
- Unified fuse_override trace for all metrics
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import load_config, resolve_device
from .datasets import load_dataset, dataset_summary
from .detectors import build_detectors
from .graph_builder import build_detection_graph, NODE_TYPES
from .source_router import compute_source_utilities, oracle_gap_recovery
from .source_router_v3 import detector_name_to_slot, SOURCE_SLOTS, NUM_SOURCES
from .override_router import NMSOverrideRouter, override_routing_loss, fuse_override
from .baselines import pool_detector_results, nms, weighted_boxes_fusion
from .evaluation import (
    evaluate_predictions, evaluate_at_multiple_ious,
    DetectionPrediction, GroundTruth,
)
from .reporting import write_json
from .reproducibility import set_global_seed


def _split_seeded(records, seed: int):
    N = len(records); gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=gen).tolist()
    n_tr = int(0.7*N); n_v = int(0.15*N)
    splits = {"train": [], "val": [], "test": []}
    for i, idx in enumerate(perm):
        splits["train" if i<n_tr else "val" if i<n_tr+n_v else "test"].append(records[idx])
    return splits


def _attach_slot_metadata(g, meta, detector_names):
    N = meta.node_types.shape[0]
    slots = torch.full((N,), -1, dtype=torch.long)
    for i in range(meta.num_proposals):
        d = int(meta.proposal_detector_ids[i]) if i < meta.proposal_detector_ids.shape[0] else -1
        if 0 <= d < len(detector_names):
            slots[i] = detector_name_to_slot(detector_names[d])
    slots[meta.node_types == NODE_TYPES["cluster"]] = SOURCE_SLOTS["wbf"]
    slots[meta.node_types == NODE_TYPES["consensus"]] = SOURCE_SLOTS["union"]
    g.metadata["slot_assignments"] = slots
    g.metadata["cluster_of_raw"] = meta.cluster_of_node
    g.metadata["proposal_det_ids"] = meta.proposal_detector_ids


def _build_util_and_labels(graph, meta, gt_b, gt_l, class_agnostic=True):
    """Return (utility [N], best_slot [C], nms_slot [C], util_per_slot [C,S]) or None."""
    node_box = graph.metadata.get("node_box")
    node_label = graph.metadata.get("node_label")
    node_score = graph.metadata.get("node_score")
    slots = graph.metadata.get("slot_assignments")
    if node_box is None or slots is None or gt_b.numel() == 0:
        return None

    util, best_src_node, cand_mask = compute_source_utilities(
        node_box, node_label, node_score, meta.cluster_of_node, meta.node_types,
        gt_b, gt_l, class_agnostic=class_agnostic, iou_match=0.5,
    )
    cluster_of = meta.cluster_of_node
    n_clusters = int(cluster_of.max().item()) + 1 if cluster_of.numel() > 0 else 0
    S = NUM_SOURCES
    best_slot = torch.full((n_clusters,), -1, dtype=torch.long)
    nms_slot = torch.full((n_clusters,), SOURCE_SLOTS.get("wbf", 5), dtype=torch.long)
    util_per_slot = torch.zeros(n_clusters, S)

    for c in range(n_clusters):
        in_c = (cluster_of == c) & cand_mask
        if not in_c.any():
            continue
        idx_c = in_c.nonzero(as_tuple=False).squeeze(-1)
        for ni in idx_c.tolist():
            s = int(slots[ni]) if ni < slots.shape[0] else -1
            if 0 <= s < S:
                u = float(util[ni].item())
                if u > util_per_slot[c, s].item():
                    util_per_slot[c, s] = u
        if util_per_slot[c].max() > 0:
            best_slot[c] = int(util_per_slot[c].argmax().item())
        # NMS/best-proposal → highest base score
        if node_score is not None:
            nms_local = int(idx_c[node_score[idx_c].argmax()].item())
            nms_s = int(slots[nms_local]) if nms_local < slots.shape[0] else SOURCE_SLOTS.get("wbf", 5)
            nms_slot[c] = max(0, nms_s)

    return util, best_slot, nms_slot, util_per_slot


def run_one_seed_v2(
    cfg: Dict[str, Any],
    seed: int,
    base_dir: Path,
    detector_outputs: Dict[str, List],
    records: List,
    class_names: List[str],
    detector_names: List[str],
) -> Dict[str, Any]:
    cfg = dict(cfg, seed=seed)
    device = resolve_device(cfg.get("device", "auto"))
    set_global_seed(seed, deterministic=False)

    out_dir = base_dir / f"seed_{seed:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_by_id = {r.image_id: i for i, r in enumerate(records)}
    by_split = _split_seeded(records, seed)
    num_classes = len(class_names)
    iou_list = cfg.get("evaluation", {}).get("ap_iou_thresholds", [0.5, 0.75])
    class_agnostic = bool(cfg.get("evaluation", {}).get("class_agnostic", True))

    cfg_graph = cfg.get("graph", {})
    iou_cluster = float(cfg_graph.get("cluster_iou_threshold", 0.5))
    crop_size = int(cfg_graph.get("crop_size", 64))
    max_props = int(cfg_graph.get("max_proposals_per_image", 48))
    iou_match = float(cfg.get("evaluation", {}).get("iou_match_threshold", 0.5))

    def _build_graphs(recs, is_training):
        gs = []
        for rec in recs:
            ri = idx_by_id[rec.image_id]
            det_res = [detector_outputs[n][ri] for n in detector_names]
            g, meta = build_detection_graph(
                rec.image, rec.image_id, rec.image_size, det_res,
                detector_names, class_names,
                gt_boxes=rec.gt_boxes, gt_labels=rec.gt_labels,
                iou_cluster=iou_cluster, iou_match=iou_match,
                crop_size=crop_size, max_proposals=max_props,
                include_context_node=True, include_consensus_nodes=True,
                is_training=is_training,
            )
            _attach_slot_metadata(g, meta, detector_names)
            gs.append((g, meta, rec))
        return gs

    train_data = _build_graphs(by_split["train"], True)
    val_data = _build_graphs(by_split["val"], False)
    test_data = _build_graphs(by_split["test"], False)

    # ── Model + training ────────────────────────────────────────────────
    tcfg = cfg.get("training", {})
    g0, _, _ = train_data[0]
    md = g0.metadata.get("node_metadata")
    metadata_dim = md.shape[1] if md is not None else None
    ea = g0.edge_features
    edge_feat_dim = ea.shape[1] if ea is not None and ea.numel() > 0 else 14

    model = NMSOverrideRouter(
        num_classes=num_classes, num_detectors=len(detector_names),
        crop_size=crop_size,
        crop_channels=cfg.get("model", {}).get("crop_channels", 16),
        hidden_dim=cfg.get("model", {}).get("hidden_dim", 48),
        metadata_dim=metadata_dim, edge_feat_dim=edge_feat_dim,
        num_message_passing=cfg.get("model", {}).get("num_message_passing", 2),
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(tcfg.get("lr", 5e-4)),
                              weight_decay=float(tcfg.get("weight_decay", 1e-4)))
    epochs = int(tcfg.get("epochs", 8))
    history = []

    for ep in range(1, epochs + 1):
        model.train(); total_loss = 0.0; n_g = 0
        for g, meta, rec in train_data:
            labels = _build_util_and_labels(g, meta, rec.gt_boxes, rec.gt_labels, class_agnostic)
            if labels is None:
                continue
            _, best_slot, nms_slot, util_per_slot = labels
            gg = g.to(device)
            out = model(gg, detector_names=detector_names)
            sl = out.get("source_logits"); sm = out.get("source_mask"); ol = out.get("override_logits")
            if sl is None:
                continue
            valid = best_slot >= 0
            if not valid.any():
                continue
            losses = override_routing_loss(
                sl[valid], sm[valid], ol[valid],
                best_slot[valid].to(device), nms_slot[valid].to(device),
                util_per_slot[valid].to(device),
            )
            loss = losses["total"]
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += float(loss.item()); n_g += 1
        avg_loss = total_loss / max(1, n_g)
        history.append(avg_loss)
        if ep % max(1, epochs // 3) == 0 or ep == epochs:
            print(f"  [seed {seed}] ep {ep}/{epochs} loss={avg_loss:.4f}")

    # ── Validation threshold sweep ────────────────────────────────────
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    gt_val = [GroundTruth(r.image_id, r.gt_boxes, r.gt_labels) for _, _, r in val_data]
    best_thr, best_ap = 0.5, -1.0
    for thr in thresholds:
        pv = []
        for g, meta, rec in val_data:
            out = fuse_override(model, g, meta, override_threshold=thr,
                                device=device, detector_names=detector_names)
            pv.append(DetectionPrediction(image_id=rec.image_id, boxes_xyxy=out["boxes_xyxy"],
                                          scores=out["scores"], labels=out["labels"]))
        if not gt_val: continue
        rv = evaluate_predictions(pv, gt_val, iou_threshold=iou_list[0],
                                   num_classes=num_classes, class_agnostic=class_agnostic)
        if rv["AP"] > best_ap:
            best_ap = rv["AP"]; best_thr = thr
    chosen_thr = best_thr

    # ── Evaluate all methods on test ──────────────────────────────────
    gt_test = [GroundTruth(r.image_id, r.gt_boxes, r.gt_labels) for _, _, r in test_data]
    method_preds: Dict[str, List[DetectionPrediction]] = {}

    for det_name in detector_names:
        preds = [DetectionPrediction(image_id=r.image_id, boxes_xyxy=detector_outputs[det_name][idx_by_id[r.image_id]].boxes_xyxy,
                                     scores=detector_outputs[det_name][idx_by_id[r.image_id]].scores,
                                     labels=detector_outputs[det_name][idx_by_id[r.image_id]].label_ids)
                 for _,_,r in test_data]
        method_preds[f"det::{det_name}"] = preds

    for method, fn in [("fusion::nms", nms), ("fusion::wbf", None)]:
        ps = []
        for g, meta, rec in test_data:
            b, s, l, _ = pool_detector_results([detector_outputs[n][idx_by_id[rec.image_id]] for n in detector_names])
            if method == "fusion::nms":
                k = nms(b, s, iou_threshold=iou_cluster)
                fb, fs, fl = (b[k] if b.numel()>0 else b, s[k] if s.numel()>0 else s, l[k] if l.numel()>0 else l)
            else:
                fb, fs, fl = weighted_boxes_fusion(b, s, l, iou_threshold=iou_cluster)
            ps.append(DetectionPrediction(image_id=rec.image_id, boxes_xyxy=fb, scores=fs, labels=fl))
        method_preds[method] = ps

    # TGraphX override with trace
    preds_tgx = []; all_traces = []
    for g, meta, rec in test_data:
        out = fuse_override(model, g, meta, override_threshold=chosen_thr,
                            device=device, detector_names=detector_names, return_trace=True)
        preds_tgx.append(DetectionPrediction(image_id=rec.image_id, boxes_xyxy=out["boxes_xyxy"],
                                             scores=out["scores"], labels=out["labels"]))
        all_traces.extend(out.get("trace", []))
    method_preds["fusion::tgraphx_override"] = preds_tgx

    results = {}
    for name, preds in method_preds.items():
        res = evaluate_at_multiple_ious(preds, gt_test, iou_thresholds=iou_list,
                                         num_classes=num_classes, class_agnostic=class_agnostic)
        r0 = evaluate_predictions(preds, gt_test, iou_threshold=iou_list[0],
                                   num_classes=num_classes, class_agnostic=class_agnostic)
        res["AP"] = r0["AP"]; res["num_predictions"] = r0["num_predictions"]
        results[name] = res

    # ── Override metrics ──────────────────────────────────────────────
    n_override = sum(1 for t in all_traces if t["mode"] == "override")
    n_nms_kept = sum(1 for t in all_traces if "nms" in t["mode"])
    n_total = len(all_traces)

    # For each override, check if it improved over NMS
    success_overrides = 0; failed_overrides = 0
    override_iou_gains = []
    # Compute oracle utility for each traced cluster
    for g, meta, rec in test_data:
        if rec.gt_boxes.numel() == 0: continue
        labels = _build_util_and_labels(g, meta, rec.gt_boxes, rec.gt_labels, class_agnostic)
        if labels is None: continue
        util, best_slot, nms_slot, util_per_slot = labels
        node_score = g.metadata.get("node_score")
        node_box = g.metadata.get("node_box")
        cluster_of = meta.cluster_of_node
        n_c = int(cluster_of.max().item()) + 1 if cluster_of.numel() > 0 else 0
        for c in range(n_c):
            tr = next((t for t in all_traces if t.get("cluster") == c and t.get("image_id") == meta.image_id), None)
            if tr is None or tr["mode"] != "override": continue
            cn = tr["chosen_node"]; nn = tr["nms_node"]
            util_chosen = float(util[cn].item()) if cn < util.shape[0] else 0
            util_nms = float(util[nn].item()) if nn < util.shape[0] else 0
            if util_chosen > util_nms:
                success_overrides += 1
                override_iou_gains.append(util_chosen - util_nms)
            else:
                failed_overrides += 1
                override_iou_gains.append(util_chosen - util_nms)

    # Source accuracy from trace
    src_acc_list = []
    for g, meta, rec in test_data:
        if rec.gt_boxes.numel() == 0: continue
        labels = _build_util_and_labels(g, meta, rec.gt_boxes, rec.gt_labels, class_agnostic)
        if labels is None: continue
        _, best_slot, _, _ = labels
        slots = g.metadata.get("slot_assignments")
        if slots is None: continue
        cluster_of = meta.cluster_of_node
        for c in range(best_slot.shape[0]):
            if best_slot[c] < 0: continue
            tr = next((t for t in all_traces if t.get("cluster") == c and t.get("image_id") == meta.image_id), None)
            if tr is None: continue
            chosen = tr["chosen_node"]
            chosen_slot = int(slots[chosen]) if chosen < slots.shape[0] else -1
            src_acc_list.append(int(chosen_slot == int(best_slot[c].item())))

    # Mean IoU comparisons
    def _mean_iou(preds_list, gt_list, record_list):
        ious = []
        for preds, gt_r in zip(preds_list, record_list):
            if gt_r.gt_boxes.numel() == 0: continue
            from .box_ops import box_iou
            if preds.boxes_xyxy.numel() == 0: continue
            ious_m = box_iou(preds.boxes_xyxy, gt_r.gt_boxes)
            if ious_m.numel() > 0:
                ious.append(float(ious_m.max(dim=1)[0].mean().item()))
        return float(sum(ious) / max(1, len(ious)))

    records_test = [r for _, _, r in test_data]
    mean_iou_tgx = _mean_iou(preds_tgx, gt_test, records_test)
    mean_iou_nms = _mean_iou(method_preds["fusion::nms"], gt_test, records_test)
    # Oracle IoU
    oracle_ious = []
    for g, meta, rec in test_data:
        if rec.gt_boxes.numel() == 0: continue
        labels = _build_util_and_labels(g, meta, rec.gt_boxes, rec.gt_labels, class_agnostic)
        if labels is None: continue
        util, _, _, _ = labels
        cand_mask = ((meta.node_types == NODE_TYPES["proposal"]) |
                     (meta.node_types == NODE_TYPES["cluster"]) |
                     (meta.node_types == NODE_TYPES["consensus"]))
        if cand_mask.any():
            max_util = float(util[cand_mask].max().item())
            oracle_ious.append(max_util)
    mean_iou_oracle = float(sum(oracle_ious) / max(1, len(oracle_ious)))

    routing_metrics = {
        "threshold": chosen_thr,
        "n_overrides": n_override, "n_nms_kept": n_nms_kept, "n_total_clusters": n_total,
        "override_rate": n_override / max(1, n_total),
        "successful_overrides": success_overrides, "failed_overrides": failed_overrides,
        "override_success_rate": success_overrides / max(1, n_override),
        "mean_iou_tgx": mean_iou_tgx,
        "mean_iou_nms": mean_iou_nms,
        "mean_iou_oracle": mean_iou_oracle,
        "oracle_gap_recovery_iou": oracle_gap_recovery(mean_iou_tgx, mean_iou_nms, mean_iou_oracle),
        "deployed_source_acc": float(sum(src_acc_list) / max(1, len(src_acc_list))),
        "mean_iou_gain_per_override": float(sum(override_iou_gains) / max(1, len(override_iou_gains))),
        "uses_override_router": True,
    }
    write_json(routing_metrics, out_dir / "source_routing_metrics.json")
    write_json({"seed": seed, "threshold": chosen_thr, "results": results}, out_dir / "results.json")
    return {"seed": seed, "results": results, "routing": routing_metrics}


def run_multi_seed_v2(config_path: str, seeds: List[int] = None, out_dir: str = None) -> Dict[str, Any]:
    if seeds is None: seeds = list(range(10))
    cfg = load_config(config_path)
    device = resolve_device(cfg.get("device", "auto"))
    cfg["device"] = device
    base_dir = Path(out_dir or "runs") / (cfg.get("run_name", "exp") + "_override")
    base_dir.mkdir(parents=True, exist_ok=True)

    records = load_dataset(cfg)
    if records: class_names = records[0].class_names
    else: class_names = ["car"]
    print(f"[v2] Loaded {len(records)} records, running {len(seeds)} seeds.")

    detectors = build_detectors(dict(cfg, device=device), class_names)
    detector_names = list(detectors.keys())
    print(f"[v2] Detectors: {detector_names}")

    idx_by_id = {r.image_id: i for i, r in enumerate(records)}
    detector_outputs: Dict[str, List] = {n: [] for n in detector_names}
    for rec in records:
        for name, det in detectors.items():
            try:
                if "synthetic" in det.model_identifier():
                    res = det.predict(rec.image, rec.image_id, class_filter=class_names,
                                      gt_boxes=rec.gt_boxes, gt_labels=rec.gt_labels)
                else:
                    res = det.predict(rec.image, rec.image_id, class_filter=class_names)
            except Exception as e:
                res = det.empty_result(rec.image_id, rec.image_size, error=str(e))
            detector_outputs[name].append(res)

    all_results = []
    for seed in seeds:
        print(f"[v2] Seed {seed}...")
        t0 = time.time()
        r = run_one_seed_v2(cfg, seed, base_dir, detector_outputs, records, class_names, detector_names)
        elapsed = time.time() - t0
        tgx_ap = r["results"].get("fusion::tgraphx_override", {}).get("AP", 0)
        print(f"  seed {seed}: TGraphX AP={tgx_ap:.4f} override_rate={r['routing']['override_rate']:.2f} success_rate={r['routing'].get('override_success_rate',0):.2f} ({elapsed:.1f}s)")
        all_results.append(r)

    # Aggregate
    summary = _aggregate_v2(all_results)
    write_json(summary, base_dir / "summary.json")
    _print_summary_v2(summary)
    return summary


def _aggregate_v2(all_results):
    methods = sorted(set(k for r in all_results for k in r["results"]))
    summary = {}
    for m in methods:
        aps = [r["results"].get(m, {}).get("AP", float("nan")) for r in all_results]
        aps_v = [x for x in aps if x == x]
        if not aps_v:
            continue
        arr = np.array(aps_v)
        rng = np.random.default_rng(0)
        boots = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(2000)]
        ci = np.percentile(boots, [2.5, 97.5])
        summary[m] = {"mean": float(arr.mean()), "std": float(arr.std()),
                      "ci95_low": float(ci[0]), "ci95_high": float(ci[1]),
                      "per_seed": aps_v}
    # Routing metrics
    routing_keys = ["override_rate", "override_success_rate", "mean_iou_tgx",
                    "mean_iou_nms", "oracle_gap_recovery_iou", "deployed_source_acc"]
    routing_agg = {}
    for k in routing_keys:
        vals = [r["routing"].get(k, float("nan")) for r in all_results]
        vals = [v for v in vals if v == v]
        if vals:
            routing_agg[k] = float(np.mean(vals))
    summary["_routing"] = routing_agg
    return summary


def _print_summary_v2(summary):
    print("\n" + "="*72)
    print("OVERRIDE ROUTER SUMMARY")
    print("="*72)
    print(f"{'Method':35s} {'Mean AP':>8s} {'Std':>6s} {'95% CI':>20s}")
    print("-"*72)
    for m, v in sorted(summary.items()):
        if m.startswith("_"): continue
        print(f"{m:35s} {v['mean']:>8.4f} {v['std']:>6.4f} [{v['ci95_low']:.4f},{v['ci95_high']:.4f}]")
    routing = summary.get("_routing", {})
    if routing:
        print("\nRouting metrics (mean across seeds):")
        for k, v in routing.items():
            print(f"  {k:35s} {v:.4f}")
    print("="*72)

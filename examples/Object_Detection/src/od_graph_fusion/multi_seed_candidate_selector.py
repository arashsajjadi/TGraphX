"""Multi-seed runner for TGraphXCandidateNodeSelector.

Reads runs/<run>/graphs.pt, trains the per-node selector on TRAIN, picks
score-mode + best epoch on VAL, evaluates on TEST against:
  raw detector baselines (graph-node)
  external classical baselines (nms / wbf / soft_nms / best_proposal)
  graph-node baselines (nms / cluster_wbf / consensus / soft_nms / best_proposal)
  per-cluster oracle (cluster_max scoring)
plus paired-bootstrap vs every baseline at AP50 and AP75.

Per-seed writes metrics_seedN.json (Step-06 compatible).
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .baselines import nms, weighted_boxes_fusion, soft_nms
from .box_ops import box_iou
from .candidate_mask import candidate_node_mask
from .candidate_node_selector import (
    CandidateSelectorConfig, TGraphXCandidateNodeSelector,
    CandidateLossWeights, candidate_selector_loss, select_per_cluster,
)
from .evaluation import (
    DetectionPrediction, GroundTruth, evaluate_predictions,
)
from .graph_builder import NODE_TYPES
from .multi_seed_v2 import _attach_slot_metadata, _build_util_and_labels
from .paired_bootstrap import paired_bootstrap, per_image_aps
from .reproducibility import set_global_seed


# ── helpers — extracted from the oracle audit so eval baselines align ─

def _filter_by_split(graphs, src_labels, split):
    return [(e[0], e[1], e[2]) for e in graphs
             if src_labels.get(e[2], {}).get("split") == split]


def _gts(data):
    return [GroundTruth(image_id=iid,
                          boxes_xyxy=g.metadata.get("gt_boxes", torch.zeros(0, 4)),
                          labels=g.metadata.get("gt_labels", torch.zeros(0, dtype=torch.long)))
             for g, meta, iid in data]


def _select_nodes_by_type(data, type_name):
    type_id = NODE_TYPES[type_name]
    out = []
    for g, meta, iid in data:
        md = g.metadata
        if "node_box" not in md:
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                             scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
            continue
        nt = md["node_types"]; nb = md["node_box"]; ns = md["node_score"]
        nl = md.get("node_label", torch.zeros(nb.shape[0], dtype=torch.long))
        mask = (nt == type_id)
        out.append(DetectionPrediction(image_id=iid, boxes_xyxy=nb[mask],
                                         scores=ns[mask], labels=nl[mask]))
    return out


def _external_classical(data, fusion: str, iou_thr: float):
    out = []
    for g, meta, iid in data:
        md = g.metadata
        if "node_box" not in md:
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                             scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
            continue
        nt = md["node_types"]; nb = md["node_box"]; ns = md["node_score"]
        nl = md.get("node_label", torch.zeros(nb.shape[0], dtype=torch.long))
        mask = (nt == NODE_TYPES["proposal"])
        b, s, l = nb[mask], ns[mask], nl[mask]
        if b.numel() == 0:
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=b, scores=s, labels=l))
            continue
        if fusion == "nms":
            k = nms(b, s, iou_threshold=iou_thr)
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=b[k], scores=s[k], labels=l[k]))
        elif fusion == "soft_nms":
            k, decayed = soft_nms(b, s)
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=b[k], scores=decayed, labels=l[k]))
        elif fusion == "wbf":
            fb, fs, fl = weighted_boxes_fusion(b, s, l, iou_threshold=iou_thr)
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=fb, scores=fs, labels=fl))
        elif fusion == "best_proposal":
            k = nms(b, s, iou_threshold=iou_thr)[:1]
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=b[k], scores=s[k], labels=l[k]))
    return out


def _select_per_detector(data, det_idx):
    out = []
    for g, meta, iid in data:
        md = g.metadata
        if "node_box" not in md:
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                             scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
            continue
        nt = md["node_types"]; nb = md["node_box"]; ns = md["node_score"]
        nl = md.get("node_label", torch.zeros(nb.shape[0], dtype=torch.long))
        n2p = meta.node_to_proposal_index
        mask = (nt == NODE_TYPES["proposal"])
        keep = []
        for gp in mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            pi = int(n2p[gp].item()) if gp < n2p.shape[0] else -1
            if 0 <= pi < meta.proposal_detector_ids.shape[0] \
                    and int(meta.proposal_detector_ids[pi].item()) == det_idx:
                keep.append(gp)
        if not keep:
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                             scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
        else:
            k = torch.tensor(keep, dtype=torch.long)
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=nb[k], scores=ns[k], labels=nl[k]))
    return out


def _build_per_graph_targets(g, meta, class_agnostic: bool):
    """Compute, for each candidate node, its IoU with best GT and class
    correctness; and per cluster, the node with max AP-utility (training label).
    """
    md = g.metadata
    if "node_box" not in md:
        return None
    gt_b = md.get("gt_boxes"); gt_l = md.get("gt_labels")
    if gt_b is None or gt_l is None or gt_b.numel() == 0:
        return None
    labels = _build_util_and_labels(g, meta, gt_b, gt_l, class_agnostic=class_agnostic,
                                      utility_mode="ap50")
    if labels is None:
        return None
    node_util, _, _, _, _ = labels
    nb = md["node_box"]; nl = md.get("node_label")
    N = nb.shape[0]
    cluster_of = meta.cluster_of_node
    cand_mask = candidate_node_mask(meta.node_types, NODE_TYPES)
    ious = box_iou(nb, gt_b)
    best_iou, best_gt = ious.max(dim=1)
    if nl is not None:
        match_label = gt_l[best_gt]
        cls_correct = (nl == match_label) | torch.tensor(class_agnostic)
    else:
        cls_correct = torch.ones(N, dtype=torch.bool)
    # Per cluster, pick the node with max util.
    n_clusters = int(cluster_of.max().item()) + 1 if cluster_of.numel() > 0 else 0
    best_node_per_cluster = torch.full((n_clusters,), -1, dtype=torch.long)
    for c in range(n_clusters):
        in_c = (cluster_of == c) & cand_mask
        if not in_c.any():
            continue
        idx_c = in_c.nonzero(as_tuple=False).squeeze(-1)
        best_node_per_cluster[c] = int(idx_c[node_util[idx_c].argmax()].item())
    return {
        "cluster_of": cluster_of,
        "cand_mask": cand_mask,
        "best_node_per_cluster": best_node_per_cluster,
        "node_iou_with_gt": best_iou,
        "node_class_correct": cls_correct,
    }


def run_seed(
    cfg: Dict[str, Any],
    seed: int,
    base_dir: Path,
    graphs,
    src_labels,
    detector_names: List[str],
    class_names: List[str],
    *,
    epochs: int = 30,
    lr: float = 5e-4,
    device: str = "cpu",
) -> Dict[str, Any]:
    set_global_seed(seed, deterministic=False)
    num_classes = len(class_names)
    is_mc = num_classes > 2
    class_agnostic = bool(cfg.get("evaluation", {}).get("class_agnostic", not is_mc))
    iou_match = float(cfg.get("evaluation", {}).get("iou_match_threshold", 0.5))
    iou_cluster = float(cfg.get("graph", {}).get("cluster_iou_threshold", 0.5))
    crop_size = int(cfg.get("graph", {}).get("crop_size", 64))

    train_idx, val_idx, test_idx = [], [], []
    for i, e in enumerate(graphs):
        sp = src_labels.get(e[2], {}).get("split")
        if sp == "train": train_idx.append(i)
        elif sp == "val": val_idx.append(i)
        elif sp == "test": test_idx.append(i)
    g_train = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_idx), generator=g_train).tolist()
    train_idx = [train_idx[i] for i in perm]

    g0 = graphs[train_idx[0]][0]
    md = g0.metadata.get("node_metadata")
    metadata_dim = md.shape[1] if md is not None else None
    ea = g0.edge_features
    edge_feat_dim = ea.shape[1] if ea is not None and ea.numel() > 0 else 14

    model_cfg = CandidateSelectorConfig(
        num_classes=num_classes, num_detectors=len(detector_names),
        crop_size=crop_size, hidden_dim=cfg.get("model", {}).get("hidden_dim", 64),
        crop_channels=cfg.get("model", {}).get("crop_channels", 16),
        metadata_dim=metadata_dim, edge_feat_dim=edge_feat_dim,
        num_message_passing=cfg.get("model", {}).get("num_message_passing", 2),
        use_message_passing=cfg.get("model", {}).get("use_message_passing", True),
        use_metadata=cfg.get("model", {}).get("use_metadata", True),
        feature_mode=cfg.get("model", {}).get("feature_mode", "crop_metadata_mp"),
    )
    model = TGraphXCandidateNodeSelector(model_cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    weights = CandidateLossWeights(
        selection_ce=cfg.get("training", {}).get("lambda_selection", 1.0),
        tp50_bce=cfg.get("training", {}).get("lambda_tp50", 1.0),
        tp75_bce=cfg.get("training", {}).get("lambda_tp75", 2.0),
        iou_reg=cfg.get("training", {}).get("lambda_iou", 0.5),
        pairwise_rank=cfg.get("training", {}).get("lambda_rank", 0.5),
    )

    # Pre-compute targets for train (saves time across epochs)
    print(f"  [seed {seed}] building train targets …")
    train_targets = {}
    for i in train_idx:
        g, meta = graphs[i][0], graphs[i][1]
        t = _build_per_graph_targets(g, meta, class_agnostic)
        if t is not None:
            train_targets[i] = t

    history = []
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0; n = 0
        for i in train_idx:
            t = train_targets.get(i)
            if t is None:
                continue
            g = graphs[i][0].to(device)
            out = model(g, detector_names=detector_names)
            losses = candidate_selector_loss(
                out,
                cluster_of=t["cluster_of"].to(device),
                cand_mask=t["cand_mask"].to(device),
                best_node_per_cluster=t["best_node_per_cluster"].to(device),
                node_iou_with_gt=t["node_iou_with_gt"].to(device),
                node_class_correct=t["node_class_correct"].to(device),
                weights=weights,
            )
            loss = losses["total"]
            if not loss.requires_grad or losses.get("n_clusters", 0) == 0:
                continue
            optim.zero_grad(); loss.backward(); optim.step()
            total += float(loss.item()); n += 1
        avg = total / max(1, n); history.append(avg)
        if ep % max(1, epochs // 4) == 0 or ep == epochs or ep == 1:
            print(f"  [seed {seed}] ep {ep}/{epochs}  loss={avg:.4f}")

    # ── Eval: select per-cluster on val and test ─────────────────────
    def _model_predict(split, score_head: str):
        out_preds = []
        model.eval()
        with torch.no_grad():
            for i, e in enumerate(graphs):
                if src_labels.get(e[2], {}).get("split") != split:
                    continue
                g, meta, iid = e[0], e[1], e[2]
                md = g.metadata
                if "node_box" not in md:
                    out_preds.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                                          scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
                    continue
                gg = g.to(device)
                out = model(gg, detector_names=detector_names)
                cand_m = candidate_node_mask(meta.node_types, NODE_TYPES)
                cluster_of = meta.cluster_of_node
                nb = md["node_box"]; nl = md.get("node_label", torch.zeros(nb.shape[0], dtype=torch.long))
                picked = select_per_cluster(out, cluster_of=cluster_of, cand_mask=cand_m,
                                              node_box=nb, node_label=nl, score_head=score_head)
                out_preds.append(DetectionPrediction(image_id=iid, **picked))
        return out_preds

    # Pick score_head on val (compare p_tp50 vs p_tp75 vs selection).
    val_data = _filter_by_split(graphs, src_labels, "val")
    val_gts = _gts(val_data)
    test_data = _filter_by_split(graphs, src_labels, "test")
    test_gts = _gts(test_data)

    def _eval(preds, gts, iou_t, agn=class_agnostic):
        return evaluate_predictions(preds, gts, iou_threshold=iou_t,
                                      num_classes=num_classes, class_agnostic=agn)["AP"]

    def _miou(preds, gts):
        gt_by_id = {g.image_id: g for g in gts}
        ious = []
        for p in preds:
            gt = gt_by_id.get(p.image_id)
            if gt is None or p.boxes_xyxy.numel() == 0 or gt.boxes_xyxy.numel() == 0:
                continue
            m = box_iou(p.boxes_xyxy, gt.boxes_xyxy)
            if m.numel() > 0:
                ious.append(float(m.max(dim=1)[0].mean().item()))
        return float(sum(ious) / max(1, len(ious)))

    selection_metric_is_ap75 = True   # since headroom only exists at AP75
    val_score_modes = {}
    best_head = "p_tp50"; best_val = -1.0
    for sh in ("p_tp50", "p_tp75", "selection"):
        vp = _model_predict("val", sh)
        ap50 = _eval(vp, val_gts, iou_match, agn=class_agnostic)
        ap75 = _eval(vp, val_gts, 0.75, agn=class_agnostic)
        sel = ap75 if selection_metric_is_ap75 else ap50
        val_score_modes[sh] = {"val_ap50": ap50, "val_ap75": ap75}
        if sel > best_val:
            best_val = sel; best_head = sh
    print(f"  [seed {seed}] selected score_head={best_head} (val_ap75={best_val:.4f})")

    # Final test predictions with frozen score_head.
    tgx_test = _model_predict("test", best_head)

    # All test baselines
    methods: Dict[str, List[DetectionPrediction]] = {}
    methods["fusion::tgraphx_candidate_selector"] = tgx_test
    for fusion in ("nms", "wbf", "soft_nms", "best_proposal"):
        methods[f"external::{fusion}"] = _external_classical(test_data, fusion, iou_cluster)
    for di, dn in enumerate(detector_names):
        methods[f"raw::{dn}"] = _select_per_detector(test_data, di)
    for tn in ("cluster", "consensus", "nms_candidate", "soft_nms_candidate", "best_proposal_candidate"):
        methods[f"graph::{tn}"] = _select_nodes_by_type(test_data, tn)

    method_results = {}
    for n, p in methods.items():
        method_results[n] = {
            "AP50": _eval(p, test_gts, iou_match, agn=class_agnostic),
            "AP75": _eval(p, test_gts, 0.75, agn=class_agnostic),
            "mIoU": _miou(p, test_gts),
        }

    # Paired bootstrap, AP50 and AP75 separately.
    pair_class_agnostic = class_agnostic
    _, tgx_aps50 = per_image_aps(tgx_test, test_gts, iou_threshold=iou_match,
                                   class_agnostic=pair_class_agnostic)
    _, tgx_aps75 = per_image_aps(tgx_test, test_gts, iou_threshold=0.75,
                                   class_agnostic=pair_class_agnostic)
    bootstrap_ap50 = {}; bootstrap_ap75 = {}
    for n, p in methods.items():
        if n == "fusion::tgraphx_candidate_selector":
            continue
        _, a50 = per_image_aps(p, test_gts, iou_threshold=iou_match, class_agnostic=pair_class_agnostic)
        _, a75 = per_image_aps(p, test_gts, iou_threshold=0.75, class_agnostic=pair_class_agnostic)
        if tgx_aps50.shape == a50.shape:
            bootstrap_ap50[n] = paired_bootstrap(tgx_aps50, a50, seed=seed)
        if tgx_aps75.shape == a75.shape:
            bootstrap_ap75[n] = paired_bootstrap(tgx_aps75, a75, seed=seed)

    tgx_test_res = method_results["fusion::tgraphx_candidate_selector"]
    metrics = {
        "seed": seed, "device": device, "detector_names": detector_names,
        "num_classes": num_classes,
        "test_metrics_selected_mode": {
            "test_ap50": tgx_test_res["AP50"],
            "test_ap75": tgx_test_res["AP75"],
            "test_miou": tgx_test_res["mIoU"],
            "headline_ap": tgx_test_res["AP75"],
        },
        "selected_score_head": best_head,
        "val_score_modes": val_score_modes,
        "test_methods": method_results,
        # Step-06 compatibility
        "baseline_methods": {
            n: {"headline_ap": v["AP75"], "test_ap_class_agnostic": v["AP50"],
                "test_ap_class_aware": v["AP50"]}
            for n, v in method_results.items()
        },
        "paired_bootstrap_vs_baselines": bootstrap_ap50,
        "paired_bootstrap_ap75_vs_baselines": bootstrap_ap75,
        "training_history": history,
        "is_multiclass": is_mc,
        "uses_candidate_node_selector": True,
        "selected_score_mode": best_head,
        "score_mode_selection_metric": "AP75" if selection_metric_is_ap75 else "AP50",
    }
    (base_dir / f"metrics_seed{seed}.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"  [seed {seed}] TGX AP50={tgx_test_res['AP50']:.4f}  AP75={tgx_test_res['AP75']:.4f}  "
          f"NMS AP75={method_results['external::nms']['AP75']:.4f}  "
          f"WBF AP50={method_results['external::wbf']['AP50']:.4f}")
    return metrics


def run_multi_seed_candidate_selector(
    *, run_dir_with_graphs: str,
    config: Optional[Dict[str, Any]] = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    out_dir: Optional[str] = None,
    epochs: int = 30,
    device: str = "cpu",
    feature_mode: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = dict(config or {})
    if feature_mode is not None:
        cfg.setdefault("model", {})["feature_mode"] = feature_mode
    rd = Path(run_dir_with_graphs)
    print(f"[candidate-selector] reading graphs from {rd}/graphs.pt …")
    graphs = torch.load(rd / "graphs.pt", weights_only=False)
    src_labels = torch.load(rd / "source_labels.pt", weights_only=False)
    manifest = json.loads((rd / "split_manifest.json").read_text())
    detector_names = manifest["detector_names"]
    class_names = manifest.get("class_names", ["car"])
    for e in graphs:
        if "slot_assignments" not in e[0].metadata:
            _attach_slot_metadata(e[0], e[1], detector_names)
    base_dir = Path(out_dir or "runs") / cfg.get("run_name", "candidate_selector")
    base_dir.mkdir(parents=True, exist_ok=True)
    all_seeds = []
    for seed in seeds:
        t0 = time.time()
        r = run_seed(cfg, seed, base_dir, graphs, src_labels, detector_names, class_names,
                       epochs=epochs, device=device)
        r["elapsed_s"] = time.time() - t0
        all_seeds.append(r)
    # Summary
    summary = {"seeds": list(seeds), "n_seeds": len(all_seeds), "detector_names": detector_names}
    if all_seeds:
        method_names = sorted(all_seeds[0]["test_methods"].keys())
        means = {}
        for n in method_names:
            ap50s = [s["test_methods"][n]["AP50"] for s in all_seeds]
            ap75s = [s["test_methods"][n]["AP75"] for s in all_seeds]
            mious = [s["test_methods"][n]["mIoU"] for s in all_seeds]
            means[n] = {
                "AP50_mean": statistics.mean(ap50s),
                "AP50_std": statistics.stdev(ap50s) if len(ap50s) > 1 else 0.0,
                "AP75_mean": statistics.mean(ap75s),
                "AP75_std": statistics.stdev(ap75s) if len(ap75s) > 1 else 0.0,
                "mIoU_mean": statistics.mean(mious),
            }
        summary["method_means"] = means
        # Bootstrap aggregate
        for tag, key in [("ap50", "paired_bootstrap_vs_baselines"),
                          ("ap75", "paired_bootstrap_ap75_vs_baselines")]:
            agg = {}
            for s in all_seeds:
                for n, b in s.get(key, {}).items():
                    if not b: continue
                    agg.setdefault(n, []).append(b)
            summary[f"bootstrap_means_{tag}"] = {
                n: {
                    "P_mean": statistics.mean([b["p_a_gt_b"] for b in vs]),
                    "P_min": min(b["p_a_gt_b"] for b in vs),
                    "P_max": max(b["p_a_gt_b"] for b in vs),
                    "Δ_mean": statistics.mean([b["mean_diff"] for b in vs]),
                    "n_seeds_clear_p95": sum(1 for b in vs if b["p_a_gt_b"] >= 0.95),
                }
                for n, vs in agg.items()
            }
    (base_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[candidate-selector] → {base_dir/'summary.json'}")
    return summary

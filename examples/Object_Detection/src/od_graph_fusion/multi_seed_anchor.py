"""Multi-seed runner for the AnchorRouter.

Produces, per seed:
  - {out_dir}/seed_{NN}/results.json — per-method AP table + override metrics
  - {out_dir}/seed_{NN}/source_routing_metrics.json — override/source diagnostics
  - {out_dir}/seed_{NN}/anchor_metrics.json — anchor-specific diagnostics
  - {out_dir}/seed_{NN}/paired_bootstrap.json — TGraphX vs each baseline
  - {out_dir}/seed_{NN}/anchor_failure_examples.json — top-K decision traces

And aggregate:
  - {out_dir}/summary.json — per-method means + paired bootstrap aggregate

Designed to be called either from a CLI script or the unit-test harness in
sanity-overfit mode.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .baselines import nms, weighted_boxes_fusion
from .config import load_config, resolve_device
from .datasets import load_dataset
from .detectors import build_detectors
from .evaluation import (
    DetectionPrediction, GroundTruth,
    evaluate_predictions, evaluate_at_multiple_ious,
)
from .graph_builder import build_detection_graph, NODE_TYPES
from .multi_seed_v2 import _attach_slot_metadata, _build_util_and_labels
from .paired_bootstrap import per_image_aps, paired_bootstrap
from .reproducibility import set_global_seed
from .source_router_v3 import NUM_SOURCES, SOURCE_SLOTS, detector_name_to_slot
from .anchor_router import AnchorRouter, AnchorRouterConfig, SPECIALIST_SLOTS, calibrate_temperature
from .anchor_training import (
    AnchorLossWeights, anchor_router_loss, build_anchor_targets, specialist_targets,
)
from .pairwise_features import (
    PAIRWISE_FEAT_DIM, SPECIALIST_EXTRA_DIM,
    pairwise_features_for_cluster, union_specialist_features, yolo_specialist_features,
    cluster_box_variance, cluster_score_entropy,
)
from .source_priors import (
    PriorTable, SIZE_BINS, SCORE_BUCKETS,
    compute_priors, select_anchor_on_validation,
    size_bin_for_box, score_bucket_for_score,
)
from .hard_cases import build_descriptors, hard_case_counts


# ── Helpers ───────────────────────────────────────────────────────────


def _cluster_meta(g, meta, anchor_slot: int) -> Dict[str, Any]:
    """Build per-cluster pairwise / prior inputs that feed into the model."""
    md = g.metadata if isinstance(g.metadata, dict) else {}
    node_box = md.get("node_box")
    node_score = md.get("node_score")
    node_label = md.get("node_label")
    cluster_of = meta.cluster_of_node
    slot_node_idx = md.get("_slot_node_idx")
    slot_assignments = md.get("slot_assignments")
    if slot_node_idx is None or node_box is None or node_score is None:
        return {}
    n_clusters = slot_node_idx.shape[0]
    slot_avail = slot_node_idx >= 0
    S = slot_node_idx.shape[1]

    # Per-cluster bookkeeping
    cluster_class = torch.zeros(n_clusters, dtype=torch.long)
    cluster_size_bin = torch.full((n_clusters,), 1, dtype=torch.long)   # default medium
    cluster_score_bucket = torch.full((n_clusters,), 1, dtype=torch.long) # default mid
    cluster_anchor_score = torch.zeros(n_clusters)
    n_proposals_in_cluster = torch.zeros(n_clusters, dtype=torch.long)
    pairwise = torch.zeros(n_clusters, S, PAIRWISE_FEAT_DIM)
    union_feats = torch.zeros(n_clusters, SPECIALIST_EXTRA_DIM)
    yolo_feats = torch.zeros(n_clusters, SPECIALIST_EXTRA_DIM)
    rtdetr_feats = torch.zeros(n_clusters, SPECIALIST_EXTRA_DIM)
    retina_feats = torch.zeros(n_clusters, SPECIALIST_EXTRA_DIM)

    image_size = meta.image_size
    for c in range(n_clusters):
        # Anchor-slot info
        anc_node = int(slot_node_idx[c, anchor_slot].item())
        if anc_node >= 0:
            anc_box = node_box[anc_node]
            cluster_anchor_score[c] = float(node_score[anc_node].item())
            cluster_class[c] = (int(node_label[anc_node].item())
                                 if node_label is not None and anc_node < node_label.shape[0] else 0)
            cluster_size_bin[c] = size_bin_for_box(anc_box, image_size)
            cluster_score_bucket[c] = score_bucket_for_score(float(node_score[anc_node].item()))
        # Cluster-level summaries
        in_c = cluster_of == c
        n_proposals_in_cluster[c] = int(((meta.node_types == NODE_TYPES["proposal"]) & in_c).sum().item())
        iou_dis = cluster_box_variance(c, slot_node_idx, slot_avail, node_box)
        score_ent = cluster_score_entropy(c, slot_node_idx, slot_avail, node_score)
        # Pairwise features.
        pairwise[c] = pairwise_features_for_cluster(
            c, slot_node_idx, slot_avail, anchor_slot, node_box, node_score,
            (node_label if node_label is not None else torch.zeros(node_box.shape[0], dtype=torch.long)),
            n_proposals_in_cluster=int(n_proposals_in_cluster[c].item()),
            detector_agreement_entropy=score_ent,
            score_entropy=score_ent,
            box_variance=iou_dis,
            proposal_max_iou=1.0 - iou_dis,
        )
        union_feats[c] = union_specialist_features(
            c, slot_node_idx, slot_avail, node_box, node_score, anchor_slot,
            n_proposals_in_cluster=int(n_proposals_in_cluster[c].item()),
            proposal_mean_pairwise_iou=1.0 - iou_dis,
            proposal_max_iou_to_union=1.0,
        )
        yolo_feats[c] = yolo_specialist_features(
            c, slot_node_idx, slot_avail, node_box, node_score, anchor_slot,
        )
        rtdetr_feats[c] = yolo_specialist_features(   # share generic shape
            c, slot_node_idx, slot_avail, node_box, node_score, anchor_slot,
        )
        retina_feats[c] = yolo_specialist_features(
            c, slot_node_idx, slot_avail, node_box, node_score, anchor_slot,
        )
    return {
        "cluster_class": cluster_class,
        "cluster_size_bin": cluster_size_bin,
        "cluster_score_bucket": cluster_score_bucket,
        "cluster_anchor_score": cluster_anchor_score,
        "n_proposals_in_cluster": n_proposals_in_cluster,
        "pairwise": pairwise,
        "specialist_extras": {
            "union": union_feats,
            "yolo_modern": yolo_feats,
            "rt_detr": rtdetr_feats,
            "retinanet": retina_feats,
        },
    }


def _compute_baseline_aps_val(test_data, detector_names, num_classes, class_agnostic, iou_match, iou_cluster):
    """Compute validation/test AP for each baseline (NMS, WBF, per-detector, etc.)."""
    method_preds: Dict[str, List[DetectionPrediction]] = {n: [] for n in
                                                            ["fusion::nms", "fusion::wbf"] + [f"det::{d}" for d in detector_names]}
    gts: List[GroundTruth] = []
    for g, meta, rec in test_data:
        md = g.metadata if isinstance(g.metadata, dict) else {}
        nb = md.get("node_box"); ns = md.get("node_score"); nl = md.get("node_label")
        nt = md.get("node_types"); pid = md.get("proposal_det_ids")
        iid = meta.image_id
        gts.append(GroundTruth(image_id=iid, boxes_xyxy=rec.gt_boxes, labels=rec.gt_labels))
        if nb is None or ns is None or nt is None or pid is None:
            for k in method_preds:
                method_preds[k].append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0,4),
                                                            scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
            continue
        prop_mask = (nt == NODE_TYPES["proposal"])
        all_b = nb[prop_mask]; all_s = ns[prop_mask]
        all_l = nl[prop_mask] if nl is not None else torch.zeros(prop_mask.sum(), dtype=torch.long)
        all_d = pid[prop_mask]
        # Per detector
        for di, dn in enumerate(detector_names):
            m = (all_d == di)
            method_preds[f"det::{dn}"].append(DetectionPrediction(
                image_id=iid,
                boxes_xyxy=all_b[m], scores=all_s[m],
                labels=all_l[m] if all_l.numel() > 0 else torch.zeros(0, dtype=torch.long),
            ))
        if all_b.numel() == 0:
            method_preds["fusion::nms"].append(DetectionPrediction(image_id=iid, boxes_xyxy=all_b, scores=all_s, labels=all_l))
            method_preds["fusion::wbf"].append(DetectionPrediction(image_id=iid, boxes_xyxy=all_b, scores=all_s, labels=all_l))
            continue
        keep = nms(all_b, all_s, iou_threshold=iou_cluster)
        method_preds["fusion::nms"].append(DetectionPrediction(image_id=iid,
            boxes_xyxy=all_b[keep], scores=all_s[keep], labels=all_l[keep]))
        fb, fs, fl = weighted_boxes_fusion(all_b, all_s, all_l, iou_threshold=iou_cluster)
        method_preds["fusion::wbf"].append(DetectionPrediction(image_id=iid, boxes_xyxy=fb, scores=fs, labels=fl))

    ap_map: Dict[str, float] = {}
    for name, preds in method_preds.items():
        ap_map[name] = evaluate_predictions(preds, gts, iou_threshold=iou_match,
                                              num_classes=num_classes, class_agnostic=class_agnostic)["AP"]
    return method_preds, gts, ap_map


# ── Main driver ───────────────────────────────────────────────────────


def run_anchor_seed(
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
    num_classes = len(class_names)
    is_multiclass = num_classes > 2
    class_agnostic = bool(cfg.get("evaluation", {}).get("class_agnostic", not is_multiclass))
    iou_list = cfg.get("evaluation", {}).get("ap_iou_thresholds", [0.5, 0.75])
    iou_match = float(cfg.get("evaluation", {}).get("iou_match_threshold", 0.5))

    cfg_graph = cfg.get("graph", {})
    iou_cluster = float(cfg_graph.get("cluster_iou_threshold", 0.5))
    crop_size = int(cfg_graph.get("crop_size", 64))
    max_props = int(cfg_graph.get("max_proposals_per_image", 48))

    # Deterministic split per seed
    from .multi_seed_v2 import _split_seeded
    by_split = _split_seeded(records, seed)
    idx_by_id = {r.image_id: i for i, r in enumerate(records)}

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
    val_data   = _build_graphs(by_split["val"], False)
    test_data  = _build_graphs(by_split["test"], False)

    # ── Anchor selection on VAL only ───────────────────────────────────
    _val_method_preds, val_gts, val_method_ap = _compute_baseline_aps_val(
        val_data, detector_names, num_classes, class_agnostic, iou_match, iou_cluster,
    )
    anchor_mode = cfg.get("model", {}).get("anchor_mode", "validation_best_global_source")
    anchor_slot, anchor_label = select_anchor_on_validation(
        val_method_ap, detector_names=detector_names, anchor_mode=anchor_mode,
    )
    print(f"[anchor] seed={seed}  anchor={anchor_label} slot={anchor_slot}")

    # ── Build per-cluster targets and descriptors on TRAIN ─────────────
    utility_mode = cfg.get("training", {}).get("utility_mode", "ap50")
    train_clusters: List[Dict[str, Any]] = []
    descriptor_input: List[Dict[str, Any]] = []
    for gi, (g, meta, rec) in enumerate(train_data):
        labels = _build_util_and_labels(g, meta, rec.gt_boxes, rec.gt_labels,
                                          class_agnostic=class_agnostic,
                                          utility_mode=utility_mode)
        if labels is None:
            continue
        _, _best_slot, _bl_slot, util_per_slot, slot_avail = labels
        cmeta = _cluster_meta(g, meta, anchor_slot)
        if not cmeta:
            continue
        C = util_per_slot.shape[0]
        for c in range(C):
            train_clusters.append({
                "graph_idx": gi, "cluster_id": c,
                "slot_utility": util_per_slot[c],
                "slot_avail": slot_avail[c],
                "cluster_class": int(cmeta["cluster_class"][c].item()),
                "cluster_size_bin": int(cmeta["cluster_size_bin"][c].item()),
                "cluster_score_bucket": int(cmeta["cluster_score_bucket"][c].item()),
            })
            descriptor_input.append({
                "graph_idx": gi, "cluster_id": c,
                "slot_utility": util_per_slot[c],
                "slot_avail": slot_avail[c],
                "anchor_score": float(cmeta["cluster_anchor_score"][c].item()),
                "iou_disagreement": float(cmeta["pairwise"][c, :, 0].abs().mean().item()),
            })
    # Priors (train + val, anchor-conditioning)
    priors = compute_priors(train_clusters, anchor_slot=anchor_slot, num_classes=num_classes)
    # Hard case audit (logging only — used by sampler in the loop too)
    descriptors = build_descriptors(descriptor_input, anchor_slot=anchor_slot)
    hc_counts = hard_case_counts(descriptors)
    (out_dir / "hard_case_counts.json").write_text(json.dumps(hc_counts, indent=2, default=str))

    # ── Model + training ───────────────────────────────────────────────
    g0, _, _ = train_data[0]
    md = g0.metadata.get("node_metadata")
    metadata_dim = md.shape[1] if md is not None else None
    ea = g0.edge_features
    edge_feat_dim = ea.shape[1] if ea is not None and ea.numel() > 0 else 14
    model_cfg = AnchorRouterConfig(
        num_classes=num_classes,
        num_detectors=len(detector_names),
        crop_size=crop_size,
        anchor_slot=anchor_slot,
        crop_channels=cfg.get("model", {}).get("crop_channels", 16),
        hidden_dim=cfg.get("model", {}).get("hidden_dim", 64),
        metadata_dim=metadata_dim,
        edge_feat_dim=edge_feat_dim,
        num_message_passing=cfg.get("model", {}).get("num_message_passing", 2),
    )
    model = AnchorRouter(model_cfg).to(device)
    tcfg = cfg.get("training", {})
    optim = torch.optim.Adam(model.parameters(),
                              lr=float(tcfg.get("lr", 5e-4)),
                              weight_decay=float(tcfg.get("weight_decay", 1e-4)))
    epochs = int(tcfg.get("epochs", 8))
    weights = AnchorLossWeights(
        false_override_penalty=float(tcfg.get("false_override_penalty", 7.0)),
    )
    history: List[float] = []
    for ep in range(1, epochs + 1):
        model.train(); total = 0.0; n = 0
        for gi, (g, meta, rec) in enumerate(train_data):
            labels = _build_util_and_labels(g, meta, rec.gt_boxes, rec.gt_labels,
                                              class_agnostic=class_agnostic,
                                              utility_mode=utility_mode)
            if labels is None:
                continue
            _, _bs, _bl, util_per_slot, slot_avail = labels
            cmeta = _cluster_meta(g, meta, anchor_slot)
            if not cmeta:
                continue
            tgt = build_anchor_targets(util_per_slot, slot_avail, anchor_slot,
                                        margin=float(tcfg.get("override_margin", 0.0)))
            spec_t = specialist_targets(util_per_slot, slot_avail, anchor_slot, SPECIALIST_SLOTS)
            gg = g.to(device)
            priors_feat = priors.as_features(
                cmeta["cluster_class"], cmeta["cluster_size_bin"], cmeta["cluster_score_bucket"],
            )
            out = model(
                gg, detector_names=detector_names,
                pairwise_feats=cmeta["pairwise"], priors_feats=priors_feat,
                cluster_class=cmeta["cluster_class"],
                specialist_extras=cmeta["specialist_extras"],
                anchor_slot_per_cluster=torch.full(
                    (util_per_slot.shape[0],), anchor_slot, dtype=torch.long),
            )
            if out["delta_ap50_hat"].numel() == 0:
                continue
            losses = anchor_router_loss(
                out,
                delta_true=tgt["delta_true"],
                slot_avail=slot_avail,
                tp50_true=tgt["tp50_true"],
                anchor_slot_per_cluster=tgt["anchor_slot_per_cluster"],
                best_alt_slot_per_cluster=tgt["best_alt_slot"],
                specialist_true=spec_t,
                weights=weights,
                override_threshold=float(tcfg.get("override_margin", 0.0)),
            )
            loss = losses["total"]
            if not loss.requires_grad:
                continue
            optim.zero_grad(); loss.backward(); optim.step()
            total += float(loss.item()); n += 1
        avg = total / max(1, n)
        history.append(avg)
        if ep % max(1, epochs // 3) == 0 or ep == epochs:
            print(f"  [seed {seed}] ep {ep}/{epochs}  loss={avg:.4f}")

    # ── Threshold + score-mode sweep on VAL only ───────────────────────
    threshold_grid = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
    val_pairs = []  # list of (cmeta, util_per_slot, slot_avail, rec, g, meta)
    for g, meta, rec in val_data:
        labels = _build_util_and_labels(g, meta, rec.gt_boxes, rec.gt_labels,
                                          class_agnostic=class_agnostic,
                                          utility_mode=utility_mode)
        if labels is None:
            continue
        cm = _cluster_meta(g, meta, anchor_slot)
        if not cm:
            continue
        val_pairs.append((cm, labels[3], labels[4], rec, g, meta))

    def _predict_for_threshold(data_pairs, threshold: float, score_fn) -> List[DetectionPrediction]:
        preds = []
        model.eval()
        with torch.no_grad():
            for cmeta, util_per_slot, slot_avail, rec, g, meta in data_pairs:
                gg = g.to(device)
                priors_feat = priors.as_features(
                    cmeta["cluster_class"], cmeta["cluster_size_bin"], cmeta["cluster_score_bucket"],
                )
                out = model(
                    gg, detector_names=detector_names,
                    pairwise_feats=cmeta["pairwise"], priors_feats=priors_feat,
                    cluster_class=cmeta["cluster_class"],
                    specialist_extras=cmeta["specialist_extras"],
                    anchor_slot_per_cluster=torch.full(
                        (util_per_slot.shape[0],), anchor_slot, dtype=torch.long, device=device),
                )
                chosen, _ = model.decide(out, override_threshold=threshold)
                sni = out["slot_node_idx"]
                nb = g.metadata.get("node_box"); ns = g.metadata.get("node_score")
                nl = g.metadata.get("node_label")
                boxes, scores, labels = [], [], []
                C = chosen.shape[0]
                tp50_hat = out["tp50_hat"]
                for c in range(C):
                    s = int(chosen[c].item())
                    if not bool(out["source_mask"][c, s].item()):
                        continue
                    n = int(sni[c, s].item())
                    if n < 0 or n >= nb.shape[0]:
                        continue
                    base = float(ns[n].item()) if ns is not None else 0.5
                    tp50 = float(torch.sigmoid(tp50_hat[c, s]).item())
                    scr = score_fn(base, tp50)
                    boxes.append(nb[n].cpu()); scores.append(torch.tensor(scr))
                    labels.append(nl[n].cpu() if nl is not None else torch.tensor(0, dtype=torch.long))
                preds.append(DetectionPrediction(
                    image_id=meta.image_id,
                    boxes_xyxy=torch.stack(boxes) if boxes else torch.zeros(0, 4),
                    scores=torch.stack(scores) if scores else torch.zeros(0),
                    labels=torch.stack(labels) if labels else torch.zeros(0, dtype=torch.long),
                ))
        return preds

    score_modes = {
        "p_tp50":      lambda b, p: float(p),
        "p_tp50*base": lambda b, p: float(p * max(b, 0.1)),
        "base":         lambda b, p: float(b),
    }
    # Pick (threshold, score_mode) on validation only.
    selection_class_agnostic = not is_multiclass
    best_thr, best_mode, best_val_ap = 0.0, "p_tp50", -1.0
    val_score_modes: Dict[str, Dict[str, float]] = {}
    for thr in threshold_grid:
        for mname, sfn in score_modes.items():
            vp = _predict_for_threshold(val_pairs, thr, sfn)
            r_agn = evaluate_predictions(vp, val_gts, iou_threshold=iou_match,
                                          num_classes=num_classes, class_agnostic=True)["AP"]
            r_aware = evaluate_predictions(vp, val_gts, iou_threshold=iou_match,
                                             num_classes=num_classes, class_agnostic=False)["AP"]
            key = f"thr{thr:.2f}_{mname}"
            val_score_modes[key] = {
                "val_ap_agn": r_agn, "val_ap_aware": r_aware,
                "selection_metric": "class_agnostic_AP" if selection_class_agnostic else "class_aware_AP",
                "threshold": thr,
            }
            sel = r_agn if selection_class_agnostic else r_aware
            if sel > best_val_ap:
                best_val_ap = sel; best_thr = thr; best_mode = mname
    print(f"  [val] best (thr={best_thr:.2f}, mode={best_mode}) val_ap={best_val_ap:.4f}")

    # ── Evaluate baselines + TGraphX on TEST once ──────────────────────
    test_method_preds, test_gts, _ = _compute_baseline_aps_val(
        test_data, detector_names, num_classes, class_agnostic, iou_match, iou_cluster,
    )
    test_pairs = []
    for g, meta, rec in test_data:
        labels = _build_util_and_labels(g, meta, rec.gt_boxes, rec.gt_labels,
                                          class_agnostic=class_agnostic, utility_mode=utility_mode)
        if labels is None:
            continue
        cm = _cluster_meta(g, meta, anchor_slot)
        if not cm:
            continue
        test_pairs.append((cm, labels[3], labels[4], rec, g, meta))
    tgx_preds = _predict_for_threshold(test_pairs, best_thr, score_modes[best_mode])
    test_method_preds["fusion::tgraphx"] = tgx_preds

    method_test_ap: Dict[str, Dict[str, float]] = {}
    for name, preds in test_method_preds.items():
        ap_agn = evaluate_predictions(preds, test_gts, iou_threshold=iou_match,
                                        num_classes=num_classes, class_agnostic=True)["AP"]
        ap_aware = evaluate_predictions(preds, test_gts, iou_threshold=iou_match,
                                          num_classes=num_classes, class_agnostic=False)["AP"]
        method_test_ap[name] = {
            "test_ap_class_agnostic": ap_agn,
            "test_ap_class_aware": ap_aware,
            "headline_ap": ap_aware if is_multiclass else ap_agn,
        }

    # ── Paired bootstrap TGraphX vs each baseline ─────────────────────
    pair_class_agnostic = not is_multiclass
    _, tgx_aps = per_image_aps(tgx_preds, test_gts, iou_threshold=iou_match,
                                 class_agnostic=pair_class_agnostic)
    bootstraps: Dict[str, Dict[str, float]] = {}
    for name, preds in test_method_preds.items():
        if name == "fusion::tgraphx":
            continue
        _, b_aps = per_image_aps(preds, test_gts, iou_threshold=iou_match,
                                   class_agnostic=pair_class_agnostic)
        if tgx_aps.shape != b_aps.shape:
            continue
        bootstraps[name] = paired_bootstrap(tgx_aps, b_aps, seed=seed)

    # ── Override + source metrics (parity with multi_seed_v2 outputs) ──
    n_override = 0; n_anchor_kept = 0; n_total = 0
    succ = 0; failed = 0; ovr_gains: List[float] = []
    src_acc_list: List[int] = []
    model.eval()
    with torch.no_grad():
        for cmeta, util_per_slot, slot_avail, rec, g, meta in test_pairs:
            gg = g.to(device)
            priors_feat = priors.as_features(
                cmeta["cluster_class"], cmeta["cluster_size_bin"], cmeta["cluster_score_bucket"],
            )
            out = model(
                gg, detector_names=detector_names,
                pairwise_feats=cmeta["pairwise"], priors_feats=priors_feat,
                cluster_class=cmeta["cluster_class"],
                specialist_extras=cmeta["specialist_extras"],
                anchor_slot_per_cluster=torch.full(
                    (util_per_slot.shape[0],), anchor_slot, dtype=torch.long, device=device),
            )
            chosen, chose_anchor = model.decide(out, override_threshold=best_thr)
            for c in range(chosen.shape[0]):
                if not bool(slot_avail[c, anchor_slot].item()):
                    continue
                n_total += 1
                cc = int(chosen[c].item())
                util_chosen = float(util_per_slot[c, cc].item()) if bool(slot_avail[c, cc].item()) else 0.0
                util_anchor = float(util_per_slot[c, anchor_slot].item())
                if cc == anchor_slot:
                    n_anchor_kept += 1
                else:
                    n_override += 1
                    if util_chosen > util_anchor:
                        succ += 1
                    else:
                        failed += 1
                    ovr_gains.append(util_chosen - util_anchor)
                # oracle for this cluster
                u = util_per_slot[c].clone()
                u[~slot_avail[c]] = float("-inf")
                oracle = int(u.argmax().item()) if torch.isfinite(u).any() else -1
                if oracle >= 0:
                    src_acc_list.append(int(cc == oracle))

    routing = {
        "anchor_slot": anchor_slot, "anchor_label": anchor_label,
        "threshold": best_thr, "score_mode": best_mode,
        "n_total_clusters": n_total, "n_overrides": n_override, "n_anchor_kept": n_anchor_kept,
        "override_rate": n_override / max(1, n_total),
        "successful_overrides": succ, "failed_overrides": failed,
        "override_success_rate": succ / max(1, n_override),
        "false_override_rate": failed / max(1, n_override),
        "deployed_source_acc": sum(src_acc_list) / max(1, len(src_acc_list)),
        "mean_iou_gain_per_override": (sum(ovr_gains) / max(1, len(ovr_gains))) if ovr_gains else 0.0,
        "uses_override_router": True, "uses_anchor_router": True,
    }
    (out_dir / "source_routing_metrics.json").write_text(json.dumps(routing, indent=2))
    (out_dir / "results.json").write_text(json.dumps({
        "seed": seed, "anchor": anchor_label, "threshold": best_thr,
        "score_mode": best_mode, "results": method_test_ap,
        "paired_bootstrap_vs_baselines": bootstraps,
        "val_score_modes": val_score_modes,
        "hard_case_counts": hc_counts,
    }, indent=2, default=str))
    # Step-05-compatible metrics file so Step 06 still works.
    tgx_test = method_test_ap["fusion::tgraphx"]
    metrics_payload = {
        "seed": seed, "device": device,
        "detector_names": detector_names,
        "num_classes": num_classes, "num_detectors": len(detector_names),
        "val_score_modes": val_score_modes,
        "selected_score_mode": best_mode,
        "score_mode_selection_metric": "class_aware_AP" if is_multiclass else "class_agnostic_AP",
        "anchor_slot": anchor_slot, "anchor_label": anchor_label,
        "override_threshold": best_thr,
        "test_metrics_selected_mode": {
            "test_ap_class_agnostic": tgx_test["test_ap_class_agnostic"],
            "test_ap_class_aware": tgx_test["test_ap_class_aware"],
            "headline_ap": tgx_test["headline_ap"],
        },
        "baseline_methods": method_test_ap,
        "paired_bootstrap_vs_baselines": bootstraps,
        "is_multiclass": is_multiclass,
        "uses_anchor_router": True,
    }
    (base_dir / f"metrics_seed{seed}.json").write_text(json.dumps(metrics_payload, indent=2))
    print(f"  [seed {seed}] TGraphX AP={tgx_test['headline_ap']:.4f}  "
          f"NMS={method_test_ap.get('fusion::nms', {}).get('headline_ap', 0):.4f}  "
          f"anchor={anchor_label}  thr={best_thr:.2f}")
    return {"seed": seed, "results": method_test_ap, "routing": routing,
            "paired_bootstrap_vs_baselines": bootstraps}


def run_multi_seed_anchor(
    config_path: str,
    seeds: Optional[Sequence[int]] = None,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Top-level entry: runs all seeds and writes per-seed + summary outputs."""
    cfg = load_config(config_path)
    device = resolve_device(cfg.get("device", "auto"))
    cfg["device"] = device
    if seeds is None:
        seeds = cfg.get("seeds", list(range(10)))
    base_dir = Path(out_dir or "runs") / (cfg.get("run_name", "anchor_exp") + "_anchor")
    base_dir.mkdir(parents=True, exist_ok=True)

    records = load_dataset(cfg)
    class_names = records[0].class_names if records else ["car"]
    detectors = build_detectors(dict(cfg, device=device), class_names)
    detector_names = list(detectors.keys())
    print(f"[anchor] Loaded {len(records)} records | detectors={detector_names} | seeds={list(seeds)}")

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
        t0 = time.time()
        r = run_anchor_seed(cfg, seed, base_dir, detector_outputs, records,
                              class_names, detector_names)
        elapsed = time.time() - t0
        r["elapsed_s"] = elapsed
        all_results.append(r)
    summary = {"seeds": list(seeds), "results": all_results,
               "detector_names": detector_names, "class_names": class_names}
    (base_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary

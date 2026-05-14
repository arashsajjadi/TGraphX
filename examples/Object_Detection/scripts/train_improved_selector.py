"""Train TGXPointerSelector and improved ablations on object-level candidate graphs.

Key improvements over the previous training script:
  1. Early stopping on val AP75 (patience=15 epochs, restore best weights)
  2. LR cosine schedule with linear warmup
  3. Crop augmentation during training (flip, brightness, noise)
  4. Gradient clipping (max_norm=1.0)
  5. AP75-focused utility (0.25*AP50 + 0.55*AP75 + 0.20*IoU)
  6. Simplified pairwise loss (best vs. top-4 negatives, not all pairs)
  7. Label smoothing on selection CE
  8. Larger weight_decay (5e-4)
  9. FP cluster supervision via tp50/tp75 heads (even without selection target)
  10. Proper val AP monitoring every epoch

Variants:
  tgx_pointer_selector     — main new TGraphX method (cross-attention, crop + meta)
  tgx_meta_only_pointer    — cross-attention, metadata only (ablation)
  flat_crop_mp             — pool-first + flat GNN (previous best)
  crop_no_mp               — CNN+metadata, no MP (ablation)
  metadata_only            — no crops (baseline)

Usage:
  python scripts/train_improved_selector.py \\
    --config configs/universal_candidate_voc_car_v2.yaml \\
    --device auto --seeds 0 1 2 3 4 \\
    --feature-mode tgx_pointer_selector
"""
import argparse, json, math, statistics, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F


# ── Utility ──────────────────────────────────────────────────────────────────

def _compute_utility(
    node_box, node_score, node_label, gt_box, gt_label,
    class_agnostic: bool,
) -> tuple:
    """AP75-focused deployable utility per candidate node.

    Returns (util [N], best_iou [N]) or (None, None) if no GT.
    """
    from od_graph_fusion.box_ops import box_iou
    if gt_box is None or gt_box.numel() == 0:
        return None, None
    N = node_box.shape[0]
    ious = box_iou(node_box, gt_box).max(dim=1)
    best_iou = ious.values          # [N]
    best_gt  = ious.indices         # [N]

    if not class_agnostic and node_label is not None:
        cls_ok = (node_label == gt_label[best_gt]).float()
    else:
        cls_ok = torch.ones(N)

    ns = node_score.clamp(0, 1)
    tp50 = (best_iou >= 0.50).float() * cls_ok
    tp75 = (best_iou >= 0.75).float() * cls_ok
    iou_u = best_iou * cls_ok

    ap50_u = tp50 + 0.05 * best_iou + 0.02 * ns
    ap75_u = tp75 + 0.05 * best_iou

    # AP75-focused: more weight on AP75
    util = 0.25 * ap50_u + 0.55 * ap75_u + 0.20 * iou_u
    return util, best_iou


def _cosine_lr(optimizer, epoch: int, total_epochs: int, base_lr: float,
               warmup_epochs: int = 5, min_lr: float = 1e-5) -> float:
    """Set LR and return current LR."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        prog = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def _make_model(feature_mode: str, model_cfg, metadata_dim, edge_feat_dim, device):
    """Build model by variant name."""
    from od_graph_fusion.improved_selector import PointerSelectorConfig, TGXPointerSelector
    from od_graph_fusion.candidate_node_selector import CandidateSelectorConfig

    if feature_mode in ("tgx_pointer_selector", "tgx_meta_only_pointer"):
        ptr_cfg = PointerSelectorConfig(
            num_classes=model_cfg.num_classes,
            num_detectors=model_cfg.num_detectors,
            crop_size=model_cfg.crop_size,
            crop_channels=model_cfg.crop_channels,
            hidden_dim=model_cfg.hidden_dim,
            metadata_dim=metadata_dim,
            num_attn_layers=model_cfg.num_message_passing,
            num_heads=2,
            ffn_expansion=2,
            dropout=0.15,
            use_crops=(feature_mode != "tgx_meta_only_pointer"),
            source_type_embed_dim=8,
        )
        return TGXPointerSelector(ptr_cfg).to(device)

    elif feature_mode in ("tgx_edge_attention", "tgx_spatial_attention",
                          "tgx_hybrid_attention", "tgx_convmp_small", "tgx_convmp_full"):
        from od_graph_fusion.attention_selector import build_selector
        cfg2 = CandidateSelectorConfig(
            num_classes=model_cfg.num_classes, num_detectors=model_cfg.num_detectors,
            crop_size=model_cfg.crop_size, crop_channels=model_cfg.crop_channels,
            hidden_dim=model_cfg.hidden_dim, metadata_dim=metadata_dim,
            edge_feat_dim=edge_feat_dim,
            num_message_passing=model_cfg.num_message_passing,
        )
        return build_selector(cfg2, feature_mode).to(device)

    else:  # flat_crop_mp, crop_no_mp, metadata_only, crop_metadata_mp
        from od_graph_fusion.candidate_node_selector import (
            TGraphXCandidateNodeSelector, CandidateSelectorConfig as CSC)
        cfg2 = CSC(
            num_classes=model_cfg.num_classes, num_detectors=model_cfg.num_detectors,
            crop_size=model_cfg.crop_size, crop_channels=model_cfg.crop_channels,
            hidden_dim=model_cfg.hidden_dim, metadata_dim=metadata_dim,
            edge_feat_dim=edge_feat_dim,
            num_message_passing=model_cfg.num_message_passing,
            feature_mode=feature_mode,
        )
        return TGraphXCandidateNodeSelector(cfg2).to(device)


# ── Augmentation ──────────────────────────────────────────────────────────────

def _augment_graph_crops(g, rng: torch.Generator):
    """Return a new Graph with augmented crop tensors. Original is unchanged."""
    from od_graph_fusion.improved_selector import augment_crops
    from tgraphx import Graph
    aug_feats = augment_crops(g.node_features.float(), rng=rng)
    return Graph(
        node_features=aug_feats,
        edge_index=g.edge_index,
        edge_features=g.edge_features,
        metadata=g.metadata,
    )


# ── Validation AP ─────────────────────────────────────────────────────────────

def _eval_val_ap75(model, val_data, obj_labels, detector_names,
                   num_classes, class_agnostic, iou_match, device):
    from od_graph_fusion.evaluation import DetectionPrediction, GroundTruth, evaluate_predictions
    from od_graph_fusion.candidate_node_selector import select_per_cluster
    from od_graph_fusion.candidate_mask import candidate_node_mask
    from od_graph_fusion.graph_builder import NODE_TYPES

    model.eval()
    preds_by_img = defaultdict(lambda: {"b": [], "s": [], "l": []})
    with torch.no_grad():
        for entry in val_data:
            g, img_id, cid, sp, cand_src, _, _ = entry
            nb = g.metadata.get("node_box")
            nl = g.metadata.get("node_label")
            nt = g.metadata.get("node_types")
            if nb is None:
                continue
            out = model(g.to(device), detector_names=detector_names)
            cand_m = candidate_node_mask(nt, NODE_TYPES) if nt is not None \
                     else torch.ones(nb.shape[0], dtype=torch.bool)
            cluster_of = torch.zeros(nb.shape[0], dtype=torch.long)
            picked = select_per_cluster(
                out, cluster_of=cluster_of, cand_mask=cand_m,
                node_box=nb,
                node_label=(nl if nl is not None
                            else torch.zeros(nb.shape[0], dtype=torch.long)),
                score_head="p_tp75",
            )
            if picked["boxes_xyxy"].numel() > 0:
                preds_by_img[img_id]["b"].append(picked["boxes_xyxy"])
                preds_by_img[img_id]["s"].append(picked["scores"])
                preds_by_img[img_id]["l"].append(picked["labels"])

    gts_by_img = {}
    for entry in val_data:
        g, img_id, cid, sp, _, _, _ = entry
        if img_id in gts_by_img:
            continue
        key = f"{img_id}_{cid}"
        lbl = obj_labels.get(key, {})
        gts_by_img[img_id] = GroundTruth(
            image_id=img_id,
            boxes_xyxy=lbl.get("gt_image_boxes", torch.zeros(0, 4)),
            labels=lbl.get("gt_image_labels", torch.zeros(0, dtype=torch.long)),
        )
    gts = list(gts_by_img.values())

    preds = [DetectionPrediction(
        image_id=img_id,
        boxes_xyxy=torch.cat(d["b"]) if d["b"] else torch.zeros(0, 4),
        scores=torch.cat(d["s"]) if d["s"] else torch.zeros(0),
        labels=torch.cat(d["l"]) if d["l"] else torch.zeros(0, dtype=torch.long),
    ) for img_id, d in preds_by_img.items()]

    if not preds:
        return 0.0
    return evaluate_predictions(
        preds, gts, iou_threshold=0.75,
        num_classes=num_classes, class_agnostic=class_agnostic
    )["AP"]


# ── Main training function per seed ──────────────────────────────────────────

def _train_one_seed(
    cfg, seed, base_dir, obj_graphs, obj_labels,
    detector_names, class_names, *,
    epochs=50, lr=3e-4, device="cpu", feature_mode="tgx_pointer_selector",
    use_augmentation=True, early_stop_patience=15,
    warmup_epochs=5,
):
    from od_graph_fusion.reproducibility import set_global_seed
    from od_graph_fusion.candidate_node_selector import (
        CandidateSelectorConfig, select_per_cluster,
    )
    from od_graph_fusion.candidate_mask import candidate_node_mask
    from od_graph_fusion.graph_builder import NODE_TYPES
    from od_graph_fusion.evaluation import DetectionPrediction, GroundTruth, evaluate_predictions
    from od_graph_fusion.baselines import nms, weighted_boxes_fusion, soft_nms
    from od_graph_fusion.box_ops import box_iou
    from od_graph_fusion.paired_bootstrap import paired_bootstrap, per_image_aps
    from od_graph_fusion.improved_selector import pointer_loss

    set_global_seed(seed, deterministic=False)
    num_classes   = len(class_names)
    is_mc         = num_classes > 2
    class_agnostic = bool(cfg.get("evaluation", {}).get("class_agnostic", not is_mc))
    iou_match     = float(cfg.get("evaluation", {}).get("iou_match_threshold", 0.5))
    iou_cluster   = float(cfg.get("graph", {}).get("cluster_iou_threshold", 0.5))
    crop_size     = int(cfg.get("graph", {}).get("crop_size", 128))

    # ── Split ──────────────────────────────────────────────────────────
    train_data, val_data, test_data = [], [], []
    for entry in obj_graphs:
        g, img_id, cid, split, cand_src, gt_box, gt_lbl = entry
        sp = obj_labels.get(f"{img_id}_{cid}", {}).get("split", split)
        (train_data if sp == "train" else val_data if sp == "val" else test_data).append(entry)

    g_rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_data), generator=g_rng).tolist()
    train_data = [train_data[i] for i in perm]
    print(f"  [seed {seed}] train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    # ── Model ──────────────────────────────────────────────────────────
    g0 = obj_graphs[0][0]
    md   = g0.metadata.get("node_metadata")
    meta_dim = md.shape[1] if md is not None else None
    ea   = g0.edge_features
    ef_dim = ea.shape[1] if ea is not None and ea.numel() > 0 else 14

    m_cfg = cfg.get("model", {})
    # For pointer selector use smaller crop (32) to reduce overfitting
    ptr_crop_size = (32 if feature_mode in ("tgx_pointer_selector", "tgx_meta_only_pointer")
                     else crop_size)
    model_cfg = CandidateSelectorConfig(
        num_classes=num_classes, num_detectors=len(detector_names),
        crop_size=ptr_crop_size,
        crop_channels=m_cfg.get("crop_channels_ptr", 8),
        hidden_dim=m_cfg.get("hidden_dim_ptr", 32),
        metadata_dim=meta_dim, edge_feat_dim=ef_dim,
        num_message_passing=m_cfg.get("num_attn_layers", 2),
    )
    model = _make_model(feature_mode, model_cfg, meta_dim, ef_dim, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [seed {seed}] model={feature_mode} params={n_params}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr,
                               weight_decay=cfg.get("training", {}).get("weight_decay", 5e-4))

    # Loss weights
    tw = cfg.get("training", {})
    w_sel  = float(tw.get("lambda_selection", 1.0))
    w_tp75 = float(tw.get("lambda_tp75", 2.0))
    w_tp50 = float(tw.get("lambda_tp50", 0.5))
    w_iou  = float(tw.get("lambda_iou", 0.5))
    w_rank = float(tw.get("lambda_rank", 0.5))
    label_smooth = float(tw.get("label_smooth", 0.05))

    # ── Pre-compute targets (GT-matched only) ──────────────────────────
    print(f"  [seed {seed}] building targets …")
    train_targets = {}
    for entry in train_data:
        g, img_id, cid, sp, cand_src, gt_box, gt_lbl = entry
        if gt_box is None or gt_box.numel() == 0:
            continue
        nb = g.metadata.get("node_box")
        ns = g.metadata.get("node_score")
        nl = g.metadata.get("node_label")
        nt = g.metadata.get("node_types")
        if nb is None:
            continue
        util, best_iou = _compute_utility(nb, ns if ns is not None else torch.zeros(nb.shape[0]),
                                           nl, gt_box, gt_lbl, class_agnostic)
        if util is None:
            continue
        cand_m = candidate_node_mask(nt, NODE_TYPES) if nt is not None \
                 else torch.ones(nb.shape[0], dtype=torch.bool)
        util_c = util.clone(); util_c[~cand_m] = -1e9
        best_node = util_c.argmax()

        if not class_agnostic and nl is not None:
            best_gt_idx = box_iou(nb, gt_box).max(dim=1).indices
            cls_ok = (nl == gt_lbl[best_gt_idx])
        else:
            cls_ok = torch.ones(nb.shape[0], dtype=torch.bool)

        train_targets[f"{img_id}_{cid}"] = {
            "best_node": best_node,
            "best_iou": best_iou,
            "cls_ok": cls_ok,
            "cand_mask": cand_m,
        }
    n_tp = sum(1 for t in train_targets.values() if int(t["best_iou"][int(t["best_node"])]) >= 0.5)
    print(f"  [seed {seed}] valid targets: {len(train_targets)}/{len(train_data)}")

    # ── Training loop ─────────────────────────────────────────────────
    aug_rng = torch.Generator(); aug_rng.manual_seed(seed * 1000)
    best_val_ap75 = -1.0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0
    history = {"train_loss": [], "val_ap75": [], "lr": []}

    for ep in range(1, epochs + 1):
        model.train()
        cur_lr = _cosine_lr(optim, ep - 1, epochs, lr, warmup_epochs)
        total = 0.0; n = 0

        for entry in train_data:
            g, img_id, cid, sp, cand_src, _, _ = entry
            key = f"{img_id}_{cid}"
            t = train_targets.get(key)
            if t is None:
                continue

            # Augment crops if enabled (training only)
            if use_augmentation and feature_mode not in ("metadata_only", "tgx_meta_only_pointer"):
                gg = _augment_graph_crops(g, aug_rng).to(device)
            else:
                gg = g.to(device)

            out = model(gg, detector_names=detector_names)
            losses = pointer_loss(
                out,
                best_node=t["best_node"].to(device),
                node_iou=t["best_iou"].to(device),
                cls_ok=t["cls_ok"].to(device),
                cand_mask=t["cand_mask"].to(device),
                w_sel=w_sel, w_tp75=w_tp75, w_tp50=w_tp50,
                w_iou=w_iou, w_rank=w_rank, label_smooth=label_smooth,
            )
            loss = losses["total"]
            if not loss.requires_grad:
                continue

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total += float(loss.item()); n += 1

        avg = total / max(1, n)
        history["train_loss"].append(avg)
        history["lr"].append(cur_lr)

        # Val AP75 every epoch (cheap since val is small)
        val_ap75 = _eval_val_ap75(
            model, val_data, obj_labels, detector_names,
            num_classes, class_agnostic, iou_match, device)
        history["val_ap75"].append(val_ap75)

        if val_ap75 > best_val_ap75:
            best_val_ap75 = val_ap75
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"  [seed {seed}] ep {ep:3d}/{epochs}  loss={avg:.4f}  "
                  f"val_ap75={val_ap75:.4f}  best={best_val_ap75:.4f}  "
                  f"lr={cur_lr:.2e}")

        # Early stopping
        if no_improve >= early_stop_patience and ep >= 2 * warmup_epochs:
            print(f"  [seed {seed}] Early stop at epoch {ep} "
                  f"(no improvement for {early_stop_patience} epochs)")
            break

    # Restore best model
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  [seed {seed}] Best val AP75={best_val_ap75:.4f}")

    # ── Full evaluation (val + test) ───────────────────────────────────
    def _predict(data, score_head="p_tp75"):
        model.eval()
        preds_by_img = defaultdict(lambda: {"b": [], "s": [], "l": []})
        with torch.no_grad():
            for entry in data:
                g, img_id, cid, sp, cand_src, _, _ = entry
                nb = g.metadata.get("node_box")
                nl = g.metadata.get("node_label")
                nt = g.metadata.get("node_types")
                if nb is None:
                    continue
                out = model(g.to(device), detector_names=detector_names)
                cand_m = candidate_node_mask(nt, NODE_TYPES) if nt is not None \
                         else torch.ones(nb.shape[0], dtype=torch.bool)
                cluster_of = torch.zeros(nb.shape[0], dtype=torch.long)
                picked = select_per_cluster(
                    out, cluster_of=cluster_of, cand_mask=cand_m,
                    node_box=nb,
                    node_label=(nl if nl is not None
                                else torch.zeros(nb.shape[0], dtype=torch.long)),
                    score_head=score_head,
                )
                if picked["boxes_xyxy"].numel() > 0:
                    preds_by_img[img_id]["b"].append(picked["boxes_xyxy"])
                    preds_by_img[img_id]["s"].append(picked["scores"])
                    preds_by_img[img_id]["l"].append(picked["labels"])
        return [DetectionPrediction(
            image_id=img_id,
            boxes_xyxy=torch.cat(d["b"]) if d["b"] else torch.zeros(0, 4),
            scores=torch.cat(d["s"]) if d["s"] else torch.zeros(0),
            labels=torch.cat(d["l"]) if d["l"] else torch.zeros(0, dtype=torch.long),
        ) for img_id, d in preds_by_img.items()]

    def _make_gts(data):
        gts = {}
        for g, img_id, cid, sp, _, _, _ in data:
            if img_id in gts:
                continue
            lbl = obj_labels.get(f"{img_id}_{cid}", {})
            gts[img_id] = GroundTruth(
                image_id=img_id,
                boxes_xyxy=lbl.get("gt_image_boxes", torch.zeros(0, 4)),
                labels=lbl.get("gt_image_labels", torch.zeros(0, dtype=torch.long)),
            )
        return list(gts.values())

    def _eval(preds, gts, iou_t):
        return evaluate_predictions(preds, gts, iou_threshold=iou_t,
                                      num_classes=num_classes,
                                      class_agnostic=class_agnostic)["AP"]

    def _miou(preds, gts):
        gts_by_id = {g.image_id: g for g in gts}
        ious = []
        for p in preds:
            gt = gts_by_id.get(p.image_id)
            if gt is None or not p.boxes_xyxy.numel() or not gt.boxes_xyxy.numel():
                continue
            m = box_iou(p.boxes_xyxy, gt.boxes_xyxy)
            if m.numel():
                ious.append(float(m.max(dim=1)[0].mean()))
        return float(sum(ious) / max(1, len(ious)))

    # Select score head on val
    best_head = "p_tp75"; best_h_ap = -1.0; val_score_modes = {}
    for sh in ("p_tp50", "p_tp75", "selection"):
        vp = _predict(val_data, sh)
        ap50 = _eval(vp, _make_gts(val_data), iou_match)
        ap75 = _eval(vp, _make_gts(val_data), 0.75)
        val_score_modes[sh] = {"val_ap50": ap50, "val_ap75": ap75}
        if ap75 > best_h_ap:
            best_h_ap = ap75; best_head = sh

    test_gts = _make_gts(test_data)
    tgx_test = _predict(test_data, best_head)
    test_ap50 = _eval(tgx_test, test_gts, iou_match)
    test_ap75 = _eval(tgx_test, test_gts, 0.75)
    test_miou = _miou(tgx_test, test_gts)

    # Baselines
    def _ext_fusion(data, fusion):
        pool = defaultdict(lambda: {"b": [], "s": [], "l": []})
        for g, img_id, cid, _, _, _, _ in data:
            nb = g.metadata.get("node_box"); ns = g.metadata.get("node_score")
            nl = g.metadata.get("node_label"); nt = g.metadata.get("node_types")
            if nb is None or nt is None:
                continue
            mask = nt == NODE_TYPES["proposal"]
            if not mask.any():
                continue
            pool[img_id]["b"].append(nb[mask]); pool[img_id]["s"].append(ns[mask])
            pool[img_id]["l"].append(nl[mask] if nl is not None
                                      else torch.zeros(mask.sum(), dtype=torch.long))
        result = []
        for img_id, d in pool.items():
            b = torch.cat(d["b"]); s = torch.cat(d["s"]); l = torch.cat(d["l"])
            if fusion == "wbf":
                fb, fs, fl = weighted_boxes_fusion(b, s, l, iou_threshold=iou_cluster)
                result.append(DetectionPrediction(image_id=img_id, boxes_xyxy=fb, scores=fs, labels=fl))
            elif fusion == "nms":
                k = nms(b, s, iou_threshold=iou_cluster)
                result.append(DetectionPrediction(image_id=img_id, boxes_xyxy=b[k], scores=s[k], labels=l[k]))
        return result

    def _select_type(data, type_name):
        type_id = NODE_TYPES[type_name]
        pool = defaultdict(lambda: {"b": [], "s": [], "l": []})
        for g, img_id, cid, _, _, _, _ in data:
            nb = g.metadata.get("node_box"); ns = g.metadata.get("node_score")
            nl = g.metadata.get("node_label"); nt = g.metadata.get("node_types")
            if nb is None or nt is None:
                continue
            mask = nt == type_id
            if not mask.any():
                continue
            pool[img_id]["b"].append(nb[mask]); pool[img_id]["s"].append(ns[mask])
            pool[img_id]["l"].append(nl[mask] if nl is not None
                                      else torch.zeros(mask.sum(), dtype=torch.long))
        return [DetectionPrediction(
            image_id=img_id,
            boxes_xyxy=torch.cat(d["b"]) if d["b"] else torch.zeros(0, 4),
            scores=torch.cat(d["s"]) if d["s"] else torch.zeros(0),
            labels=torch.cat(d["l"]) if d["l"] else torch.zeros(0, dtype=torch.long),
        ) for img_id, d in pool.items()]

    test_methods = {
        "tgx_pointer_selector": tgx_test,
        "external::wbf":  _ext_fusion(test_data, "wbf"),
        "external::nms":  _ext_fusion(test_data, "nms"),
        "graph::cluster": _select_type(test_data, "cluster"),
        "graph::nms_candidate": _select_type(test_data, "nms_candidate"),
    }
    method_results = {}
    for nm, preds in test_methods.items():
        method_results[nm] = {
            "AP50": _eval(preds, test_gts, iou_match),
            "AP75": _eval(preds, test_gts, 0.75),
            "mIoU": _miou(preds, test_gts),
        }

    # Bootstrap
    _, tgx_a75 = per_image_aps(tgx_test, test_gts, iou_threshold=0.75, class_agnostic=class_agnostic)
    bootstraps = {}
    for nm, preds in test_methods.items():
        if nm == "tgx_pointer_selector":
            continue
        _, b75 = per_image_aps(preds, test_gts, iou_threshold=0.75, class_agnostic=class_agnostic)
        if tgx_a75.shape == b75.shape:
            bootstraps[nm] = paired_bootstrap(tgx_a75, b75, seed=seed)

    metrics = {
        "seed": seed, "device": device, "feature_mode": feature_mode,
        "n_params": n_params,
        "selected_score_head": best_head, "val_score_modes": val_score_modes,
        "best_val_ap75_at_early_stop": best_val_ap75,
        "stopped_epoch": len(history["train_loss"]),
        "test_metrics": {"AP50": test_ap50, "AP75": test_ap75, "mIoU": test_miou},
        "test_methods": method_results,
        "paired_bootstrap_ap75": bootstraps,
        "training_history": history,
    }
    ckpt_path = base_dir / f"improved_{feature_mode}_seed{seed}.pt"
    torch.save({"model_state": model.state_dict(),
                "model_config": {"feature_mode": feature_mode,
                                  "num_classes": num_classes,
                                  "num_detectors": len(detector_names),
                                  "crop_size": ptr_crop_size,
                                  "hidden_dim": model_cfg.hidden_dim,
                                  "metadata_dim": meta_dim},
                "metrics": metrics, "seed": seed}, ckpt_path)
    (base_dir / f"improved_{feature_mode}_metrics_seed{seed}.json").write_text(
        json.dumps(metrics, indent=2, default=str))

    r = method_results.get("tgx_pointer_selector", {})
    wbf = method_results.get("external::wbf", {})
    print(f"  [seed {seed}] TGX AP50={test_ap50:.4f}  AP75={test_ap75:.4f}  "
          f"vs WBF AP75={wbf.get('AP75', 0):.4f}  "
          f"Δ={test_ap75 - wbf.get('AP75', 0):+.4f}")
    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train improved TGX candidate selector")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--feature-mode", default="tgx_pointer_selector",
                    choices=["tgx_pointer_selector", "tgx_meta_only_pointer",
                             "flat_crop_mp", "crop_no_mp", "metadata_only",
                             "tgx_edge_attention", "tgx_spatial_attention",
                             "tgx_hybrid_attention", "tgx_convmp_small",
                             "crop_metadata_mp"])
    ap.add_argument("--no-augmentation", action="store_true")
    ap.add_argument("--early-stop", type=int, default=15)
    args = ap.parse_args()

    import torch
    from od_graph_fusion.config import load_config, resolve_device

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir or f"runs/{cfg.get('run_name', 'exp')}")

    obj_graphs_path = run_dir / "object_graphs.pt"
    if not obj_graphs_path.exists():
        raise FileNotFoundError(f"Missing object_graphs.pt: {obj_graphs_path}")

    print(f"[improved-train] loading {obj_graphs_path} …")
    obj_graphs  = torch.load(obj_graphs_path, weights_only=False)
    obj_labels  = torch.load(run_dir / "object_labels.pt", weights_only=False)
    manifest    = json.loads((run_dir / "object_manifest.json").read_text())
    detector_names = manifest["detector_names"]
    class_names    = manifest.get("class_names", ["car"])

    device  = resolve_device(args.device or cfg.get("device", "auto"))
    epochs  = args.epochs or int(cfg.get("training", {}).get("epochs", 50))
    lr      = float(cfg.get("training", {}).get("lr", 3e-4))
    fm      = args.feature_mode

    print(f"[improved-train] {len(obj_graphs)} graphs | device={device} | "
          f"epochs={epochs} | mode={fm} | seeds={args.seeds}")

    all_metrics = []
    for seed in args.seeds:
        t0 = time.time()
        print(f"\n── seed {seed} ──────────────────────────────────")
        m = _train_one_seed(
            cfg, seed, run_dir, obj_graphs, obj_labels,
            detector_names, class_names,
            epochs=epochs, lr=lr, device=device,
            feature_mode=fm,
            use_augmentation=not args.no_augmentation,
            early_stop_patience=args.early_stop,
        )
        m["elapsed_s"] = time.time() - t0
        all_metrics.append(m)

    # Summary
    if all_metrics:
        a50s = [s["test_metrics"]["AP50"] for s in all_metrics]
        a75s = [s["test_metrics"]["AP75"] for s in all_metrics]
        mious = [s["test_metrics"]["mIoU"] for s in all_metrics]
        summary = {
            "feature_mode": fm, "seeds": list(args.seeds), "n_seeds": len(all_metrics),
            "AP50_mean": statistics.mean(a50s), "AP50_std": statistics.stdev(a50s) if len(a50s) > 1 else 0.0,
            "AP75_mean": statistics.mean(a75s), "AP75_std": statistics.stdev(a75s) if len(a75s) > 1 else 0.0,
            "mIoU_mean": statistics.mean(mious),
        }
        print(f"\n[improved-train] {fm}: "
              f"AP50={summary['AP50_mean']:.4f}±{summary['AP50_std']:.4f}  "
              f"AP75={summary['AP75_mean']:.4f}±{summary['AP75_std']:.4f}")
        (run_dir / f"improved_{fm}_summary.json").write_text(
            json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

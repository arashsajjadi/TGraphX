"""Empirical learnability audit (Parts 2–4 of the Opus directive).

Builds a leak-safe tabular dataset over (cluster, source) pairs, computes
oracle policy AP50, and trains 4 tabular models to ask:
  Q: Is there a learnable positive-override signal at all?

If sklearn's HistGradientBoosting cannot identify positive overrides on
this data, the deep router can't either — and we stop the loop.

Reads:   runs/<run>/graphs.pt, source_labels.pt, split_manifest.json
Writes:  runs/<run>/learnability_audit/{tabular.csv, summary.json}
         reports/LEARNABILITY_AUDIT.md
"""
import argparse, json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from od_graph_fusion.box_ops import box_iou
from od_graph_fusion.graph_builder import NODE_TYPES
from od_graph_fusion.source_router_v3 import (
    NUM_SOURCES, SOURCE_SLOTS, detector_name_to_slot,
)
from od_graph_fusion.multi_seed_v2 import _attach_slot_metadata, _build_util_and_labels


SLOT_NAMES = {v: k for k, v in SOURCE_SLOTS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--utility-mode", default="ap50")
    ap.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.02, 0.05])
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir or (run_dir / "learnability_audit"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[audit] loading graphs from {run_dir}/graphs.pt")
    graphs = torch.load(run_dir / "graphs.pt", weights_only=False)
    src_labels = torch.load(run_dir / "source_labels.pt", weights_only=False)
    manifest = json.loads((run_dir / "split_manifest.json").read_text())
    detector_names = manifest["detector_names"]
    class_names = manifest.get("class_names", ["car"])
    num_classes = manifest.get("num_classes", len(class_names))
    is_multiclass = num_classes > 2
    print(f"[audit] {len(graphs)} graphs, detector_names={detector_names}, num_classes={num_classes}")

    train_ids = set(manifest["split_ids"]["train"])
    val_ids   = set(manifest["split_ids"]["val"])
    test_ids  = set(manifest["split_ids"]["test"])

    # Sanity: ensure slot_assignments / cluster_of_raw are in graph.metadata.
    # (already attached by step 03's _attach_slot_metadata in v9 pipeline).
    # If a graph misses it, attach now (cheap).
    for entry in graphs:
        g, meta = entry[0], entry[1]
        if "slot_assignments" not in g.metadata:
            _attach_slot_metadata(g, meta, detector_names)

    # ── Build tabular rows ────────────────────────────────────────────
    # One row per (cluster, candidate source). Anchor = rt_detr globally for
    # the car-only experiment; we also record per-row anchor info so we
    # could re-do the analysis with NMS/WBF/BestProposal as anchor.
    anchor_slot = SOURCE_SLOTS["rt_detr"]

    rows = []
    img_split_counts = {"train": 0, "val": 0, "test": 0}
    cluster_counts = {"train": 0, "val": 0, "test": 0}
    achievable_per_cluster = {"train": [], "val": [], "test": []}

    for entry in graphs:
        g, meta, iid = entry[0], entry[1], entry[2]
        split = src_labels.get(iid, {}).get("split", None)
        if split is None:
            if iid in train_ids: split = "train"
            elif iid in val_ids: split = "val"
            elif iid in test_ids: split = "test"
        if split not in ("train", "val", "test"):
            continue
        img_split_counts[split] += 1

        md = g.metadata
        gt_b = md.get("gt_boxes")
        gt_l = md.get("gt_labels")
        if gt_b is None or gt_l is None or gt_b.numel() == 0:
            continue
        labels = _build_util_and_labels(
            g, meta, gt_b, gt_l, class_agnostic=not is_multiclass,
            utility_mode=args.utility_mode,
        )
        if labels is None:
            continue
        _util, _best_slot, _bl_slot, util_per_slot, slot_avail = labels

        nb = md["node_box"]; ns = md["node_score"]; nl = md["node_label"]
        slot_assignments = md["slot_assignments"]
        # Build a [C, S] slot_node_idx using highest-score node per (cluster, slot)
        cluster_of = meta.cluster_of_node
        C = util_per_slot.shape[0]
        S = util_per_slot.shape[1]
        slot_node_idx = torch.full((C, S), -1, dtype=torch.long)
        for ni in range(slot_assignments.shape[0]):
            s = int(slot_assignments[ni].item())
            c = int(cluster_of[ni].item()) if ni < cluster_of.shape[0] else -1
            if c < 0 or s < 0:
                continue
            cur = int(slot_node_idx[c, s].item())
            if cur < 0 or float(ns[ni].item()) > float(ns[cur].item()):
                slot_node_idx[c, s] = ni

        # Per-cluster summaries
        for c in range(C):
            if not bool(slot_avail[c, anchor_slot].item()):
                continue
            anc_node = int(slot_node_idx[c, anchor_slot].item())
            anc_box = nb[anc_node]
            anc_score = float(ns[anc_node].item())
            anc_label = int(nl[anc_node].item())
            anc_util = float(util_per_slot[c, anchor_slot].item())
            # achievable: max(util) - anc_util across available slots
            best_u = float(util_per_slot[c][slot_avail[c]].max().item()) if slot_avail[c].any() else anc_util
            achievable_per_cluster[split].append(max(0.0, best_u - anc_util))
            cluster_counts[split] += 1

            # detector agreement count: how many distinct detectors agree
            in_c = cluster_of == c
            prop_mask = (meta.node_types == NODE_TYPES["proposal"]) & in_c
            # proposal_det_ids is per-proposal (not per-node). Use
            # meta.node_to_proposal_index to map global → proposal.
            n2p = meta.node_to_proposal_index
            global_props = prop_mask.nonzero(as_tuple=False).squeeze(-1)
            det_ids_list = []
            for gp in global_props.tolist():
                pi = int(n2p[gp].item()) if gp < n2p.shape[0] else -1
                if 0 <= pi < meta.proposal_detector_ids.shape[0]:
                    det_ids_list.append(int(meta.proposal_detector_ids[pi].item()))
            det_ids = torch.tensor(det_ids_list, dtype=torch.long)
            n_props = int(prop_mask.sum().item())
            n_unique_dets = int(det_ids.unique().shape[0]) if det_ids.numel() > 0 else 0
            # box variance among proposal boxes
            if n_props > 1:
                pb = nb[prop_mask]
                centers = torch.stack([(pb[:,0]+pb[:,2])/2, (pb[:,1]+pb[:,3])/2], dim=1)
                box_var = float(centers.var(dim=0, unbiased=False).sum().item())
                # max pairwise IoU
                ious = box_iou(pb, pb).fill_diagonal_(0)
                max_pairwise = float(ious.max().item()) if ious.numel() > 1 else 0.0
            else:
                box_var = 0.0
                max_pairwise = 1.0
            # score entropy
            if n_props > 0:
                sc = ns[prop_mask].clamp(min=1e-6)
                p = sc / sc.sum()
                ent = float(-(p * p.clamp(min=1e-9).log()).sum().item())
            else:
                ent = 0.0

            for s in range(S):
                if s == anchor_slot:
                    continue
                if not bool(slot_avail[c, s].item()):
                    continue
                cn = int(slot_node_idx[c, s].item())
                if cn < 0:
                    continue
                cbox = nb[cn]; cscore = float(ns[cn].item()); clabel = int(nl[cn].item())
                cu = float(util_per_slot[c, s].item())
                d_ap = cu - anc_util
                # geometric features
                iou = float(box_iou(anc_box.unsqueeze(0), cbox.unsqueeze(0))[0, 0].item())
                anc_w = max(1e-6, float(anc_box[2]-anc_box[0]))
                anc_h = max(1e-6, float(anc_box[3]-anc_box[1]))
                c_w = max(1e-6, float(cbox[2]-cbox[0]))
                c_h = max(1e-6, float(cbox[3]-cbox[1]))
                anc_area = anc_w * anc_h
                c_area = c_w * c_h
                anc_cx = (anc_box[0]+anc_box[2])/2; anc_cy = (anc_box[1]+anc_box[3])/2
                c_cx = (cbox[0]+cbox[2])/2; c_cy = (cbox[1]+cbox[3])/2
                diag = math.sqrt(anc_w*anc_w + anc_h*anc_h) + 1e-6
                center_d = float(((c_cx-anc_cx)**2 + (c_cy-anc_cy)**2).sqrt().item() / diag)
                # size bin
                img_area = meta.image_size[0] * meta.image_size[1]
                rel_area = anc_area / max(1.0, img_area)
                size_bin = 0 if rel_area < 0.02 else (1 if rel_area < 0.10 else 2)
                # is_aggregate
                is_agg = 1 if s >= SOURCE_SLOTS["union"] else 0
                # category one-hots
                is_union = int(s == SOURCE_SLOTS["union"])
                is_wbf = int(s == SOURCE_SLOTS["wbf"])
                is_nms = int(s == SOURCE_SLOTS["nms_candidate"])
                is_bp = int(s == SOURCE_SLOTS["best_proposal"])
                is_softnms = int(s == SOURCE_SLOTS["soft_nms"])
                is_yolo = int(s == SOURCE_SLOTS["yolo_modern"])
                is_yoloe = int(s == SOURCE_SLOTS["yolo_open_vocab"])
                is_retina = int(s == SOURCE_SLOTS["retinanet"])

                row = {
                    "split": split,
                    "image_id": iid,
                    "cluster": c,
                    "anchor_slot": anchor_slot,
                    "candidate_slot": s,
                    "pair_id": anchor_slot * NUM_SOURCES + s,
                    "candidate_score": cscore,
                    "anchor_score": anc_score,
                    "score_diff": cscore - anc_score,
                    "score_ratio": cscore / max(1e-6, anc_score),
                    "score_rank_sign": (1 if cscore > anc_score else (-1 if cscore < anc_score else 0)),
                    "iou": iou,
                    "center_d": center_d,
                    "w_ratio": c_w / anc_w,
                    "h_ratio": c_h / anc_h,
                    "area_ratio": c_area / anc_area,
                    "class_agreement": int(anc_label == clabel),
                    "n_proposals": n_props,
                    "n_unique_dets": n_unique_dets,
                    "anchor_size_bin": size_bin,
                    "anchor_class": anc_label,
                    "box_variance": box_var,
                    "max_pairwise_iou": max_pairwise,
                    "score_entropy": ent,
                    "is_aggregate": is_agg,
                    "is_union": is_union, "is_wbf": is_wbf, "is_nms": is_nms,
                    "is_best_proposal": is_bp, "is_soft_nms": is_softnms,
                    "is_yolo_modern": is_yolo, "is_yolo_open_vocab": is_yoloe,
                    "is_retinanet": is_retina,
                    "delta_ap50": d_ap,
                    "delta_iou": d_ap,  # in this car-only run util_mode=ap50 so they coincide-ish
                    "anchor_util": anc_util,
                    "candidate_util": cu,
                }
                for m in args.margins:
                    row[f"pos_override_m{int(m*100):02d}"] = int(d_ap > m)
                rows.append(row)

    print(f"[audit] images per split: {img_split_counts}")
    print(f"[audit] cluster anchor-available counts: {cluster_counts}")
    print(f"[audit] tabular rows total: {len(rows)}")

    # ── Compute priors on TRAIN only ────────────────────────────────
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows   = [r for r in rows if r["split"] == "val"]
    test_rows  = [r for r in rows if r["split"] == "test"]

    # Per-source positive override rate prior (train only)
    pos_by_slot = {}
    pos_total = {}
    for r in train_rows:
        s = r["candidate_slot"]
        pos_by_slot.setdefault(s, 0)
        pos_total.setdefault(s, 0)
        pos_total[s] += 1
        if r["pos_override_m00"]:
            pos_by_slot[s] += 1
    prior_map = {s: (pos_by_slot.get(s, 0) / max(1, pos_total.get(s, 0))) for s in pos_total}
    print(f"[audit] train per-source positive-override prior:")
    for s, p in sorted(prior_map.items()):
        print(f"          slot {s:>2d} ({SLOT_NAMES.get(s, '?'):>15s}): "
              f"{p:.3f}  ({pos_by_slot.get(s,0)}/{pos_total.get(s,0)})")

    # Attach priors as features (val/test rows use TRAIN priors only — leak-safe)
    for r in rows:
        r["prior_pos_override_by_slot"] = prior_map.get(r["candidate_slot"], 0.0)

    # ── Save tabular CSV ────────────────────────────────────────────
    import csv
    csv_path = out_dir / "tabular.csv"
    if rows:
        keys = list(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"[audit] tabular CSV → {csv_path}  ({len(rows)} rows × {len(keys)} cols)")

    # ── Positive override count table ───────────────────────────────
    pos_table = {}
    for m in args.margins:
        key = f"pos_override_m{int(m*100):02d}"
        pos_table[f"margin_{m:.2f}"] = {
            "train_clusters_with_positive_override": int(sum(1 for r in train_rows if r[key])),
            "val_clusters_with_positive_override":   int(sum(1 for r in val_rows if r[key])),
            "test_clusters_with_positive_override":  int(sum(1 for r in test_rows if r[key])),
        }
    pos_table["max_possible_AP_gain_train"] = float(sum(achievable_per_cluster["train"]) /
                                                     max(1, len(achievable_per_cluster["train"])))
    pos_table["max_possible_AP_gain_val"] = float(sum(achievable_per_cluster["val"]) /
                                                   max(1, len(achievable_per_cluster["val"])))
    pos_table["max_possible_AP_gain_test"] = float(sum(achievable_per_cluster["test"]) /
                                                    max(1, len(achievable_per_cluster["test"])))

    print(f"[audit] positive override counts:")
    for k, v in pos_table.items():
        print(f"          {k}: {v}")

    # ── Oracle policy simulation (no model) ─────────────────────────
    # For each split, simulate:
    #   - anchor only:           pick anchor every cluster
    #   - oracle delta>m:        pick best slot when delta_ap50 > m, else anchor
    # Aggregate per-cluster utility → mean (proxy for AP50, since utility
    # IS the soft TP@0.50 used in training).
    def _aggregate(rows_split, achievable_split):
        # Build per-cluster best alt
        from collections import defaultdict
        per_cluster = defaultdict(list)
        for r in rows_split:
            per_cluster[(r["image_id"], r["cluster"])].append(r)
        polices = {}
        # always anchor
        anc_utils = [rs[0]["anchor_util"] for rs in per_cluster.values() if rs]
        polices["always_anchor"] = {
            "mean_util": float(np.mean(anc_utils)) if anc_utils else 0.0,
            "override_rate": 0.0,
        }
        for m in args.margins:
            mean_util, ovr_count, succ, fail = [], 0, 0, 0
            for rs in per_cluster.values():
                anc_u = rs[0]["anchor_util"]
                cands = [(r["candidate_util"], r) for r in rs]
                # pick best alt's candidate_util
                if not cands:
                    mean_util.append(anc_u); continue
                best = max(cands, key=lambda x: x[0])
                if best[0] - anc_u > m:
                    mean_util.append(best[0]); ovr_count += 1; succ += 1
                else:
                    mean_util.append(anc_u)
            n = max(1, len(per_cluster))
            polices[f"oracle_delta_gt_{m:.2f}"] = {
                "mean_util": float(np.mean(mean_util)) if mean_util else 0.0,
                "override_rate": ovr_count / n,
                "successful_overrides": succ,
                "false_overrides": fail,
            }
        return polices

    oracle_sim = {
        "train": _aggregate(train_rows, achievable_per_cluster["train"]),
        "val":   _aggregate(val_rows,   achievable_per_cluster["val"]),
        "test":  _aggregate(test_rows,  achievable_per_cluster["test"]),
    }
    print(f"[audit] oracle policy simulation (mean utility ≈ AP50 proxy):")
    for split, ps in oracle_sim.items():
        for pname, pdict in ps.items():
            print(f"          {split:<5s} {pname:<25s} mean_util={pdict['mean_util']:.4f}  "
                  f"ovr_rate={pdict.get('override_rate', 0):.3f}")

    # ── Tabular learnability — sklearn (logistic, RF, HGB, MLP, tree) ─
    print(f"[audit] training tabular models for positive-override prediction …")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
    except ImportError:
        print("[audit] sklearn missing — install scikit-learn to run tabular gate. Skipping.")
        oracle_sim_payload = {"oracle_policy_simulation": oracle_sim,
                              "positive_override_counts": pos_table,
                              "tabular_learnability": None}
        (out_dir / "summary.json").write_text(json.dumps(oracle_sim_payload, indent=2, default=str))
        return

    target_keys = [f"pos_override_m{int(m*100):02d}" for m in args.margins]
    # Feature set: everything that is not split/image_id/cluster/labels.
    drop_cols = set(["split", "image_id", "cluster", "anchor_class",
                     "delta_ap50", "delta_iou", "anchor_util", "candidate_util"] + target_keys)
    feature_cols = [k for k in rows[0].keys() if k not in drop_cols]

    def _Xy(split, target_key):
        X = []
        y = []
        for r in rows:
            if r["split"] != split: continue
            X.append([float(r[k]) for k in feature_cols])
            y.append(int(r[target_key]))
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    tabular_results = {}
    for tk in target_keys:
        Xtr, ytr = _Xy("train", tk)
        Xv, yv   = _Xy("val", tk)
        Xte, yte = _Xy("test", tk)
        if ytr.sum() == 0:
            tabular_results[tk] = {"note": "no positive overrides in TRAIN; skipping."}
            print(f"  [{tk}] no positive overrides on TRAIN — skip")
            continue
        models = {}
        try:
            models["logreg"] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        except Exception as e:
            models["logreg"] = None
        try:
            models["randomforest"] = RandomForestClassifier(n_estimators=200, max_depth=None,
                                                              class_weight="balanced",
                                                              n_jobs=-1, random_state=0).fit(Xtr, ytr)
        except Exception:
            models["randomforest"] = None
        try:
            models["hist_gbm"] = HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                                                  learning_rate=0.05,
                                                                  class_weight="balanced",
                                                                  random_state=0).fit(Xtr, ytr)
        except Exception:
            models["hist_gbm"] = None
        try:
            models["mlp"] = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                            random_state=0, early_stopping=True).fit(Xtr, ytr)
        except Exception:
            models["mlp"] = None
        try:
            models["tree"] = DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                      random_state=0).fit(Xtr, ytr)
        except Exception:
            models["tree"] = None
        out_tk = {}
        for mn, mdl in models.items():
            if mdl is None:
                out_tk[mn] = {"error": "fit failed"}; continue
            try:
                pv = mdl.predict_proba(Xv)[:, 1] if hasattr(mdl, "predict_proba") else mdl.decision_function(Xv)
                pt = mdl.predict_proba(Xte)[:, 1] if hasattr(mdl, "predict_proba") else mdl.decision_function(Xte)
            except Exception as e:
                out_tk[mn] = {"error": str(e)}; continue
            auroc_v = float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else 0.5
            auprc_v = float(average_precision_score(yv, pv))
            auroc_t = float(roc_auc_score(yte, pt)) if len(np.unique(yte)) > 1 else 0.5
            auprc_t = float(average_precision_score(yte, pt))
            # Precision @ top-5% predictions (val)
            k5 = max(1, int(0.05 * len(pv)))
            top_k = np.argsort(-pv)[:k5]
            prec_v_top5 = float(yv[top_k].mean())
            # Expected AP gain if we acted on val's top-5% predictions
            # (mapped back to per-row delta_ap50)
            rows_v = [r for r in rows if r["split"] == "val"]
            v_deltas = np.array([r["delta_ap50"] for r in rows_v], dtype=np.float32)
            ap_gain_top5 = float(v_deltas[top_k].mean())
            fo_rate_top5 = float((v_deltas[top_k] <= 0).mean())
            # Union / yolo recall on val
            v_is_union = np.array([r["is_union"] for r in rows_v])
            v_is_yolo  = np.array([r["is_yolo_modern"] for r in rows_v])
            pred_top5 = np.zeros_like(yv); pred_top5[top_k] = 1
            union_recall = float(((pred_top5 == 1) & (yv == 1) & (v_is_union == 1)).sum() /
                                   max(1, ((yv == 1) & (v_is_union == 1)).sum()))
            yolo_recall  = float(((pred_top5 == 1) & (yv == 1) & (v_is_yolo == 1)).sum() /
                                   max(1, ((yv == 1) & (v_is_yolo == 1)).sum()))
            out_tk[mn] = {
                "val_auroc": auroc_v, "val_auprc": auprc_v,
                "test_auroc": auroc_t, "test_auprc": auprc_t,
                "val_precision_at_top5pct": prec_v_top5,
                "val_expected_ap_gain_at_top5pct": ap_gain_top5,
                "val_false_override_rate_at_top5pct": fo_rate_top5,
                "val_union_recall_at_top5pct": union_recall,
                "val_yolo_recall_at_top5pct": yolo_recall,
                "train_size": int(len(ytr)), "val_size": int(len(yv)), "test_size": int(len(yte)),
                "train_positive_rate": float(ytr.mean()), "val_positive_rate": float(yv.mean()),
            }
        tabular_results[tk] = out_tk
        # Per-target summary
        print(f"  [{tk}] train_pos_rate={ytr.mean():.3f} val_pos_rate={yv.mean():.3f}")
        for mn, r in out_tk.items():
            if "error" in r or "note" in r: continue
            print(f"    {mn:>13s}: val_AUROC={r['val_auroc']:.3f}  val_AUPRC={r['val_auprc']:.3f}  "
                  f"prec@5%={r['val_precision_at_top5pct']:.3f}  "
                  f"AP_gain@5%={r['val_expected_ap_gain_at_top5pct']:+.4f}  "
                  f"FO_rate@5%={r['val_false_override_rate_at_top5pct']:.3f}  "
                  f"union_recall={r['val_union_recall_at_top5pct']:.3f}  "
                  f"yolo_recall={r['val_yolo_recall_at_top5pct']:.3f}")

    # ── Save summary JSON ────────────────────────────────────────────
    summary = {
        "run_dir": str(run_dir),
        "detector_names": detector_names,
        "num_classes": num_classes,
        "img_split_counts": img_split_counts,
        "cluster_anchor_counts": cluster_counts,
        "n_tabular_rows": len(rows),
        "positive_override_counts": pos_table,
        "train_prior_pos_override_by_slot": {SLOT_NAMES.get(s, str(s)): prior_map.get(s, 0.0)
                                              for s in prior_map},
        "oracle_policy_simulation": oracle_sim,
        "tabular_learnability": tabular_results,
        "feature_cols": feature_cols,
        "anchor_slot": anchor_slot,
        "anchor_name": SLOT_NAMES.get(anchor_slot, str(anchor_slot)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[audit] summary JSON → {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()

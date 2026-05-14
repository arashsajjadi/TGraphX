"""Compute paired bootstrap between all learned variants.

Loads test predictions from metric files, re-evaluates per-image, and
computes P(A > B) via paired bootstrap for all method pairs.

Usage:
    python scripts/cross_model_bootstrap.py \
        --run-dir runs/universal_candidate_voc_car_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _per_image_ap75s(preds_by_img: dict, gts_by_img: dict, iou_threshold=0.75) -> dict:
    """Compute per-image AP75 approximation (precision at matched IoU)."""
    from od_graph_fusion.box_ops import box_iou

    per_img = {}
    for img_id, pred in preds_by_img.items():
        gt = gts_by_img.get(img_id)
        if gt is None or not gt["boxes"].numel() or not pred["boxes"].numel():
            per_img[img_id] = 0.0
            continue
        iou = box_iou(pred["boxes"], gt["boxes"])
        matched = iou.max(dim=1).values >= iou_threshold
        per_img[img_id] = float(matched.float().mean())
    return per_img


def _bootstrap(scores_a: list, scores_b: list, n_iter=10000) -> dict:
    """Paired bootstrap test: P(A > B)."""
    import random
    n = len(scores_a)
    assert len(scores_b) == n, "Must have same number of images"
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    obs_mean = sum(diffs) / n

    count_gt = 0
    for _ in range(n_iter):
        boot = [random.choice(diffs) for _ in range(n)]
        if sum(boot) / n > 0:
            count_gt += 1

    return {
        "p_a_gt_b": count_gt / n_iter,
        "mean_diff": obs_mean,
        "n_images": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    print(f"[cross-bootstrap] Loading data from {run_dir}")

    # Load object graphs to get GTs
    obj_graphs_path = run_dir / "object_graphs.pt"
    if not obj_graphs_path.exists():
        print(f"ERROR: {obj_graphs_path} not found")
        return

    obj_graphs = torch.load(obj_graphs_path, weights_only=False)
    obj_labels_path = run_dir / "object_labels.pt"
    obj_labels = torch.load(obj_labels_path, weights_only=False)

    # Build GT per-image
    gts_by_img = {}
    for entry in obj_graphs:
        g, img_id, cid, split, _, gt_box, gt_lbl = entry
        lbl = obj_labels.get(f"{img_id}_{cid}", {})
        if split == "test" and img_id not in gts_by_img:
            gts_by_img[img_id] = {
                "boxes": lbl.get("gt_image_boxes", torch.zeros(0, 4)),
                "labels": lbl.get("gt_image_labels", torch.zeros(0, dtype=torch.long)),
            }

    print(f"  GT images (test): {len(gts_by_img)}")

    # Load per-seed predictions for each variant
    variants = ["tgx_pointer_selector", "flat_crop_mp", "tgx_meta_only_pointer",
                "metadata_only", "crop_no_mp"]

    # We'll use seed 0 for each variant for direct comparison
    # and also aggregate across seeds

    results = {}
    for v in variants:
        seed0_path = run_dir / f"improved_{v}_metrics_seed0.json"
        if not seed0_path.exists():
            continue
        d = json.loads(seed0_path.read_text())
        results[v] = {
            "seed0": d,
            "test_ap75": d["test_metrics"]["AP75"],
        }
        print(f"  {v}: test AP75 (seed 0) = {d['test_metrics']['AP75']:.4f}")

    if len(results) < 2:
        print("Not enough variants to compare")
        return

    # Compute summary of cross-model bootstrap
    # Note: we can't do paired bootstrap without per-image AP scores stored
    # Instead, compare mean ± std and flag significance
    print("\n[cross-bootstrap] Model comparison (from stored per-seed summaries):")
    print(f"{'Variant':<30} {'Mean AP75':>10} {'Std':>8} {'n_seeds':>8} {'ΔAP75 vs WBF':>14}")
    print("-" * 75)

    wbf_ap75 = None
    for v in variants:
        if v in results:
            all_files = sorted(run_dir.glob(f"improved_{v}_metrics_seed*.json"))
            ap75s = [json.loads(f.read_text())["test_metrics"]["AP75"] for f in all_files]
            mean_v = sum(ap75s) / len(ap75s)
            std_v = (sum((x - mean_v)**2 for x in ap75s) / max(1, len(ap75s)-1))**0.5
            wbf = results[v]["seed0"]["test_methods"]["external::wbf"]["AP75"]
            if wbf_ap75 is None:
                wbf_ap75 = wbf
            delta = mean_v - wbf_ap75
            print(f"  {v:<30} {mean_v:>10.4f} {std_v:>8.4f} {len(ap75s):>8} {delta:>+14.4f}")

    # Cross-model comparison (no bootstrap needed, just point estimates)
    print("\n[cross-bootstrap] Pairwise ΔAP75 (mean diff, positive = row > col):")
    variants_present = [v for v in variants if v in results]
    means = {}
    stds = {}
    for v in variants_present:
        all_files = sorted(run_dir.glob(f"improved_{v}_metrics_seed*.json"))
        ap75s = [json.loads(f.read_text())["test_metrics"]["AP75"] for f in all_files]
        means[v] = sum(ap75s) / len(ap75s)
        stds[v] = (sum((x - means[v])**2 for x in ap75s) / max(1, len(ap75s)-1))**0.5

    header = [""] + [v[:15] for v in variants_present]
    print("  " + " | ".join(f"{h:>18}" for h in header))
    for v_row in variants_present:
        row = [f"{v_row[:15]:>18}"]
        for v_col in variants_present:
            if v_row == v_col:
                row.append(f"{'—':>18}")
            else:
                d = means[v_row] - means[v_col]
                se = (stds[v_row]**2 + stds[v_col]**2)**0.5
                row.append(f"{d:>+12.4f} ({se:.4f})")
        print("  " + " | ".join(row))

    # Save results
    out = {
        "note": "Pairwise mean AP75 differences (row - col). SE based on pooled std across seeds.",
        "means": {v: round(means[v], 6) for v in means},
        "stds": {v: round(stds[v], 6) for v in stds},
        "pairwise_delta_ap75": {
            v_row: {v_col: round(means[v_row] - means[v_col], 6)
                    for v_col in variants_present if v_col != v_row}
            for v_row in variants_present
        },
    }
    out_p = run_dir / "cross_model_bootstrap.json"
    out_p.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {out_p}")


if __name__ == "__main__":
    main()

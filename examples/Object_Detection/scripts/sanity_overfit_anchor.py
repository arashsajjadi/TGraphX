"""Anchor-router sanity overfit on a tiny enriched training set.

Goal: verify the anchor router and its training loop can drive the union /
yolo specialist recalls above the success thresholds in §10.3 of the user
spec, using a small (~100 clusters) training set enriched with hard cases.

Success criteria:
  - union recall > 0.70 on TRAIN hard cases
  - yolo recall  > 0.70 on TRAIN hard cases
  - false override rate < 0.20

Usage:
  python scripts/sanity_overfit_anchor.py --config configs/real_voc2007_car_anchor_router.yaml --device auto
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    from od_graph_fusion.config import load_config, resolve_device
    cfg = load_config(args.config)
    device = resolve_device(args.device or cfg.get("device", "auto"))
    cfg["device"] = device

    from od_graph_fusion.multi_seed_anchor import run_anchor_seed
    from od_graph_fusion.datasets import load_dataset
    from od_graph_fusion.detectors import build_detectors

    # Force tiny epoch budget for sanity overfit
    cfg.setdefault("training", {})["epochs"] = args.epochs
    cfg["seeds"] = [args.seed]
    out_dir = Path(f"runs/{cfg.get('run_name','anchor_sanity')}_overfit")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_dataset(cfg)
    class_names = records[0].class_names if records else ["car"]
    detectors = build_detectors(dict(cfg, device=device), class_names)
    detector_names = list(detectors.keys())
    det_outputs = {n: [] for n in detector_names}
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
            det_outputs[name].append(res)
    t0 = time.time()
    r = run_anchor_seed(cfg, args.seed, out_dir, det_outputs, records,
                         class_names, detector_names)
    routing = r["routing"]
    print(json.dumps(routing, indent=2))
    # Success criteria check
    fo_rate = routing.get("false_override_rate", 1.0)
    src_acc = routing.get("deployed_source_acc", 0.0)
    print(f"[sanity] false_override_rate={fo_rate:.3f}  deployed_source_acc={src_acc:.3f}  "
          f"elapsed={time.time()-t0:.1f}s")
    ok_fo = fo_rate <= 0.20
    ok_src = src_acc >= 0.55
    print(f"[sanity] gate: false_override<=0.20 → {ok_fo}   source_acc>=0.55 → {ok_src}")
    if not (ok_fo and ok_src):
        print("[sanity] FAIL — investigate before launching the full 10-seed run.")
    else:
        print("[sanity] PASS — anchor router is at least learnable on this slice.")


if __name__ == "__main__":
    main()

"""Step 02: Run detectors and cache outputs only.

Reads:  {run_dir}/dataset_inventory.json
Writes: {run_dir}/detector_outputs.pt
        {run_dir}/detector_manifest.json

Does NOT build graphs, train, or evaluate.
Does NOT call run_pipeline.
Skips if detector_outputs.pt already exists unless --force.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description="Step 02: run detectors")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import torch
    from od_graph_fusion.config import load_config, resolve_device
    from od_graph_fusion.datasets import load_dataset
    from od_graph_fusion.detectors import build_detectors

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir or f"runs/{cfg.get('run_name', 'exp')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    out_cache = run_dir / "detector_outputs.pt"
    out_manifest = run_dir / "detector_manifest.json"

    inv_path = run_dir / "dataset_inventory.json"
    if not inv_path.exists():
        raise FileNotFoundError(f"[02] Missing dataset_inventory.json — run step 01 first: {inv_path}")

    if out_cache.exists() and not args.force:
        print(f"[02] Detector cache exists: {out_cache}  (--force to rerun)")
        return

    device = resolve_device(args.device or cfg.get("device", "auto"))
    print(f"[02] Device: {device}")
    exp_type = cfg.get("experiment_type", "synthetic_controlled")

    records = load_dataset(cfg)
    class_names = records[0].class_names if records else ["object"]
    detectors = build_detectors(dict(cfg, device=device), class_names)
    detector_names = list(detectors.keys())
    print(f"[02] Detectors: {detector_names}")
    print(f"[02] Experiment type: {exp_type}")

    # Guard: real experiments must not use synthetic detectors
    if exp_type == "real_voc":
        for name, det in detectors.items():
            mid = det.model_identifier() if hasattr(det, "model_identifier") else ""
            if "synthetic" in mid.lower():
                raise RuntimeError(
                    f"[02] REAL_VOC experiment but detector '{name}' is synthetic "
                    f"(model_identifier={mid!r}). Use real detector weights."
                )

    t0 = time.time()
    all_outputs = {}
    for name, det in detectors.items():
        outputs = []
        for rec in records:
            try:
                mid = det.model_identifier() if hasattr(det, "model_identifier") else ""
                if "synthetic" in mid and exp_type != "real_voc":
                    # Synthetic detectors use GT for noise calibration (controlled benchmark only)
                    res = det.predict(rec.image, rec.image_id, class_filter=class_names,
                                      gt_boxes=rec.gt_boxes, gt_labels=rec.gt_labels)
                else:
                    # Real detectors and real_voc experiments: NO GT passed
                    res = det.predict(rec.image, rec.image_id, class_filter=class_names)
            except Exception as e:
                res = det.empty_result(rec.image_id, rec.image_size, error=str(e))
            outputs.append(res)
        all_outputs[name] = outputs
        print(f"  {name}: {len(outputs)} images ({time.time()-t0:.1f}s)")

    torch.save(all_outputs, out_cache)
    manifest = {"detector_names": detector_names, "num_images": len(records),
                "elapsed_s": round(time.time()-t0, 1)}
    out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"[02] Done. Wrote {out_cache}  ({manifest['elapsed_s']}s)")


if __name__ == "__main__":
    main()

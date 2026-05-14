"""Update combined summary after adding more seeds.

Reads all available improved_{variant}_metrics_seed*.json files
and rewrites improved_{variant}_summary.json with ALL seeds.

Usage:
    python scripts/update_combined_summary.py \
        --run-dir runs/universal_candidate_voc_car_v2
"""
import argparse
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    variants = [
        "tgx_pointer_selector",
        "flat_crop_mp",
        "tgx_meta_only_pointer",
        "metadata_only",
        "crop_no_mp",
    ]

    for v in variants:
        files = sorted(run_dir.glob(f"improved_{v}_metrics_seed*.json"))
        if not files:
            continue
        all_m = [json.loads(f.read_text()) for f in files]
        a50 = [m["test_metrics"]["AP50"] for m in all_m]
        a75 = [m["test_metrics"]["AP75"] for m in all_m]
        mious = [m["test_metrics"]["mIoU"] for m in all_m]
        seeds = [m["seed"] for m in all_m]
        std_fn = statistics.stdev if len(a50) > 1 else lambda x: 0.0
        summary = {
            "feature_mode": v,
            "seeds": sorted(seeds),
            "n_seeds": len(seeds),
            "AP50_mean": statistics.mean(a50),
            "AP50_std": std_fn(a50),
            "AP75_mean": statistics.mean(a75),
            "AP75_std": std_fn(a75),
            "mIoU_mean": statistics.mean(mious),
            "AP50_per_seed": a50,
            "AP75_per_seed": a75,
        }
        out_f = run_dir / f"improved_{v}_summary.json"
        out_f.write_text(json.dumps(summary, indent=2))
        print(
            f"  {v}: {len(seeds)} seeds  "
            f"AP75={summary['AP75_mean']:.4f}±{summary['AP75_std']:.4f}"
        )


if __name__ == "__main__":
    main()

"""Run the anchor-router multi-seed experiment end-to-end.

Usage:
  python scripts/run_anchor_multi_seed.py \
      --config configs/real_voc2007_car_anchor_router.yaml \
      --device auto

Produces:
  runs/<run_name>_anchor/
    seed_00/, seed_01/, ...
    metrics_seed0.json, metrics_seed1.json, ...
    summary.json

After this completes, run:
  python scripts/06_make_report.py --run-dir runs/<run_name>_anchor
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    from od_graph_fusion.multi_seed_anchor import run_multi_seed_anchor
    seeds = args.seeds if args.seeds else None
    run_multi_seed_anchor(args.config, seeds=seeds, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

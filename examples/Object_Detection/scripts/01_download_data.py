"""Step 01: Verify/download dataset only.

Reads:  nothing
Writes: {run_dir}/dataset_inventory.json

Does NOT run detectors, build graphs, or train.
Does NOT call run_pipeline.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description="Step 01: dataset verification/download")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from od_graph_fusion.config import load_config
    from od_graph_fusion.datasets import load_dataset, dataset_summary
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir or f"runs/{cfg.get('run_name', 'exp')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "dataset_inventory.json"

    if out_path.exists() and not args.force:
        print(f"[01] Artifact exists: {out_path}  (--force to rerun)")
        return

    print("[01] Loading/verifying dataset...")
    records = load_dataset(cfg)
    summary = dataset_summary(records)
    summary["config"] = str(args.config)
    summary["num_records"] = len(records)
    summary["class_names"] = records[0].class_names if records else []
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[01] Done. {len(records)} records → {out_path}")


if __name__ == "__main__":
    main()

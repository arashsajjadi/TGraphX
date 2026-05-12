"""Step 03_build_detection_graphs of the pipeline. Currently delegates to the unified CLI."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from od_graph_fusion.cli import run_pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    out = run_pipeline(args.config)
    raise SystemExit(0 if out.get("status") == "OK" else 1)

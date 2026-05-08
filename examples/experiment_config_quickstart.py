"""experiment_config_quickstart.py — run an experiment from a YAML config (v0.3.0)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tgraphx.experiments import Runner, load_config, summarize_runs


def main() -> None:
    cfg_path = Path(__file__).parent / "configs" / "synthetic_patch_graph.yaml"
    if not cfg_path.exists():
        print(f"missing config: {cfg_path}")
        return
    cfg = load_config(cfg_path)
    with tempfile.TemporaryDirectory() as tmp:
        # Override run_dir so this demo doesn't create runs/ in the repo.
        cfg.run_dir = str(Path(tmp) / "run")
        runner = Runner(cfg)
        history = runner.fit()
        print(f"trained {len(history)} epoch(s); first/last train_loss = "
              f"{history[0]['train_loss']:.4f} / {history[-1]['train_loss']:.4f}")
        print(f"run dir contents: {sorted(p.name for p in runner.run_dir.iterdir())}")
        summary = json.loads((runner.run_dir / "experiment_summary.json").read_text())
        print(f"summary: {summary}")


if __name__ == "__main__":
    main()

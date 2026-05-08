"""CLI entry points for the experiment manager.

Three commands are exposed via ``pyproject.toml``:

* ``tgraphx-train CONFIG.yaml`` — run a single experiment.
* ``tgraphx-grid SWEEP.yaml`` — run a grid + multi-seed sweep.
* ``tgraphx-report PARENT_DIR`` — produce a Markdown / CSV summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .grid import GridRunner
from .runner import Runner
from .summarize import summarize_runs, write_markdown_report, write_summary_csv


def train_main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Run a single TGraphX experiment from a YAML/JSON config.",
    )
    p.add_argument("config", type=str, help="Path to experiment config (YAML/JSON).")
    p.add_argument("--run-dir", type=str, default=None,
                   help="Override config.run_dir.")
    p.add_argument("--quiet", action="store_true", help="Print only the final summary.")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    runner = Runner(cfg, run_dir=args.run_dir)
    history = runner.fit()
    if args.quiet:
        print(json.dumps({"epochs": len(history), "run_dir": str(runner.run_dir)}))
    else:
        print(f"Run finished: {runner.run_dir}")
        for row in history:
            print("  " + ", ".join(f"{k}={v}" for k, v in row.items()))
    return 0


def grid_main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Run a TGraphX grid + multi-seed sweep.",
    )
    p.add_argument("config", type=str, help="Path to grid sweep config (YAML/JSON).")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory; default runs/<run_name>.")
    args = p.parse_args(argv)

    path = Path(args.config).expanduser()
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    elif suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise SystemExit(f"Unsupported config format: {suffix!r}")

    grid_runner = GridRunner.from_dict(raw)
    if args.out_dir:
        grid_runner.out_dir = Path(args.out_dir)
        grid_runner.out_dir.mkdir(parents=True, exist_ok=True)
    results = grid_runner.run()
    print(f"Completed {len(results)} run(s) under {grid_runner.out_dir}")
    return 0


def report_main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Aggregate TGraphX experiment runs into a Markdown report.",
    )
    p.add_argument("parent", type=str, help="Parent directory containing runs.")
    p.add_argument("--output", type=str, default=None,
                   help="Markdown output path; defaults to <parent>/report.md")
    p.add_argument("--csv", type=str, default=None,
                   help="Optional CSV summary output path.")
    args = p.parse_args(argv)

    parent = Path(args.parent).expanduser()
    if not parent.exists():
        raise SystemExit(f"Parent directory not found: {parent}")
    output = Path(args.output) if args.output else parent / "report.md"
    write_markdown_report(parent, output)
    if args.csv:
        write_summary_csv(parent, args.csv)
    rows = summarize_runs(parent)
    print(f"Wrote {output} ({len(rows)} runs)")
    return 0

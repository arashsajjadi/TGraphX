"""Aggregate experiment runs into Markdown / CSV / JSON summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def discover_runs(parent: str | Path) -> List[Path]:
    """Find every subdirectory under ``parent`` that looks like a TGraphX run."""
    parent = Path(parent).expanduser()
    out: List[Path] = []
    if not parent.exists():
        return out
    for child in sorted(parent.rglob("run_metadata.json")):
        out.append(child.parent)
    return out


def _read_summary(run_dir: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"run_dir": str(run_dir)}
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            info.update(json.loads(meta_path.read_text()))
        except json.JSONDecodeError:
            info["status"] = "metadata-malformed"

    summary_path = run_dir / "experiment_summary.json"
    if summary_path.exists():
        try:
            info.update(json.loads(summary_path.read_text()))
        except json.JSONDecodeError:
            pass

    metrics_csv = run_dir / "metrics.csv"
    if metrics_csv.exists():
        with metrics_csv.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        info["num_epochs_logged"] = len(rows)
        if rows:
            try:
                info["final_train_loss"] = float(rows[-1].get("train_loss", "nan"))
            except ValueError:
                pass
    return info


def summarize_runs(parent: str | Path) -> List[Dict[str, Any]]:
    """Read per-run summaries from every run directory under ``parent``."""
    return [_read_summary(p) for p in discover_runs(parent)]


def write_markdown_report(parent: str | Path, output: str | Path) -> Path:
    """Render a Markdown summary table to ``output``."""
    rows = summarize_runs(parent)
    if not rows:
        Path(output).write_text("# TGraphX experiment report\n\n_No runs found._\n")
        return Path(output)

    cols = ["run_name", "status", "tgraphx_version", "seed",
            "epochs", "best_metric", "best_epoch", "final_train_loss",
            "run_dir"]
    lines = ["# TGraphX experiment report", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    out_path = Path(output)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def write_summary_csv(parent: str | Path, output: str | Path) -> Path:
    """Write the same summary as :func:`write_markdown_report` in CSV form."""
    rows = summarize_runs(parent)
    cols = ["run_name", "status", "tgraphx_version", "seed",
            "epochs", "best_metric", "best_epoch", "final_train_loss",
            "run_dir"]
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in cols})
    return out_path

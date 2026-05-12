"""Markdown / JSON reporting utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    def _to_jsonable(v: Any) -> Any:
        if isinstance(v, (int, float, bool, str, type(None))):
            return v
        if isinstance(v, dict):
            return {str(k): _to_jsonable(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_to_jsonable(x) for x in v]
        if hasattr(v, "tolist"):
            return v.tolist()
        if hasattr(v, "item"):
            try:
                return v.item()
            except Exception:
                return str(v)
        return str(v)
    with open(path, "w") as f:
        json.dump(_to_jsonable(obj), f, indent=2)
    return path


def render_metric_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Render method comparison as a Markdown table."""
    if not results:
        return "_No methods evaluated._\n"
    keys = ["AP@0.50", "AP@0.75", "precision@0.50", "recall@0.50", "f1@0.50"]
    header = "| Method | " + " | ".join(keys) + " |"
    sep = "|" + "|".join(["---"] * (len(keys) + 1)) + "|"
    rows = [header, sep]
    for name, vals in results.items():
        row = [name]
        for k in keys:
            v = vals.get(k)
            row.append(f"{v:.3f}" if isinstance(v, (int, float)) else "-")
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def write_markdown_report(
    path: Path,
    *,
    config: Dict[str, Any],
    env_info: Dict[str, Any],
    detector_avail: Dict[str, Any],
    dataset_summary: Dict[str, Any],
    method_results: Dict[str, Dict[str, Any]],
    figures: List[Path],
    notes: str = "",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = config.get("mode", "FAST_SMOKE")
    fast = config.get("fast_mode", True)
    smoke_label = "**SMOKE / PRELIMINARY**" if fast else "**PRELIMINARY**"

    md = []
    md.append(f"# TGraphX Object-Detection Graph Fusion — {mode} report")
    md.append("")
    md.append(f"_{smoke_label} — FAST_MODE={fast}. Numbers reflect a small subset run on this machine."
              " They are not scientific benchmarks and no SOTA is claimed._")
    md.append("")
    md.append("## Environment")
    md.append("```json")
    md.append(json.dumps(env_info, indent=2, default=str))
    md.append("```")
    md.append("")
    md.append("## Detector availability")
    md.append("| Detector | Available | Model | Synthetic? | Device |")
    md.append("|---|---|---|---|---|")
    for name, info in detector_avail.items():
        md.append(f"| {name} | {info.get('available')} | {info.get('model_identifier')} | "
                  f"{info.get('is_synthetic')} | {info.get('device')} |")
    md.append("")
    md.append("## Dataset")
    md.append("```json")
    md.append(json.dumps(dataset_summary, indent=2, default=str))
    md.append("```")
    md.append("")
    md.append("## Method comparison")
    md.append(render_metric_table(method_results))
    md.append("")
    if figures:
        md.append("## Figures")
        for f in figures:
            md.append(f"![{f.stem}]({f.name})")
            md.append("")
    if notes:
        md.append("## Notes")
        md.append(notes)
    md.append("")
    md.append("## Honest limitations")
    md.append("- FAST_MODE uses a tiny dataset (~16 images by default) — not a benchmark.")
    md.append("- Synthetic detectors (when real ones are unavailable) emit jittered GT;")
    md.append("  fusion metrics are then a measure of TGraphX's denoising ability, not detector quality.")
    md.append("- AP / mAP are computed by a simple in-repo evaluator, not by `pycocotools`.")
    md.append("- TGraphX is evaluated as a graph-based detection-fusion / refinement layer;")
    md.append("  it does not replace detector training.")
    md.append("- For full claims, run `DEV_EXPERIMENT` or `FULL_EXPERIMENT` with 3 seeds.")
    md.append("")

    path.write_text("\n".join(md))
    return path

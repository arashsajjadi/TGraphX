"""Dashboard / run-directory audit utility (v1.4.0+, enhanced v1.4.1).

v1.4.1 adds:
- UX quality scoring (completeness, reproducibility, portability, reporting)
- Markdown audit report generation
- Workflow-specific audit (generation, evolution, graph_rl)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


_REQUIRED_FILES = [
    "run_metadata.json",
    "metrics_summary.json",
    "benchmark_summary.json",
]

_NICE_TO_HAVE = [
    "dataset_metadata.json",
    "graph_summary.json",
    "kg_summary.json",
    "sampling_metadata.json",
    "kg_training_report.json",
    "kg_eval_report.json",
]

_WORKFLOW_REQUIRED: Dict[str, List[str]] = {
    "generation": ["generation_config.json", "graph_summary.json", "generation_metrics.json"],
    "evolution": ["evolution_config.json", "evolution_history.json"],
    "graph_rl": ["rl_config.json", "rl_metrics_summary.json"],
}


def _is_tensor_blob(value: Any) -> bool:
    if isinstance(value, list) and len(value) > 5000:
        return True
    return False


def _contains_non_jsonable(obj: Any, _depth: int = 0) -> bool:
    if _depth > 30:
        return True
    if isinstance(obj, (str, int, float, bool, type(None))):
        return False
    if isinstance(obj, list):
        return any(_contains_non_jsonable(v, _depth + 1) for v in obj)
    if isinstance(obj, dict):
        return any(_contains_non_jsonable(v, _depth + 1) for v in obj.values())
    return True


def _compute_scores(path: Path, present: list, issues: list) -> Dict[str, int]:
    completeness = 100 - max(0, min(50, len([f for f in _NICE_TO_HAVE if f not in present]) * 5))
    reproducibility = 100
    portability = 100
    reporting = 100

    rm = path / "run_metadata.json"
    if rm.exists():
        try:
            data = json.loads(rm.read_text())
            if "seed" not in data:
                reproducibility -= 30
            if "tgraphx_version" not in data:
                reproducibility -= 20
            if "device" not in data:
                reproducibility -= 10
        except Exception:
            reproducibility -= 40
    else:
        reproducibility = 30

    for f in present:
        try:
            text = (path / f).read_text()
            if any(bad in text for bad in ("/home/", "/Users/", "C:\\Users\\")):
                portability -= 20
                break
        except Exception:
            pass

    reporting -= min(50, len(issues) * 10)
    return {
        "completeness_score": max(0, completeness),
        "reproducibility_score": max(0, reproducibility),
        "portability_score": max(0, portability),
        "scientific_reporting_score": max(0, reporting),
    }


def _to_markdown(path: Path, result: Dict[str, Any]) -> str:
    lines = [f"# TGraphX Dashboard Audit: `{path}`", "",
             f"**Status:** {'✓ PASS' if result['ok'] else '✗ FAIL'}",
             f"**Files present:** {len(result.get('files_present', []))}",
             f"**Files missing:** {result.get('files_missing', [])}",
             "", "## Quality Scores"]
    for k in ("completeness_score", "reproducibility_score",
               "portability_score", "scientific_reporting_score"):
        v = result.get(k, "?")
        lines.append(f"- {k.replace('_', ' ').title()}: {v}/100")
    if result.get("issues"):
        lines.extend(["", "## Issues"])
        for issue in result["issues"]:
            lines.append(f"- {issue}")
    lines.extend(["", "## Recommendations"])
    if result.get("completeness_score", 100) < 80:
        lines.append("- Write more artifact files (dataset_metadata.json, graph_summary.json)")
    if result.get("reproducibility_score", 100) < 80:
        lines.append("- Include `seed`, `tgraphx_version`, and `device` in run_metadata.json")
    return "\n".join(lines)


def audit_run_dir(
    path: Union[str, Path],
    *,
    strict: bool = False,
    return_markdown: bool = False,
    workflow: Optional[str] = None,
) -> Dict[str, Any]:
    """Audit a TGraphX run directory for required artifact files, JSON validity, and UX quality.

    v1.4.1+: Returns quality scores (completeness, reproducibility, portability, reporting).

    Args:
        path: Path to the run directory (e.g. ``runs/advanced_notebooks/31_mnist``).
        strict: If True, raise ValueError on any failure.
        return_markdown: Include a markdown string in the result.
        workflow: Optional workflow type for workflow-specific required files.
            Supported: ``"generation"``, ``"evolution"``, ``"graph_rl"``.

    Returns:
        Dict with ``ok``, ``issues``, ``files_present``, ``files_missing``,
        ``schema``, and quality score fields.
    """
    path = Path(path)
    issues: List[str] = []

    if not path.exists():
        issues.append(f"Run directory not found: {path}")
        if strict:
            raise FileNotFoundError(issues[-1])
        return {"ok": False, "issues": issues, "files_present": [],
                "files_missing": _REQUIRED_FILES, "schema": {},
                "completeness_score": 0, "reproducibility_score": 0,
                "portability_score": 0, "scientific_reporting_score": 0}

    present = sorted(p.name for p in path.glob("*.json"))
    base_required = list(_REQUIRED_FILES)
    if workflow and workflow in _WORKFLOW_REQUIRED:
        base_required = list(_WORKFLOW_REQUIRED[workflow])
    missing = [f for f in base_required if f not in present]
    for f in missing:
        issues.append(f"Missing required artifact: {f}")

    schema: Dict[str, Any] = {}
    for f in present:
        fp = path / f
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(f"{f}: invalid JSON ({exc})")
            schema[f] = {"error": str(exc)}
            continue
        if not isinstance(data, dict):
            issues.append(f"{f}: expected JSON object, got {type(data).__name__}")
            schema[f] = {"type": type(data).__name__}
            continue
        schema[f] = {"keys": sorted(data.keys()), "size_bytes": fp.stat().st_size}
        if fp.stat().st_size > 5_000_000:
            issues.append(f"{f}: unusually large ({fp.stat().st_size} bytes)")
        if _contains_non_jsonable(data):
            issues.append(f"{f}: contains non-JSON-serializable values")
        for k, v in data.items():
            if _is_tensor_blob(v):
                issues.append(f"{f}: key {k!r} looks like a raw tensor dump")

    rm = path / "run_metadata.json"
    if rm.exists():
        try:
            data = json.loads(rm.read_text(encoding="utf-8"))
            for k in ("tgraphx_version", "seed"):
                if k not in data:
                    issues.append(f"run_metadata.json: missing recommended key {k!r}")
        except Exception:
            pass

    ok = len(issues) == 0
    scores = _compute_scores(path, present, issues)
    if strict and not ok:
        raise ValueError("Run-dir audit failed: " + "; ".join(issues))

    result: Dict[str, Any] = {
        "ok": ok, "issues": issues, "files_present": present,
        "files_missing": missing, "schema": schema, **scores
    }
    if return_markdown:
        result["markdown"] = _to_markdown(path, result)
    return result


def dashboard_audit(path: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """Alias for :func:`audit_run_dir` (v1.4.0+)."""
    return audit_run_dir(path, **kwargs)

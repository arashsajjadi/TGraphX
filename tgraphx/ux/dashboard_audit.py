"""Dashboard / run-directory audit utility."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union


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


def _is_tensor_blob(value: Any) -> bool:
    """Detect accidental raw-tensor dumps (long lists of floats nested deeply)."""
    if isinstance(value, list):
        if len(value) > 5000:
            return True
    return False


def _contains_non_jsonable(obj: Any, _depth: int = 0) -> bool:
    """Return True if the object contains values that won't JSON-round-trip cleanly."""
    if _depth > 30:
        return True
    if isinstance(obj, (str, int, float, bool, type(None))):
        return False
    if isinstance(obj, list):
        return any(_contains_non_jsonable(v, _depth + 1) for v in obj)
    if isinstance(obj, dict):
        return any(_contains_non_jsonable(v, _depth + 1) for v in obj.values())
    return True  # tensors, Path, device, etc.


def audit_run_dir(
    path: Union[str, Path],
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    """Audit a run directory for required artifact files and JSON correctness.

    Args:
        path: Path to the run directory (e.g. ``runs/advanced_notebooks/31_mnist``).
        strict: If True, raise ValueError instead of returning ``ok=False``.

    Returns:
        Dict with keys ``ok``, ``issues``, ``files_present``, ``files_missing``,
        ``schema``.
    """
    path = Path(path)
    issues: List[str] = []
    if not path.exists():
        issues.append(f"Run directory not found: {path}")
        if strict:
            raise FileNotFoundError(issues[-1])
        return {"ok": False, "issues": issues, "files_present": [],
                "files_missing": _REQUIRED_FILES, "schema": {}}

    present = sorted(p.name for p in path.glob("*.json"))
    missing = [f for f in _REQUIRED_FILES if f not in present]
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
            issues.append(f"{f}: artifact unusually large ({fp.stat().st_size} bytes)")
        if _contains_non_jsonable(data):
            issues.append(f"{f}: contains non-JSON-serializable values")
        for k, v in data.items():
            if _is_tensor_blob(v):
                issues.append(f"{f}: key {k!r} looks like a raw tensor dump")

    # Run metadata sanity
    rm = path / "run_metadata.json"
    if rm.exists():
        try:
            data = json.loads(rm.read_text(encoding="utf-8"))
            for required_key in ("tgraphx_version", "seed"):
                if required_key not in data:
                    issues.append(
                        f"run_metadata.json: missing recommended key {required_key!r}"
                    )
        except Exception:
            pass

    ok = len(issues) == 0
    if strict and not ok:
        raise ValueError("Run-dir audit failed: " + "; ".join(issues))
    return {
        "ok": ok,
        "issues": issues,
        "files_present": present,
        "files_missing": missing,
        "schema": schema,
    }


def dashboard_audit(*args, **kwargs):
    """Alias for :func:`audit_run_dir`."""
    return audit_run_dir(*args, **kwargs)

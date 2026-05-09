#!/usr/bin/env python3
"""TGraphX Technical Debt Audit Tool.

Generates reproducible debt metrics, scores, and CI-ready reports.

Usage:
    python tools/technical_debt_audit.py --out-dir reports/technical_debt
    python tools/technical_debt_audit.py --ci --baseline reports/technical_debt/baseline.json
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "max_file_lines": 1200,
    "warn_file_lines": 800,
    "max_function_lines": 150,
    "warn_function_lines": 80,
    "max_cyclomatic_complexity": 15,
    "warn_cyclomatic_complexity": 10,
    "min_module_coverage": 50,
    "warn_module_coverage": 70,
    "max_public_api_without_reference": 0,
    "fail_on_broken_links": True,
    "fail_on_version_mismatch": True,
}

WEIGHTS = {
    "complexity_debt": 15,
    "dead_code_debt": 12,
    "test_coverage_debt": 15,
    "docs_api_drift_debt": 15,
    "type_debt": 10,
    "architecture_import_debt": 10,
    "performance_guard_debt": 10,
    "security_debt": 5,
    "packaging_release_debt": 3,
    "ai_code_smell_debt": 5,
}

# O(N²) / dense patterns to flag
QUADRATIC_PATTERNS = [
    r"for \w+ in range.*\n.*for \w+ in range",
    r"torch\.zeros\(num_nodes,\s*num_nodes",
    r"torch\.ones\(num_nodes,\s*num_nodes",
    r"full_adjacency",
    r"all_pairs",
    r"pairwise_distances",
    r"\.all_combinations",
    r"dense_adj",
]

SECURITY_PATTERNS = [
    # Only flag Python builtin eval() — not `def eval(`, `.eval()`, or method calls named `eval`
    # Pattern: eval( followed by a quoted string or expression, but NOT preceded by 'def ' or '.'
    (r"(?<!def )(?<!\.)(?<!\w)eval\s*\((?!self)", "eval() builtin call (not def/method)"),
    (r"(?<!def )(?<!\.)exec\s*\((?!self)", "exec() call"),
    (r"pickle\.loads\s*\(", "pickle.loads (unsafe deserialization)"),
    (r"new Function\s*\(", "new Function() (JS eval equivalent)"),
]

# AI-code smells: patterns that suggest copy-paste or boilerplate generation
AI_SMELL_PATTERNS = [
    r"# TODO: implement",
    r"# FIXME: not implemented",
    r"raise NotImplementedError.*pass",
    r"# placeholder",
    r"# stub",
    r"def \w+\(self\):\s*\n\s*pass$",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    category: str
    severity: str  # blocker / high / medium / low
    file: str
    line: int
    message: str
    rule: str = ""


@dataclass
class FileMetrics:
    path: str
    lines: int
    code_lines: int
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    long_functions: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class AuditReport:
    timestamp: str
    version: str
    total_score: float
    severity: str
    category_scores: Dict[str, float] = field(default_factory=dict)
    category_raw: Dict[str, Any] = field(default_factory=dict)
    blockers: List[Dict] = field(default_factory=list)
    high_debt: List[Dict] = field(default_factory=list)
    medium_debt: List[Dict] = field(default_factory=list)
    low_debt: List[Dict] = field(default_factory=list)
    top_risk_files: List[Dict] = field(default_factory=list)
    long_functions: List[Dict] = field(default_factory=list)
    docs_drift: List[Dict] = field(default_factory=list)
    api_coverage: List[Dict] = field(default_factory=list)
    tensor_debt: List[Dict] = field(default_factory=list)
    performance_debt: List[Dict] = field(default_factory=list)
    dashboard_drift: List[Dict] = field(default_factory=list)
    benchmark_drift: List[Dict] = field(default_factory=list)
    security_findings: List[Dict] = field(default_factory=list)
    ruff_summary: Dict[str, Any] = field(default_factory=dict)
    mypy_summary: Dict[str, Any] = field(default_factory=dict)
    tools_available: Dict[str, bool] = field(default_factory=dict)
    cleanup_plan: Dict[str, List[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _run(cmd: List[str], cwd: Path = ROOT) -> Tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd), timeout=120)
        return r.returncode, r.stdout, r.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return -1, "", str(e)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _py_files(directory: Path) -> List[Path]:
    return sorted(directory.rglob("*.py"))


def _count_lines(text: str) -> Tuple[int, int]:
    lines = text.splitlines()
    code = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    return len(lines), len(code)


def _parse_ast_metrics(path: Path) -> FileMetrics:
    text = _read(path)
    total, code = _count_lines(text)
    fm = FileMetrics(path=str(path.relative_to(ROOT)), lines=total, code_lines=code)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return fm
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            fm.functions.append(node.name)
            fn_lines = (node.end_lineno or node.lineno) - node.lineno + 1
            if fn_lines > DEFAULT_THRESHOLDS["warn_function_lines"]:
                fm.long_functions.append((node.name, fn_lines))
        elif isinstance(node, ast.AsyncFunctionDef):
            fm.functions.append(node.name)
            fn_lines = (node.end_lineno or node.lineno) - node.lineno + 1
            if fn_lines > DEFAULT_THRESHOLDS["warn_function_lines"]:
                fm.long_functions.append((node.name, fn_lines))
        elif isinstance(node, ast.ClassDef):
            fm.classes.append(node.name)
    return fm


# ---------------------------------------------------------------------------
# Check 1: File metrics
# ---------------------------------------------------------------------------

def check_file_metrics(package_dir: Path) -> Tuple[List[FileMetrics], List[Finding]]:
    findings: List[Finding] = []
    metrics: List[FileMetrics] = []
    for path in _py_files(package_dir):
        fm = _parse_ast_metrics(path)
        metrics.append(fm)
        rel = str(path.relative_to(ROOT))

        if fm.lines > DEFAULT_THRESHOLDS["max_file_lines"]:
            findings.append(Finding(
                "complexity_debt", "high", rel, 0,
                f"File has {fm.lines} lines (max {DEFAULT_THRESHOLDS['max_file_lines']})",
                "file-too-long",
            ))
        elif fm.lines > DEFAULT_THRESHOLDS["warn_file_lines"]:
            findings.append(Finding(
                "complexity_debt", "medium", rel, 0,
                f"File has {fm.lines} lines (warn at {DEFAULT_THRESHOLDS['warn_file_lines']})",
                "file-long",
            ))

        for fn_name, fn_lines in fm.long_functions:
            sev = "high" if fn_lines > DEFAULT_THRESHOLDS["max_function_lines"] else "medium"
            findings.append(Finding(
                "complexity_debt", sev, rel, 0,
                f"Function '{fn_name}' has {fn_lines} lines",
                "function-too-long",
            ))
    return metrics, findings


# ---------------------------------------------------------------------------
# Check 2: Ruff lint
# ---------------------------------------------------------------------------

def check_ruff(package_dir: Path) -> Tuple[Dict[str, Any], List[Finding]]:
    findings: List[Finding] = []
    summary: Dict[str, Any] = {"available": False, "total": 0, "by_code": {}, "by_file": {}}

    rc, out, err = _run(["ruff", "check", str(package_dir), "--output-format=json"])
    if rc == -1:
        return summary, findings

    summary["available"] = True
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        data = []

    summary["total"] = len(data)
    by_code: Dict[str, int] = defaultdict(int)
    by_file: Dict[str, int] = defaultdict(int)

    for item in data:
        code = item.get("code", "UNK")
        filename = item.get("filename", "?")
        rel = str(Path(filename).relative_to(ROOT)) if os.path.isabs(filename) else filename
        by_code[code] += 1
        by_file[rel] += 1

        msg = item.get("message", "")
        row = item.get("location", {}).get("row", 0)
        sev = "low"
        if code.startswith("E") or code.startswith("F"):
            sev = "medium"
        if code in ("F401", "F811", "E501"):
            sev = "low"

        findings.append(Finding("architecture_import_debt", sev, rel, row, f"[{code}] {msg}", code))

    summary["by_code"] = dict(sorted(by_code.items(), key=lambda x: -x[1])[:30])
    summary["by_file"] = dict(sorted(by_file.items(), key=lambda x: -x[1])[:20])
    return summary, findings


# ---------------------------------------------------------------------------
# Check 3: Mypy type checking
# ---------------------------------------------------------------------------

def check_mypy(package_dir: Path) -> Tuple[Dict[str, Any], List[Finding]]:
    findings: List[Finding] = []
    summary: Dict[str, Any] = {"available": False, "total": 0, "by_module": {}, "error_types": {}}

    rc, out, err = _run([
        "python", "-m", "mypy", str(package_dir),
        "--ignore-missing-imports", "--no-error-summary",
        "--follow-imports=skip", "--show-column-numbers",
        "--no-incremental",
    ])
    if "mypy" in err.lower() and "not found" in err.lower():
        return summary, findings

    summary["available"] = True
    by_module: Dict[str, int] = defaultdict(int)
    error_types: Dict[str, int] = defaultdict(int)
    count = 0

    for line in out.splitlines():
        m = re.match(r"(.+\.py):(\d+):\d+: (error|note|warning): (.+)\s+\[(.+)\]$", line)
        if not m:
            continue
        fpath, lineno, level, msg, code = m.groups()
        if level != "error":
            continue
        rel = str(Path(fpath).relative_to(ROOT)) if os.path.isabs(fpath) else fpath
        module = Path(rel).parts[0] if "/" in rel else rel
        by_module[module] += 1
        error_types[code] = error_types.get(code, 0) + 1
        count += 1
        findings.append(Finding("type_debt", "low", rel, int(lineno), f"[{code}] {msg}", code))

    summary["total"] = count
    summary["by_module"] = dict(sorted(by_module.items(), key=lambda x: -x[1])[:15])
    summary["error_types"] = dict(sorted(error_types.items(), key=lambda x: -x[1])[:15])
    return summary, findings


# ---------------------------------------------------------------------------
# Check 4: Docs/link drift
# ---------------------------------------------------------------------------

def check_docs_drift(readme: Path, docs_dir: Path, examples_dir: Path, tutorials_dir: Path) -> List[Finding]:
    findings: List[Finding] = []

    def _check_file_links(src: Path, source_label: str) -> None:
        text = _read(src)
        links = re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)', text)
        for link in links:
            if link.startswith("http") or link.startswith("mailto"):
                continue
            # Resolve relative to file's directory
            base = src.parent if src.is_file() else ROOT
            target = (base / link).resolve()
            if not target.exists():
                sev = "blocker" if "README" in source_label or "index" in source_label else "high"
                findings.append(Finding(
                    "docs_api_drift_debt", sev,
                    str(src.relative_to(ROOT)), 0,
                    f"Broken link: {link}",
                    "broken-link",
                ))

    if readme.exists():
        _check_file_links(readme, "README")
    if (docs_dir / "index.md").exists():
        _check_file_links(docs_dir / "index.md", "docs/index")

    # Check examples referenced in README exist
    if readme.exists():
        text = _read(readme)
        for m in re.finditer(r'python\s+(examples/\S+\.py)', text):
            ex_path = ROOT / m.group(1)
            if not ex_path.exists():
                findings.append(Finding(
                    "docs_api_drift_debt", "blocker",
                    "README.md", 0,
                    f"README references missing example: {m.group(1)}",
                    "missing-example",
                ))
        for m in re.finditer(r'python\s+(tutorials/\S+\.py)', text):
            tut_path = ROOT / m.group(1)
            if not tut_path.exists():
                findings.append(Finding(
                    "docs_api_drift_debt", "blocker",
                    "README.md", 0,
                    f"README references missing tutorial: {m.group(1)}",
                    "missing-tutorial",
                ))

    return findings


# ---------------------------------------------------------------------------
# Check 5: API drift — exports vs tests/examples/docs
# ---------------------------------------------------------------------------

def check_api_drift(package_dir: Path, tests_dir: Path, examples_dir: Path, docs_dir: Path) -> List[Finding]:
    findings: List[Finding] = []

    # Collect all public exports from __init__.py files
    exports: Dict[str, str] = {}  # symbol -> file
    for init in package_dir.rglob("__init__.py"):
        text = _read(init)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    exports[elt.value] = str(init.relative_to(ROOT))

    # Check each export is importable
    import_failures: List[str] = []
    for sym in list(exports.keys())[:50]:  # sample first 50 for speed
        try:
            mod_path = exports[sym].replace("/", ".").replace(".py", "").replace(".__init__", "")
            mod = importlib.import_module(mod_path)
            if not hasattr(mod, sym):
                import_failures.append(sym)
        except Exception:
            import_failures.append(sym)

    for sym in import_failures:
        findings.append(Finding(
            "docs_api_drift_debt", "blocker",
            exports.get(sym, "?"), 0,
            f"Public export '{sym}' not importable from its module",
            "export-not-importable",
        ))

    # Check exported symbols have at least one reference in tests/examples/docs
    all_ref_text = ""
    for d in [tests_dir, examples_dir, docs_dir]:
        if d.exists():
            for f in d.rglob("*.py"):
                all_ref_text += _read(f) + "\n"
            for f in d.rglob("*.md"):
                all_ref_text += _read(f) + "\n"

    unreferenced: List[str] = []
    for sym in exports:
        if sym.startswith("_"):
            continue
        if sym not in all_ref_text:
            unreferenced.append(sym)

    for sym in unreferenced[:30]:  # cap at 30
        findings.append(Finding(
            "docs_api_drift_debt", "medium",
            exports.get(sym, "?"), 0,
            f"Public export '{sym}' has no reference in tests/examples/docs",
            "unreferenced-export",
        ))

    return findings, len(exports), len(unreferenced)


# ---------------------------------------------------------------------------
# Check 6: Security patterns
# ---------------------------------------------------------------------------

def check_security(package_dir: Path) -> List[Finding]:
    findings: List[Finding] = []
    for path in _py_files(package_dir):
        text = _read(path)
        rel = str(path.relative_to(ROOT))
        # Build a set of line numbers that are inside docstrings
        in_docstring_lines: set[int] = set()
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                    if (node.body and isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Constant)
                            and isinstance(node.body[0].value.value, str)):
                        ds = node.body[0]
                        for ln in range(ds.lineno, (ds.end_lineno or ds.lineno) + 1):
                            in_docstring_lines.add(ln)
        except SyntaxError:
            pass

        lines = text.splitlines()
        in_ml_string = False
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            # Skip comment-only lines
            if stripped.startswith("#"):
                continue
            # Skip docstring lines
            if lineno in in_docstring_lines:
                continue
            # Skip multi-line string blocks (''' / """)
            if stripped.startswith(('"""', "'''")):
                in_ml_string = not in_ml_string
                continue
            if in_ml_string:
                continue
            # Skip .eval() patterns (PyTorch model evaluation mode)
            if re.search(r"\.\s*eval\s*\(\s*\)", line):
                continue

            for pattern, label in SECURITY_PATTERNS:
                if not re.search(pattern, line):
                    continue
                # Skip negation comments: "No eval()" means it's NOT present
                if re.search(r"[Nn]o\s+eval|[Nn]ot\s+eval|without\s+eval", line):
                    continue
                sev = "high"
                findings.append(Finding(
                    "security_debt", sev, rel, lineno,
                    f"Security pattern: {label} — {stripped[:80]}",
                    "security-pattern",
                ))
    return findings


# ---------------------------------------------------------------------------
# Check 7: Performance guard debt (O(N²) patterns)
# ---------------------------------------------------------------------------

def check_performance_guards(package_dir: Path) -> List[Finding]:
    findings: List[Finding] = []
    nested_loop = re.compile(r"for \w+ in range\(\w*\).*\n(?:\s+.*\n)*\s+for \w+ in range\(\w*\)")

    for path in _py_files(package_dir):
        text = _read(path)
        rel = str(path.relative_to(ROOT))
        lines = text.splitlines()

        for pattern in [
            r"\btorch\.zeros\s*\(\s*num_nodes\s*,\s*num_nodes",
            r"\btorch\.ones\s*\(\s*num_nodes\s*,\s*num_nodes",
            r"\bfull_adjacency\b",
            r"\ball_pairs\b",
            r"\bpairwise_distances\b",
            r"\bdense_adj\b",
        ]:
            for lineno, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    if "# noqa" in line or "# O(N" in line:
                        continue
                    findings.append(Finding(
                        "performance_guard_debt", "medium", rel, lineno,
                        f"Potential O(N²) pattern without guard: {line.strip()[:80]}",
                        "quadratic-pattern",
                    ))

        # Detect nested for-loops over nodes
        for lineno, line in enumerate(lines, 1):
            if re.search(r"for \w+ in range\(n\b|for \w+ in range\(num_nodes", line):
                if lineno < len(lines):
                    next_line = lines[lineno]
                    if re.search(r"for \w+ in range\(n\b|for \w+ in range\(num_nodes", next_line):
                        findings.append(Finding(
                            "performance_guard_debt", "medium", rel, lineno,
                            f"Nested O(N²) loop: {line.strip()[:60]}",
                            "nested-loop",
                        ))

    return findings


# ---------------------------------------------------------------------------
# Check 8: Tensor-native debt
# ---------------------------------------------------------------------------

def check_tensor_debt(tests_dir: Path, package_dir: Path) -> List[Finding]:
    findings: List[Finding] = []
    tensor_apis = [
        "ImageNodeEncoder", "VolumeNodeEncoder", "TensorGATLayer",
        "TensorGraphSAGELayer", "TensorGINLayer", "ConvMessagePassing",
        "VectorNodeProjector", "EdgeFeatureProjector",
    ]
    # Search tests + examples + docs for references
    all_text = ""
    root = package_dir.parent
    for search_dir in [tests_dir, root / "examples", root / "docs", root / "tutorials"]:
        if search_dir.exists():
            for f in search_dir.rglob("*"):
                if f.suffix in (".py", ".md"):
                    all_text += _read(f)

    for api in tensor_apis:
        if api not in all_text:
            findings.append(Finding(
                "docs_api_drift_debt", "high",
                "tests/", 0,
                f"Tensor-aware API '{api}' has no reference in tests/examples/docs",
                "tensor-api-no-test",
            ))
            continue
        # Check for device/dtype test patterns in tests only
        test_text = ""
        if tests_dir.exists():
            for f in tests_dir.rglob("*.py"):
                test_text += _read(f)
        device_tested = any(
            keyword in test_text for keyword in ["cuda", ".to(device", "device=torch"]
        )
        if not device_tested:
            findings.append(Finding(
                "docs_api_drift_debt", "low",
                "tests/", 0,
                f"Tensor API test suite may lack explicit device/dtype assertions",
                "tensor-api-device-test",
            ))

    # Check for raw tensor dump in dashboard JSON
    dashboard_dir = package_dir / "dashboard"
    if dashboard_dir.exists():
        for path in dashboard_dir.rglob("*.py"):
            text = _read(path)
            if "tensor(" in text.lower() and "json.dump" in text:
                findings.append(Finding(
                    "security_debt", "high",
                    str(path.relative_to(ROOT)), 0,
                    "Possible raw tensor dump to JSON in dashboard",
                    "raw-tensor-json",
                ))

    return findings


# ---------------------------------------------------------------------------
# Check 9: Benchmark drift
# ---------------------------------------------------------------------------

def check_benchmark_drift(benchmarks_dir: Path) -> List[Finding]:
    findings: List[Finding] = []
    if not benchmarks_dir.exists():
        findings.append(Finding(
            "docs_api_drift_debt", "high",
            "benchmarks/", 0,
            "Benchmarks directory missing",
            "missing-benchmarks",
        ))
        return findings

    for path in benchmarks_dir.rglob("*.py"):
        if path.name.startswith("_"):
            continue
        rel = str(path.relative_to(ROOT))
        text = _read(path)

        if "--help" not in text and "add_argument" not in text:
            findings.append(Finding(
                "docs_api_drift_debt", "medium", rel, 0,
                "Benchmark lacks --help / argparse",
                "benchmark-no-argparse",
            ))

        if "--small" not in text:
            findings.append(Finding(
                "docs_api_drift_debt", "medium", rel, 0,
                "Benchmark lacks --small flag for fast CI runs",
                "benchmark-no-small",
            ))

        if "--json" not in text or "json.dumps" not in text:
            findings.append(Finding(
                "docs_api_drift_debt", "medium", rel, 0,
                "Benchmark lacks --json / machine-readable output",
                "benchmark-no-json",
            ))

        # Check for required fields in JSON output
        required = ["package_version", "status", "limitations"]
        for req in required:
            if req not in text:
                findings.append(Finding(
                    "docs_api_drift_debt", "low", rel, 0,
                    f"Benchmark JSON output missing field: {req}",
                    f"benchmark-missing-{req}",
                ))

    return findings


# ---------------------------------------------------------------------------
# Check 10: Dashboard drift
# ---------------------------------------------------------------------------

def check_dashboard_drift(package_dir: Path, tests_dir: Path) -> List[Finding]:
    findings: List[Finding] = []

    # Expected artifact names from reports writers
    artifact_writers = {}
    for path in package_dir.rglob("reports.py"):
        text = _read(path)
        for m in re.finditer(r'os\.path\.join\([^,]+,\s*["\']([^"\']+\.json)["\']', text):
            artifact_writers[m.group(1)] = str(path.relative_to(ROOT))

    # Check tests reference dashboard validation
    test_text = ""
    if tests_dir.exists():
        for f in tests_dir.rglob("*.py"):
            test_text += _read(f)

    if "dashboard_artifact_validation" not in test_text and "write_graph_generation_report" not in test_text:
        findings.append(Finding(
            "docs_api_drift_debt", "medium",
            "tests/", 0,
            "No dashboard artifact validation test found",
            "dashboard-no-validation-test",
        ))

    # Check for eval/CDN in dashboard app
    app_path = package_dir / "dashboard" / "app.py"
    if app_path.exists():
        app_text = _read(app_path)
        if "cdn.jsdelivr" in app_text or "cdnjs.cloudflare" in app_text:
            findings.append(Finding(
                "security_debt", "blocker",
                "tgraphx/dashboard/app.py", 0,
                "External CDN reference in dashboard (privacy violation)",
                "dashboard-cdn",
            ))
        # Only flag new Function if it appears in actual JS context (not in comments/strings)
        for lineno, line in enumerate(app_text.splitlines(), 1):
            if re.search(r"new Function\s*\(", line) and not line.strip().startswith(("#", "*", "/")):
                findings.append(Finding(
                    "security_debt", "blocker",
                    "tgraphx/dashboard/app.py", lineno,
                    "new Function() eval equivalent in dashboard",
                    "dashboard-eval",
                ))

    return findings, artifact_writers


# ---------------------------------------------------------------------------
# Check 11: AI code smells
# ---------------------------------------------------------------------------

def check_ai_smells(package_dir: Path) -> List[Finding]:
    findings: List[Finding] = []
    smell_patterns = [
        (r"# TODO: implement", "Unimplemented TODO stub"),
        (r"# FIXME: not implemented", "Unimplemented FIXME stub"),
        (r"raise NotImplementedError\(.*\)\s*$", "NotImplementedError in public API"),
        (r"# placeholder", "Placeholder comment"),
        (r"# This is (a|an) (stub|placeholder|skeleton)", "Skeleton stub comment"),
        (r"pass\s*#\s*(stub|placeholder|not implemented)", "Stub pass statement"),
    ]
    for path in _py_files(package_dir):
        text = _read(path)
        rel = str(path.relative_to(ROOT))
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern, label in smell_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(Finding(
                        "ai_code_smell_debt", "medium", rel, lineno,
                        f"AI-code smell: {label}",
                        "ai-smell",
                    ))
    return findings


# ---------------------------------------------------------------------------
# Check 12: Packaging and release
# ---------------------------------------------------------------------------

def check_packaging(root: Path) -> List[Finding]:
    findings: List[Finding] = []

    # Version sync
    pyproject = _read(root / "pyproject.toml")
    init_py = _read(root / "tgraphx" / "__init__.py")

    pv = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', pyproject, re.MULTILINE)
    iv = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_py, re.MULTILINE)

    pv_str = pv.group(1) if pv else "?"
    iv_str = iv.group(1) if iv else "?"

    if pv_str != iv_str:
        findings.append(Finding(
            "packaging_release_debt", "blocker",
            "pyproject.toml", 0,
            f"Version mismatch: pyproject.toml={pv_str}, __init__.py={iv_str}",
            "version-mismatch",
        ))

    # CHANGELOG check
    changelog = _read(root / "CHANGELOG.md")
    if not changelog:
        findings.append(Finding(
            "packaging_release_debt", "blocker",
            "CHANGELOG.md", 0,
            "CHANGELOG.md missing",
            "no-changelog",
        ))
    else:
        # Check for current version or 1.0.0 entry
        if pv_str not in changelog and "1.0.0" not in changelog:
            findings.append(Finding(
                "packaging_release_debt", "high",
                "CHANGELOG.md", 0,
                f"CHANGELOG missing entry for current version {pv_str}",
                "changelog-missing-version",
            ))

    # Check no large binary files in wheel-candidates
    for path in (root / "tgraphx").rglob("*"):
        if path.is_file() and path.suffix in (".pt", ".pkl", ".bin", ".npy", ".npz", ".h5"):
            findings.append(Finding(
                "packaging_release_debt", "high",
                str(path.relative_to(root)), 0,
                f"Data/model file in package directory (will bloat wheel)",
                "data-in-package",
            ))

    # Check optional deps are not imported at top level
    optional_deps = ["torch_geometric", "dgl", "ogb", "mlflow", "tensorboard", "psutil", "pynvml"]
    init_text = _read(root / "tgraphx" / "__init__.py")
    for dep in optional_deps:
        pattern = rf"^import {dep}|^from {dep}"
        if re.search(pattern, init_text, re.MULTILINE):
            findings.append(Finding(
                "architecture_import_debt", "blocker",
                "tgraphx/__init__.py", 0,
                f"Optional dependency '{dep}' imported at package top level",
                "optional-dep-toplevel",
            ))

    return findings, pv_str, iv_str


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_score(all_findings: List[Finding], file_metrics: List[FileMetrics]) -> Dict[str, float]:
    cat_blockers: Dict[str, int] = defaultdict(int)
    cat_high: Dict[str, int] = defaultdict(int)
    cat_medium: Dict[str, int] = defaultdict(int)
    cat_low: Dict[str, int] = defaultdict(int)

    for f in all_findings:
        if f.severity == "blocker":
            cat_blockers[f.category] += 1
        elif f.severity == "high":
            cat_high[f.category] += 1
        elif f.severity == "medium":
            cat_medium[f.category] += 1
        else:
            cat_low[f.category] += 1

    # Scale by codebase size to avoid penalizing large projects
    n_files = max(1, len(file_metrics))
    # For a 200+ file codebase, normalisation caps are higher
    # Caps: blockers->very tight, high->moderate, medium/low->scale with size
    size_factor = max(1.0, n_files / 50)

    scores: Dict[str, float] = {}
    for cat in WEIGHTS:
        b = cat_blockers.get(cat, 0)
        h = cat_high.get(cat, 0)
        m = cat_medium.get(cat, 0)
        l = cat_low.get(cat, 0)
        # Debt formula: blockers dominate; low findings are soft signal
        raw = b * 100 + h * 20 + m * 3 + l * 0.5
        # Normalise: cap = 200 for a 50-file project, scales with size
        # At 200-file project: cap=800, so 10 high = 200/800 = 25 score
        cap = 200 * size_factor
        score = min(100.0, (raw / cap) * 100.0)
        scores[cat] = round(score, 1)

    # Weighted total
    total = sum(scores.get(cat, 0) * w / 100 for cat, w in WEIGHTS.items())
    scores["total"] = round(total, 1)
    return scores


def severity_label(score: float) -> str:
    if score <= 15:
        return "excellent"
    if score <= 30:
        return "acceptable"
    if score <= 50:
        return "needs-cleanup"
    if score <= 70:
        return "risky"
    return "do-not-release"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _finding_to_dict(f: Finding) -> Dict:
    return {"category": f.category, "severity": f.severity, "file": f.file,
            "line": f.line, "message": f.message, "rule": f.rule}


def build_report(
    all_findings: List[Finding],
    file_metrics: List[FileMetrics],
    ruff_summary: Dict,
    mypy_summary: Dict,
    tools_available: Dict,
    api_stats: Tuple,
    dashboard_artifact_writers: Dict,
    version_info: Tuple,
    out_dir: Path,
) -> AuditReport:
    scores = compute_score(all_findings, file_metrics)
    total = scores["total"]

    blockers = [_finding_to_dict(f) for f in all_findings if f.severity == "blocker"]
    high = [_finding_to_dict(f) for f in all_findings if f.severity == "high"]
    medium = [_finding_to_dict(f) for f in all_findings if f.severity == "medium"]
    low = [_finding_to_dict(f) for f in all_findings if f.severity == "low"]

    # Top risk files by finding count + line count
    file_risk: Dict[str, int] = defaultdict(int)
    for f in all_findings:
        file_risk[f.file] += {"blocker": 10, "high": 5, "medium": 2, "low": 1}[f.severity]
    top_files = sorted(file_risk.items(), key=lambda x: -x[1])[:20]

    long_fns = []
    for fm in file_metrics:
        for fn_name, fn_lines in fm.long_functions:
            long_fns.append({"file": fm.path, "function": fn_name, "lines": fn_lines})
    long_fns.sort(key=lambda x: -x["lines"])
    long_fns = long_fns[:20]

    # Docs drift
    docs_drift = [_finding_to_dict(f) for f in all_findings if f.rule in (
        "broken-link", "missing-example", "missing-tutorial", "unreferenced-export",
    )]

    # API coverage
    api_total, api_unreferenced = api_stats[1], api_stats[2]
    api_coverage = [_finding_to_dict(f) for f in all_findings if "export" in f.rule]

    # Tensor debt
    tensor_debt = [_finding_to_dict(f) for f in all_findings if "tensor" in f.rule]

    # Performance debt
    performance_debt = [_finding_to_dict(f) for f in all_findings if f.category == "performance_guard_debt"]

    # Dashboard drift
    dashboard_drift = [_finding_to_dict(f) for f in all_findings if "dashboard" in f.rule]

    # Benchmark drift
    benchmark_drift = [_finding_to_dict(f) for f in all_findings if "benchmark" in f.rule]

    # Security
    security_findings = [_finding_to_dict(f) for f in all_findings if f.category == "security_debt"]

    # Cleanup plan
    p0 = [f["message"] for f in blockers[:10]]
    p1 = [f["message"] for f in high[:15]]
    p2 = [f["message"] for f in medium[:15]]
    p3 = [f["message"] for f in low[:10]]

    import datetime
    report = AuditReport(
        timestamp=datetime.datetime.now().isoformat(),
        version=version_info[0],
        total_score=total,
        severity=severity_label(total),
        category_scores=scores,
        blockers=blockers,
        high_debt=high,
        medium_debt=medium,
        low_debt=low,
        top_risk_files=[{"file": f, "risk_score": s} for f, s in top_files],
        long_functions=long_fns,
        docs_drift=docs_drift,
        api_coverage=api_coverage,
        tensor_debt=tensor_debt,
        performance_debt=performance_debt,
        dashboard_drift=dashboard_drift,
        benchmark_drift=benchmark_drift,
        security_findings=security_findings,
        ruff_summary=ruff_summary,
        mypy_summary=mypy_summary,
        tools_available=tools_available,
        cleanup_plan={"P0_blockers": p0, "P1_high": p1, "P2_medium": p2, "P3_low": p3},
        category_raw={
            "total_exports": api_total,
            "unreferenced_exports": api_unreferenced,
            "total_files": len(file_metrics),
            "total_findings": len(all_findings),
            "dashboard_writers": list(dashboard_artifact_writers.keys()),
        },
    )
    return report


def write_json(report: AuditReport, out_dir: Path) -> Path:
    out = out_dir / "technical_debt_report.json"
    data = asdict(report)
    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return out


def write_markdown(report: AuditReport, out_dir: Path) -> Path:
    out = out_dir / "technical_debt_report.md"
    md = []

    md.append("# TGraphX Technical Debt Report")
    md.append(f"\n**Generated:** {report.timestamp}  ")
    md.append(f"**Package version:** {report.version}  ")
    md.append(f"**Total Debt Score:** {report.total_score}/100 — **{report.severity.upper()}**\n")

    md.append("---\n")
    md.append("## 1. Executive Summary\n")
    md.append(f"- **Blockers:** {len(report.blockers)}")
    md.append(f"- **High debt:** {len(report.high_debt)}")
    md.append(f"- **Medium debt:** {len(report.medium_debt)}")
    md.append(f"- **Low debt:** {len(report.low_debt)}")
    md.append(f"- **Total findings:** {sum([len(report.blockers), len(report.high_debt), len(report.medium_debt), len(report.low_debt)])}")
    md.append("")

    # Verdict
    if len(report.blockers) == 0 and report.total_score <= 30:
        md.append("> **Verdict:** Release quality is acceptable. No blockers found.")
    elif len(report.blockers) > 0:
        md.append(f"> **Verdict:** {len(report.blockers)} BLOCKER(s) must be resolved before release.")
    else:
        md.append(f"> **Verdict:** Debt score {report.total_score} — consider cleanup before release.")

    md.append("\n---\n")
    md.append("## 2. Technical Debt Index\n")
    md.append("| Category | Score (0–100) | Weight |")
    md.append("|----------|:---:|:---:|")
    for cat, w in WEIGHTS.items():
        s = report.category_scores.get(cat, 0.0)
        bar = "🟢" if s <= 15 else ("🟡" if s <= 40 else "🔴")
        md.append(f"| {cat.replace('_', ' ').title()} | {s} {bar} | {w}% |")
    md.append(f"| **Total (weighted)** | **{report.total_score}** | 100% |")

    md.append("\n---\n")
    md.append("## 3. Tools Available\n")
    for tool, avail in report.tools_available.items():
        md.append(f"- {tool}: {'✅' if avail else '⬜ (skipped)'}")

    md.append("\n---\n")
    md.append("## 4. Blockers\n")
    if not report.blockers:
        md.append("_No blockers found._")
    else:
        for b in report.blockers:
            md.append(f"- **[{b['rule']}]** `{b['file']}:{b['line']}` — {b['message']}")

    md.append("\n---\n")
    md.append("## 5. Top 20 Highest-Risk Files\n")
    md.append("| File | Risk Score |")
    md.append("|------|:---:|")
    for item in report.top_risk_files[:20]:
        md.append(f"| `{item['file']}` | {item['risk_score']} |")

    md.append("\n---\n")
    md.append("## 6. Top 20 Long Functions\n")
    md.append("| File | Function | Lines |")
    md.append("|------|----------|:---:|")
    for item in report.long_functions[:20]:
        md.append(f"| `{item['file']}` | `{item['function']}` | {item['lines']} |")

    md.append("\n---\n")
    md.append("## 7. Docs / API Drift\n")
    if not report.docs_drift:
        md.append("_No docs/API drift detected._")
    else:
        for d in report.docs_drift[:20]:
            md.append(f"- **[{d['severity']}]** `{d['file']}` — {d['message']}")

    md.append("\n---\n")
    md.append("## 8. Public API Coverage\n")
    raw = report.category_raw
    md.append(f"- Total public exports found: {raw.get('total_exports', '?')}")
    md.append(f"- Exports without test/doc/example reference: {raw.get('unreferenced_exports', '?')}")
    if report.api_coverage:
        md.append("\n**Unreferenced exports (sample):**")
        for a in report.api_coverage[:15]:
            md.append(f"- `{a['message']}`")

    md.append("\n---\n")
    md.append("## 9. Tensor-Native Debt\n")
    if not report.tensor_debt:
        md.append("_No tensor-native debt detected._")
    else:
        for t in report.tensor_debt:
            md.append(f"- **[{t['severity']}]** {t['message']}")

    md.append("\n---\n")
    md.append("## 10. Performance Guard Debt\n")
    if not report.performance_debt:
        md.append("_No O(N²) patterns detected._")
    else:
        for p in report.performance_debt[:15]:
            md.append(f"- **[{p['severity']}]** `{p['file']}:{p['line']}` — {p['message']}")

    md.append("\n---\n")
    md.append("## 11. Dashboard Drift\n")
    if not report.dashboard_drift:
        md.append("_No dashboard drift detected._")
    else:
        for d in report.dashboard_drift[:10]:
            md.append(f"- **[{d['severity']}]** `{d['file']}` — {d['message']}")

    md.append("\n---\n")
    md.append("## 12. Benchmark Drift\n")
    if not report.benchmark_drift:
        md.append("_No benchmark drift detected._")
    else:
        for b in report.benchmark_drift[:15]:
            md.append(f"- **[{b['severity']}]** `{b['file']}` — {b['message']}")

    md.append("\n---\n")
    md.append("## 13. Security Findings\n")
    if not report.security_findings:
        md.append("_No security patterns detected._")
    else:
        for s in report.security_findings:
            md.append(f"- **[{s['severity']}]** `{s['file']}:{s['line']}` — {s['message']}")

    md.append("\n---\n")
    md.append("## 14. Lint Summary (ruff)\n")
    rs = report.ruff_summary
    if rs.get("available"):
        md.append(f"Total lint findings: {rs.get('total', 0)}")
        md.append("\n**Top rule codes:**")
        for code, count in list(rs.get("by_code", {}).items())[:10]:
            md.append(f"- `{code}`: {count}")
    else:
        md.append("_ruff not available or no findings._")

    md.append("\n---\n")
    md.append("## 15. Type Debt (mypy)\n")
    ms = report.mypy_summary
    if ms.get("available"):
        md.append(f"Total mypy errors: {ms.get('total', 0)}")
        md.append("\n**By module:**")
        for mod, count in list(ms.get("by_module", {}).items())[:10]:
            md.append(f"- `{mod}`: {count}")
    else:
        md.append("_mypy not available or no errors._")

    md.append("\n---\n")
    md.append("## 16. Suggested Cleanup Plan\n")
    if report.cleanup_plan.get("P0_blockers"):
        md.append("### P0 — Blockers (must fix before release)")
        for item in report.cleanup_plan["P0_blockers"]:
            md.append(f"- {item}")
    if report.cleanup_plan.get("P1_high"):
        md.append("\n### P1 — High Debt (fix in v1.x patch)")
        for item in report.cleanup_plan["P1_high"][:10]:
            md.append(f"- {item}")
    if report.cleanup_plan.get("P2_medium"):
        md.append("\n### P2 — Medium Debt (tech debt sprint)")
        for item in report.cleanup_plan["P2_medium"][:10]:
            md.append(f"- {item}")
    if report.cleanup_plan.get("P3_low"):
        md.append("\n### P3 — Low (nice-to-have)")
        for item in report.cleanup_plan["P3_low"][:5]:
            md.append(f"- {item}")

    out.write_text("\n".join(md) + "\n")
    return out


# ---------------------------------------------------------------------------
# CI mode
# ---------------------------------------------------------------------------

def run_ci_mode(report: AuditReport, baseline_path: Path) -> int:
    """Return 0 if CI passes, 1 if it fails."""
    fails = []
    warns = []

    if report.blockers:
        fails.append(f"{len(report.blockers)} blockers found: " +
                     "; ".join(b["message"][:60] for b in report.blockers[:3]))

    # Version mismatch
    for b in report.blockers:
        if b["rule"] == "version-mismatch":
            fails.append("Version mismatch between pyproject.toml and __init__.py")

    # Broken links
    for b in report.blockers:
        if b["rule"] == "broken-link":
            fails.append(f"Broken link: {b['message']}")

    # Import failures
    for b in report.blockers:
        if b["rule"] == "export-not-importable":
            fails.append(f"Import failure: {b['message']}")

    # Complexity and dead code: warn but don't fail
    if report.category_scores.get("complexity_debt", 0) > 50:
        warns.append(f"Complexity debt score {report.category_scores['complexity_debt']} > 50")

    # Compare against baseline
    if baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text())
            prev_blockers = len(baseline.get("blockers", []))
            curr_blockers = len(report.blockers)
            if curr_blockers > prev_blockers:
                fails.append(f"New blockers added: {curr_blockers} vs baseline {prev_blockers}")
        except Exception as e:
            warns.append(f"Could not load baseline: {e}")

    print("\n=== CI Mode Results ===")
    if fails:
        print(f"FAIL — {len(fails)} failure(s):")
        for f in fails:
            print(f"  ✗ {f}")
    else:
        print("PASS — No CI failures")

    if warns:
        print(f"\nWARNINGS — {len(warns)} warning(s):")
        for w in warns:
            print(f"  ⚠ {w}")

    return 1 if fails else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="TGraphX Technical Debt Audit")
    parser.add_argument("--package", default="tgraphx")
    parser.add_argument("--tests", default="tests")
    parser.add_argument("--examples", default="examples")
    parser.add_argument("--benchmarks", default="benchmarks")
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--out-dir", default="reports/technical_debt")
    parser.add_argument("--ci", action="store_true", help="CI mode: fail on blockers")
    parser.add_argument("--baseline", default="reports/technical_debt/baseline.json")
    parser.add_argument("--run-coverage", action="store_true", help="Run pytest-cov (slow)")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    package_dir = ROOT / args.package
    tests_dir = ROOT / args.tests
    examples_dir = ROOT / args.examples
    benchmarks_dir = ROOT / args.benchmarks
    docs_dir = ROOT / args.docs
    readme = ROOT / args.readme
    tutorials_dir = ROOT / "tutorials"

    print(f"TGraphX Technical Debt Audit — {package_dir}")
    print("=" * 60)

    all_findings: List[Finding] = []
    tools_available: Dict[str, bool] = {}

    # Check tools
    tools_available["ruff"] = _run(["ruff", "--version"])[0] == 0
    tools_available["mypy"] = _run(["python", "-m", "mypy", "--version"])[0] == 0
    tools_available["radon"] = _run(["python", "-c", "import radon"])[0] == 0
    tools_available["vulture"] = _run(["python", "-c", "import vulture"])[0] == 0
    tools_available["bandit"] = _run(["python", "-c", "import bandit"])[0] == 0

    print(f"Tools: " + " | ".join(f"{k}={'ok' if v else 'skip'}" for k, v in tools_available.items()))

    # 1. File metrics
    print("  [1/12] Collecting file metrics...")
    file_metrics, fm_findings = check_file_metrics(package_dir)
    all_findings.extend(fm_findings)

    # 2. Ruff
    print("  [2/12] Running ruff lint...")
    ruff_summary, ruff_findings = check_ruff(package_dir)
    all_findings.extend(ruff_findings)

    # 3. Mypy
    print("  [3/12] Running mypy type check...")
    mypy_summary, mypy_findings = check_mypy(package_dir)
    all_findings.extend(mypy_findings)

    # 4. Docs drift
    print("  [4/12] Checking docs/link drift...")
    docs_findings = check_docs_drift(readme, docs_dir, examples_dir, tutorials_dir)
    all_findings.extend(docs_findings)

    # 5. API drift
    print("  [5/12] Checking API drift...")
    api_result = check_api_drift(package_dir, tests_dir, examples_dir, docs_dir)
    api_findings, n_exports, n_unreferenced = api_result[0], api_result[1], api_result[2]
    all_findings.extend(api_findings)
    api_stats = (None, n_exports, n_unreferenced)

    # 6. Security
    print("  [6/12] Scanning security patterns...")
    sec_findings = check_security(package_dir)
    all_findings.extend(sec_findings)

    # 7. Performance guards
    print("  [7/12] Checking performance guards...")
    perf_findings = check_performance_guards(package_dir)
    all_findings.extend(perf_findings)

    # 8. Tensor debt
    print("  [8/12] Checking tensor-native debt...")
    tensor_findings = check_tensor_debt(tests_dir, package_dir)
    all_findings.extend(tensor_findings)

    # 9. Benchmark drift
    print("  [9/12] Checking benchmark drift...")
    bench_findings = check_benchmark_drift(benchmarks_dir)
    all_findings.extend(bench_findings)

    # 10. Dashboard drift
    print("  [10/12] Checking dashboard drift...")
    dash_result = check_dashboard_drift(package_dir, tests_dir)
    dash_findings, artifact_writers = dash_result
    all_findings.extend(dash_findings)

    # 11. AI smells
    print("  [11/12] Checking AI-code smells...")
    ai_findings = check_ai_smells(package_dir)
    all_findings.extend(ai_findings)

    # 12. Packaging
    print("  [12/12] Checking packaging/release...")
    pkg_result = check_packaging(ROOT)
    pkg_findings, pv, iv = pkg_result
    all_findings.extend(pkg_findings)

    # Build report
    print("\nBuilding report...")
    report = build_report(
        all_findings, file_metrics,
        ruff_summary, mypy_summary,
        tools_available, api_stats,
        artifact_writers,
        (pv,),
        out_dir,
    )

    # Write outputs
    json_path = write_json(report, out_dir)
    md_path = write_markdown(report, out_dir)
    print(f"  JSON:     {json_path}")
    print(f"  Markdown: {md_path}")

    # Write baseline if it doesn't exist yet
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump({
                "timestamp": report.timestamp,
                "version": report.version,
                "total_score": report.total_score,
                "blockers": report.blockers,
                "blocker_count": len(report.blockers),
                "high_count": len(report.high_debt),
                "medium_count": len(report.medium_debt),
            }, f, indent=2)
        print(f"  Baseline: {baseline_path} (created)")

    # Summary
    print(f"\n{'='*60}")
    print(f"Technical Debt Score: {report.total_score}/100 — {report.severity.upper()}")
    print(f"  Blockers: {len(report.blockers)}")
    print(f"  High:     {len(report.high_debt)}")
    print(f"  Medium:   {len(report.medium_debt)}")
    print(f"  Low:      {len(report.low_debt)}")
    print(f"{'='*60}")

    if args.ci:
        return run_ci_mode(report, baseline_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

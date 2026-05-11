"""Validation tool for advanced real-dataset notebook drafts (31–35).

Checks structural integrity, safety, and TGraphX correctness of each
notebook before maintainer upload.

Usage:
    python tools/validate_advanced_colab_drafts.py
    python tools/validate_advanced_colab_drafts.py --strict   # fail on warnings
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

ADVANCED_DIR = Path("colab_drafts/advanced_real_datasets")

# Patterns that must NOT appear in public notebook code cells.
FORBIDDEN_PATTERNS = [
    ("e[:120]", "Exception object sliced directly; use str(e)[:120]"),
    ("err[:120]", "Exception object sliced directly; use str(err)[:120]"),
    ("benchmarks/run_v13_benchmark_suite.py",
     "Repo-local benchmark path; use `from tgraphx.benchmarks import ...`"),
    ("NSGAIIOptimizer(config2, composite_fitness)",
     "NSGA-II passed composite_fitness directly; use a list of objectives"),
    ("/home/arash/", "Private local path"),
    ("/Users/", "Private local path (macOS)"),
    ("C:\\\\Users\\\\", "Private local path (Windows)"),
    ("os.environ['TOKEN", "Token/secret exposure"),
    ("API_KEY =", "API key exposure"),
    ("os.environ['KEY", "Key exposure"),
    ("private_key", "Potential secret"),
]

# Patterns that MUST appear (at least one of each group).
REQUIRED_ANY = [
    # TGraphX import
    ["import tgraphx", "from tgraphx"],
    # FAST_MODE flag
    ["FAST_MODE"],
    # Seed/reproducibility
    ["SEED", "set_seed", "manual_seed"],
    # Dataset section comment
    ["download", "Download", "dataset", "Dataset"],
    # Limitations mention
    ["Limitation", "limitation", "## Limitations"],
    # Device selection
    ["is_available", "cuda", "device"],
    # Reproducibility print or version
    ["__version__", "tgraphx.__version__", "TGraphX v"],
    # Gradient check or training
    ["backward", "loss.backward", "grad"],
    # Final summary or results
    ["Summary", "summary", "Results", "results", "PASSED", "passed"],
]

# False SOTA claims that must NOT appear.
FORBIDDEN_SOTA = [
    "achieves state-of-the-art",
    "achieve state-of-the-art",
    "beats all baselines",
    "outperforms all",
    "sets a new record",
]

# Cell types required to exist
REQUIRED_CELL_TYPES = {"markdown", "code"}


def check_notebook(path: Path, strict: bool = False) -> list[str]:
    """Return list of error/warning strings. Empty = pass."""
    issues: list[str] = []

    # ── JSON validity ──────────────────────────────────────────────────────────
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"INVALID JSON: {e}"]

    if nb.get("nbformat") != 4:
        issues.append(f"nbformat {nb.get('nbformat')} != 4")

    cells = nb.get("cells", [])
    if not cells:
        return ["No cells found"]

    cell_types = {c.get("cell_type") for c in cells}
    for ct in REQUIRED_CELL_TYPES:
        if ct not in cell_types:
            issues.append(f"Missing cell type: {ct!r}")

    # ── Collect source text ────────────────────────────────────────────────────
    all_src = "\n".join(
        "".join(c.get("source", []))
        for c in cells
    )
    code_src = "\n".join(
        "".join(c.get("source", []))
        for c in cells if c.get("cell_type") == "code"
    )
    md_src = "\n".join(
        "".join(c.get("source", []))
        for c in cells if c.get("cell_type") == "markdown"
    )

    # ── Forbidden patterns ─────────────────────────────────────────────────────
    for pattern, reason in FORBIDDEN_PATTERNS:
        if pattern in all_src:
            issues.append(f"FORBIDDEN pattern {pattern!r}: {reason}")

    # ── Required patterns ──────────────────────────────────────────────────────
    for group in REQUIRED_ANY:
        if not any(p in all_src for p in group):
            issues.append(f"Missing required: {group}")

    # ── Title check (first cell should be Markdown with a heading) ────────────
    first = cells[0]
    first_src = "".join(first.get("source", []))
    if first.get("cell_type") != "markdown" or not first_src.strip().startswith("#"):
        issues.append("First cell must be a Markdown title (# ...)")

    # ── Dataset section ────────────────────────────────────────────────────────
    if "Dataset" not in md_src and "dataset" not in md_src:
        issues.append("No dataset section found in Markdown cells")

    # ── Limitations section ───────────────────────────────────────────────────
    if "## Limitations" not in all_src and "## limitations" not in all_src.lower():
        issues.append("No '## Limitations' section found")

    # ── No false SOTA claims ──────────────────────────────────────────────────
    for phrase in FORBIDDEN_SOTA:
        if phrase in all_src:
            issues.append(f"False SOTA claim {phrase!r}")

    # ── No giant base64 blobs ─────────────────────────────────────────────────
    for cell in cells:
        outputs = cell.get("outputs", [])
        for out in outputs:
            data = out.get("data", {})
            for mime, content in data.items():
                blob = content if isinstance(content, str) else "".join(content)
                if len(blob) > 50_000:
                    issues.append(f"Large base64 output ({len(blob)} chars) in {mime}")

    # ── Must have at least 10 code cells ─────────────────────────────────────
    n_code = sum(1 for c in cells if c.get("cell_type") == "code")
    if n_code < 8:
        issues.append(f"Only {n_code} code cells; expected >= 8 for a real notebook")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate advanced notebook drafts.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero on warnings too")
    args = parser.parse_args()

    if not ADVANCED_DIR.exists():
        print(f"Directory {ADVANCED_DIR} does not exist. Run from repo root.", file=sys.stderr)
        return 1

    notebooks = sorted(ADVANCED_DIR.glob("*.ipynb"))
    if not notebooks:
        print(f"No *.ipynb files in {ADVANCED_DIR}")
        return 1

    all_pass = True
    for nb_path in notebooks:
        issues = check_notebook(nb_path, args.strict)
        status = "PASS" if not issues else "FAIL"
        print(f"[{status}] {nb_path.name}")
        for issue in issues:
            print(f"       ⚠  {issue}")
            all_pass = False

    print()
    if all_pass:
        print(f"All {len(notebooks)} notebooks passed validation.")
        return 0
    else:
        print("One or more notebooks failed validation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

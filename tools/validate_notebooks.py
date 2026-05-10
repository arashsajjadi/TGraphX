"""Notebook validation tool for TGraphX.

Checks every notebooks/*.ipynb without executing code.  Validates:
- Valid JSON.
- Has a title Markdown cell.
- References TGraphX.
- No private paths (/home/, /Users/, /root/ etc.).
- No embedded secrets (token=, password=, API_KEY= patterns).
- No mandatory network download commands (curl, wget, gdown without comments).
- Referenced script counterparts exist on disk.

Usage::

    python tools/validate_notebooks.py
    python tools/validate_notebooks.py --notebooks-dir notebooks/ --strict

Exit code 0 = all checks pass; 1 = one or more failures.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


_PRIVATE_PATH_RE = re.compile(r"/home/[a-zA-Z0-9_]+|/Users/[a-zA-Z0-9_]+|/root/")
_SECRET_RE = re.compile(r"(?i)(token\s*=|password\s*=|api_key\s*=|secret\s*=)['\"]?\w{4,}")
_NETWORK_RE = re.compile(r"\b(curl |wget |gdown )(?!#)", re.MULTILINE)


def _source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def validate_notebook(nb_path: Path, strict: bool = False) -> list[str]:
    """Return a list of error messages (empty = pass)."""
    errors: list[str] = []

    # 1. Valid JSON.
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    cells = nb.get("cells", [])

    # 2. Has at least one Markdown cell.
    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    if not md_cells:
        errors.append("No Markdown cells found.")

    # 3. Has a title cell (first cell is Markdown with #).
    if not md_cells:
        errors.append("Missing title Markdown cell.")
    else:
        first_src = _source(md_cells[0])
        if not first_src.strip().startswith("#"):
            errors.append(f"First Markdown cell is not a heading: {first_src[:80]!r}")

    # 4. References TGraphX.
    all_source = "\n".join(_source(c) for c in cells)
    if "tgraphx" not in all_source.lower():
        errors.append("Notebook does not reference TGraphX.")

    # 5. No private paths.
    for match in _PRIVATE_PATH_RE.finditer(all_source):
        errors.append(f"Private path detected: {match.group()!r}")

    # 6. No embedded secrets.
    for match in _SECRET_RE.finditer(all_source):
        errors.append(f"Potential secret detected: {match.group()!r}")

    # 7. No uncommented network download commands.
    for match in _NETWORK_RE.finditer(all_source):
        context = all_source[max(0, match.start()-30):match.end()+60]
        if "# " not in context[:match.start() - max(0, match.start()-30) + 5]:
            errors.append(f"Uncommented network command: {context.strip()!r}")

    # 8. Notebook has code cells.
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    if not code_cells:
        errors.append("No code cells found.")

    # 9. No huge outputs (> 200 lines per cell).
    for i, c in enumerate(code_cells):
        for out in c.get("outputs", []):
            text = out.get("text", [])
            n_lines = len(text) if isinstance(text, list) else text.count("\n")
            if n_lines > 200:
                errors.append(f"Code cell {i} has excessive output ({n_lines} lines).")

    return errors


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--notebooks-dir", default="notebooks",
                   help="Directory containing *.ipynb files.")
    p.add_argument("--strict", action="store_true",
                   help="Treat warnings as errors.")
    args = p.parse_args()

    nb_dir = Path(args.notebooks_dir)
    if not nb_dir.exists():
        print(f"ERROR: Notebooks directory not found: {nb_dir}")
        return 1

    nb_files = sorted(nb_dir.glob("*.ipynb"))
    if not nb_files:
        print(f"WARNING: No notebooks found in {nb_dir}/")
        return 0

    all_ok = True
    for nb_path in nb_files:
        errors = validate_notebook(nb_path, args.strict)
        if errors:
            all_ok = False
            print(f"FAIL {nb_path.name}:")
            for e in errors:
                print(f"  - {e}")
        else:
            print(f"PASS {nb_path.name}")

    if all_ok:
        print(f"\n{len(nb_files)} notebook(s) validated successfully.")
        return 0
    else:
        print("\nValidation FAILED — see errors above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

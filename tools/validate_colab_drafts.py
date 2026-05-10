"""Validate TGraphX Colab draft notebooks.

Usage::

    python tools/validate_colab_drafts.py [--dir colab_drafts]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_PRIVATE_RE = re.compile(r"/home/[a-zA-Z0-9_]+|/Users/[a-zA-Z0-9_]+")
_SECRET_RE  = re.compile(r"(?i)(token\s*=|password\s*=|api_key\s*=)['\"]?\w{8,}")
_FAKE_URL_RE = re.compile(r"colab\.research\.google\.com/drive/[A-Za-z0-9_-]{10,}")


def validate(nb_path: Path) -> list[str]:
    errs: list[str] = []
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    cells = nb.get("cells", [])
    src_all = "\n".join("".join(c.get("source", [])) for c in cells)

    if not cells:
        errs.append("No cells.")
    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    if not md_cells:
        errs.append("No Markdown cells.")
    elif not "".join(md_cells[0].get("source", [])).strip().startswith("#"):
        errs.append("First Markdown cell is not a heading.")
    if "tgraphx" not in src_all.lower():
        errs.append("Does not reference tgraphx.")
    for m in _PRIVATE_RE.finditer(src_all):
        errs.append(f"Private path: {m.group()!r}")
    for m in _SECRET_RE.finditer(src_all):
        errs.append(f"Potential secret: {m.group()!r}")
    for m in _FAKE_URL_RE.finditer(src_all):
        errs.append(f"Potential fake Colab URL: {m.group()!r}")
    return errs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="colab_drafts")
    args = p.parse_args()

    d = Path(args.dir)
    if not d.exists():
        print(f"WARNING: {d} does not exist. Generate with: python tools/generate_colab_drafts.py")
        return 0

    nbs = sorted(d.glob("*.ipynb"))
    if not nbs:
        print(f"WARNING: No .ipynb files in {d}/")
        return 0

    ok = True
    for nb in nbs:
        errs = validate(nb)
        if errs:
            ok = False
            print(f"FAIL {nb.name}:")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"PASS {nb.name}")

    if ok:
        print(f"\n{len(nbs)} draft(s) validated successfully.")
        return 0
    print("\nValidation FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

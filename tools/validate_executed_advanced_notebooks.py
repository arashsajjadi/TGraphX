"""Validate that advanced real-dataset notebooks have been executed.

Checks:
- execution_count is set on code cells
- at least one code cell has outputs
- no cell has output_type=='error'
- "Notebook" + ("passed" OR "completed") appears in final outputs
- no cell still has the placeholder execution_count=None for a non-empty cell
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def validate_one(nb_path: Path) -> tuple[bool, list[str]]:
    issues: list[str] = []
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, [f"Invalid JSON: {exc}"]

    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if not code_cells:
        return False, ["No code cells"]

    unexecuted = 0
    has_outputs = False
    error_cells = []
    last_output_text = ""
    for i, cell in enumerate(code_cells):
        src = "".join(cell.get("source", [])).strip()
        if not src:
            continue
        if cell.get("execution_count") is None:
            unexecuted += 1
        outputs = cell.get("outputs", [])
        if outputs:
            has_outputs = True
        for out in outputs:
            ot = out.get("output_type")
            if ot == "error":
                error_cells.append(i)
            elif ot == "stream":
                t = out.get("text", "")
                if isinstance(t, list):
                    t = "".join(t)
                last_output_text += t

    if unexecuted > 0:
        issues.append(f"{unexecuted} code cell(s) not executed (execution_count is None)")
    if not has_outputs:
        issues.append("No code cell has outputs")
    if error_cells:
        issues.append(f"Cells with error outputs: {error_cells}")
    if "passed all checks" not in last_output_text and "Notebook completed" not in last_output_text:
        issues.append(
            "Missing final completion message ('passed all checks' or 'Notebook completed')"
        )

    return len(issues) == 0, issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="colab_drafts/advanced_real_datasets")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Directory not found: {root}", file=sys.stderr)
        return 1

    notebooks = sorted(root.glob("3*.ipynb"))
    all_ok = True
    for nb_path in notebooks:
        ok, issues = validate_one(nb_path)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {nb_path.name}")
        for issue in issues:
            print(f"       ⚠  {issue}")
            all_ok = False
    print()
    if all_ok:
        print(f"All {len(notebooks)} notebooks executed and validated.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

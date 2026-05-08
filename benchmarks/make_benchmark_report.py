"""Combine multiple benchmark JSONs into a single Markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _format_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    if not rows:
        return "_(no rows)_\n"
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
    sep = "|-" + "-|-".join("-" * widths[c] for c in cols) + "-|"
    body = "\n".join(
        "| " + " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) + " |"
        for r in rows
    )
    return f"{header}\n{sep}\n{body}\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="*", type=Path,
                   help="One or more benchmark JSON files.")
    p.add_argument("--output", type=Path, default=None,
                   help="Write the markdown report to this path.")
    args = p.parse_args(argv)

    sections: List[str] = ["# TGraphX benchmark report\n"]
    for f in args.inputs:
        if not f.exists():
            sections.append(f"## {f.name}\n\n_File not found._\n")
            continue
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError as exc:
            sections.append(f"## {f.name}\n\nInvalid JSON: {exc}\n")
            continue
        sections.append(f"## {f.name}\n")
        version = data.get("version")
        small = data.get("small")
        sections.append(f"- TGraphX version: `{version}`\n- Small mode: `{small}`\n\n")
        rows = data.get("results")
        if isinstance(rows, list) and rows:
            cols = sorted({k for row in rows for k in row.keys()})
            sections.append(_format_table(rows, cols))
        else:
            # Some benchmarks store top-level dicts (tensor/flatten); render those.
            for k, v in data.items():
                if isinstance(v, dict):
                    sections.append(f"### {k}\n\n")
                    sections.append(_format_table([v], sorted(v.keys())))
        sections.append("\n")

    sections.append(
        "\n_TGraphX synthetic benchmarks are reproducibility / sanity tools. "
        "They do not constitute real-world performance claims._\n"
    )
    text = "".join(sections)
    if args.output:
        args.output.write_text(text)
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

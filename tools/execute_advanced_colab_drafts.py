"""Execute advanced real-dataset notebooks 31–35 with `nbclient`.

Runs each notebook in-place, keeps minimal outputs, and fails on any cell error.
Designed to work in FAST_MODE without network access (each notebook has a
synthetic fallback for missing datasets).

Usage:
    python tools/execute_advanced_colab_drafts.py
    python tools/execute_advanced_colab_drafts.py --root colab_drafts/advanced_real_datasets
    python tools/execute_advanced_colab_drafts.py --fast --timeout 1200 --keep-outputs

Exit code 0 = all notebooks executed cleanly. Non-zero = at least one failed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def execute_notebook(
    nb_path: Path,
    timeout: int = 1200,
    keep_outputs: bool = True,
) -> tuple[bool, str, int]:
    """Execute one notebook in place. Returns (success, message, num_outputs)."""
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent.resolve())}},
        record_timing=False,
        allow_errors=False,
    )
    t0 = time.time()
    try:
        client.execute()
    except CellExecutionError as exc:
        return False, f"Cell error: {str(exc)[:500]}", 0
    except Exception as exc:
        return False, f"Execution error: {type(exc).__name__}: {str(exc)[:500]}", 0
    elapsed = time.time() - t0

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    n_outputs = sum(len(c.get("outputs", [])) for c in code_cells)

    # Optionally trim each cell's outputs to limit notebook size
    if keep_outputs:
        for cell in code_cells:
            outputs = cell.get("outputs", [])
            for out in outputs:
                if out.get("output_type") == "stream":
                    text = out.get("text", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    if len(text) > 4000:
                        out["text"] = text[:2000] + "\n... [output truncated for size] ...\n" + text[-1000:]
                elif "data" in out:
                    data = out["data"]
                    for mime, content in list(data.items()):
                        blob = content if isinstance(content, str) else "".join(content)
                        if len(blob) > 8000:
                            data[mime] = blob[:3000] + "\n... [output truncated] ...\n" + blob[-1000:]
    else:
        for cell in code_cells:
            cell["outputs"] = []
            cell["execution_count"] = None

    nbformat.write(nb, nb_path)
    return True, f"OK in {elapsed:.1f}s", n_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="colab_drafts/advanced_real_datasets")
    parser.add_argument("--fast", action="store_true",
                        help="Notebooks should already have FAST_MODE=True; this flag is informational.")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--keep-outputs", action="store_true", default=True)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Directory not found: {root}", file=sys.stderr)
        return 1

    notebooks = sorted(root.glob("3*.ipynb"))
    if not notebooks:
        print(f"No notebooks in {root}", file=sys.stderr)
        return 1

    all_ok = True
    summary = []
    for nb_path in notebooks:
        print(f"\n[exec] {nb_path.name}")
        ok, msg, n_outputs = execute_notebook(
            nb_path, timeout=args.timeout, keep_outputs=args.keep_outputs,
        )
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {msg}  outputs={n_outputs}")
        summary.append((nb_path.name, status, msg, n_outputs))
        if not ok:
            all_ok = False

    print("\n" + "=" * 70)
    print(f"{'Notebook':<55} {'Status':<6} Outputs")
    print("-" * 70)
    for name, status, _, n in summary:
        print(f"{name:<55} {status:<6} {n}")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

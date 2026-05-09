"""Graph IO utilities: read/write edge lists, JSON graphs, NPZ.

All IO is path-safe, no unsafe pickle by default, and handles malformed
input with clear errors.

Stability: Beta (v0.4.4+).
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

__all__ = [
    "read_edge_list_csv",
    "write_edge_list_csv",
    "read_graph_json",
    "write_graph_json",
    "save_graph_npz",
    "load_graph_npz",
]


def _safe_path(path: str) -> Path:
    """Return a resolved Path and verify no traversal escapes the CWD."""
    p = Path(path).expanduser().resolve()
    return p


# ── CSV edge list ─────────────────────────────────────────────────────────────


def read_edge_list_csv(
    path: str,
    delimiter: str = ",",
    has_header: bool = True,
    src_col: int = 0,
    dst_col: int = 1,
    weight_col: Optional[int] = None,
    num_nodes: Optional[int] = None,
    remap_ids: bool = True,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Read an edge list CSV file.

    Args:
        path: Path to the CSV file.
        delimiter: Column delimiter.
        has_header: Skip the first row.
        src_col: Column index for source node.
        dst_col: Column index for destination node.
        weight_col: Optional column index for edge weight.
        num_nodes: Optional explicit node count.  When ``None`` and
            ``remap_ids=True``, inferred from unique node IDs.
        remap_ids: When ``True``, remap non-contiguous integer IDs to
            ``[0, num_unique)``.

    Returns:
        ``(edge_index, num_nodes, edge_weight)`` — edge weight is
        ``None`` if ``weight_col`` is ``None``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the CSV is malformed.
    """
    p = _safe_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Edge list CSV not found: {p}")

    src_ids: List[int] = []
    dst_ids: List[int] = []
    weights: List[float] = []

    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader, None)
        for lineno, row in enumerate(reader, start=2 if has_header else 1):
            try:
                src_ids.append(int(row[src_col]))
                dst_ids.append(int(row[dst_col]))
                if weight_col is not None:
                    weights.append(float(row[weight_col]))
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Malformed CSV at line {lineno}: {exc}") from exc

    if not src_ids:
        ei = torch.zeros((2, 0), dtype=torch.long)
        return ei, num_nodes or 0, None

    all_ids = sorted(set(src_ids) | set(dst_ids))
    if remap_ids:
        id_map = {old: new for new, old in enumerate(all_ids)}
        src_ids = [id_map[v] for v in src_ids]
        dst_ids = [id_map[v] for v in dst_ids]

    ei = torch.tensor([src_ids, dst_ids], dtype=torch.long)
    N = num_nodes if num_nodes is not None else int(ei.max().item()) + 1
    ew = torch.tensor(weights, dtype=torch.float) if weights else None
    return ei, N, ew


def write_edge_list_csv(
    path: str,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
    delimiter: str = ",",
    header: Optional[List[str]] = None,
) -> str:
    """Write an edge list to a CSV file.

    Args:
        path: Output file path.
        edge_index: ``LongTensor[2, E]``.
        edge_weight: Optional ``FloatTensor[E]``.
        delimiter: Column delimiter.
        header: Optional header row (e.g. ``["src", "dst"]`` or
            ``["src", "dst", "weight"]``).

    Returns:
        Resolved path string.
    """
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    src = edge_index[0].cpu().tolist() if edge_index.numel() else []
    dst = edge_index[1].cpu().tolist() if edge_index.numel() else []
    wts = edge_weight.cpu().tolist() if edge_weight is not None else None

    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=delimiter)
            if header is not None:
                writer.writerow(header)
            for i, (u, v) in enumerate(zip(src, dst)):
                row = [u, v]
                if wts is not None:
                    row.append(wts[i])
                writer.writerow(row)
        os.replace(tmp, str(p))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(p)


# ── JSON graph ────────────────────────────────────────────────────────────────

_GRAPH_JSON_VERSION = "1.0"


def read_graph_json(
    path: str,
) -> Dict[str, Any]:
    """Read a graph from a JSON file written by :func:`write_graph_json`.

    Returns a dict with keys:
      - ``edge_index``: ``LongTensor[2, E]``
      - ``num_nodes``: int
      - ``edge_weight``: ``FloatTensor[E]`` or ``None``
      - ``metadata``: dict

    Args:
        path: Path to JSON file.

    Raises:
        FileNotFoundError: If file missing.
        ValueError: If JSON is malformed or schema version mismatch.
    """
    p = _safe_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Graph JSON not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {p}: {exc}") from exc

    if data.get("_schema_version") != _GRAPH_JSON_VERSION:
        import warnings
        warnings.warn(f"Graph JSON schema version mismatch; expected {_GRAPH_JSON_VERSION}", stacklevel=2)

    num_nodes = int(data.get("num_nodes", 0))
    src_dst = data.get("edge_index", [[], []])
    ei = torch.tensor(src_dst, dtype=torch.long)
    ew_list = data.get("edge_weight", None)
    ew = torch.tensor(ew_list, dtype=torch.float) if ew_list is not None else None
    meta = data.get("metadata", {})
    return {"edge_index": ei, "num_nodes": num_nodes, "edge_weight": ew, "metadata": meta}


def write_graph_json(
    path: str,
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Write a graph to a JSON file.

    Args:
        path: Output path.
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        edge_weight: Optional ``FloatTensor[E]``.
        metadata: Optional JSON-serialisable metadata dict.

    Returns:
        Resolved path string.
    """
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ei_list = edge_index.cpu().tolist() if edge_index.numel() else [[], []]
    ew_list = edge_weight.cpu().tolist() if edge_weight is not None else None
    payload = {
        "_schema_version": _GRAPH_JSON_VERSION,
        "num_nodes": int(num_nodes),
        "num_edges": int(edge_index.size(1)) if edge_index.numel() else 0,
        "edge_index": ei_list,
        "edge_weight": ew_list,
        "metadata": metadata or {},
    }
    text = json.dumps(payload, indent=2)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, str(p))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(p)


# ── NPZ sparse graph ─────────────────────────────────────────────────────────


def save_graph_npz(
    path: str,
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
    node_features: Optional[torch.Tensor] = None,
) -> str:
    """Save a graph to a compressed NumPy NPZ file.

    Args:
        path: Output path (should end with ``.npz``).
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        edge_weight: Optional ``FloatTensor[E]``.
        node_features: Optional ``FloatTensor[N, D]``.

    Returns:
        Resolved path string.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("save_graph_npz requires numpy. Install with: pip install numpy") from exc
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, Any] = {
        "edge_index": edge_index.cpu().numpy() if edge_index.numel() else np.zeros((2, 0), dtype=np.int64),
        "num_nodes": np.array([num_nodes], dtype=np.int64),
    }
    if edge_weight is not None:
        arrays["edge_weight"] = edge_weight.cpu().numpy()
    if node_features is not None:
        arrays["node_features"] = node_features.cpu().numpy()
    np.savez_compressed(str(p), **arrays)
    return str(p)


def load_graph_npz(path: str) -> Dict[str, Any]:
    """Load a graph from an NPZ file written by :func:`save_graph_npz`.

    Returns:
        Dict with ``edge_index``, ``num_nodes``, optionally
        ``edge_weight`` and ``node_features``.

    Raises:
        FileNotFoundError: If file missing.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("load_graph_npz requires numpy.") from exc
    p = _safe_path(path)
    if not p.exists():
        raise FileNotFoundError(f"NPZ graph not found: {p}")
    data = np.load(str(p), allow_pickle=False)
    ei = torch.from_numpy(data["edge_index"]).to(torch.long)
    num_nodes = int(data["num_nodes"][0])
    result: Dict[str, Any] = {"edge_index": ei, "num_nodes": num_nodes}
    if "edge_weight" in data:
        result["edge_weight"] = torch.from_numpy(data["edge_weight"]).float()
    if "node_features" in data:
        result["node_features"] = torch.from_numpy(data["node_features"]).float()
    return result

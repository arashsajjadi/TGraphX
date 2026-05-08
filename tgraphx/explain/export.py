"""Explanation export helpers — explicit paths only, no hidden writes."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


def export_explanation_metadata(
    path: str | Path,
    *,
    method: str,
    target: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a JSON metadata record for an explanation artefact.

    Dashboard reads ``explanation_metadata.json`` when present.
    """
    payload: Dict[str, Any] = {"method": str(method), "target": target}
    if extra:
        payload.update({k: v for k, v in extra.items()
                        if isinstance(v, (str, int, float, bool, list, dict))})
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out


def export_edge_scores_csv(
    path: str | Path,
    edge_index: torch.Tensor,
    scores: torch.Tensor,
    *,
    method: Optional[str] = None,
    top_k: Optional[int] = None,
) -> Path:
    """Write per-edge scores to CSV: ``edge_id,src,dst,score`` (+ optional method).

    The file format matches what the TGraphX dashboard's explanation
    panel reads.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must be [2, E]; got {tuple(edge_index.shape)}"
        )
    if scores.dim() != 1 or scores.numel() != edge_index.size(1):
        raise ValueError(
            f"scores must be 1-D of length {edge_index.size(1)}; "
            f"got {tuple(scores.shape)}"
        )
    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()
    sc = scores.detach().cpu().tolist()
    rows = sorted(
        ({"edge_id": i, "src": src[i], "dst": dst[i], "score": float(sc[i]),
          **({"method": method} if method else {})}
         for i in range(len(sc))),
        key=lambda r: -abs(r["score"]),
    )
    if top_k is not None:
        rows = rows[: int(top_k)]
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["edge_id", "src", "dst", "score"]
    if method:
        fieldnames.append("method")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    return out


def export_patch_heatmap_json(
    path: str | Path,
    heatmap: torch.Tensor,
    *,
    grid_shape: Optional[Iterable[int]] = None,
    method: Optional[str] = None,
) -> Path:
    """Persist a 2-D heatmap as JSON ``{shape, values, method, grid_shape}``.

    Numbers are serialised as floats (no NumPy / no PyTorch dependency
    on the dashboard side).
    """
    if heatmap.dim() != 2:
        raise ValueError(
            f"heatmap must be 2-D; got {tuple(heatmap.shape)}"
        )
    payload: Dict[str, Any] = {
        "shape": list(heatmap.shape),
        "values": heatmap.detach().cpu().tolist(),
    }
    if method:
        payload["method"] = str(method)
    if grid_shape:
        payload["grid_shape"] = list(grid_shape)
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    return out

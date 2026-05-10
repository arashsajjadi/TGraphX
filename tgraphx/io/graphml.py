"""GraphML read/write for TGraphX Graph objects.

Implementation notes:

- Uses only the Python standard library (``xml.etree.ElementTree``).
- Round-trip preserves: node count, edge index, directedness, edge_weight,
  and 1-D node/edge feature tensors when ``include_tensor_features=True``.
- Multi-dimensional node/edge feature tensors are rejected by default with
  a clear error.  GraphML cannot express arbitrary tensor shapes safely.
- Node/edge labels (``y``) are written as a ``y`` data attribute on each
  node when ``include_labels=True``.
- Floats use full IEEE-754 representation; integers preserve type.

Limitations:
- ``graph_features``, ``graph_label``, ``edge_features`` with rank > 1, and
  metadata dicts are **not** serialized.  This is a documented limitation.
- Reading an unknown attribute key emits a warning, not an error.
"""
from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional, Union

import torch

from ..core.graph import Graph

__all__ = ["read_graphml", "write_graphml"]


_GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"


def _is_directed(graph: Graph) -> bool:
    """Best-effort directedness check: graph is undirected only if edge_index
    is symmetric.  When uncertain, default to directed (the safer choice)."""
    if graph.edge_index is None:
        return True
    if not graph.is_undirected():
        return True
    return False


def _tensor_to_str(t: torch.Tensor) -> str:
    """Serialize a 1-D tensor as a comma-separated string."""
    return ",".join(str(float(x)) for x in t.tolist())


def _str_to_tensor(s: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    parts = [p for p in s.split(",") if p.strip()]
    return torch.tensor([float(p) for p in parts], dtype=dtype)


def write_graphml(
    graph: Graph,
    path: Union[str, Path],
    *,
    include_labels: bool = True,
    include_tensor_features: bool = False,
) -> Path:
    """Write a TGraphX :class:`Graph` to a GraphML file.

    Args:
        graph: Graph to serialize.
        path: Output file path (``str`` or ``Path``).
        include_labels: When ``True`` (default), serialize ``node_labels``
            as ``<data key="y">`` per node and ``edge_labels`` as
            ``<data key="edge_label">`` per edge.
        include_tensor_features: When ``True``, serialize 1-D
            ``node_features``/``edge_features`` as comma-separated strings.
            For tensors with rank > 1 this still raises; multi-dim tensors
            cannot be safely round-tripped through GraphML.

    Returns:
        ``Path`` of the written file.

    Raises:
        ValueError: If ``include_tensor_features=True`` is requested but the
            graph has multi-dimensional node/edge feature tensors.
    """
    path = Path(path)
    if graph.node_features.dim() > 2 and include_tensor_features:
        raise ValueError(
            f"write_graphml: node_features has shape "
            f"{tuple(graph.node_features.shape)} (rank > 2 unflattenable). "
            f"GraphML cannot safely round-trip multi-dimensional tensor "
            f"features.  Either omit include_tensor_features=True or save "
            f"node features separately (e.g. torch.save)."
        )
    if (graph.edge_features is not None and graph.edge_features.dim() > 2
            and include_tensor_features):
        raise ValueError(
            f"write_graphml: edge_features has shape "
            f"{tuple(graph.edge_features.shape)} (rank > 2 unflattenable)."
        )

    directed = _is_directed(graph)
    N = graph.num_nodes

    root = ET.Element("graphml", attrib={"xmlns": _GRAPHML_NS})

    # ── key definitions ──
    keys: list = []
    if include_labels and graph.node_labels is not None:
        keys.append({
            "id": "y", "for": "node", "attr.name": "y",
            "attr.type": "double",
        })
    if include_labels and graph.edge_labels is not None:
        keys.append({
            "id": "edge_label", "for": "edge", "attr.name": "edge_label",
            "attr.type": "double",
        })
    if graph.edge_weight is not None:
        keys.append({
            "id": "weight", "for": "edge", "attr.name": "weight",
            "attr.type": "double",
        })
    if include_tensor_features and graph.node_features is not None:
        keys.append({
            "id": "node_features", "for": "node", "attr.name": "node_features",
            "attr.type": "string",
        })
    if include_tensor_features and graph.edge_features is not None:
        keys.append({
            "id": "edge_features", "for": "edge", "attr.name": "edge_features",
            "attr.type": "string",
        })

    for k in keys:
        ET.SubElement(root, "key", attrib=k)

    g_el = ET.SubElement(
        root, "graph",
        attrib={
            "id": "G",
            "edgedefault": "directed" if directed else "undirected",
        },
    )

    # ── nodes ──
    for i in range(N):
        n_el = ET.SubElement(g_el, "node", attrib={"id": f"n{i}"})
        if include_labels and graph.node_labels is not None:
            d = ET.SubElement(n_el, "data", attrib={"key": "y"})
            d.text = str(float(graph.node_labels[i].item()))
        if include_tensor_features and graph.node_features is not None:
            row = graph.node_features[i]
            if row.dim() > 1:
                row = row.flatten()
            d = ET.SubElement(n_el, "data", attrib={"key": "node_features"})
            d.text = _tensor_to_str(row)

    # ── edges ──
    if graph.edge_index is not None:
        ei = graph.edge_index
        E = ei.size(1)
        for j in range(E):
            src = int(ei[0, j].item())
            dst = int(ei[1, j].item())
            e_el = ET.SubElement(
                g_el, "edge",
                attrib={"id": f"e{j}", "source": f"n{src}", "target": f"n{dst}"},
            )
            if graph.edge_weight is not None:
                d = ET.SubElement(e_el, "data", attrib={"key": "weight"})
                d.text = str(float(graph.edge_weight[j].item()))
            if include_labels and graph.edge_labels is not None:
                d = ET.SubElement(e_el, "data", attrib={"key": "edge_label"})
                d.text = str(float(graph.edge_labels[j].item()))
            if include_tensor_features and graph.edge_features is not None:
                row = graph.edge_features[j]
                if row.dim() > 1:
                    row = row.flatten()
                d = ET.SubElement(e_el, "data", attrib={"key": "edge_features"})
                d.text = _tensor_to_str(row)

    # Pretty-print indent (ET.indent is Python 3.9+).
    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def _strip_ns(tag: str) -> str:
    """Strip the XML namespace prefix from an element tag."""
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def read_graphml(
    path: Union[str, Path],
    *,
    feature_dtype: torch.dtype = torch.float32,
) -> Graph:
    """Read a GraphML file into a TGraphX :class:`Graph`.

    Args:
        path: Input file path.
        feature_dtype: dtype for parsed node/edge feature tensors when
            present.  Default ``torch.float32``.

    Returns:
        :class:`Graph` with structure, optional ``edge_weight``, and any
        ``node_labels`` / ``edge_labels`` / 1-D feature tensors recovered
        from the file.  ``node_features`` defaults to a ``[N, 1]`` zero
        tensor when no per-node feature data is present (so the result is
        a valid ``Graph``).

    Raises:
        ValueError: If the file is not a valid GraphML or has no graph element.
    """
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()
    if _strip_ns(root.tag) != "graphml":
        raise ValueError(f"read_graphml: root tag is {root.tag!r}, expected 'graphml'")

    # Build key id → (target, name, dtype) map.
    key_map: dict = {}
    for el in root:
        if _strip_ns(el.tag) == "key":
            kid = el.attrib.get("id", "")
            kfor = el.attrib.get("for", "node")
            ktype = el.attrib.get("attr.type", "string")
            key_map[kid] = {"for": kfor, "type": ktype}

    g_el = None
    for el in root:
        if _strip_ns(el.tag) == "graph":
            g_el = el
            break
    if g_el is None:
        raise ValueError("read_graphml: no <graph> element found")

    directed = g_el.attrib.get("edgedefault", "directed") == "directed"

    # ── nodes ──
    node_ids: list = []
    node_y: list = []
    node_feats_list: list = []

    for el in g_el:
        if _strip_ns(el.tag) != "node":
            continue
        nid = el.attrib.get("id", "")
        node_ids.append(nid)
        y_val = None
        nf_val = None
        for d in el:
            if _strip_ns(d.tag) != "data":
                continue
            key = d.attrib.get("key", "")
            text = (d.text or "").strip()
            if key == "y" and text:
                y_val = float(text)
            elif key == "node_features" and text:
                nf_val = text
        node_y.append(y_val)
        node_feats_list.append(nf_val)

    N = len(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # ── edges ──
    src_list: list = []
    dst_list: list = []
    weight_list: list = []
    edge_label_list: list = []
    edge_feats_list: list = []

    for el in g_el:
        if _strip_ns(el.tag) != "edge":
            continue
        s = el.attrib.get("source", "")
        t = el.attrib.get("target", "")
        if s not in id_to_idx or t not in id_to_idx:
            warnings.warn(f"Edge references unknown node {s}/{t}; skipped",
                          stacklevel=2)
            continue
        src_list.append(id_to_idx[s])
        dst_list.append(id_to_idx[t])
        w_val: Optional[float] = None
        el_val: Optional[float] = None
        ef_val: Optional[str] = None
        for d in el:
            if _strip_ns(d.tag) != "data":
                continue
            key = d.attrib.get("key", "")
            text = (d.text or "").strip()
            if key == "weight" and text:
                w_val = float(text)
            elif key == "edge_label" and text:
                el_val = float(text)
            elif key == "edge_features" and text:
                ef_val = text
        weight_list.append(w_val)
        edge_label_list.append(el_val)
        edge_feats_list.append(ef_val)

    # ── tensors ──
    if any(v is not None for v in node_feats_list):
        # Use the first non-None as a shape reference.
        rows = []
        for s in node_feats_list:
            if s is None:
                # Zero-fill missing rows with the same length.
                rows.append(None)
            else:
                rows.append(_str_to_tensor(s, dtype=feature_dtype))
        first_len = next((r.numel() for r in rows if r is not None), 1)
        node_features = torch.stack([
            r if r is not None else torch.zeros(first_len, dtype=feature_dtype)
            for r in rows
        ])
    else:
        node_features = torch.zeros(N, 1, dtype=feature_dtype)

    edge_index = (
        torch.tensor([src_list, dst_list], dtype=torch.long)
        if src_list else None
    )

    edge_weight = None
    if any(v is not None for v in weight_list) and src_list:
        edge_weight = torch.tensor(
            [w if w is not None else 0.0 for w in weight_list],
            dtype=torch.float32,
        )

    node_labels = None
    if any(v is not None for v in node_y):
        # Use long if all integer-valued; otherwise float.
        if all(v is None or float(v).is_integer() for v in node_y):
            node_labels = torch.tensor(
                [int(v) if v is not None else 0 for v in node_y],
                dtype=torch.long,
            )
        else:
            node_labels = torch.tensor(
                [float(v) if v is not None else 0.0 for v in node_y],
                dtype=torch.float32,
            )

    edge_labels = None
    if any(v is not None for v in edge_label_list) and src_list:
        if all(v is None or float(v).is_integer() for v in edge_label_list):
            edge_labels = torch.tensor(
                [int(v) if v is not None else 0 for v in edge_label_list],
                dtype=torch.long,
            )
        else:
            edge_labels = torch.tensor(
                [float(v) if v is not None else 0.0 for v in edge_label_list],
                dtype=torch.float32,
            )

    edge_features = None
    if any(v is not None for v in edge_feats_list) and src_list:
        rows = []
        for s in edge_feats_list:
            if s is None:
                rows.append(None)
            else:
                rows.append(_str_to_tensor(s, dtype=feature_dtype))
        first_len = next((r.numel() for r in rows if r is not None), 1)
        edge_features = torch.stack([
            r if r is not None else torch.zeros(first_len, dtype=feature_dtype)
            for r in rows
        ])

    metadata = {"graphml_directed": directed, "graphml_path": str(path)}

    return Graph(
        node_features=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_features=edge_features,
        node_labels=node_labels,
        edge_labels=edge_labels,
        metadata=metadata,
    )

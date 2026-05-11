"""Object-aware describe/summary utility for TGraphX."""
from __future__ import annotations

from typing import Any

import torch


def _tensor_info(t: torch.Tensor) -> dict:
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "requires_grad": bool(t.requires_grad) if t.is_floating_point() else False,
    }


def describe(obj: Any) -> dict:
    """Return a JSON-serializable summary dict for any TGraphX object.

    Supports:
      - :class:`tgraphx.Graph`
      - :class:`tgraphx.GraphBatch`
      - :class:`tgraphx.KnowledgeGraph`
      - dataset wrapper objects with a ``.summary()`` method
      - ``torch.nn.Module`` (returns parameter and module summary)
    """
    # Graph or GraphBatch
    if hasattr(obj, "node_features") and hasattr(obj, "edge_index"):
        return _describe_graph(obj)
    # KnowledgeGraph
    if hasattr(obj, "triples") and hasattr(obj, "num_entities"):
        return _describe_kg(obj)
    # Dataset
    if hasattr(obj, "summary") and callable(obj.summary):
        try:
            s = obj.summary()
            if isinstance(s, dict):
                return s
        except Exception:
            pass
    if hasattr(obj, "_build_metadata") and callable(obj._build_metadata):
        try:
            meta = obj._build_metadata()
            return meta.__dict__ if hasattr(meta, "__dict__") else {"meta": str(meta)}
        except Exception:
            pass
    # Module
    if isinstance(obj, torch.nn.Module):
        return _describe_module(obj)
    return {"type": type(obj).__name__, "repr": repr(obj)[:200]}


def _describe_graph(g: Any) -> dict:
    out = {"type": type(g).__name__}
    nf = g.node_features
    out["num_nodes"] = int(nf.size(0))
    out["node_features"] = _tensor_info(nf)
    ei = getattr(g, "edge_index", None)
    if ei is not None:
        out["num_edges"] = int(ei.shape[1])
        out["edge_index"] = _tensor_info(ei)
    ea = getattr(g, "edge_features", None)
    if ea is not None:
        out["edge_attr"] = _tensor_info(ea)
    y = getattr(g, "node_labels", None)
    if y is None:
        y = getattr(g, "y", None)
    if y is not None and isinstance(y, torch.Tensor):
        out["y"] = _tensor_info(y)
    gl = getattr(g, "graph_label", None)
    if gl is not None and isinstance(gl, torch.Tensor):
        out["graph_label"] = _tensor_info(gl)
    for m in ("train_mask", "val_mask", "test_mask"):
        v = getattr(g, m, None)
        if v is not None and isinstance(v, torch.Tensor):
            out[m] = {"shape": list(v.shape), "true_count": int(v.sum().item())}
    if hasattr(g, "num_graphs"):
        out["num_graphs"] = int(g.num_graphs)
    return out


def _describe_kg(kg: Any) -> dict:
    out = {
        "type": type(kg).__name__,
        "num_entities": int(kg.num_entities),
        "num_relations": int(kg.num_relations),
        "num_triples": int(kg.num_triples) if hasattr(kg, "num_triples") else int(kg.triples.size(0)),
    }
    if hasattr(kg, "entity_features") and kg.entity_features:
        out["entity_features"] = {
            k: _tensor_info(v) for k, v in kg.entity_features.items()
        }
    if hasattr(kg, "relation_features") and kg.relation_features:
        out["relation_features"] = {
            k: _tensor_info(v) for k, v in kg.relation_features.items()
        }
    return out


def _describe_module(m: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return {
        "type": type(m).__name__,
        "parameters_total": total,
        "parameters_trainable": trainable,
        "submodules": [type(c).__name__ for _, c in m.named_children()],
    }


def summary(obj: Any) -> dict:
    """Alias for :func:`describe`."""
    return describe(obj)

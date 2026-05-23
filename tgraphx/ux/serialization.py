"""Native save/load for TGraphX Graph and KnowledgeGraph objects.

Uses ``torch.save`` so tensor features of any rank are preserved exactly.
Files are NOT GraphML — they can carry ``[N, C, H, W]`` tensors that GraphML
cannot represent.

File layout (single torch.save bundle):
    {
      "tgraphx_format": "tgx-v1",
      "type": "Graph" | "KnowledgeGraph",
      "payload": {... CPU tensors ...},
      "metadata": {"tgraphx_version": ..., ...},
    }
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Union

import torch


_FORMAT = "tgx-v1"


class TGraphXSerializationError(IOError):
    """Raised when a TGraphX bundle is corrupted, version-mismatched, or wrong type."""


def _graph_to_payload(graph: Any) -> dict:
    payload: dict[str, Any] = {}
    if hasattr(graph, "node_features") and graph.node_features is not None:
        payload["node_features"] = graph.node_features.detach().cpu()
    if hasattr(graph, "edge_index") and graph.edge_index is not None:
        payload["edge_index"] = graph.edge_index.detach().cpu()
    if hasattr(graph, "edge_weight") and graph.edge_weight is not None:
        payload["edge_weight"] = graph.edge_weight.detach().cpu()
    if getattr(graph, "edge_features", None) is not None:
        payload["edge_features"] = graph.edge_features.detach().cpu()
    if getattr(graph, "node_labels", None) is not None:
        payload["node_labels"] = graph.node_labels.detach().cpu()
    if getattr(graph, "edge_labels", None) is not None:
        payload["edge_labels"] = graph.edge_labels.detach().cpu()
    if getattr(graph, "graph_label", None) is not None:
        payload["graph_label"] = graph.graph_label.detach().cpu()
    if getattr(graph, "graph_features", None) is not None:
        payload["graph_features"] = graph.graph_features.detach().cpu()
    for m in ("train_mask", "val_mask", "test_mask"):
        v = getattr(graph, m, None)
        if v is not None and isinstance(v, torch.Tensor):
            payload[m] = v.detach().cpu()
    md = getattr(graph, "metadata", None)
    payload["metadata"] = md if md is not None else {}
    return payload


def _payload_to_graph(payload: dict) -> Any:
    from ..core.graph import Graph
    kwargs: dict[str, Any] = {}
    kwargs["node_features"] = payload["node_features"]
    if "edge_index" in payload:
        kwargs["edge_index"] = payload["edge_index"]
    if "edge_weight" in payload:
        kwargs["edge_weight"] = payload["edge_weight"]
    if "edge_features" in payload:
        kwargs["edge_features"] = payload["edge_features"]
    if "node_labels" in payload:
        kwargs["node_labels"] = payload["node_labels"]
    if "edge_labels" in payload:
        kwargs["edge_labels"] = payload["edge_labels"]
    if "graph_label" in payload:
        kwargs["graph_label"] = payload["graph_label"]
    if "graph_features" in payload:
        kwargs["graph_features"] = payload["graph_features"]
    for m in ("train_mask", "val_mask", "test_mask"):
        if m in payload:
            kwargs[m] = payload[m]
    if payload.get("metadata"):
        kwargs["metadata"] = payload["metadata"]
    return Graph(**kwargs)


def _kg_to_payload(kg: Any) -> dict:
    p: dict[str, Any] = {
        "triples": kg.triples.detach().cpu(),
        "num_entities": int(kg.num_entities),
        "num_relations": int(kg.num_relations),
    }
    if getattr(kg, "entity_features", None):
        p["entity_features"] = {
            k: v.detach().cpu() for k, v in kg.entity_features.items()
        }
    if getattr(kg, "relation_features", None):
        p["relation_features"] = {
            k: v.detach().cpu() for k, v in kg.relation_features.items()
        }
    if getattr(kg, "triple_features", None):
        p["triple_features"] = {
            k: v.detach().cpu() for k, v in kg.triple_features.items()
        }
    if getattr(kg, "metadata", None):
        p["metadata"] = kg.metadata
    return p


def _payload_to_kg(payload: dict) -> Any:
    from ..kg import KnowledgeGraph
    kwargs: dict[str, Any] = {
        "triples": payload["triples"],
        "num_entities": payload["num_entities"],
        "num_relations": payload["num_relations"],
    }
    for k in ("entity_features", "relation_features", "triple_features"):
        if k in payload:
            kwargs[k] = payload[k]
    if payload.get("metadata"):
        kwargs["metadata"] = payload["metadata"]
    return KnowledgeGraph(**kwargs)


def save_tgraphx(obj: Any, path: Union[str, Path]) -> str:
    """Save a Graph or KnowledgeGraph to a `.tgx` file (single torch.save bundle).

    All tensors are moved to CPU before serialization so the file is portable.
    """
    from .. import __version__
    if hasattr(obj, "node_features") and hasattr(obj, "edge_index"):
        payload = _graph_to_payload(obj)
        kind = "Graph"
    elif hasattr(obj, "triples") and hasattr(obj, "num_entities"):
        payload = _kg_to_payload(obj)
        kind = "KnowledgeGraph"
    else:
        raise TypeError(
            f"save_tgraphx: unsupported object type {type(obj).__name__}. "
            "Supported: tgraphx.Graph, tgraphx.KnowledgeGraph."
        )
    bundle = {
        "tgraphx_format": _FORMAT,
        "tgraphx_version": __version__,
        "type": kind,
        "payload": payload,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, str(path))
    return str(path)


def load_tgraphx(
    path: Union[str, Path],
    map_location: str = "cpu",
    *,
    trust_source: bool = True,
) -> Any:
    """Load a Graph or KnowledgeGraph from a `.tgx` file.

    Args:
        path: Path to a file written by :func:`save_tgraphx`.
        map_location: torch.load map_location; defaults to "cpu" for portability.
        trust_source: ``.tgx`` bundles are pickled via :func:`torch.save` and
            can contain arbitrary Python objects in their ``metadata`` field.
            Loading such files executes pickle code, so only load bundles from
            trusted sources. Defaults to ``True`` to preserve backwards
            compatibility with existing bundles; set ``False`` to refuse the
            load and require manual review.

    Returns:
        A :class:`Graph` or :class:`KnowledgeGraph` instance.

    Security:
        See :func:`torch.load` docs — TGraphX bundles include user metadata,
        so ``weights_only=True`` would reject them. Only load ``.tgx`` files
        that you produced yourself or received from a trusted collaborator.
    """
    if not os.path.exists(str(path)):
        raise FileNotFoundError(f"TGraphX bundle not found: {path}")
    if not trust_source:
        raise TGraphXSerializationError(
            f"Refusing to load {path}: trust_source=False. "
            "TGraphX bundles can execute arbitrary pickle code; "
            "review the file and re-call with trust_source=True if safe."
        )
    try:
        bundle = torch.load(str(path), map_location=map_location, weights_only=False)
    except Exception as exc:
        raise TGraphXSerializationError(
            f"Failed to load {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(bundle, dict) or bundle.get("tgraphx_format") != _FORMAT:
        raise TGraphXSerializationError(
            f"File {path} is not a TGraphX bundle (expected tgraphx_format={_FORMAT!r})."
        )
    kind = bundle["type"]
    payload = bundle["payload"]
    if kind == "Graph":
        return _payload_to_graph(payload)
    if kind == "KnowledgeGraph":
        return _payload_to_kg(payload)
    raise TGraphXSerializationError(f"Unknown bundle type: {kind!r}")


# Friendly aliases
save = save_tgraphx
load = load_tgraphx

"""Dataset metadata records.

A :class:`DatasetMetadata` is a small JSON-friendly object that travels
alongside every TGraphX dataset.  It stores upstream credit, license,
task, split sizes, version, and anything else the dataset wants to
carry.

The container is intentionally minimal — heavy data lives in tensors
inside the dataset; this object is for the things you would otherwise
write down in a README.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DatasetMetadata:
    """JSON-serialisable record of dataset provenance and shape.

    All fields are optional so different dataset families (synthetic,
    folder-backed, third-party-wrapped) can populate the parts that are
    meaningful for them without lying about what they don't know.

    Attributes:
        name: Canonical dataset name (e.g. ``"synthetic:patch_graph"``).
        source: Free-form human description of where the data came from.
        source_url: Authoritative URL for the dataset (if any).
        upstream_library: Name of an upstream library this dataset
            depends on for download/loading (``"torchvision"``,
            ``"torch_geometric"``, ``"dgl"``, ``"ogb"``, or ``None``).
        citation: Citation text or BibTeX key the user is expected to
            cite when publishing results.
        license: SPDX-style license identifier or free-form description.
        task: Task family string (``"graph_classification"``,
            ``"node_classification"``, ``"link_prediction"``,
            ``"graph_regression"``, ``"node_regression"``,
            ``"edge_prediction"``, ``"hetero_*"``, ``"temporal_*"``).
        graph_type: ``"homogeneous"``, ``"heterogeneous"``,
            or ``"temporal"``.
        num_graphs: Number of graphs (single-graph datasets use ``1``).
        num_nodes: Total number of nodes (sum across graphs if
            applicable).
        num_edges: Total number of edges.
        num_classes: Number of label classes for classification tasks.
        splits: Mapping of split name to size, e.g.
            ``{"train": 1208, "val": 500, "test": 1000}``.
        version: Dataset version string (free-form).
        processed_at: ISO-8601 timestamp of the last processing run.
        extra: Any additional JSON-friendly metadata.
    """

    name: str
    source: Optional[str] = None
    source_url: Optional[str] = None
    upstream_library: Optional[str] = None
    citation: Optional[str] = None
    license: Optional[str] = None
    task: Optional[str] = None
    graph_type: Optional[str] = None
    num_graphs: Optional[int] = None
    num_nodes: Optional[int] = None
    num_edges: Optional[int] = None
    num_classes: Optional[int] = None
    splits: Optional[Dict[str, int]] = None
    version: Optional[str] = None
    processed_at: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # ── (de)serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        # Defensive: ignore unknown keys rather than crashing on schema
        # drift across releases.
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        cleaned = {k: v for k, v in data.items() if k in known}
        return cls(**cleaned)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "DatasetMetadata":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    # ── Pretty-printing ──────────────────────────────────────────────────────

    def short_summary(self) -> str:
        bits: List[str] = [self.name]
        if self.task:
            bits.append(self.task)
        if self.graph_type:
            bits.append(self.graph_type)
        if self.num_graphs is not None:
            bits.append(f"{self.num_graphs} graphs")
        if self.num_nodes is not None:
            bits.append(f"{self.num_nodes} nodes")
        if self.num_edges is not None:
            bits.append(f"{self.num_edges} edges")
        if self.num_classes is not None:
            bits.append(f"{self.num_classes} classes")
        return " · ".join(bits)

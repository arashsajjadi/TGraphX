"""Optional OGB-backed dataset adapter.

OGB is **optional** — install with ``pip install ogb``.  This module
imports OGB lazily inside ``__init__`` and exposes:

* :class:`OGBDatasetAdapter` — generic wrapper.
* :class:`OGBNodePropertyDatasetAdapter` /
  :class:`OGBLinkPropertyDatasetAdapter` /
  :class:`OGBGraphPropertyDatasetAdapter` — task-specific wrappers
  that pick the right OGB class.
* :class:`OGBEvaluatorWrapper` — small wrapper around
  ``ogb.nodeproppred.Evaluator`` / ``LinkPropPredDataset`` /
  ``GraphPropPredDataset`` evaluators.

The wrappers preserve the OGB-supplied split indices via
:meth:`get_idx_split` and expose :meth:`get_evaluator` so user code
can run the official protocol.  TGraphX makes **no SOTA / leaderboard
claims**; the evaluator simply forwards predictions to OGB.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .base import ExternalDatasetAdapter, TargetTransformFn, TransformFn
from .converters import from_pyg_data, ogb_item_to_graph
from .errors import OptionalDependencyError
from .metadata import DatasetMetadata

__all__ = [
    "OGBDatasetAdapter",
    "OGBNodePropertyDatasetAdapter",
    "OGBLinkPropertyDatasetAdapter",
    "OGBGraphPropertyDatasetAdapter",
    "OGBEvaluatorWrapper",
]


_OGB_HINT = "OGB-backed datasets require ogb. Install with `pip install ogb`."


def _require_ogb():
    try:
        import ogb  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("ogb", _OGB_HINT) from exc


# ── Generic adapter ──────────────────────────────────────────────────────────


_OGB_TASK_MODULES = {
    "node": ("ogb.nodeproppred", "PygNodePropPredDataset", "Evaluator"),
    "link": ("ogb.linkproppred", "PygLinkPropPredDataset", "Evaluator"),
    "graph": ("ogb.graphproppred", "PygGraphPropPredDataset", "Evaluator"),
}


def _detect_task(name: str) -> str:
    if name.startswith("ogbn-"):
        return "node"
    if name.startswith("ogbl-"):
        return "link"
    if name.startswith("ogbg-"):
        return "graph"
    raise ValueError(
        f"Cannot infer OGB task type from {name!r}; expected ogbn-*, "
        f"ogbl-*, or ogbg-*"
    )


class OGBDatasetAdapter(ExternalDatasetAdapter):
    """Generic OGB wrapper.

    Args:
        name: OGB dataset name (``"ogbn-arxiv"``, ``"ogbl-collab"``,
            ``"ogbg-molhiv"``, ...).
        root: Cache directory.  Passed to OGB as ``root``.
        download: Forwarded informationally.  OGB downloads on its own
            when ``root`` is empty; pass an existing root to avoid
            network access.
        task_type: Override task auto-detection.
        transform: TGraphX-side transform applied after conversion.
    """

    upstream_library = "ogb"
    upstream_install_hint = _OGB_HINT

    def __init__(
        self,
        name: str,
        root: Optional[str | Path] = None,
        download: bool = False,  # noqa: ARG002
        task_type: Optional[str] = None,
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
    ) -> None:
        _require_ogb()
        self._task = task_type or _detect_task(name)
        module_name, dataset_cls_name, evaluator_cls_name = _OGB_TASK_MODULES[self._task]

        import importlib
        module = importlib.import_module(module_name)
        DatasetCls = getattr(module, dataset_cls_name)
        EvaluatorCls = getattr(module, evaluator_cls_name)
        self._evaluator_cls = EvaluatorCls

        from .cache import resolve_dataset_root
        upstream_root = resolve_dataset_root(root, f"ogb/{name}")
        upstream_root.mkdir(parents=True, exist_ok=True)

        self.name = name
        self._upstream = DatasetCls(name=name, root=str(upstream_root))
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )

    def __len__(self) -> int:
        try:
            return len(self._upstream)
        except TypeError:
            return 1

    def get(self, idx: int):
        item = self._upstream[idx]
        return ogb_item_to_graph(item, task_type=self._task)

    # ── Splits / evaluator passthroughs ──────────────────────────────────────

    def get_idx_split(self) -> Dict[str, Any]:
        return self._upstream.get_idx_split()

    def get_evaluator(self) -> "OGBEvaluatorWrapper":
        return OGBEvaluatorWrapper(self._evaluator_cls, self.name)

    def _build_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name=f"ogb:{self.name}",
            source="ogb",
            upstream_library="ogb",
            source_url="https://ogb.stanford.edu/",
            citation="Hu et al., NeurIPS 2020; see https://ogb.stanford.edu/",
            license="See upstream dataset card.",
            task={"node": "node_classification",
                  "link": "link_prediction",
                  "graph": "graph_classification"}[self._task],
            graph_type="homogeneous",
            num_graphs=len(self._upstream),
        )


# ── Task-specific subclasses (no extra logic; just for registry clarity) ─────


class OGBNodePropertyDatasetAdapter(OGBDatasetAdapter):
    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name=name, task_type="node", **kwargs)


class OGBLinkPropertyDatasetAdapter(OGBDatasetAdapter):
    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name=name, task_type="link", **kwargs)


class OGBGraphPropertyDatasetAdapter(OGBDatasetAdapter):
    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name=name, task_type="graph", **kwargs)


# ── Evaluator wrapper ────────────────────────────────────────────────────────


class OGBEvaluatorWrapper:
    """Thin wrapper around an OGB ``Evaluator`` for a given dataset name.

    The wrapper exists so that:

    * users get a stable import path
      (``tgraphx.datasets.OGBEvaluatorWrapper``);
    * tests can monkey-patch ``_evaluator_cls`` without importing OGB.
    """

    def __init__(self, evaluator_cls: Any, name: str) -> None:
        self._evaluator = evaluator_cls(name=name)
        self.name = name

    def eval(self, input_dict: Dict[str, Any]) -> Dict[str, float]:
        return self._evaluator.eval(input_dict)

    @property
    def expected_input_format(self) -> Optional[str]:
        return getattr(self._evaluator, "expected_input_format", None)

    @property
    def expected_output_format(self) -> Optional[str]:
        return getattr(self._evaluator, "expected_output_format", None)

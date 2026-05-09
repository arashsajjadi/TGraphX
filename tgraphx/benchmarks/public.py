"""OGB / TGB evaluator wrappers.

Thin wrappers around the official ``ogb`` and ``tgb`` evaluators.  All
wrappers follow the same contract:

* Importing this module never triggers a network call; the optional
  dependency is imported lazily on first use.
* Datasets are not downloaded automatically.  Callers must construct
  the dataset themselves (with explicit ``download=True``) and pass
  predictions/labels to ``.eval(...)``.
* When the optional dependency is missing, instantiation raises
  :class:`OptionalDependencyError` with a clear install hint, and
  ``is_available`` is ``False``.

Returned metrics are JSON-serialisable dicts with the metric name(s)
that the upstream evaluator computes (e.g. ``"acc"``, ``"rocauc"``,
``"mrr"``, ``"hits@10"``).

Stability: Beta.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from ..datasets.errors import OptionalDependencyError

__all__ = [
    "OGBNodeEvaluator",
    "OGBLinkEvaluator",
    "OGBGraphEvaluator",
    "TGBLinkEvaluator",
]


_OGB_HINT = "OGB evaluator requires `pip install ogb`."
_TGB_HINT = "TGB evaluator requires `pip install py-tgb` (or `pip install tgb`)."


def _require_ogb() -> Any:
    try:
        import ogb  # noqa: F401
        return ogb
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("ogb", _OGB_HINT) from exc


def _require_tgb() -> Any:
    try:
        import tgb  # noqa: F401
        return tgb
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("tgb", _TGB_HINT) from exc


def _is_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# ── OGB node ─────────────────────────────────────────────────────────────────


class OGBNodeEvaluator:
    """Wraps ``ogb.nodeproppred.Evaluator``.

    Args:
        name: OGB dataset name (e.g. ``"ogbn-arxiv"``).
        download: Forwarded as a no-op flag for caller intent — the
            wrapper does NOT download.  Callers must download via
            ``PygNodePropPredDataset(name=..., root=...)`` themselves.

    Stability: Beta.
    """

    is_available: bool = _is_available("ogb")

    def __init__(self, name: str, download: bool = False) -> None:
        _require_ogb()
        from ogb.nodeproppred import Evaluator
        self.name = name
        self._eval = Evaluator(name=name)
        self.expected_metric = self._eval.expected_input_format

    def eval(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        """Return ``{metric_name: value}`` from the OGB evaluator."""
        result = self._eval.eval({
            "y_true": y_true.detach().cpu().numpy(),
            "y_pred": y_pred.detach().cpu().numpy(),
        })
        return {k: float(v) for k, v in result.items()}


# ── OGB link ─────────────────────────────────────────────────────────────────


class OGBLinkEvaluator:
    """Wraps ``ogb.linkproppred.Evaluator``.

    Stability: Beta.
    """

    is_available: bool = _is_available("ogb")

    def __init__(self, name: str, download: bool = False) -> None:
        _require_ogb()
        from ogb.linkproppred import Evaluator
        self.name = name
        self._eval = Evaluator(name=name)

    def eval(
        self,
        y_pred_pos: torch.Tensor,
        y_pred_neg: torch.Tensor,
    ) -> Dict[str, float]:
        """Return ``{metric_name: value}``.

        OGB's link evaluator typically returns ``hits@K`` or ``mrr``.
        """
        result = self._eval.eval({
            "y_pred_pos": y_pred_pos.detach().cpu().numpy(),
            "y_pred_neg": y_pred_neg.detach().cpu().numpy(),
        })
        out: Dict[str, float] = {}
        for k, v in result.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                # Some entries are arrays (per-positive-edge scores);
                # collapse to mean for a single scalar summary.
                out[str(k)] = float(torch.tensor(v).float().mean().item())
        return out


# ── OGB graph ────────────────────────────────────────────────────────────────


class OGBGraphEvaluator:
    """Wraps ``ogb.graphproppred.Evaluator``.

    Stability: Beta.
    """

    is_available: bool = _is_available("ogb")

    def __init__(self, name: str, download: bool = False) -> None:
        _require_ogb()
        from ogb.graphproppred import Evaluator
        self.name = name
        self._eval = Evaluator(name=name)

    def eval(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        result = self._eval.eval({
            "y_true": y_true.detach().cpu().numpy(),
            "y_pred": y_pred.detach().cpu().numpy(),
        })
        return {k: float(v) for k, v in result.items()}


# ── TGB link ─────────────────────────────────────────────────────────────────


class TGBLinkEvaluator:
    """Optional wrapper around the TGB temporal link evaluator.

    The TGB API differs across releases; this wrapper accepts a
    pre-built evaluator instance for caller flexibility, and exposes a
    consistent ``eval(...)`` method that returns a JSON-serialisable
    dict.  When ``tgb`` is missing, instantiation raises a clear
    ``OptionalDependencyError``.

    Stability: Beta.
    """

    is_available: bool = _is_available("tgb")

    def __init__(self, evaluator: Optional[Any] = None, name: Optional[str] = None) -> None:
        _require_tgb()
        if evaluator is None and name is None:
            raise ValueError(
                "TGBLinkEvaluator requires either an explicit `evaluator` instance "
                "or a `name` so the wrapper can construct one."
            )
        if evaluator is None:
            # Best-effort: try the most common upstream pattern.
            try:
                from tgb.linkproppred.evaluate import Evaluator  # type: ignore
                evaluator = Evaluator(name=name)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to construct a tgb evaluator automatically. "
                    "Pass `evaluator=...` explicitly."
                ) from exc
        self._eval = evaluator
        self.name = name

    def eval(self, **kwargs) -> Dict[str, float]:
        """Forward ``**kwargs`` to the upstream evaluator.

        The expected keys depend on the TGB version; common patterns
        include ``y_pred_pos`` / ``y_pred_neg`` arrays or a dict
        containing the model predictions for the test edges.

        Returns a JSON-serialisable dict.
        """
        result = self._eval.eval(**kwargs)
        if isinstance(result, dict):
            return {k: float(v) if not hasattr(v, "__iter__") else
                    float(torch.tensor(list(v)).float().mean().item())
                    for k, v in result.items()}
        return {"value": float(result)}

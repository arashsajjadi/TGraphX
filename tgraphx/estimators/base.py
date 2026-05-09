"""Base estimator and pipeline classes."""
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Tuple

__all__ = ["BaseGraphEstimator", "GraphPipeline"]


class BaseGraphEstimator:
    """Minimal sklearn-like estimator base.

    Subclasses must implement :meth:`fit` and :meth:`predict`, and may
    override :meth:`predict_proba`, :meth:`score`, :meth:`transform`.
    Construction parameters are introspected via ``__init__`` to make
    :meth:`get_params` / :meth:`set_params` work.

    Stability: Beta.
    """

    # Subclasses populate ``_param_names`` from ``__init__`` signature.
    _is_fitted: bool = False

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return the constructor parameters as a dict."""
        names = self._init_param_names()
        out: Dict[str, Any] = {}
        for name in names:
            if hasattr(self, name):
                out[name] = getattr(self, name)
        return out

    def set_params(self, **params: Any) -> "BaseGraphEstimator":
        """Set constructor parameters in-place; returns ``self``."""
        names = set(self._init_param_names())
        for k, v in params.items():
            if k not in names:
                raise ValueError(
                    f"Invalid parameter {k!r} for {type(self).__name__}; "
                    f"valid: {sorted(names)}"
                )
            setattr(self, k, v)
        return self

    def _init_param_names(self) -> List[str]:
        sig = inspect.signature(type(self).__init__)
        return [p for p, prm in sig.parameters.items()
                if p != "self" and prm.kind != inspect.Parameter.VAR_KEYWORD]

    # ── Abstract surface ────────────────────────────────────────────────────

    def fit(self, graph: Any, y: Any = None) -> "BaseGraphEstimator":
        raise NotImplementedError

    def predict(self, graph: Any) -> Any:
        raise NotImplementedError

    def predict_proba(self, graph: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement predict_proba"
        )

    def score(self, graph: Any, y: Any) -> float:
        """Default: classification accuracy on graph-level y."""
        import torch
        preds = self.predict(graph)
        if isinstance(preds, torch.Tensor) and isinstance(y, torch.Tensor):
            if preds.dim() == 2:
                preds = preds.argmax(dim=-1)
            return float((preds == y).float().mean().item())
        raise NotImplementedError("Override score() for non-tensor predictions")

    def transform(self, graph: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement transform"
        )

    def fit_transform(self, graph: Any, y: Any = None) -> Any:
        return self.fit(graph, y).transform(graph)


class GraphPipeline:
    """Chain a sequence of named estimators.

    Args:
        steps: List of ``(name, estimator)`` tuples.  All but the last
            step must implement :meth:`transform`; the last step must
            implement :meth:`fit` and :meth:`predict`.

    Stability: Beta.
    """

    def __init__(self, steps: List[Tuple[str, BaseGraphEstimator]]) -> None:
        names = [s[0] for s in steps]
        if len(names) != len(set(names)):
            raise ValueError("step names must be unique")
        self.steps = list(steps)

    @property
    def named_steps(self) -> Dict[str, BaseGraphEstimator]:
        return {n: e for n, e in self.steps}

    def fit(self, graph: Any, y: Any = None) -> "GraphPipeline":
        cur = graph
        for name, est in self.steps[:-1]:
            cur = est.fit_transform(cur, y)
        last_name, last_est = self.steps[-1]
        last_est.fit(cur, y)
        return self

    def predict(self, graph: Any) -> Any:
        cur = graph
        for name, est in self.steps[:-1]:
            cur = est.transform(cur)
        return self.steps[-1][1].predict(cur)

    def score(self, graph: Any, y: Any) -> float:
        cur = graph
        for name, est in self.steps[:-1]:
            cur = est.transform(cur)
        return self.steps[-1][1].score(cur, y)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, est in self.steps:
            out[name] = est
            if deep:
                for k, v in est.get_params(deep=True).items():
                    out[f"{name}__{k}"] = v
        return out

    def set_params(self, **params: Any) -> "GraphPipeline":
        named = self.named_steps
        for k, v in params.items():
            if "__" in k:
                step, sub = k.split("__", 1)
                if step not in named:
                    raise ValueError(f"unknown step {step!r}")
                named[step].set_params(**{sub: v})
            else:
                if k not in named:
                    raise ValueError(f"unknown step {k!r}")
                # Replace estimator outright.
                self.steps = [(name, v if name == k else est) for name, est in self.steps]
        return self

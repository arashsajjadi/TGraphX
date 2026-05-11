"""Data-leakage guards for graph / KG splits."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class LeakageError(ValueError):
    """Raised when train/val/test splits leak labels."""


def check_leakage(
    graph: Any = None,
    train_mask: Optional[torch.Tensor] = None,
    val_mask: Optional[torch.Tensor] = None,
    test_mask: Optional[torch.Tensor] = None,
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """Verify that node-classification masks do not overlap.

    Args:
        graph: Optional :class:`Graph` from which masks are pulled if not given.
        train_mask, val_mask, test_mask: Boolean tensors of shape ``[N]``.
        strict: If True, raise :class:`LeakageError` on any overlap.

    Returns:
        Dict with ``ok`` (bool), ``overlaps`` (dict), and per-mask counts.
    """
    if graph is not None:
        if train_mask is None:
            train_mask = getattr(graph, "train_mask", None)
        if val_mask is None:
            val_mask = getattr(graph, "val_mask", None)
        if test_mask is None:
            test_mask = getattr(graph, "test_mask", None)

    issues = []
    overlaps = {}

    def _overlap(a: torch.Tensor, b: torch.Tensor) -> int:
        return int((a & b).sum().item())

    if train_mask is not None and val_mask is not None:
        o = _overlap(train_mask.bool(), val_mask.bool())
        overlaps["train_val"] = o
        if o > 0:
            issues.append(f"train_mask and val_mask overlap on {o} nodes")
    if train_mask is not None and test_mask is not None:
        o = _overlap(train_mask.bool(), test_mask.bool())
        overlaps["train_test"] = o
        if o > 0:
            issues.append(f"train_mask and test_mask overlap on {o} nodes")
    if val_mask is not None and test_mask is not None:
        o = _overlap(val_mask.bool(), test_mask.bool())
        overlaps["val_test"] = o
        if o > 0:
            issues.append(f"val_mask and test_mask overlap on {o} nodes")

    counts = {}
    for name, m in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        if m is not None:
            counts[name] = int(m.bool().sum().item())

    result = {"ok": len(issues) == 0, "overlaps": overlaps, "counts": counts,
              "issues": issues}
    if strict and not result["ok"]:
        raise LeakageError("; ".join(issues))
    return result


def leakage_report(
    train_triples: Optional[torch.Tensor] = None,
    val_triples: Optional[torch.Tensor] = None,
    test_triples: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Report KG triple overlap between train/val/test splits.

    KG entity IDs are normally shared across splits (transductive); this
    function specifically detects TRIPLE-level overlap, which is a real leak.
    """
    issues = []
    counts = {}

    def _to_set(t: Optional[torch.Tensor]):
        if t is None:
            return None
        rows = t.tolist() if isinstance(t, torch.Tensor) else list(t)
        return {tuple(r) for r in rows}

    s_train = _to_set(train_triples)
    s_val = _to_set(val_triples)
    s_test = _to_set(test_triples)
    if s_train is not None:
        counts["train"] = len(s_train)
    if s_val is not None:
        counts["val"] = len(s_val)
    if s_test is not None:
        counts["test"] = len(s_test)

    overlaps = {}
    if s_train and s_val:
        ov = len(s_train & s_val)
        overlaps["train_val_triple_overlap"] = ov
        if ov > 0:
            issues.append(f"train/val share {ov} identical triples — leak!")
    if s_train and s_test:
        ov = len(s_train & s_test)
        overlaps["train_test_triple_overlap"] = ov
        if ov > 0:
            issues.append(f"train/test share {ov} identical triples — leak!")
    if s_val and s_test:
        ov = len(s_val & s_test)
        overlaps["val_test_triple_overlap"] = ov
        if ov > 0:
            issues.append(f"val/test share {ov} identical triples — leak!")

    return {
        "ok": len(issues) == 0,
        "counts": counts,
        "overlaps": overlaps,
        "issues": issues,
        "policy": "Transductive KG link prediction: entity IDs may be shared, but triples must be disjoint.",
    }


def validate_split_policy(
    obj: Any,
    setting: str = "transductive",
) -> Dict[str, Any]:
    """Document/validate the split policy of a dataset or graph.

    Args:
        obj: A graph, KG, or dataset.
        setting: ``"transductive"``, ``"inductive"``, or ``"semi_supervised"``.

    Returns:
        Dict with the declared policy and any detected violations.
    """
    valid = ("transductive", "inductive", "semi_supervised")
    if setting not in valid:
        raise ValueError(
            f"setting must be one of {valid}; got {setting!r}"
        )
    return {
        "policy": setting,
        "obj_type": type(obj).__name__,
        "notes": (
            "Transductive: nodes/entities shared across splits; only labels withheld. "
            "Inductive: completely disjoint subgraphs/graphs per split. "
            "Semi-supervised: subset of nodes labeled; others unlabeled but observed."
        ),
    }

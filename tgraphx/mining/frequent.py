"""Frequent pattern primitives for graph collections.

These are exploratory mining primitives — **not** a full gSpan
implementation.  They count label/degree/neighbourhood frequencies
across a collection of graphs.

Stability: Experimental (v0.3.2+).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

import torch

__all__ = [
    "frequent_node_labels",
    "frequent_degree_bins",
    "support_count",
]


def _to_fraction(support, n_graphs: int) -> int:
    """Convert support (fraction or int) to an absolute count."""
    if isinstance(support, float):
        if not 0.0 < support <= 1.0:
            raise ValueError(f"support fraction must be in (0, 1]; got {support}")
        return max(1, int(math.ceil(support * n_graphs)))
    return int(support)


def frequent_node_labels(
    label_lists: List[List[int]],
    min_support: Union[int, float] = 1,
) -> Dict[int, int]:
    """Find labels that appear in at least ``min_support`` graphs.

    Args:
        label_lists: One list of integer node labels per graph.
        min_support: Minimum number of graphs containing the label.
            Pass a float in (0, 1] for a fraction.

    Returns:
        ``{label: support_count}`` for frequent labels, sorted by count desc.
    """
    import math  # noqa (used in _to_fraction)
    G = len(label_lists)
    if G == 0:
        return {}
    thresh = _to_fraction(min_support, G)
    label_counts: Dict[int, int] = {}
    for labels in label_lists:
        for lab in set(labels):
            label_counts[lab] = label_counts.get(lab, 0) + 1
    return dict(sorted(
        {k: v for k, v in label_counts.items() if v >= thresh}.items(),
        key=lambda x: -x[1],
    ))


def frequent_degree_bins(
    graphs: list,
    bins: Optional[List[int]] = None,
    min_support: Union[int, float] = 1,
) -> Dict[str, int]:
    """Find degree bins that are frequent across graphs.

    Args:
        graphs: List of graph dicts (``edge_index``, ``num_nodes``) or
            TGraphX :class:`~tgraphx.Graph` objects.
        bins: Sorted list of degree thresholds, e.g. ``[0, 1, 5, 10]``
            creates bins ``[0,1)``, ``[1,5)``, ``[5,10)``, ``>=10``.
            When ``None``, uses ``[0, 1, 5, 10, 50]``.
        min_support: Minimum number of graphs with a node in the bin.

    Returns:
        ``{"bin_label": count}`` for frequent bins.
    """
    import math
    if bins is None:
        bins = [0, 1, 5, 10, 50]
    G = len(graphs)
    if G == 0:
        return {}
    thresh = _to_fraction(min_support, G)

    def _bin_label(deg: int, bins: List[int]) -> str:
        for i in range(len(bins) - 1):
            if bins[i] <= deg < bins[i + 1]:
                return f"[{bins[i]},{bins[i+1]})"
        return f">={bins[-1]}"

    bin_counts: Dict[str, int] = {}
    for g in graphs:
        if hasattr(g, "edge_index"):
            ei = g.edge_index
            n = g.num_nodes
        else:
            ei = g["edge_index"]
            n = g["num_nodes"]
        deg = torch.zeros(n, dtype=torch.long)
        if ei.numel() and n > 0:
            ones = torch.ones(ei.size(1), dtype=torch.long)
            deg.scatter_add_(0, ei[0].to(torch.long), ones)
        present_bins = set(_bin_label(int(d), bins) for d in deg.tolist())
        for b in present_bins:
            bin_counts[b] = bin_counts.get(b, 0) + 1

    return dict(sorted(
        {k: v for k, v in bin_counts.items() if v >= thresh}.items(),
        key=lambda x: -x[1],
    ))


def support_count(
    pattern_labels: List[int],
    label_lists: List[List[int]],
) -> int:
    """Count how many graphs in the collection contain all pattern labels.

    Args:
        pattern_labels: Labels that must all be present in a graph.
        label_lists: Per-graph node label lists.

    Returns:
        Integer count.
    """
    pattern_set = set(pattern_labels)
    return sum(1 for labels in label_lists if pattern_set <= set(labels))



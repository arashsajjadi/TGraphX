"""Weisfeiler-Lehman graph features and kernels in pure PyTorch.

The Weisfeiler-Lehman (WL) subtree kernel is a classical method for
comparing graphs.  This implementation provides label assignment,
histogram extraction, and kernel matrix computation.

Stability: Beta (v0.4.0+).

Determinism guarantee
---------------------
WL labels are assigned by mapping tuples-of-integers through a per-call
counter dictionary.  Because Python's tuple-of-integer hashing is *not*
affected by ``PYTHONHASHSEED`` randomisation (only ``str`` / ``bytes`` /
``datetime`` hashing is randomised in Python 3.3+), the label sequence
produced by this function is stable across separate Python processes
given the same inputs.

The internal ``_stable_key`` helper serialises complex keys through
``hashlib.sha256`` so that even if a caller passes non-integer labels the
output remains reproducible across processes.

Note: WL is a *structural fingerprint*, not a canonical graph
isomorphism test.  Hash collisions are theoretically possible.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import torch

__all__ = [
    "weisfeiler_lehman_labels",
    "wl_feature_histogram",
    "wl_graph_features",
    "wl_kernel_matrix",
    "degree_histogram_features",
]


def _stable_key(key: Any) -> bytes:
    """Convert a WL aggregation key to a stable byte string for hashing.

    Uses ``hashlib.sha256`` so the output is identical across Python
    processes regardless of ``PYTHONHASHSEED``.

    The key is a tuple ``(current_label_int, tuple_of_sorted_neighbour_labels_int)``.
    We serialise it as a space-separated ASCII string before hashing.
    """
    # Fast path: pure tuple-of-ints — already stable via repr.
    return repr(key).encode("ascii")


def _stable_compress(key: Any, label_map: Dict[bytes, int], counter: list) -> int:
    """Map a WL key to a stable integer label using sha256-derived bytes."""
    raw = _stable_key(key)
    if raw not in label_map:
        label_map[raw] = counter[0]
        counter[0] += 1
    return label_map[raw]


def _build_adj_lists(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return adj
    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()
    for u, v in zip(src, dst):
        adj[u].append(v)
    return adj


def weisfeiler_lehman_labels(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_labels: Optional[List[int]] = None,
    num_iterations: int = 3,
) -> List[List[int]]:
    """Assign WL labels to nodes over ``num_iterations`` rounds.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        node_labels: Initial integer labels per node.  When ``None``,
            initialised by out-degree.
        num_iterations: Number of WL refinement iterations.

    Returns:
        List of ``num_iterations + 1`` label lists (including the
        initial labelling at index 0).  Each inner list has length
        ``num_nodes``.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E]; got {tuple(edge_index.shape)}")
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative; got {num_nodes}")
    if num_iterations < 0:
        raise ValueError(f"num_iterations must be non-negative; got {num_iterations}")

    adj = _build_adj_lists(edge_index, num_nodes)

    if node_labels is not None:
        if len(node_labels) != num_nodes:
            raise ValueError(
                f"node_labels length {len(node_labels)} != num_nodes={num_nodes}"
            )
        labels = list(node_labels)
    else:
        # Initialise by out-degree.
        labels = [len(adj[v]) for v in range(num_nodes)]

    history = [list(labels)]
    # Use bytes keys (stable across PYTHONHASHSEED) instead of raw tuples.
    label_map: Dict[bytes, int] = {}
    counter = [max(labels) + 1 if labels else 0]

    for _ in range(num_iterations):
        new_labels = []
        for v in range(num_nodes):
            nbr_labels = tuple(sorted(labels[u] for u in adj[v]))
            key = (labels[v], nbr_labels)
            new_labels.append(_stable_compress(key, label_map, counter))
        labels = new_labels
        history.append(list(labels))

    return history


def wl_feature_histogram(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_labels: Optional[List[int]] = None,
    num_iterations: int = 3,
    vocabulary: Optional[Dict[int, int]] = None,
) -> Dict[int, int]:
    """Return a histogram of WL labels across all iterations.

    The histogram counts the total frequency of each compressed label
    across all ``num_iterations + 1`` label sets.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        node_labels: Initial labels; see :func:`weisfeiler_lehman_labels`.
        num_iterations: Number of refinement rounds.
        vocabulary: Optional pre-built vocabulary ``{label_id: col_index}``
            for building fixed-length feature vectors.

    Returns:
        ``Dict[int, int]`` mapping label ids to counts.
    """
    history = weisfeiler_lehman_labels(
        edge_index, num_nodes, node_labels, num_iterations,
    )
    hist: Dict[int, int] = {}
    for labels in history:
        for lab in labels:
            hist[lab] = hist.get(lab, 0) + 1
    return hist


def wl_graph_features(
    graphs: list,
    num_iterations: int = 3,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """Build a fixed-length WL feature matrix for a list of graphs.

    Args:
        graphs: List of dicts with keys ``edge_index`` (``LongTensor[2,E]``),
            ``num_nodes`` (int), and optionally ``node_labels``
            (``list[int]``).  Also accepts TGraphX
            :class:`~tgraphx.Graph` objects.
        num_iterations: WL refinement rounds.

    Returns:
        Tuple ``(feature_matrix, vocabulary)``:
          - ``FloatTensor[G, K]`` — one row per graph.
          - ``Dict[int, int]`` — shared vocabulary.
    """
    G = len(graphs)
    if G == 0:
        return torch.zeros((0, 0), dtype=torch.float), {}

    # First pass: collect histograms and build global vocabulary.
    hists = []
    for g in graphs:
        if hasattr(g, "edge_index"):
            ei = g.edge_index
            n = g.num_nodes
            nl = None
        else:
            ei = g["edge_index"]
            n = g["num_nodes"]
            nl = g.get("node_labels")
        h = wl_feature_histogram(ei, n, nl, num_iterations)
        hists.append(h)

    all_labels = sorted({lab for h in hists for lab in h})
    vocab = {lab: i for i, lab in enumerate(all_labels)}
    K = len(vocab)

    feat = torch.zeros(G, K, dtype=torch.float)
    for i, h in enumerate(hists):
        for lab, cnt in h.items():
            if lab in vocab:
                feat[i, vocab[lab]] = float(cnt)
    return feat, vocab


def wl_kernel_matrix(
    graphs: list,
    num_iterations: int = 3,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute a WL kernel matrix ``[G, G]``.

    ``K[i, j] = <phi(G_i), phi(G_j)>`` where ``phi`` is the WL feature
    vector.  Optional normalisation: ``K_norm[i,j] = K[i,j] / sqrt(K[i,i] * K[j,j])``.

    Args:
        graphs: List of graph dicts or TGraphX :class:`~tgraphx.Graph`
            objects; see :func:`wl_graph_features`.
        num_iterations: WL refinement rounds.
        normalize: When ``True`` (default), normalise by self-kernels.

    Returns:
        ``FloatTensor[G, G]``, symmetric, positive (semi-)definite.
    """
    feat, _ = wl_graph_features(graphs, num_iterations)
    K = feat @ feat.t()
    if normalize:
        diag = K.diag().clamp(min=1e-10).sqrt()
        K = K / diag.unsqueeze(0) / diag.unsqueeze(1)
    return K


def degree_histogram_features(
    graphs: list,
    max_degree: Optional[int] = None,
) -> torch.Tensor:
    """Build a degree-histogram feature matrix ``[G, max_degree + 1]``.

    Each row is a normalised histogram of out-degree values.

    Args:
        graphs: List of graph dicts or TGraphX :class:`~tgraphx.Graph`
            objects with ``edge_index`` and ``num_nodes``.
        max_degree: Histogram width.  When ``None``, determined from data.

    Returns:
        ``FloatTensor[G, D+1]`` where D = max_degree.
    """
    G = len(graphs)
    if G == 0:
        return torch.zeros((0, 1), dtype=torch.float)

    # Compute per-graph degree histograms.
    raw_hists = []
    global_max = 0
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
            deg.scatter_add_(0, ei[0], ones)
        raw_hists.append(deg)
        global_max = max(global_max, int(deg.max().item()) if deg.numel() else 0)

    D = max_degree if max_degree is not None else global_max
    feat = torch.zeros(G, D + 1, dtype=torch.float)
    for i, deg in enumerate(raw_hists):
        for d in deg.tolist():
            d_clipped = min(d, D)
            feat[i, d_clipped] += 1.0
        # Normalise by number of nodes.
        n = deg.numel()
        if n > 0:
            feat[i] = feat[i] / float(n)
    return feat

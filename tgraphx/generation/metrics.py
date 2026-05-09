"""Graph generation quality metrics.

All distances are:
    - Symmetric: d(A, B) = d(B, A)
    - Non-negative: d(A, B) >= 0
    - Zero iff distributions are identical

Mathematical definitions are given in each function's docstring.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Dict, List, Optional

import torch

from .data_model import GeneratedGraph

__all__ = [
    "graph_wl_hash",
    "validity_score",
    "uniqueness_score",
    "novelty_score",
    "diversity_score",
    "degree_distribution_distance",
    "motif_distribution_distance",
    "spectral_distance",
    "mmd_degree",
    "mmd_clustering",
    "constraint_satisfaction_rate",
]

_N_WARN_SPECTRAL = 500


def graph_wl_hash(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_labels: Optional[torch.Tensor] = None,
    iterations: int = 3,
) -> str:
    """Weisfeiler-Lehman hash for approximate graph isomorphism.

    Uses 1-WL colour refinement:
        label_new(v) = hash(label(v), sorted(label(u) for u in N(v)))

    Args:
        edge_index: LongTensor [2, E].
        num_nodes: Number of nodes.
        node_labels: Optional LongTensor [N] initial labels.
        iterations: Number of refinement iterations.

    Returns:
        String hash of the final label multiset.
    """
    if num_nodes == 0:
        return "empty"

    # Initial labels
    if node_labels is not None:
        labels: List[int] = node_labels.tolist()
    else:
        labels = [0] * num_nodes

    if edge_index.numel() == 0:
        label_multiset = tuple(sorted(labels))
        return str(hash(label_multiset))

    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()

    # Build adjacency lists
    adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    for s, d in zip(src_list, dst_list):
        adj[s].append(d)

    for _ in range(iterations):
        new_labels: List[int] = []
        for v in range(num_nodes):
            neighbor_labels = sorted(labels[u] for u in adj[v])
            combined = (labels[v], tuple(neighbor_labels))
            new_labels.append(hash(combined))
        labels = new_labels

    label_multiset = tuple(sorted(labels))
    return str(hash(label_multiset))


def validity_score(
    graphs: List[GeneratedGraph],
    constraint_fn: Callable[[GeneratedGraph], bool],
) -> float:
    """Fraction of graphs satisfying the constraint function.

    validity = |{G in graphs : constraint_fn(G) = True}| / |graphs|

    Args:
        graphs: List of GeneratedGraphs.
        constraint_fn: Callable returning True if graph is valid.

    Returns:
        Float in [0, 1].
    """
    if not graphs:
        return 0.0
    return sum(1 for g in graphs if constraint_fn(g)) / len(graphs)


def uniqueness_score(graphs: List[GeneratedGraph]) -> float:
    """Fraction of unique graphs by WL hash.

    uniqueness = |unique WL hashes| / |graphs|

    Args:
        graphs: List of GeneratedGraphs.

    Returns:
        Float in [0, 1].
    """
    if not graphs:
        return 0.0
    hashes = [
        graph_wl_hash(g.edge_index, g.num_nodes, g.node_types)
        for g in graphs
    ]
    return len(set(hashes)) / len(hashes)


def novelty_score(
    generated_graphs: List[GeneratedGraph],
    reference_graphs: List[GeneratedGraph],
) -> float:
    """Fraction of generated graphs not seen in the reference set (WL-hash).

    novelty = |{G in generated : WL(G) not in {WL(R) for R in reference}}| / |generated|

    Args:
        generated_graphs: Generated graphs.
        reference_graphs: Training/reference graphs.

    Returns:
        Float in [0, 1].
    """
    if not generated_graphs:
        return 0.0
    ref_hashes = set(
        graph_wl_hash(g.edge_index, g.num_nodes, g.node_types)
        for g in reference_graphs
    )
    novel = sum(
        1 for g in generated_graphs
        if graph_wl_hash(g.edge_index, g.num_nodes, g.node_types) not in ref_hashes
    )
    return novel / len(generated_graphs)


def diversity_score(graphs: List[GeneratedGraph]) -> float:
    """Average pairwise WL hash distance (Jaccard complement).

    For two graphs with label sets H_a and H_b:
        d(a, b) = 1 - |H_a ∩ H_b| / |H_a ∪ H_b|   (Jaccard distance on multisets)

    diversity = mean_{a≠b} d(a, b)

    This implementation uses the WL label multisets (not just the final hash)
    for a richer comparison.

    Args:
        graphs: List of GeneratedGraphs.

    Returns:
        Float in [0, 1]. Returns 0.0 for fewer than 2 graphs.
    """
    if len(graphs) < 2:
        return 0.0

    def _label_multiset(g: GeneratedGraph) -> Counter:
        labels = g.node_types.tolist() if g.node_types is not None else [0] * g.num_nodes
        if g.num_edges == 0:
            return Counter(labels)
        src_list = g.edge_index[0].tolist()
        dst_list = g.edge_index[1].tolist()
        adj: Dict[int, List[int]] = {i: [] for i in range(g.num_nodes)}
        for s, d in zip(src_list, dst_list):
            adj[s].append(d)
        refined = list(labels)
        for _ in range(3):
            new_labels = []
            for v in range(g.num_nodes):
                nb = sorted(refined[u] for u in adj[v])
                new_labels.append(hash((refined[v], tuple(nb))) % (2**20))
            refined = new_labels
        return Counter(refined)

    multisets = [_label_multiset(g) for g in graphs]
    total_dist = 0.0
    count = 0
    n = len(graphs)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = multisets[i], multisets[j]
            # Jaccard distance on multisets
            inter = sum((a & b).values())
            union = sum((a | b).values())
            if union == 0:
                d = 0.0
            else:
                d = 1.0 - inter / union
            total_dist += d
            count += 1

    return total_dist / count if count > 0 else 0.0


def _degree_hist(graphs: List[GeneratedGraph], max_degree: Optional[int] = None) -> torch.Tensor:
    """Compute averaged degree histogram over a list of graphs."""
    all_degrees: List[float] = []
    for g in graphs:
        if g.num_edges == 0:
            continue
        deg = torch.zeros(g.num_nodes, dtype=torch.float)
        deg.scatter_add_(0, g.edge_index[1], torch.ones(g.num_edges, dtype=torch.float))
        all_degrees.extend(deg.tolist())

    if not all_degrees:
        return torch.zeros(1)

    md = max_degree if max_degree is not None else int(max(all_degrees)) + 1
    hist = torch.zeros(md + 1)
    for d in all_degrees:
        idx = min(int(d), md)
        hist[idx] += 1.0
    hist = hist / hist.sum().clamp(min=1e-9)
    return hist


def degree_distribution_distance(
    graphs_a: List[GeneratedGraph],
    graphs_b: List[GeneratedGraph],
    method: str = "l1",
) -> float:
    """Compare degree distributions between two sets of graphs.

    Methods:
        l1:  ||p - q||_1
        l2:  ||p - q||_2
        js:  Jensen-Shannon divergence = 0.5 KL(p||m) + 0.5 KL(q||m) where m = (p+q)/2

    Args:
        graphs_a: First set of graphs.
        graphs_b: Second set of graphs.
        method: One of 'l1', 'l2', 'js'.

    Returns:
        Non-negative float distance.
    """
    ha = _degree_hist(graphs_a)
    hb = _degree_hist(graphs_b)

    # Align sizes
    max_len = max(len(ha), len(hb))
    if len(ha) < max_len:
        ha = torch.cat([ha, torch.zeros(max_len - len(ha))])
    if len(hb) < max_len:
        hb = torch.cat([hb, torch.zeros(max_len - len(hb))])

    if method == "l1":
        return float((ha - hb).abs().sum().item())
    elif method == "l2":
        return float(((ha - hb) ** 2).sum().sqrt().item())
    elif method == "js":
        m = (ha + hb) / 2.0
        eps = 1e-10

        def kl(p: torch.Tensor, q: torch.Tensor) -> float:
            mask = p > 0
            return float((p[mask] * (p[mask] / (q[mask] + eps)).log()).sum().item())

        return 0.5 * kl(ha, m) + 0.5 * kl(hb, m)
    else:
        raise ValueError(f"Unknown method={method!r}. Choose 'l1', 'l2', 'js'.")


def _count_triangles_wedges(g: GeneratedGraph) -> tuple:
    """Count triangles and wedges in an undirected graph."""
    if g.num_nodes == 0 or g.num_edges == 0:
        return 0, 0

    adj: Dict[int, set] = {i: set() for i in range(g.num_nodes)}
    for s, d in zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()):
        adj[s].add(d)
        adj[d].add(s)

    triangles = 0
    wedges = 0
    for v in range(g.num_nodes):
        nb = list(adj[v])
        k = len(nb)
        wedges += k * (k - 1) // 2
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                if nb[j] in adj[nb[i]]:
                    triangles += 1
    triangles //= 3
    return triangles, wedges


def motif_distribution_distance(
    graphs_a: List[GeneratedGraph],
    graphs_b: List[GeneratedGraph],
) -> float:
    """Compare triangle and wedge counts between two sets of graphs.

    Computes L1 distance on the (triangle_count, wedge_count) normalized histogram.

    Args:
        graphs_a: First set of graphs.
        graphs_b: Second set of graphs.

    Returns:
        Non-negative float.
    """
    def _stats(graphs: List[GeneratedGraph]) -> torch.Tensor:
        counts = []
        for g in graphs:
            tri, wed = _count_triangles_wedges(g)
            counts.append([float(tri), float(wed)])
        if not counts:
            return torch.zeros(2)
        t = torch.tensor(counts).mean(dim=0)
        norm = t.norm().clamp(min=1e-9)
        return t / norm

    sa = _stats(graphs_a)
    sb = _stats(graphs_b)
    return float((sa - sb).abs().sum().item())


def spectral_distance(
    graphs_a: List[GeneratedGraph],
    graphs_b: List[GeneratedGraph],
) -> float:
    """Laplacian eigenvalue distance between two sets of graphs.

    Computes the mean L2 distance between sorted Laplacian spectra.

    d(A, B) = mean_G [ ||eig(L_A)||_2 - ||eig(L_B)||_2 ||_2 ]

    Warning: O(N^3) per graph via full eigendecomposition. Warns if N > 500.

    Args:
        graphs_a: First set.
        graphs_b: Second set.

    Returns:
        Non-negative float.
    """
    def _spectrum(g: GeneratedGraph) -> torch.Tensor:
        n = g.num_nodes
        if n > _N_WARN_SPECTRAL:
            import warnings
            warnings.warn(
                f"spectral_distance: graph has {n} nodes, O(N^3) cost.",
                RuntimeWarning,
                stacklevel=3,
            )
        L = torch.zeros(n, n, dtype=torch.float)
        if g.num_edges > 0:
            for s, d in zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()):
                L[s, s] += 1.0
                L[s, d] -= 1.0
        eigvals = torch.linalg.eigvalsh(L)
        return eigvals.sort().values

    def _mean_spectrum(graphs: List[GeneratedGraph]) -> Optional[torch.Tensor]:
        spectra = []
        for g in graphs:
            try:
                spectra.append(_spectrum(g))
            except Exception:
                pass
        if not spectra:
            return None
        max_len = max(s.shape[0] for s in spectra)
        padded = [
            torch.cat([s, torch.zeros(max_len - s.shape[0])]) for s in spectra
        ]
        return torch.stack(padded).mean(dim=0)

    sa = _mean_spectrum(graphs_a)
    sb = _mean_spectrum(graphs_b)
    if sa is None or sb is None:
        return 0.0

    max_len = max(sa.shape[0], sb.shape[0])
    if sa.shape[0] < max_len:
        sa = torch.cat([sa, torch.zeros(max_len - sa.shape[0])])
    if sb.shape[0] < max_len:
        sb = torch.cat([sb, torch.zeros(max_len - sb.shape[0])])

    return float(((sa - sb) ** 2).sum().sqrt().item())


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> float:
    return float(torch.exp(-((x - y) ** 2).sum() / (2 * sigma ** 2)).item())


def _linear_kernel(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x * y).sum().item())


def _mmd_on_stats(
    stats_a: List[torch.Tensor],
    stats_b: List[torch.Tensor],
    kernel: str = "linear",
) -> float:
    """Compute MMD between two lists of feature vectors.

    MMD^2 = E[k(x,x')] - 2 E[k(x,y)] + E[k(y,y')]
    """
    if not stats_a or not stats_b:
        return 0.0

    if kernel == "linear":
        kfn = _linear_kernel
    elif kernel == "rbf":
        kfn = lambda x, y: _rbf_kernel(x, y, sigma=1.0)
    else:
        raise ValueError(f"Unknown kernel={kernel!r}. Choose 'linear' or 'rbf'.")

    def _k_mean(xs: List[torch.Tensor], ys: List[torch.Tensor]) -> float:
        total = 0.0
        for x in xs:
            for y in ys:
                total += kfn(x, y)
        return total / (len(xs) * len(ys))

    kxx = _k_mean(stats_a, stats_a)
    kxy = _k_mean(stats_a, stats_b)
    kyy = _k_mean(stats_b, stats_b)
    return max(0.0, kxx - 2 * kxy + kyy)


def mmd_degree(
    graphs_a: List[GeneratedGraph],
    graphs_b: List[GeneratedGraph],
    kernel: str = "linear",
) -> float:
    """Maximum Mean Discrepancy on degree distributions.

    MMD^2 = E_{x,x' ~ p}[k(x,x')] - 2 E_{x~p,y~q}[k(x,y)] + E_{y,y' ~ q}[k(y,y')]

    Args:
        graphs_a: First set.
        graphs_b: Second set.
        kernel: 'linear' or 'rbf'.

    Returns:
        Non-negative float (MMD^2).
    """
    def _degree_vector(g: GeneratedGraph) -> torch.Tensor:
        if g.num_nodes == 0:
            return torch.zeros(1)
        deg = torch.zeros(g.num_nodes, dtype=torch.float)
        if g.num_edges > 0:
            deg.scatter_add_(0, g.edge_index[1], torch.ones(g.num_edges))
        return deg.float()

    stats_a = [_degree_vector(g) for g in graphs_a]
    stats_b = [_degree_vector(g) for g in graphs_b]

    # Pad to same length
    if stats_a and stats_b:
        max_len = max(max(s.shape[0] for s in stats_a), max(s.shape[0] for s in stats_b))
        stats_a = [torch.cat([s, torch.zeros(max_len - s.shape[0])]) for s in stats_a]
        stats_b = [torch.cat([s, torch.zeros(max_len - s.shape[0])]) for s in stats_b]

    return _mmd_on_stats(stats_a, stats_b, kernel=kernel)


def _clustering_coefficients(g: GeneratedGraph) -> torch.Tensor:
    """Compute per-node clustering coefficients."""
    n = g.num_nodes
    if n == 0:
        return torch.zeros(1)
    adj: Dict[int, set] = {i: set() for i in range(n)}
    for s, d in zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()):
        adj[s].add(d)
        adj[d].add(s)
    coeffs = []
    for v in range(n):
        nb = list(adj[v])
        k = len(nb)
        if k < 2:
            coeffs.append(0.0)
            continue
        tri = sum(
            1 for i in range(len(nb))
            for j in range(i + 1, len(nb))
            if nb[j] in adj[nb[i]]
        )
        coeffs.append(tri / (k * (k - 1) / 2))
    return torch.tensor(coeffs)


def mmd_clustering(
    graphs_a: List[GeneratedGraph],
    graphs_b: List[GeneratedGraph],
) -> float:
    """Maximum Mean Discrepancy on clustering coefficients.

    Args:
        graphs_a: First set.
        graphs_b: Second set.

    Returns:
        Non-negative float (MMD^2).
    """
    stats_a = [_clustering_coefficients(g) for g in graphs_a]
    stats_b = [_clustering_coefficients(g) for g in graphs_b]

    if stats_a and stats_b:
        max_len = max(max(s.shape[0] for s in stats_a), max(s.shape[0] for s in stats_b))
        stats_a = [torch.cat([s, torch.zeros(max_len - s.shape[0])]) for s in stats_a]
        stats_b = [torch.cat([s, torch.zeros(max_len - s.shape[0])]) for s in stats_b]

    return _mmd_on_stats(stats_a, stats_b, kernel="linear")


def constraint_satisfaction_rate(
    graphs: List[GeneratedGraph],
    constraints: Dict[str, object],
) -> Dict[str, float]:
    """Detailed breakdown of constraint satisfaction rates.

    Args:
        graphs: List of GeneratedGraphs.
        constraints: Dict with constraint keys (same as validate_generated_graph).

    Returns:
        Dict mapping constraint name -> satisfaction rate (float in [0, 1]).
        Also includes 'overall' key.
    """
    from .data_model import validate_generated_graph

    if not graphs:
        return {"overall": 0.0}

    results: Dict[str, List[bool]] = {k: [] for k in constraints}
    results["overall"] = []

    for g in graphs:
        valid, violations = validate_generated_graph(g, constraints)
        results["overall"].append(valid)
        violated_keys = set()
        for v in violations:
            for k in constraints:
                if k in v.lower():
                    violated_keys.add(k)
        for k in constraints:
            results[k].append(k not in violated_keys)

    return {
        k: sum(v) / len(v)
        for k, v in results.items()
    }

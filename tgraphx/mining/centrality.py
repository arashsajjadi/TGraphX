"""Graph centrality and node-importance algorithms in pure PyTorch.

All algorithms are designed for GNN-training-oriented workflows on
small-to-medium graphs.  Where an exact algorithm is O(N²) or worse,
a size guard is enforced and the function raises a ``ValueError``
rather than silently running out of memory.

Stability: Beta (v0.4.2+).

Note: TGraphX is not a NetworkX replacement.  Optional NetworkX
parity tests are run when NetworkX is installed.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional, Tuple

import torch

__all__ = [
    "degree_centrality",
    "in_degree_centrality",
    "out_degree_centrality",
    "pagerank",
    "personalized_pagerank",
    "hits",
    "katz_centrality",
    "closeness_centrality",
    "harmonic_centrality",
    "betweenness_centrality",
    "eigenvector_centrality",
    "k_core_numbers",
]

_LARGE_GRAPH = 5_000  # nodes above this trigger a warning for O(N²) methods


def _validate(edge_index: torch.Tensor, num_nodes: int) -> None:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be [2, E]; got {tuple(edge_index.shape)}")
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be >= 0; got {num_nodes}")
    if edge_index.numel() and int(edge_index.max().item()) >= num_nodes:
        raise ValueError(
            f"edge_index max node id >= num_nodes={num_nodes}"
        )


def _degrees(
    edge_index: torch.Tensor, num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (out_degree, in_degree) LongTensors."""
    device = edge_index.device
    out_d = torch.zeros(num_nodes, dtype=torch.long, device=device)
    in_d = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if edge_index.numel():
        ones = torch.ones(edge_index.size(1), dtype=torch.long, device=device)
        out_d.scatter_add_(0, edge_index[0], ones)
        in_d.scatter_add_(0, edge_index[1], ones)
    return out_d, in_d


# ── Degree centrality ─────────────────────────────────────────────────────────


def degree_centrality(
    edge_index: torch.Tensor,
    num_nodes: int,
    directed: bool = False,
) -> torch.Tensor:
    """Degree centrality: ``deg(v) / (N-1)`` (normalised).

    For directed graphs returns ``(in_deg + out_deg) / (2*(N-1))``.
    For N <= 1 returns zeros.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        directed: When ``True``, treats the graph as directed.

    Returns:
        ``FloatTensor[N]`` in ``[0, 1]``.
    """
    _validate(edge_index, num_nodes)
    out_d, in_d = _degrees(edge_index, num_nodes)
    if directed:
        deg = (out_d + in_d).float()
        max_deg = 2.0 * float(num_nodes - 1) if num_nodes > 1 else 1.0
    else:
        # For undirected assume symmetric edge_index; each edge counted once.
        deg = out_d.float()
        max_deg = float(num_nodes - 1) if num_nodes > 1 else 1.0
    return deg / max(max_deg, 1.0)


def in_degree_centrality(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Normalised in-degree centrality: ``in_deg(v) / (N-1)``."""
    _validate(edge_index, num_nodes)
    _, in_d = _degrees(edge_index, num_nodes)
    max_deg = float(num_nodes - 1) if num_nodes > 1 else 1.0
    return in_d.float() / max(max_deg, 1.0)


def out_degree_centrality(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Normalised out-degree centrality: ``out_deg(v) / (N-1)``."""
    _validate(edge_index, num_nodes)
    out_d, _ = _degrees(edge_index, num_nodes)
    max_deg = float(num_nodes - 1) if num_nodes > 1 else 1.0
    return out_d.float() / max(max_deg, 1.0)


# ── PageRank ─────────────────────────────────────────────────────────────────


def pagerank(
    edge_index: torch.Tensor,
    num_nodes: int,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Power-iteration PageRank.

    ``PR(v) = (1-alpha)/N + alpha * Σ_{u→v} PR(u) / out_deg(u)``

    Args:
        edge_index: ``LongTensor[2, E]`` (directed).
        num_nodes: Node count.
        alpha: Damping factor (default 0.85).
        max_iter: Maximum power iterations.
        tol: Convergence tolerance (L1 norm of change).
        weight: Optional ``FloatTensor[E]`` edge weights (unnormalised).

    Returns:
        ``FloatTensor[N]`` of PageRank scores summing to 1.

    Complexity: O(iter × (N + E)).  No dense matrix is allocated.
    """
    _validate(edge_index, num_nodes)
    if num_nodes == 0:
        return torch.zeros(0, dtype=torch.float)
    device = edge_index.device
    N = num_nodes
    pr = torch.full((N,), 1.0 / N, dtype=torch.float, device=device)
    src, dst = edge_index[0], edge_index[1]

    # Compute out-degree (for normalisation).
    out_d = torch.zeros(N, dtype=torch.float, device=device)
    ones = torch.ones(src.size(0), dtype=torch.float, device=device)
    w = weight.to(dtype=torch.float, device=device) if weight is not None else ones
    out_d.scatter_add_(0, src, w)

    for _ in range(max_iter):
        # Normalised edge message: PR(u) * w(u,v) / out_deg(u).
        safe_out = out_d[src].clamp(min=1e-12)
        msg = pr[src] * w / safe_out  # [E]
        new_pr = torch.zeros(N, dtype=torch.float, device=device)
        new_pr.scatter_add_(0, dst, msg)
        new_pr = alpha * new_pr + (1.0 - alpha) / N
        diff = (new_pr - pr).abs().sum().item()
        pr = new_pr
        if diff < tol:
            break

    return pr


def personalized_pagerank(
    edge_index: torch.Tensor,
    num_nodes: int,
    personalization: torch.Tensor,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Personalised PageRank with a seed node distribution.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        personalization: ``FloatTensor[N]`` of restart probabilities
            (need not sum to 1; normalised internally).
        alpha: Damping factor.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        ``FloatTensor[N]`` of PPR scores.
    """
    _validate(edge_index, num_nodes)
    if num_nodes == 0:
        return torch.zeros(0, dtype=torch.float)
    device = edge_index.device
    N = num_nodes
    pers = personalization.to(dtype=torch.float, device=device)
    pers = pers / pers.sum().clamp(min=1e-12)
    pr = pers.clone()
    src, dst = edge_index[0], edge_index[1]
    out_d = torch.zeros(N, dtype=torch.float, device=device)
    ones = torch.ones(src.size(0), dtype=torch.float, device=device)
    out_d.scatter_add_(0, src, ones)
    for _ in range(max_iter):
        safe_out = out_d[src].clamp(min=1e-12)
        msg = pr[src] / safe_out
        new_pr = torch.zeros(N, dtype=torch.float, device=device)
        new_pr.scatter_add_(0, dst, msg)
        new_pr = alpha * new_pr + (1.0 - alpha) * pers
        diff = (new_pr - pr).abs().sum().item()
        pr = new_pr
        if diff < tol:
            break
    return pr


# ── HITS ─────────────────────────────────────────────────────────────────────


def hits(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    normalized: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HITS (Hyperlink-Induced Topic Search) hubs and authorities.

    ``auth(v) ← Σ_{u→v} hub(u)``
    ``hub(u)  ← Σ_{u→v} auth(v)``

    Args:
        edge_index: ``LongTensor[2, E]`` (directed).
        num_nodes: Node count.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        normalized: When ``True``, normalise to unit L2 norm.

    Returns:
        ``(hub_scores, authority_scores)`` — two ``FloatTensor[N]``.
    """
    _validate(edge_index, num_nodes)
    if num_nodes == 0:
        empty = torch.zeros(0, dtype=torch.float)
        return empty, empty
    device = edge_index.device
    N = num_nodes
    hub = torch.ones(N, dtype=torch.float, device=device)
    auth = torch.ones(N, dtype=torch.float, device=device)
    src, dst = edge_index[0], edge_index[1]
    for _ in range(max_iter):
        new_auth = torch.zeros(N, dtype=torch.float, device=device)
        new_auth.scatter_add_(0, dst, hub[src])
        new_hub = torch.zeros(N, dtype=torch.float, device=device)
        new_hub.scatter_add_(0, src, new_auth[dst])
        if normalized:
            auth_n = new_auth.norm().clamp(min=1e-12)
            hub_n = new_hub.norm().clamp(min=1e-12)
            new_auth = new_auth / auth_n
            new_hub = new_hub / hub_n
        diff = (new_auth - auth).abs().sum() + (new_hub - hub).abs().sum()
        auth = new_auth
        hub = new_hub
        if float(diff.item()) < tol:
            break
    return hub, auth


# ── Katz centrality ───────────────────────────────────────────────────────────


def katz_centrality(
    edge_index: torch.Tensor,
    num_nodes: int,
    alpha: float = 0.1,
    beta: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    normalized: bool = True,
) -> torch.Tensor:
    """Katz centrality via power iteration.

    ``c(v) = alpha * Σ_u A_{uv} * c(u) + beta``

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        alpha: Attenuation factor.  Must be less than 1/λ_max.
        beta: Intrinsic node importance.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        normalized: When ``True``, divide by L2 norm.

    Returns:
        ``FloatTensor[N]`` Katz centrality scores.
    """
    _validate(edge_index, num_nodes)
    if num_nodes == 0:
        return torch.zeros(0, dtype=torch.float)
    device = edge_index.device
    N = num_nodes
    c = torch.full((N,), float(beta), dtype=torch.float, device=device)
    src, dst = edge_index[0], edge_index[1]
    for _ in range(max_iter):
        new_c = torch.full((N,), float(beta), dtype=torch.float, device=device)
        new_c.scatter_add_(0, dst, alpha * c[src])
        diff = (new_c - c).abs().sum().item()
        c = new_c
        if diff < tol:
            break
    if normalized:
        c_n = c.norm().clamp(min=1e-12)
        c = c / c_n
    return c


# ── Closeness and harmonic centrality ─────────────────────────────────────────


def closeness_centrality(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_nodes_exact: int = 2_000,
) -> torch.Tensor:
    """Closeness centrality: ``(N-1) / Σ_v d(u, v)`` for reachable nodes.

    Exact computation using BFS for all-pairs shortest paths.
    Only computed for ``num_nodes <= max_nodes_exact``.

    Args:
        edge_index: ``LongTensor[2, E]`` (treated as undirected).
        num_nodes: Node count.
        max_nodes_exact: Size guard; raises for larger graphs.

    Returns:
        ``FloatTensor[N]`` in ``[0, 1]``.  Isolated nodes get 0.
    """
    _validate(edge_index, num_nodes)
    if num_nodes == 0:
        return torch.zeros(0, dtype=torch.float)
    if num_nodes > max_nodes_exact:
        raise ValueError(
            f"closeness_centrality: num_nodes={num_nodes} > max_nodes_exact={max_nodes_exact}. "
            "Use harmonic_centrality for large graphs (handles disconnected components)."
        )
    from tgraphx.algorithms.traversal import bfs_layers
    # Make undirected.
    if edge_index.numel():
        ei_sym = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        ei_sym = torch.unique(ei_sym, dim=1)
    else:
        ei_sym = edge_index

    scores = torch.zeros(num_nodes, dtype=torch.float)
    for v in range(num_nodes):
        layers = bfs_layers(ei_sym, source=v, num_nodes=num_nodes)
        total_dist = 0
        reachable = 0
        for hop, layer in enumerate(layers):
            if hop == 0:
                continue
            total_dist += hop * layer.numel()
            reachable += layer.numel()
        if reachable > 0:
            scores[v] = float(reachable) / float(total_dist)
    return scores


def harmonic_centrality(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_nodes_exact: int = 2_000,
) -> torch.Tensor:
    """Harmonic centrality: ``Σ_{v≠u} 1/d(u,v)`` normalised by ``N-1``.

    Handles disconnected graphs gracefully (unreachable nodes contribute 0).

    Args:
        edge_index: ``LongTensor[2, E]`` (treated as undirected).
        num_nodes: Node count.
        max_nodes_exact: Size guard.

    Returns:
        ``FloatTensor[N]`` in ``[0, 1]``.
    """
    _validate(edge_index, num_nodes)
    if num_nodes == 0:
        return torch.zeros(0, dtype=torch.float)
    if num_nodes > max_nodes_exact:
        raise ValueError(
            f"harmonic_centrality: num_nodes={num_nodes} > max_nodes_exact={max_nodes_exact}."
        )
    from tgraphx.algorithms.traversal import bfs_layers
    if edge_index.numel():
        ei_sym = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        ei_sym = torch.unique(ei_sym, dim=1)
    else:
        ei_sym = edge_index

    scores = torch.zeros(num_nodes, dtype=torch.float)
    norm = float(num_nodes - 1) if num_nodes > 1 else 1.0
    for v in range(num_nodes):
        layers = bfs_layers(ei_sym, source=v, num_nodes=num_nodes)
        h = 0.0
        for hop, layer in enumerate(layers):
            if hop == 0:
                continue
            h += layer.numel() / float(hop)
        scores[v] = h / norm
    return scores


# ── Betweenness centrality ────────────────────────────────────────────────────


def betweenness_centrality(
    edge_index: torch.Tensor,
    num_nodes: int,
    normalized: bool = True,
    endpoints: bool = False,
    max_nodes_exact: int = 500,
) -> torch.Tensor:
    """Exact unweighted betweenness centrality via Brandes' algorithm.

    Only computed exactly for ``num_nodes <= max_nodes_exact``.

    BC(v) = Σ_{s≠v≠t} σ(s,t|v) / σ(s,t)

    Args:
        edge_index: ``LongTensor[2, E]`` (treated as undirected).
        num_nodes: Node count.
        normalized: Divide by ``(N-1)(N-2)`` (directed) or
            ``(N-1)(N-2)/2`` (undirected).
        endpoints: Include endpoints in the count.
        max_nodes_exact: Size guard.

    Returns:
        ``FloatTensor[N]``, non-negative.
    """
    _validate(edge_index, num_nodes)
    if num_nodes == 0:
        return torch.zeros(0, dtype=torch.float)
    if num_nodes > max_nodes_exact:
        raise ValueError(
            f"betweenness_centrality: num_nodes={num_nodes} > max_nodes_exact={max_nodes_exact}. "
            "For large graphs use approximate methods or sampling."
        )
    N = num_nodes
    # Build undirected adjacency lists.
    adj: list = [[] for _ in range(N)]
    if edge_index.numel():
        src = edge_index[0].cpu().tolist()
        dst = edge_index[1].cpu().tolist()
        for u, v in zip(src, dst):
            if v not in adj[u]:
                adj[u].append(v)
            if u not in adj[v]:
                adj[v].append(u)

    bc = [0.0] * N
    for s in range(N):
        # Brandes' BFS.
        stack = []
        pred = [[] for _ in range(N)]
        sigma = [0] * N
        sigma[s] = 1
        dist = [-1] * N
        dist[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            v = queue[qi]; qi += 1
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = [0.0] * N
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w]:
                    delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]

    bc_t = torch.tensor(bc, dtype=torch.float)
    if normalized and N > 2:
        scale = 1.0 / ((N - 1) * (N - 2) / 2.0)
        bc_t = bc_t * scale
    return bc_t


# ── Eigenvector centrality ────────────────────────────────────────────────────


def eigenvector_centrality(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    normalized: bool = True,
) -> torch.Tensor:
    """Eigenvector centrality via power iteration.

    The leading eigenvector of the adjacency matrix corresponds to nodes
    with the most influential neighbours.

    Args:
        edge_index: ``LongTensor[2, E]`` (treated as undirected).
        num_nodes: Node count.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        normalized: Normalise to unit L2 norm.

    Returns:
        ``FloatTensor[N]``, non-negative.
    """
    _validate(edge_index, num_nodes)
    if num_nodes == 0:
        return torch.zeros(0, dtype=torch.float)
    device = edge_index.device
    N = num_nodes
    x = torch.ones(N, dtype=torch.float, device=device)
    # Make symmetric.
    if edge_index.numel():
        ei_sym = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        src, dst = ei_sym[0], ei_sym[1]
    else:
        src = dst = torch.zeros(0, dtype=torch.long, device=device)
    for _ in range(max_iter):
        new_x = torch.zeros(N, dtype=torch.float, device=device)
        if src.numel():
            new_x.scatter_add_(0, dst, x[src])
        new_norm = new_x.norm().clamp(min=1e-12)
        new_x = new_x / new_norm
        diff = (new_x - x).abs().sum().item()
        x = new_x
        if diff < tol:
            break
    if normalized:
        x = x / x.norm().clamp(min=1e-12)
    return x.clamp(min=0.0)  # eigenvector entries are non-negative for connected


# ── k-core ────────────────────────────────────────────────────────────────────


def k_core_numbers(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Compute the core number of each node.

    The k-core of a graph is the maximal subgraph where every node has
    degree >= k.  The core number of a node is the largest k such that
    the node belongs to the k-core.

    Args:
        edge_index: ``LongTensor[2, E]`` (treated as undirected).
        num_nodes: Node count.

    Returns:
        ``LongTensor[N]`` of core numbers.
    """
    _validate(edge_index, num_nodes)
    if num_nodes == 0:
        return torch.zeros(0, dtype=torch.long)
    N = num_nodes
    # Build undirected adjacency sets.
    adj: list = [set() for _ in range(N)]
    if edge_index.numel():
        src = edge_index[0].cpu().tolist()
        dst = edge_index[1].cpu().tolist()
        for u, v in zip(src, dst):
            if u != v:
                adj[u].add(v)
                adj[v].add(u)
    deg = [len(adj[v]) for v in range(N)]
    core = [0] * N
    # Iterative peeling.
    k = 0
    removed = [False] * N
    while True:
        changed = True
        while changed:
            changed = False
            for v in range(N):
                if not removed[v] and deg[v] <= k:
                    removed[v] = True
                    core[v] = k
                    for u in adj[v]:
                        if not removed[u]:
                            deg[u] -= 1
                    changed = True
        if all(removed):
            break
        k += 1
    return torch.tensor(core, dtype=torch.long)

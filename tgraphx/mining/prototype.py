"""Prototype graph membership and class-graph utilities.

This module provides the core primitives for the *class-graph membership*
pattern recognition paradigm — a TGraphX-native alternative to flat
nearest-neighbour classification for structured image/volume inputs:

1. Build a **support graph** per class from training embeddings.
2. For each query, build a **candidate graph** by adding the query node
   to a class support graph.
3. A GNN then scores each candidate graph to predict class membership.

**Key design principles:**

- Tensor-aware: node features may be ``[N, D]``, ``[N, C, H, W]``, or
  ``[N, C, D, H, W]``.  Embeddings (for topology construction) are
  always ``[N, D_embed]`` and are separate from the raw features.
- No hidden training: this module builds graphs and datasets; model
  training is the user's responsibility.
- No data leakage: the builder ensures query nodes are not in the
  support graph.
- No mandatory heavy dependency.

Stability: Experimental (v0.3.2+).  API may evolve in v0.3.4 once
real MNIST class-graph experiments are run.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

__all__ = [
    "ClassGraphBuilder",
    "CandidateGraphBuilder",
    "GraphMembershipDataset",
    "MembershipEvaluator",
    "cosine_graph_membership_baseline",
]


# ── Internal helpers ─────────────────────────────────────────────────────────


def _cosine_knn(
    embeddings: torch.Tensor,
    k: int,
    mutual: bool = False,
    exclude_self: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build cosine kNN graph.  Returns (src, dst) tensors."""
    N = embeddings.size(0)
    if N == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    k_actual = min(k, N - 1 if exclude_self else N)
    if k_actual <= 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)

    emb_norm = embeddings / (embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8))
    sim = emb_norm @ emb_norm.t()
    if exclude_self:
        sim.fill_diagonal_(-2.0)

    # Take top-k.
    _, topk_idx = sim.topk(k_actual, dim=1, largest=True, sorted=False)
    src = torch.arange(N, dtype=torch.long).unsqueeze(1).expand_as(topk_idx).reshape(-1)
    dst = topk_idx.reshape(-1)

    if mutual:
        # Keep only edges that appear in both directions.
        edge_set = set(zip(src.tolist(), dst.tolist()))
        keep = [(u, v) for u, v in edge_set if (v, u) in edge_set]
        if not keep:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
        src = torch.tensor([e[0] for e in keep], dtype=torch.long)
        dst = torch.tensor([e[1] for e in keep], dtype=torch.long)
    return src, dst


class ClassGraphBuilder:
    """Build one class-support graph per class from training data.

    The class graph for class ``c`` is a kNN graph over the training
    samples belonging to class ``c``.  Edges are built using cosine
    similarity over ``embeddings`` (which may be precomputed model
    activations, raw features, etc.).

    Args:
        k_support: Number of nearest neighbours per node.
        max_neighbor_fraction: Cap k at ``int(n_c * max_neighbor_fraction)``
            where ``n_c`` is the class size.  Prevents dense graphs for
            small classes.
        mutual_knn: When ``True``, keep only mutually-agreed edges
            (both endpoints selected each other).
        ensure_connected: When ``True``, connect isolated nodes to their
            single nearest neighbour to avoid completely disconnected
            class graphs.
    """

    def __init__(
        self,
        k_support: int = 5,
        max_neighbor_fraction: float = 0.5,
        mutual_knn: bool = False,
        ensure_connected: bool = True,
    ) -> None:
        if k_support <= 0:
            raise ValueError(f"k_support must be positive; got {k_support}")
        if not 0 < max_neighbor_fraction <= 1.0:
            raise ValueError(
                f"max_neighbor_fraction must be in (0, 1]; got {max_neighbor_fraction}"
            )
        self.k_support = int(k_support)
        self.max_neighbor_fraction = float(max_neighbor_fraction)
        self.mutual_knn = bool(mutual_knn)
        self.ensure_connected = bool(ensure_connected)
        self.class_graphs_: Dict[int, Dict[str, Any]] = {}

    def fit(
        self,
        node_features: torch.Tensor,
        labels: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
    ) -> "ClassGraphBuilder":
        """Build class graphs from training data.

        Args:
            node_features: ``Tensor[N, *]`` — raw node features (any layout).
            labels: ``LongTensor[N]`` — class labels.
            embeddings: ``FloatTensor[N, D]`` — topology embeddings.
                When ``None``, the first two dimensions of ``node_features``
                are used (i.e. ``[N, D]`` assumed).

        Returns:
            self (fluent API).
        """
        N = node_features.size(0)
        labels = labels.to(torch.long)
        if labels.shape[0] != N:
            raise ValueError(
                f"node_features has {N} samples but labels has {labels.shape[0]}"
            )
        if embeddings is None:
            if node_features.dim() != 2:
                raise ValueError(
                    "embeddings must be provided for non-vector node_features"
                )
            embeddings = node_features.float()
        elif embeddings.shape[0] != N:
            raise ValueError(
                f"embeddings has {embeddings.shape[0]} rows but N={N}"
            )

        unique_classes = labels.unique().tolist()
        self.class_graphs_ = {}

        for cls in unique_classes:
            cls = int(cls)
            mask = labels == cls
            idx = mask.nonzero(as_tuple=False).view(-1)
            n_c = len(idx)
            feat_c = node_features[idx]
            emb_c = embeddings[idx].float()

            k_eff = min(
                self.k_support,
                max(1, int(math.floor(n_c * self.max_neighbor_fraction))),
                n_c - 1,
            )

            if n_c <= 1 or k_eff <= 0:
                # Degenerate class: no edges.
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                bridge_edges = 0
            else:
                src, dst = _cosine_knn(emb_c, k_eff, mutual=self.mutual_knn)
                edge_index = torch.stack([src, dst], dim=0) if src.numel() else \
                    torch.zeros((2, 0), dtype=torch.long)

                bridge_edges = 0
                if self.ensure_connected and n_c > 1:
                    # Add bridge edges for isolated nodes.
                    in_deg = torch.zeros(n_c, dtype=torch.long)
                    out_deg = torch.zeros(n_c, dtype=torch.long)
                    if edge_index.numel():
                        ones = torch.ones(edge_index.size(1), dtype=torch.long)
                        out_deg.scatter_add_(0, edge_index[0], ones)
                        in_deg.scatter_add_(0, edge_index[1], ones)
                    total_deg = in_deg + out_deg
                    isolated = (total_deg == 0).nonzero(as_tuple=False).view(-1)
                    if isolated.numel() > 0:
                        emb_norm = emb_c / (emb_c.norm(dim=1, keepdim=True).clamp(min=1e-8))
                        for iso in isolated.tolist():
                            sim_row = (emb_norm[iso] * emb_norm).sum(dim=1)
                            sim_row[iso] = -2.0
                            nn_idx = int(sim_row.argmax().item())
                            bridge = torch.tensor([[iso, nn_idx]], dtype=torch.long).t()
                            edge_index = torch.cat([edge_index, bridge], dim=1)
                            bridge_edges += 1

            # Compute density stat.
            E = int(edge_index.size(1))
            max_edges = n_c * (n_c - 1) if n_c > 1 else 1
            density = float(E) / float(max_edges)

            # Mean edge similarity.
            if edge_index.numel() > 0:
                emb_norm = emb_c / (emb_c.norm(dim=1, keepdim=True).clamp(min=1e-8))
                sim_vals = (emb_norm[edge_index[0]] * emb_norm[edge_index[1]]).sum(dim=1)
                mean_edge_sim = float(sim_vals.mean().item())
            else:
                mean_edge_sim = 0.0

            self.class_graphs_[cls] = {
                "node_features": feat_c,
                "embeddings": emb_c,
                "edge_index": edge_index,
                "original_indices": idx,
                "num_nodes": n_c,
                "k_effective": k_eff,
                "bridge_edges_added": bridge_edges,
                "density": round(density, 6),
                "mean_edge_similarity": round(mean_edge_sim, 6),
            }

        return self

    def get_class_graph(self, cls: int) -> Dict[str, Any]:
        """Return the class graph dict for ``cls``."""
        if cls not in self.class_graphs_:
            raise KeyError(f"Class {cls} not found; call fit() first.")
        return self.class_graphs_[cls]

    def report(self) -> Dict[str, Any]:
        """Return a JSON-serializable build report."""
        report = {}
        for cls, cg in self.class_graphs_.items():
            report[int(cls)] = {
                "num_nodes": cg["num_nodes"],
                "num_edges": int(cg["edge_index"].size(1)),
                "k_effective": cg["k_effective"],
                "bridge_edges_added": cg["bridge_edges_added"],
                "density": cg["density"],
                "mean_edge_similarity": cg["mean_edge_similarity"],
            }
        return report


class CandidateGraphBuilder:
    """Add a query node to a class support graph and return a candidate graph.

    The candidate graph is the class support graph augmented with a
    query node connected to the top-k most similar support nodes.

    Args:
        top_k_query: Number of edges from the query node to support nodes.
        use_ego_subgraph: When ``True``, only keep the top-m support
            nodes (most similar to query) and their edges.
        ego_top_m: Support nodes to retain when ``use_ego_subgraph=True``.
    """

    def __init__(
        self,
        top_k_query: int = 5,
        use_ego_subgraph: bool = False,
        ego_top_m: int = 30,
    ) -> None:
        self.top_k_query = int(top_k_query)
        self.use_ego_subgraph = bool(use_ego_subgraph)
        self.ego_top_m = int(ego_top_m)

    def build(
        self,
        class_graph: Dict[str, Any],
        query_features: torch.Tensor,
        query_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """Build a candidate graph by adding the query node.

        Args:
            class_graph: Output of :meth:`ClassGraphBuilder.get_class_graph`.
            query_features: ``Tensor[*]`` — raw features for the query node.
            query_embedding: ``FloatTensor[D]`` — topology embedding for the
                query.  When ``None``, ``query_features.flatten()`` is used.

        Returns:
            Tuple ``(candidate_graph_dict, query_node_index)``.  The query
            node has index ``n_support`` in the candidate graph (0-indexed
            after the support nodes).
        """
        support_feats = class_graph["node_features"]
        support_embs = class_graph["embeddings"]
        n_support = class_graph["num_nodes"]

        if query_embedding is None:
            query_embedding = query_features.float().flatten()
        query_embedding = query_embedding.float().unsqueeze(0)  # [1, D]

        # Ego subgraph: select top-m support nodes.
        if self.use_ego_subgraph:
            m = min(self.ego_top_m, n_support)
            emb_norm = support_embs / (
                support_embs.norm(dim=1, keepdim=True).clamp(min=1e-8)
            )
            q_norm = query_embedding / (
                query_embedding.norm(dim=1, keepdim=True).clamp(min=1e-8)
            )
            sims = (emb_norm @ q_norm.t()).squeeze(1)
            _, top_support = sims.topk(m, largest=True, sorted=False)
            top_support = top_support.sort().values

            # Filter edge_index to keep only top-m nodes.
            support_feats = support_feats[top_support]
            support_embs = support_embs[top_support]
            n_support = m

            # Remap edge_index.
            old_to_new = torch.full((class_graph["num_nodes"],), -1, dtype=torch.long)
            old_to_new[top_support] = torch.arange(m, dtype=torch.long)
            old_ei = class_graph["edge_index"]
            if old_ei.numel():
                src_new = old_to_new[old_ei[0]]
                dst_new = old_to_new[old_ei[1]]
                keep = (src_new >= 0) & (dst_new >= 0)
                support_ei = torch.stack([src_new[keep], dst_new[keep]], dim=0)
            else:
                support_ei = torch.zeros((2, 0), dtype=torch.long)
        else:
            support_ei = class_graph["edge_index"]

        # Query node index.
        query_idx = n_support

        # Build query → support edges.
        k_q = min(self.top_k_query, n_support)
        if k_q > 0 and n_support > 0:
            emb_norm = support_embs / (
                support_embs.norm(dim=1, keepdim=True).clamp(min=1e-8)
            )
            q_norm = query_embedding / (
                query_embedding.norm(dim=1, keepdim=True).clamp(min=1e-8)
            )
            sims = (emb_norm @ q_norm.t()).squeeze(1)
            _, topk_support = sims.topk(k_q, largest=True, sorted=False)

            q_src = torch.full((k_q,), query_idx, dtype=torch.long)
            q_dst = topk_support.to(torch.long)
            # Bidirectional: support → query too.
            q_edges = torch.stack([
                torch.cat([q_src, q_dst]),
                torch.cat([q_dst, q_src]),
            ], dim=0)
        else:
            q_edges = torch.zeros((2, 0), dtype=torch.long)

        # Combine support + query features.
        if query_features.dim() == 0:
            query_features = query_features.unsqueeze(0)
        query_feat = query_features.unsqueeze(0) if query_features.dim() == support_feats.dim() - 1 \
            else query_features.unsqueeze(0) if support_feats.dim() > 1 and query_features.dim() == support_feats.dim() - 1 \
            else query_features.reshape(1, *support_feats.shape[1:]) if support_feats.dim() > 1 \
            else query_features.unsqueeze(0)

        combined_feats = torch.cat([support_feats, query_feat], dim=0)
        combined_ei = torch.cat([support_ei, q_edges], dim=1) \
            if q_edges.numel() else support_ei

        return {
            "node_features": combined_feats,
            "edge_index": combined_ei,
            "num_nodes": n_support + 1,
            "query_idx": query_idx,
            "num_support": n_support,
        }, query_idx


class GraphMembershipDataset:
    """Dataset of positive and negative membership samples.

    For each query, produce one candidate graph per class (positive for
    the true class, negative for others).

    Args:
        class_builder: A fitted :class:`ClassGraphBuilder`.
        candidate_builder: A :class:`CandidateGraphBuilder`.
        hard_negative_fraction: Fraction of negatives chosen as hard
            negatives (highest cosine similarity to query centroid).
    """

    def __init__(
        self,
        class_builder: ClassGraphBuilder,
        candidate_builder: CandidateGraphBuilder,
        hard_negative_fraction: float = 0.5,
    ) -> None:
        self.class_builder = class_builder
        self.candidate_builder = candidate_builder
        self.hard_negative_fraction = float(hard_negative_fraction)
        self._samples: List[Dict[str, Any]] = []

    def build(
        self,
        query_features: torch.Tensor,
        query_labels: torch.Tensor,
        query_embeddings: Optional[torch.Tensor] = None,
    ) -> "GraphMembershipDataset":
        """Build the dataset from query samples.

        Query samples must not overlap with support samples.

        Args:
            query_features: ``Tensor[Q, *]`` — query node features.
            query_labels: ``LongTensor[Q]`` — true class labels.
            query_embeddings: ``FloatTensor[Q, D]`` — topology embeddings.
                When ``None``, ``query_features`` (must be 2-D) is used.

        Returns:
            self.
        """
        Q = query_features.size(0)
        query_labels = query_labels.to(torch.long)
        if query_embeddings is None:
            if query_features.dim() != 2:
                raise ValueError("query_embeddings required for non-vector features")
            query_embeddings = query_features.float()

        self._samples = []
        classes = sorted(self.class_builder.class_graphs_.keys())

        for qi in range(Q):
            qf = query_features[qi]
            qe = query_embeddings[qi]
            true_cls = int(query_labels[qi].item())
            for cls in classes:
                cg = self.class_builder.get_class_graph(cls)
                candidate, q_idx = self.candidate_builder.build(cg, qf, qe)
                self._samples.append({
                    "candidate_graph": candidate,
                    "target": 1 if cls == true_cls else 0,
                    "candidate_class": cls,
                    "true_class": true_cls,
                    "query_idx": q_idx,
                    "query_index": qi,
                })
        return self

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._samples[idx]


class MembershipEvaluator:
    """Evaluate prototype graph membership predictions.

    Wraps a scoring function that maps each candidate graph to a score,
    then picks the class with the highest score as the prediction.
    """

    @staticmethod
    def evaluate(
        score_fn,
        query_features: torch.Tensor,
        query_labels: torch.Tensor,
        class_builder: ClassGraphBuilder,
        candidate_builder: CandidateGraphBuilder,
        query_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run per-query evaluation over all classes.

        Args:
            score_fn: Callable ``(candidate_dict) -> float``; higher =
                more likely to be the true class.
            query_features, query_labels, query_embeddings: Query split.
            class_builder: Fitted support graphs.
            candidate_builder: Candidate graph builder.

        Returns:
            Dict with ``accuracy``, ``balanced_accuracy``, ``confusion_matrix``,
            ``classification_report``, ``top_confusion_pairs`` — all JSON-
            serializable.
        """
        Q = query_features.size(0)
        classes = sorted(class_builder.class_graphs_.keys())
        C = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        if query_embeddings is None:
            if query_features.dim() != 2:
                raise ValueError("query_embeddings required for non-vector features")
            query_embeddings = query_features.float()

        y_true, y_pred = [], []
        for qi in range(Q):
            qf = query_features[qi]
            qe = query_embeddings[qi]
            true_cls = int(query_labels[qi].item())

            scores = []
            for cls in classes:
                cg = class_builder.get_class_graph(cls)
                cand, _ = candidate_builder.build(cg, qf, qe)
                scores.append(float(score_fn(cand)))
            pred_cls = classes[int(torch.tensor(scores).argmax().item())]
            y_true.append(true_cls)
            y_pred.append(pred_cls)

        # Build confusion matrix.
        conf = [[0] * C for _ in range(C)]
        for yt, yp in zip(y_true, y_pred):
            if yt in class_to_idx and yp in class_to_idx:
                conf[class_to_idx[yt]][class_to_idx[yp]] += 1

        # Per-class metrics.
        report: Dict[str, Any] = {}
        all_recalls = []
        for i, cls in enumerate(classes):
            tp = conf[i][i]
            fn = sum(conf[i]) - tp
            fp = sum(conf[r][i] for r in range(C)) - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            all_recalls.append(rec)
            report[str(cls)] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": sum(1 for yt in y_true if yt == cls),
            }

        accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / max(Q, 1)
        balanced_acc = sum(all_recalls) / max(len(all_recalls), 1)

        # Top confusion pairs.
        confusion_pairs = []
        for i in range(C):
            for j in range(C):
                if i != j and conf[i][j] > 0:
                    confusion_pairs.append((classes[i], classes[j], conf[i][j]))
        confusion_pairs.sort(key=lambda x: -x[2])
        top_pairs = [{"true": p[0], "pred": p[1], "count": p[2]}
                     for p in confusion_pairs[:10]]

        return {
            "accuracy": round(float(accuracy), 4),
            "balanced_accuracy": round(float(balanced_acc), 4),
            "classification_report": report,
            "confusion_matrix": conf,
            "top_confusion_pairs": top_pairs,
            "num_queries": Q,
            "num_classes": C,
        }


def cosine_graph_membership_baseline(
    query_embedding: torch.Tensor,
    class_builder: ClassGraphBuilder,
) -> Dict[int, float]:
    """Return per-class cosine similarity of query to class centroid.

    A simple non-graph baseline for comparison.

    Args:
        query_embedding: ``FloatTensor[D]`` — query embedding.
        class_builder: Fitted :class:`ClassGraphBuilder`.

    Returns:
        Dict ``{class_id: cosine_similarity_to_centroid}``.
    """
    q = query_embedding.float().flatten()
    q_norm = q / (q.norm().clamp(min=1e-8))
    result = {}
    for cls, cg in class_builder.class_graphs_.items():
        centroid = cg["embeddings"].mean(dim=0).float()
        c_norm = centroid / (centroid.norm().clamp(min=1e-8))
        result[int(cls)] = round(float((q_norm * c_norm).sum().item()), 6)
    return result

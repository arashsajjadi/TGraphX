"""Benchmark prototype graph membership (classical + neural).

Usage::

    python benchmarks/mining/benchmark_prototype_membership.py --small
    python benchmarks/mining/benchmark_prototype_membership.py --json
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from _common import make_parser, timer, print_result
import tgraphx
import torch


def run(args):
    torch.manual_seed(args.seed)
    N_support = 20 if args.small else (args.num_nodes or 60)
    N_query = 6 if args.small else 20
    D = 8; C = 3

    from tgraphx.mining import (
        ClassGraphBuilder, CandidateGraphBuilder,
        cosine_graph_membership_baseline,
        PrototypeMembershipScorer,
    )

    # Generate synthetic data.
    protos = torch.randn(C, D)
    support_embs = torch.cat([
        protos[c].unsqueeze(0) + 0.3 * torch.randn(N_support // C, D)
        for c in range(C)
    ])
    support_labels = torch.tensor([c for c in range(C) for _ in range(N_support // C)])
    support_feats = support_embs.clone()
    query_embs = torch.cat([
        protos[c].unsqueeze(0) + 0.3 * torch.randn(N_query // C, D)
        for c in range(C)
    ])
    query_feats = query_embs.clone()

    # Classical: build class graphs.
    def _build_graphs():
        builder = ClassGraphBuilder(k_support=2, max_neighbor_fraction=0.4)
        builder.fit(support_feats, support_labels, embeddings=support_embs)
        return builder

    t_build, builder = timer(_build_graphs)

    # Classical: cosine baseline.
    def _cosine_baseline():
        results = []
        for qi in range(N_query):
            scores = cosine_graph_membership_baseline(query_embs[qi], builder)
            pred = max(scores, key=scores.get)
            results.append(pred)
        return results

    t_baseline, preds_baseline = timer(_cosine_baseline)
    true_labels = [c for c in range(C) for _ in range(N_query // C)]
    acc_baseline = sum(p == t for p, t in zip(preds_baseline, true_labels)) / N_query

    # Neural: PrototypeMembershipScorer forward speed.
    cand_builder = CandidateGraphBuilder(top_k_query=2)
    model = PrototypeMembershipScorer(in_dim=D, hidden_dim=16, out_dim=8)
    model.eval()

    cg0 = builder.get_class_graph(0)
    qf0 = query_feats[0]
    qe0 = query_embs[0]
    cand0, q_idx0 = cand_builder.build(cg0, qf0, qe0)

    def _neural_forward():
        with torch.no_grad():
            return model(cand0["node_features"], cand0["edge_index"], cand0["query_idx"])

    t_neural, _ = timer(_neural_forward)

    result = {
        "benchmark": "prototype_membership",
        "tgraphx_version": tgraphx.__version__,
        "num_support": N_support,
        "num_query": N_query,
        "feature_dim": D,
        "num_classes": C,
        "device": str(args.device),
        "seed": args.seed,
        "class_graph_build_time_s": round(t_build, 6),
        "cosine_baseline_query_time_s": round(t_baseline, 6),
        "cosine_baseline_accuracy": round(acc_baseline, 4),
        "neural_forward_time_s": round(t_neural, 6),
    }
    print_result(result, args.json)
    return result


if __name__ == "__main__":
    parser = make_parser("benchmark_prototype_membership", "Prototype graph membership benchmark.")
    run(parser.parse_args())

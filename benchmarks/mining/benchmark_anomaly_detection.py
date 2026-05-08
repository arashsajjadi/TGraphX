"""Benchmark graph anomaly detection (classical + neural).

Usage::

    python benchmarks/mining/benchmark_anomaly_detection.py --small
    python benchmarks/mining/benchmark_anomaly_detection.py --json
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from _common import make_parser, make_synthetic_graph, timer, print_result
import tgraphx
import torch


def run(args):
    torch.manual_seed(args.seed)
    N = 30 if args.small else (args.num_nodes or 100)
    ei, num_nodes = make_synthetic_graph(N, density=0.1, seed=args.seed)
    D = 8
    x = torch.randn(num_nodes, D)

    from tgraphx.mining import (
        DegreeAnomalyScorer,
        EgoDensityAnomalyScorer,
        GraphAutoencoderAnomalyDetector,
    )

    # Classical: degree anomaly.
    scorer = DegreeAnomalyScorer()
    scorer.fit(ei, num_nodes)

    def _degree_score():
        return scorer.score_nodes(ei, num_nodes)

    t_deg, deg_scores = timer(_degree_score)

    # Classical: ego-density anomaly.
    ego_scorer = EgoDensityAnomalyScorer(min_ego_size=2)
    ego_scorer.fit(ei, num_nodes)

    def _ego_score():
        return ego_scorer.score_nodes(ei, num_nodes)

    t_ego, _ = timer(_ego_score, n_runs=3)

    # Neural: graph autoencoder.
    ae = GraphAutoencoderAnomalyDetector(in_dim=D, latent_dim=8, hidden_dim=16)
    ae.eval()

    def _ae_score():
        return ae.node_anomaly_scores(x, ei, num_nodes)

    t_ae, ae_scores = timer(_ae_score)

    result = {
        "benchmark": "anomaly_detection",
        "tgraphx_version": tgraphx.__version__,
        "num_nodes": num_nodes,
        "num_edges": int(ei.size(1)),
        "feature_dim": D,
        "device": str(args.device),
        "seed": args.seed,
        "degree_anomaly_time_s": round(t_deg, 6),
        "ego_density_anomaly_time_s": round(t_ego, 6),
        "autoencoder_anomaly_time_s": round(t_ae, 6),
        "degree_scores_max": round(float(deg_scores.max().item()), 4),
        "ae_scores_max": round(float(ae_scores.max().item()), 4),
        "ae_scores_finite": bool(torch.isfinite(ae_scores).all().item()),
    }
    print_result(result, args.json)
    return result


if __name__ == "__main__":
    parser = make_parser("benchmark_anomaly_detection", "Anomaly detection benchmark.")
    run(parser.parse_args())

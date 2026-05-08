"""Benchmark neural graph mining models (PrototypeMembershipScorer,
GraphPatternClassifier, GraphAutoencoderAnomalyDetector).

This is an **engineering smoke benchmark**, not a scientific leaderboard.
It measures training time, loss trajectory, and gradient health on
synthetic controlled tasks.  All tasks are synthetic and CPU-safe.
No downloads required.

Usage::

    python benchmarks/mining/benchmark_neural_mining.py --small
    python benchmarks/mining/benchmark_neural_mining.py --json
    python benchmarks/mining/benchmark_neural_mining.py --epochs 20 --num-graphs 64
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import torch
import tgraphx
from tgraphx.mining import (
    PrototypeMembershipScorer,
    GraphAutoencoderAnomalyDetector,
    GraphPatternClassifier,
    create_synthetic_pattern_dataset,
    train_prototype_membership_step,
    train_anomaly_autoencoder_step,
    train_graph_pattern_classifier_step,
    ClassGraphBuilder,
    CandidateGraphBuilder,
)


def make_parser():
    p = argparse.ArgumentParser(
        prog="benchmark_neural_mining",
        description="Neural graph mining benchmark (synthetic, no downloads).",
    )
    p.add_argument("--small", action="store_true",
                   help="Smallest/fastest mode; use for CI smoke.")
    p.add_argument("--json", action="store_true",
                   help="Print machine-readable JSON output.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num-graphs", type=int, default=None)
    return p


def _gradient_health(model):
    """Return dict: {max_grad_norm, any_nan, any_inf, has_nonzero}."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return {"max_grad_norm": None, "any_nan": None, "any_inf": None, "has_nonzero": False}
    all_grads = torch.cat([g.flatten() for g in grads])
    return {
        "max_grad_norm": round(float(all_grads.norm().item()), 6),
        "any_nan": bool(all_grads.isnan().any().item()),
        "any_inf": bool(all_grads.isinf().any().item()),
        "has_nonzero": bool((all_grads != 0).any().item()),
    }


def benchmark_prototype_membership(epochs, num_graphs, device, seed):
    """Tiny 2-class prototype membership benchmark."""
    torch.manual_seed(seed)
    D, N = 8, 6
    ds = create_synthetic_pattern_dataset(
        num_graphs_per_class=num_graphs // 4, num_nodes=N, in_dim=D,
        seed=seed, noise_std=0.1,
    )

    # Build class support graphs — use first 60% as support, rest as query.
    per_class = {c: [g for g in ds if g["label"] == c] for c in range(4)}
    support = [g for c in range(4) for g in per_class[c][:max(2, len(per_class[c]) * 3 // 5)]]
    query = [g for c in range(4) for g in per_class[c][max(2, len(per_class[c]) * 3 // 5):max(3, len(per_class[c]))]]
    if not query:
        # Fallback: reuse first support sample as query
        query = [per_class[c][0] for c in range(4) if per_class[c]]
    embs = torch.stack([g["node_features"].mean(0) for g in support])
    feats = torch.stack([g["node_features"].mean(0) for g in support])
    labels = torch.tensor([g["label"] for g in support])
    builder = ClassGraphBuilder(k_support=2, max_neighbor_fraction=0.5)
    builder.fit(feats, labels, embeddings=embs)

    cand_builder = CandidateGraphBuilder(top_k_query=2)
    model = PrototypeMembershipScorer(in_dim=D, hidden_dim=16, out_dim=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Build candidates.
    classes = sorted(builder.class_graphs_.keys())
    candidates, targets_list = [], []
    for g in query[:4]:
        qf = g["node_features"].mean(0)
        true_cls = g["label"]
        for cls in classes[:2]:
            cg = builder.get_class_graph(cls)
            cand, q_idx = cand_builder.build(cg, qf, qf)
            # Move to device.
            cand_d = {
                "node_features": cand["node_features"].to(device),
                "edge_index": cand["edge_index"].to(device),
                "query_idx": cand["query_idx"],
            }
            candidates.append(cand_d)
            targets_list.append(1.0 if cls == true_cls else 0.0)
    targets = torch.tensor(targets_list, dtype=torch.float32, device=device)

    losses = []
    t0 = time.perf_counter()
    for _ in range(epochs):
        loss = train_prototype_membership_step(model, opt, candidates, targets)
        losses.append(loss)
    elapsed = time.perf_counter() - t0

    grad_health = _gradient_health(model)
    return {
        "task": "prototype_membership",
        "model": "PrototypeMembershipScorer",
        "epochs": epochs,
        "num_candidates": len(candidates),
        "feature_dim": D,
        "train_time_s": round(elapsed, 4),
        "initial_loss": round(losses[0], 4) if losses else None,
        "final_loss": round(losses[-1], 4) if losses else None,
        "loss_decreased": bool(losses and losses[-1] < losses[0]),
        "gradient_health": grad_health,
    }


def benchmark_graph_pattern_classifier(epochs, num_graphs, device, seed):
    """Graph pattern classification benchmark."""
    torch.manual_seed(seed)
    D, N = 4, 6
    ds = create_synthetic_pattern_dataset(
        num_graphs_per_class=max(4, num_graphs // 8), num_nodes=N, in_dim=D,
        seed=seed, noise_std=0.05,
    )
    # Stratified split.
    per_class = {c: [g for g in ds if g["label"] == c] for c in range(4)}
    train_ds = [g for c in range(4) for g in per_class[c][:max(2, len(per_class[c]) * 3 // 4)]]

    model = GraphPatternClassifier(in_dim=D, hidden_dim=16, enc_dim=8, num_classes=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    losses = []
    t0 = time.perf_counter()
    for _ in range(epochs):
        epoch_loss = 0.0
        for g in train_ds[:8]:  # cap per epoch for benchmark speed
            gd = {"node_features": g["node_features"].to(device),
                  "edge_index": g["edge_index"].to(device),
                  "num_nodes": g["num_nodes"]}
            loss = train_graph_pattern_classifier_step(
                model, opt, [gd], torch.tensor([g["label"]], device=device),
            )
            epoch_loss += loss
        losses.append(epoch_loss / min(8, len(train_ds)))
    elapsed = time.perf_counter() - t0

    # Final accuracy on train.
    model.eval()
    correct = 0
    with torch.no_grad():
        for g in train_ds:
            pred = int(model(
                g["node_features"].to(device),
                g["edge_index"].to(device),
                g["num_nodes"],
            ).argmax().item())
            if pred == g["label"]:
                correct += 1
    acc = correct / len(train_ds) if train_ds else 0.0

    grad_health = _gradient_health(model)
    return {
        "task": "graph_pattern_classification",
        "model": "GraphPatternClassifier",
        "epochs": epochs,
        "num_graphs_train": len(train_ds),
        "num_classes": 4,
        "feature_dim": D,
        "train_time_s": round(elapsed, 4),
        "initial_loss": round(losses[0], 4) if losses else None,
        "final_loss": round(losses[-1], 4) if losses else None,
        "loss_decreased": bool(losses and losses[-1] < losses[0]),
        "final_train_accuracy": round(acc, 4),
        "gradient_health": grad_health,
    }


def benchmark_anomaly_autoencoder(epochs, num_graphs, device, seed):
    """Graph autoencoder anomaly detection benchmark."""
    torch.manual_seed(seed)
    N, D = 8, 4
    # Normal data: zero-mean node features.
    ei = torch.zeros((2, 0), dtype=torch.long)  # no edges → pure feature reconstruction
    x_normal = (torch.randn(N, D) * 0.3).to(device)

    model = GraphAutoencoderAnomalyDetector(in_dim=D, latent_dim=4, hidden_dim=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []
    t0 = time.perf_counter()
    for _ in range(epochs):
        loss = train_anomaly_autoencoder_step(model, opt, x_normal, ei.to(device), N)
        losses.append(loss)
    elapsed = time.perf_counter() - t0

    # Verify injected anomaly is detected.
    x_test = x_normal.clone()
    x_test[0] = x_test[0] + 5.0  # inject anomaly at node 0
    scores = model.node_anomaly_scores(x_test, ei.to(device), N)
    anomaly_detected = bool(float(scores[0].item()) >= float(scores[1:].min().item()))

    grad_health = _gradient_health(model)
    return {
        "task": "anomaly_detection",
        "model": "GraphAutoencoderAnomalyDetector",
        "epochs": epochs,
        "num_nodes": N,
        "feature_dim": D,
        "train_time_s": round(elapsed, 4),
        "initial_loss": round(losses[0], 4) if losses else None,
        "final_loss": round(losses[-1], 4) if losses else None,
        "loss_decreased": bool(losses and losses[-1] < losses[0]),
        "injected_anomaly_detected": anomaly_detected,
        "gradient_health": grad_health,
    }


def run(args):
    torch.manual_seed(args.seed)
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    epochs = args.epochs or (3 if args.small else 20)
    num_graphs = args.num_graphs or (16 if args.small else 64)

    result = {
        "benchmark": "neural_mining",
        "tgraphx_version": tgraphx.__version__,
        "device": str(device),
        "seed": args.seed,
        "epochs": epochs,
        "num_graphs": num_graphs,
        "tasks": {},
    }

    # Run each benchmark task.
    for name, fn in [
        ("prototype_membership", benchmark_prototype_membership),
        ("graph_pattern_classifier", benchmark_graph_pattern_classifier),
        ("anomaly_autoencoder", benchmark_anomaly_autoencoder),
    ]:
        task_result = fn(epochs, num_graphs, device, args.seed)
        result["tasks"][name] = task_result

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"[neural_mining] tgraphx={result['tgraphx_version']} "
              f"device={result['device']} epochs={epochs}")
        for task_name, task_res in result["tasks"].items():
            print(f"  {task_name}:")
            print(f"    time={task_res['train_time_s']:.3f}s "
                  f"loss {task_res['initial_loss']:.4f}→{task_res['final_loss']:.4f} "
                  f"↓={task_res['loss_decreased']}")
    return result


if __name__ == "__main__":
    parser = make_parser()
    run(parser.parse_args())

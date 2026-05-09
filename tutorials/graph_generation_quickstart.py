"""Graph Generation Quickstart — CPU runnable, deterministic, ~30 seconds.

This tutorial shows how to:
1. Generate ER / BA / SBM graphs using the one-line API.
2. Compute validity, uniqueness, and diversity metrics.
3. Write a dashboard artifact to a local directory.

Stability: Experimental (v0.7.0+)
"""
import json
import os
import tempfile

import torch

from tgraphx.generation import (
    run_graph_generation,
    list_graph_generation_methods,
)
from tgraphx.generation.metrics import (
    validity_score,
    uniqueness_score,
    diversity_score,
)

SEED = 42

# ---------------------------------------------------------------------------
# 1. List available methods
# ---------------------------------------------------------------------------
print("Available generation methods:")
for name, info in list_graph_generation_methods().items():
    print(f"  {name:25s}  [{info['stability']}]  {info['description']}")

print()

# ---------------------------------------------------------------------------
# 2. Erdős-Rényi
# ---------------------------------------------------------------------------
print("--- Erdős-Rényi (erdos_renyi) ---")
result_er = run_graph_generation(
    method="erdos_renyi",
    num_graphs=32,
    num_nodes=20,
    node_feature_dim=8,
    seed=SEED,
    p=0.3,
)
print(f"  Generated {len(result_er.graphs)} graphs")
print(f"  Validity:    {result_er.metrics['validity']:.3f}")
print(f"  Uniqueness:  {result_er.metrics['uniqueness']:.3f}")
print(f"  Diversity:   {result_er.metrics['diversity']:.3f}")
print(f"  Mean nodes:  {result_er.metrics['mean_num_nodes']:.1f}")
print(f"  Mean edges:  {result_er.metrics['mean_num_edges']:.1f}")

# ---------------------------------------------------------------------------
# 3. Barabási-Albert
# ---------------------------------------------------------------------------
print()
print("--- Barabási-Albert (barabasi_albert) ---")
result_ba = run_graph_generation(
    method="barabasi_albert",
    num_graphs=32,
    num_nodes=20,
    node_feature_dim=8,
    seed=SEED,
    m=2,
)
print(f"  Validity:    {result_ba.metrics['validity']:.3f}")
print(f"  Uniqueness:  {result_ba.metrics['uniqueness']:.3f}")
print(f"  Mean edges:  {result_ba.metrics['mean_num_edges']:.1f}")

# ---------------------------------------------------------------------------
# 4. Stochastic Block Model
# ---------------------------------------------------------------------------
print()
print("--- Stochastic Block Model (stochastic_block_model) ---")
result_sbm = run_graph_generation(
    method="stochastic_block_model",
    num_graphs=32,
    num_nodes=20,
    node_feature_dim=0,
    seed=SEED,
    n_blocks=4,
    p_in=0.7,
    p_out=0.05,
)
print(f"  Validity:    {result_sbm.metrics['validity']:.3f}")
print(f"  Uniqueness:  {result_sbm.metrics['uniqueness']:.3f}")
print(f"  Diversity:   {result_sbm.metrics['diversity']:.3f}")

# ---------------------------------------------------------------------------
# 5. Write dashboard artifact
# ---------------------------------------------------------------------------
print()
with tempfile.TemporaryDirectory() as tmpdir:
    result_dash = run_graph_generation(
        method="erdos_renyi",
        num_graphs=16,
        num_nodes=15,
        seed=SEED,
        dashboard_dir=tmpdir,
    )
    files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
    assert files, "No dashboard artifact written"
    artifact_path = os.path.join(tmpdir, files[0])
    with open(artifact_path) as fh:
        artifact = json.load(fh)
    assert "metrics" in artifact, "Artifact missing 'metrics'"
    print(f"Dashboard artifact: {files[0]}")
    print(f"  Keys: {sorted(artifact.keys())}")
    print(f"  Validity in artifact: {artifact['metrics']['validity']:.3f}")

# ---------------------------------------------------------------------------
# 6. Determinism check
# ---------------------------------------------------------------------------
r1 = run_graph_generation(method="erdos_renyi", num_graphs=4, seed=7)
r2 = run_graph_generation(method="erdos_renyi", num_graphs=4, seed=7)
assert all(
    torch.equal(g1.edge_index, g2.edge_index) for g1, g2 in zip(r1.graphs, r2.graphs)
), "Seed did not produce deterministic results"
print()
print("Seed determinism: OK")

print()
print("=== graph_generation_quickstart COMPLETE ===")

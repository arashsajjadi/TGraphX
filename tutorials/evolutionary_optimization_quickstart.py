"""Evolutionary Graph Optimization Quickstart — CPU runnable, deterministic, ~30 seconds.

This tutorial shows how to:
1. Optimize a graph toward a target objective using GA and NSGA-II.
2. Inspect best fitness and Pareto front.
3. Write a dashboard artifact.

Stability: Experimental (v0.7.0+)
"""
import json
import os
import tempfile

from tgraphx.evolutionary import (
    run_evolutionary_optimization,
    list_evolutionary_optimizers,
)

SEED = 42

# ---------------------------------------------------------------------------
# 1. List available algorithms
# ---------------------------------------------------------------------------
print("Available evolutionary algorithms:")
for name, info in list_evolutionary_optimizers().items():
    multi = "multi-objective" if info["multi_objective"] == "yes" else "single-objective"
    print(f"  {name:15s}  [{info['stability']}]  {multi}  — {info['description']}")

print()

# ---------------------------------------------------------------------------
# 2. Genetic Algorithm — maximize connectivity
# ---------------------------------------------------------------------------
print("--- Genetic Algorithm (ga) ---")
result_ga = run_evolutionary_optimization(
    algorithm="ga",
    objective="connectivity",
    population_size=20,
    generations=15,
    num_nodes=10,
    max_edges=30,
    seed=SEED,
)
print(f"  Best fitness:    {result_ga.best_fitness:.4f}")
print(f"  Generations run: {len(result_ga.fitness_history)}")
print(f"  Best genome:     {result_ga.best_genome.num_nodes} nodes, "
      f"{int(result_ga.best_genome.edge_index.shape[1]) // 2} undirected edges")

# ---------------------------------------------------------------------------
# 3. Simulated Annealing — maximize density
# ---------------------------------------------------------------------------
print()
print("--- Simulated Annealing (sa) ---")
result_sa = run_evolutionary_optimization(
    algorithm="sa",
    objective="density",
    generations=30,
    num_nodes=12,
    seed=SEED,
)
print(f"  Best fitness: {result_sa.best_fitness:.4f}")
print(f"  Steps run:    {len(result_sa.fitness_history)}")

# ---------------------------------------------------------------------------
# 4. NSGA-II — multi-objective (connectivity + density)
# ---------------------------------------------------------------------------
print()
print("--- NSGA-II multi-objective (nsga2) ---")
result_nsga2 = run_evolutionary_optimization(
    algorithm="nsga2",
    objective=["connectivity", "density"],
    population_size=20,
    generations=10,
    num_nodes=10,
    seed=SEED,
)
print(f"  Best fitness (primary): {result_nsga2.best_fitness:.4f}")
if result_nsga2.pareto_front is not None:
    pf_size = len(result_nsga2.pareto_front.genomes) if result_nsga2.pareto_front.genomes else 0
    print(f"  Pareto front size: {pf_size}")
    if pf_size > 0:
        print("  First 3 Pareto solutions:")
        for g in result_nsga2.pareto_front.genomes[:3]:
            n_edges = int(g.edge_index.shape[1]) // 2
            print(f"    {g.num_nodes} nodes, {n_edges} edges")

# ---------------------------------------------------------------------------
# 5. Dashboard artifact
# ---------------------------------------------------------------------------
print()
with tempfile.TemporaryDirectory() as tmpdir:
    result_dash = run_evolutionary_optimization(
        algorithm="ga",
        objective="connectivity",
        population_size=10,
        generations=5,
        seed=SEED,
        dashboard_dir=tmpdir,
    )
    files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
    assert files, "No dashboard artifact written"
    artifact_path = os.path.join(tmpdir, files[0])
    with open(artifact_path) as fh:
        artifact = json.load(fh)
    assert "config" in artifact and "best_fitness" in artifact
    print(f"Dashboard artifact: {files[0]}")
    print(f"  best_fitness: {artifact['best_fitness']:.4f}")
    print(f"  algorithm:    {artifact['config']['algorithm']}")

# ---------------------------------------------------------------------------
# 6. Fitness should improve over generations (or stay same on trivial problems)
# ---------------------------------------------------------------------------
result_check = run_evolutionary_optimization(
    algorithm="ga",
    objective="density",
    population_size=15,
    generations=10,
    num_nodes=8,
    seed=42,
)
first = result_check.fitness_history[0] if result_check.fitness_history else 0.0
last = result_check.fitness_history[-1] if result_check.fitness_history else 0.0
print()
print(f"Fitness history: {first:.3f} -> {last:.3f}")

print()
print("=== evolutionary_optimization_quickstart COMPLETE ===")

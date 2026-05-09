"""Demonstrate run_graph_rl, run_graph_generation, and run_evolutionary_optimization.

Usage:
    python examples/generation_rl_high_level_api_demo.py
"""
from __future__ import annotations


def demo_graph_rl():
    print("--- Graph RL (DQN on navigation) ---")
    from tgraphx.rl import run_graph_rl, list_graph_rl_algorithms

    algs = list_graph_rl_algorithms()
    print(f"Available algorithms: {list(algs.keys())}")

    result = run_graph_rl(
        env="graph_navigation",
        algorithm="dqn",
        episodes=20,
        seed=42,
        verbose=False,
    )
    print(f"Mean return: {result.metrics['mean_return']:.2f}")
    print(f"Success rate: {result.metrics['success_rate']:.2%}")
    print(f"Algorithm: {result.metrics['algorithm']}")
    print()


def demo_graph_generation():
    print("--- Graph Generation (Barabasi-Albert) ---")
    from tgraphx.generation import run_graph_generation, list_graph_generation_methods

    methods = list_graph_generation_methods()
    print(f"Available methods: {list(methods.keys())[:5]}...")

    result = run_graph_generation(
        method="barabasi_albert",
        num_graphs=16,
        num_nodes=20,
        m=2,
        node_feature_dim=4,
        seed=42,
    )
    print(f"Generated {len(result.graphs)} graphs")
    print(f"Validity: {result.metrics['validity']:.2f}")
    print(f"Uniqueness: {result.metrics['uniqueness']:.2f}")
    print(f"Mean nodes: {result.metrics['mean_num_nodes']:.1f}")
    print(f"Mean edges: {result.metrics['mean_num_edges']:.1f}")
    print()


def demo_evolutionary_optimization():
    print("--- Evolutionary Optimization (NSGA-II) ---")
    from tgraphx.evolutionary import (
        run_evolutionary_optimization,
        list_evolutionary_optimizers,
    )

    optimizers = list_evolutionary_optimizers()
    print(f"Available algorithms: {list(optimizers.keys())}")

    result = run_evolutionary_optimization(
        algorithm="nsga2",
        objective=["connectivity", "density"],
        population_size=10,
        generations=5,
        num_nodes=10,
        seed=42,
    )
    print(f"Best fitness: {result.best_fitness:.4f}")
    if result.pareto_front is not None:
        print(f"Pareto front size: {len(result.pareto_front.genomes)}")
    print(f"Fitness history: {[f'{f:.3f}' for f in result.fitness_history[:5]]}")
    print()


if __name__ == "__main__":
    demo_graph_rl()
    demo_graph_generation()
    demo_evolutionary_optimization()
    print("All demos completed successfully.")

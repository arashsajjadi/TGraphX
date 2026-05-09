"""TGraphX Evolutionary Graph Optimization Subpackage.

Provides genetic algorithms, simulated annealing, NSGA-II multi-objective
optimization, and supporting mutation/crossover/selection operators.

Stability: Experimental (v0.7.0+).
"""
from .genome import GraphGenome
from .operators import (
    mutate_add_node,
    mutate_remove_node,
    mutate_add_edge,
    mutate_remove_edge,
    mutate_rewire_edge,
    mutate_node_feature,
    mutate_edge_feature,
    mutate_node_type,
    mutate_edge_type,
    apply_mutation,
    edge_set_crossover,
    node_induced_crossover,
    feature_crossover,
)
from .selection import (
    tournament_selection,
    roulette_wheel_selection,
    rank_selection,
    elitism_selection,
    diversity_preserving_selection,
)
from .fitness import (
    connectivity_fitness,
    density_fitness,
    clustering_fitness,
    motif_count_fitness,
    constraint_penalty,
    composite_fitness,
)
from .multi_objective import (
    pareto_dominates,
    non_dominated_sort,
    crowding_distance,
    nsga2_select,
    ParetoFront,
    compute_hypervolume_2d,
)
from .algorithms import (
    GeneticAlgorithmConfig,
    EvolutionResult,
    GeneticAlgorithmOptimizer,
    SimulatedAnnealingOptimizer,
    NSGAIIOptimizer,
    HillClimbingOptimizer,
    RandomSearchOptimizer,
)
from .config import EvolutionConfig
from .metrics import (
    best_fitness_curve,
    mean_fitness_curve,
    diversity_curve,
    mutation_acceptance_rate,
    constraint_violation_rate,
    pareto_front_size,
    hypervolume_2d,
)
from .reports import write_evolution_report
from .high_level_api import (
    run_evolutionary_optimization,
    make_evolutionary_optimizer,
    list_evolutionary_optimizers,
    OptimizationResult,
)

__all__ = [
    "GraphGenome",
    # Operators
    "mutate_add_node",
    "mutate_remove_node",
    "mutate_add_edge",
    "mutate_remove_edge",
    "mutate_rewire_edge",
    "mutate_node_feature",
    "mutate_edge_feature",
    "mutate_node_type",
    "mutate_edge_type",
    "apply_mutation",
    "edge_set_crossover",
    "node_induced_crossover",
    "feature_crossover",
    # Selection
    "tournament_selection",
    "roulette_wheel_selection",
    "rank_selection",
    "elitism_selection",
    "diversity_preserving_selection",
    # Fitness
    "connectivity_fitness",
    "density_fitness",
    "clustering_fitness",
    "motif_count_fitness",
    "constraint_penalty",
    "composite_fitness",
    # Multi-objective
    "pareto_dominates",
    "non_dominated_sort",
    "crowding_distance",
    "nsga2_select",
    "ParetoFront",
    "compute_hypervolume_2d",
    # Algorithms
    "GeneticAlgorithmConfig",
    "EvolutionResult",
    "GeneticAlgorithmOptimizer",
    "SimulatedAnnealingOptimizer",
    "NSGAIIOptimizer",
    "HillClimbingOptimizer",
    "RandomSearchOptimizer",
    # Config
    "EvolutionConfig",
    # Metrics
    "best_fitness_curve",
    "mean_fitness_curve",
    "diversity_curve",
    "mutation_acceptance_rate",
    "constraint_violation_rate",
    "pareto_front_size",
    "hypervolume_2d",
    # Reports
    "write_evolution_report",
    # High-level API
    "run_evolutionary_optimization",
    "make_evolutionary_optimizer",
    "list_evolutionary_optimizers",
    "OptimizationResult",
]

"""TGraphX Graph Generation Subpackage.

Provides classical graph generators with tensor features, neural generative
models, action spaces for MDP-based generation, and quality metrics.

Stability: Experimental (v0.7.0+).
"""
from .data_model import (
    GeneratedGraph,
    GraphEditState,
    GraphGenerationTrajectory,
    GraphGenerationBatch,
    graph_to_generation_state,
    generation_state_to_graph,
    validate_generated_graph,
    graph_generation_summary,
)
from .actions import (
    GraphActionType,
    GraphAction,
    GraphActionSpace,
    enumerate_valid_actions,
    sample_valid_action,
    apply_graph_action,
    batch_action_masks,
    action_to_index,
    index_to_action,
)
from .classical import (
    FeatureAwareERGraph,
    FeatureAwareBAGraph,
    TemporalEvolvingGraph,
    TypedGeneratedGraph,
    AnomalyInjectedGraph,
    MotifInjectedGraph,
)
from .metrics import (
    graph_wl_hash,
    validity_score,
    uniqueness_score,
    novelty_score,
    diversity_score,
    degree_distribution_distance,
    motif_distribution_distance,
    spectral_distance,
    mmd_degree,
    mmd_clustering,
    constraint_satisfaction_rate,
)
from .neural import (
    VGAEGraphGenerator,
    AutoregressiveEdgeGenerator,
    GraphTransformerGenerator,
)
from .projectors import (
    VectorNodeProjector,
    ImageNodeEncoder,
    VolumeNodeEncoder,
    EdgeFeatureProjector,
    GraphFeatureProjector,
    TensorFeatureFusion,
)
from .config import GraphGenerationConfig
from .high_level_api import (
    run_graph_generation,
    make_graph_generator,
    list_graph_generation_methods,
    GenerationResult,
)
from .reports import (
    write_graph_generation_report,
    write_generation_metrics_report,
    write_neural_generation_report,
    write_sequence_model_report,
)

__all__ = [
    # Data model
    "GeneratedGraph",
    "GraphEditState",
    "GraphGenerationTrajectory",
    "GraphGenerationBatch",
    "graph_to_generation_state",
    "generation_state_to_graph",
    "validate_generated_graph",
    "graph_generation_summary",
    # Actions
    "GraphActionType",
    "GraphAction",
    "GraphActionSpace",
    "enumerate_valid_actions",
    "sample_valid_action",
    "apply_graph_action",
    "batch_action_masks",
    "action_to_index",
    "index_to_action",
    # Classical generators
    "FeatureAwareERGraph",
    "FeatureAwareBAGraph",
    "TemporalEvolvingGraph",
    "TypedGeneratedGraph",
    "AnomalyInjectedGraph",
    "MotifInjectedGraph",
    # Metrics
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
    # Neural
    "VGAEGraphGenerator",
    "AutoregressiveEdgeGenerator",
    "GraphTransformerGenerator",
    # Projectors
    "VectorNodeProjector",
    "ImageNodeEncoder",
    "VolumeNodeEncoder",
    "EdgeFeatureProjector",
    "GraphFeatureProjector",
    "TensorFeatureFusion",
    # Config
    "GraphGenerationConfig",
    # High-level API
    "run_graph_generation",
    "make_graph_generator",
    "list_graph_generation_methods",
    "GenerationResult",
    # Reports
    "write_graph_generation_report",
    "write_generation_metrics_report",
    "write_neural_generation_report",
    "write_sequence_model_report",
]

"""TGraphX: tensor-aware graph neural networks preserving spatial node feature layouts.

The package keeps multi-dimensional node features (e.g. ``[C, H, W]`` tensors)
intact through message passing, allowing CNN-style spatial reasoning to be
combined with GNN-style relational reasoning.

Common one-liner imports::

    from tgraphx import Graph, build_model, fit, CSVLogger, env_report
"""

# Keep this in sync with [project].version in pyproject.toml.
__version__ = "1.3.1"

# ── Core data structures ──────────────────────────────────────────────────────
from .core.graph import Graph, GraphBatch
from .core.dataloader import GraphDataset, GraphDataLoader
from .core.utils import load_config, get_device
from .core.graph_utils import (
    add_self_loops,
    coalesce_edges,
    is_undirected,
    make_undirected,
    remove_self_loops,
)

# ── GNN layers ────────────────────────────────────────────────────────────────
from .layers.base import TensorMessagePassingLayer, LinearMessagePassing
from .layers.conv_message import ConvMessagePassing
from .layers.attention_message import AttentionMessagePassing
from .layers.gat import TensorGATLayer
from .layers.sage import TensorGraphSAGELayer
from .layers.gin import TensorGINLayer
# v0.3.0 model-zoo additions (vector node features).
from .layers.vector_gcn import GCNConv
from .layers.gatv2 import GATv2Conv
from .layers.appnp import APPNP
from .layers.pooling import global_mean_pool, global_sum_pool, global_max_pool

# ── Graph builders and patch helpers ─────────────────────────────────────────
from .graph_builders import (
    build_grid_graph,
    build_grid_graph_3d,
    build_fully_connected_graph,
    build_knn_graph,
    build_radius_graph,
    build_iou_graph,
    build_random_graph,
    patch_grid_shape,
    image_to_patches,
    volume_patch_grid_shape,
    volume_to_patches,
)

# ── Factories ─────────────────────────────────────────────────────────────────
from .layers.factory import make_layer
from .models.factory import build_model, build_model_from_config

# ── Model classes ─────────────────────────────────────────────────────────────
from .models.cnn_encoder import CNNEncoder
from .models.edge_predictor import EdgePredictor
from .models.regressors import NodeRegressor, GraphRegressor
from .models.graph_classifier import GraphClassifier
from .models.node_classifier import NodeClassifier

# ── Training utilities ────────────────────────────────────────────────────────
# These only import torch.nn — no heavy optional dependencies.
from .training import (
    set_seed,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    accuracy,
    mean_absolute_error,
    mean_squared_error,
    train_epoch,
    evaluate,
    fit,
)

# ── Metric logging ────────────────────────────────────────────────────────────
# CSVLogger: zero optional dependencies.
# TensorBoardLogger/MLflowLogger: class objects imported here; their heavy
# dependencies (tensorboard/mlflow) are loaded lazily only when instantiated.
from .tracking import (
    CSVLogger,
    MLflowLogger,
    TensorBoardLogger,
    write_benchmark_results,
    write_dataset_metadata,
    write_experiment_config,
    write_explanation_metadata,
    write_graph_stats,
    write_hardware_report,
    write_hetero_graph_metadata,
    write_metrics_summary,
    write_run_metadata,
    write_sampling_metadata,
    write_temporal_metadata,
    write_transform_metadata,
)

# ── Performance utilities ─────────────────────────────────────────────────────
from .performance import env_report, recommended_device, estimate_message_memory

# ── Experimental hetero / temporal containers and batches (v0.2.4 / v0.2.5) ──
from .core.hetero_graph import HeteroGraph
from .core.hetero_batch import HeteroGraphBatch
from .core.temporal import TemporalGraphSequence
from .core.temporal_batch import TemporalGraphBatch

# ── Sampling utilities (v0.2.6, extended in v0.2.8) ──────────────────────────
from .sampling import (
    induced_subgraph,
    edge_subgraph,
    k_hop_subgraph,
    sample_nodes,
    sample_edges,
    neighbor_sample,
    random_walk_sample,
)
from .sampling_loaders import SubgraphDataLoader, NeighborSamplerLoader
from .hetero_sampling import hetero_induced_subgraph, hetero_neighbor_sample
from .temporal_sampling import (
    temporal_window_sample,
    temporal_window_sample_batch,
)
# Negative sampling primitives for link-prediction (v0.3.2 beta).
from .sampling_negative import (
    negative_sampling,
    structured_negative_sampling,
    batched_negative_sampling,
    hard_negative_sampling,
)

# ── KG subsystem (v0.6.0) — imported lazily from tgraphx.kg ─────────────────
# tgraphx.kg is available as `from tgraphx.kg import KnowledgeGraph, ...`
# It is NOT re-exported at the top level to avoid heavy namespace pollution.

# ── Production-scale loaders + feature store (v0.5.0 Beta) ───────────────────
from .loaders import (
    GraphMiniBatch,
    NeighborLoader,
    LinkNeighborLoader,
    GraphLoader,
    fetch_features_for_subgraph,
    make_neighbor_loader,
    make_link_loader,
    make_graph_loader,
    map_global_to_local,
    seed_logits,
)
from .feature_store import (
    InMemoryFeatureStore,
    MemmapFeatureStore,
    FeatureStoreError,
)

# ── GraphSAINT and Cluster-GCN samplers/loaders (v0.5.0 Beta) ────────────────
from .graphsaint import (
    GraphSAINTNodeSampler,
    GraphSAINTEdgeSampler,
    GraphSAINTRandomWalkSampler,
    GraphSAINTLoader,
    estimate_norm_coefficients,
)
from .cluster_gcn import (
    PartitionResult,
    RandomBalancedPartitioner,
    BFSPartitioner,
    ConnectedComponentPartitioner,
    SpectralPartitioner,
    ClusterLoader,
)

# ── Graph generation (Experimental, v0.7.0+) ─────────────────────────────────
from .generation import (
    GeneratedGraph, GraphEditState, GraphGenerationTrajectory, GraphGenerationBatch,
    GraphActionType, GraphAction, GraphActionSpace,
    graph_to_generation_state, generation_state_to_graph, validate_generated_graph,
    apply_graph_action, sample_valid_action,
    VGAEGraphGenerator, AutoregressiveEdgeGenerator,
    VectorNodeProjector, ImageNodeEncoder, EdgeFeatureProjector,
    validity_score, uniqueness_score, novelty_score, diversity_score,
    GraphGenerationConfig,
    write_graph_generation_report, write_generation_metrics_report,
    run_graph_generation, list_graph_generation_methods, GenerationResult,
)

# ── Evolutionary optimization (Experimental, v0.7.0+) ────────────────────────
from .evolutionary import (
    GraphGenome, GeneticAlgorithmOptimizer, SimulatedAnnealingOptimizer,
    NSGAIIOptimizer, HillClimbingOptimizer,
    EvolutionConfig, EvolutionResult, ParetoFront,
    mutate_add_node, mutate_add_edge, mutate_node_feature,
    edge_set_crossover, composite_fitness,
    write_evolution_report,
    run_evolutionary_optimization, list_evolutionary_optimizers, OptimizationResult,
)

# ── Graph RL (Experimental, v0.7.0+) ─────────────────────────────────────────
from .rl import (
    GraphNavigationEnv, GraphColoringEnv, MaxCutEnv, GraphGenerationEnv,
    KGPathReasoningEnv, ShortestPathEnv,
    ContinuousNavigationEnv, ContinuousGraphEditEnv, ContinuousGraphActionSpace,
    GraphPolicyNetwork, GraphValueNetwork, GraphQNetwork, GraphActorCriticNetwork,
    MaskedCategoricalPolicy,
    REINFORCEAgent, ActorCriticAgent, A2CAgent, DQNAgent, DoubleDQNAgent, PPOAgent,
    GraphDDPGAgent, GraphTD3Agent, GraphSACAgent,
    ContinuousGraphActor, StochasticGraphActor, ContinuousGraphCritic, TwinContinuousGraphCritic,
    RandomPolicy, GreedyPolicy,
    ReplayBuffer, RolloutBuffer, EpsilonGreedy,
    soft_update, OUNoise, GaussianNoise,
    RLTrainingConfig, PolicyConfig,
    write_graph_rl_training_report, write_graph_rl_env_report,
    run_graph_rl, list_graph_rl_algorithms, make_graph_env, RLResult,
)

# ── Easy mode (optional high-level API, v1.0.1+) ─────────────────────────────
# Import as `import tgraphx as tgx; tgx.easy.train_node_classifier(...)`.
# Imported here as a module so users can do `tgx.easy.xxx`.
from . import easy

__all__ = [
    "__version__",
    "easy",
    # Core
    "Graph",
    "GraphBatch",
    "GraphDataset",
    "GraphDataLoader",
    "load_config",
    "get_device",
    # Graph utilities
    "add_self_loops",
    "coalesce_edges",
    "is_undirected",
    "make_undirected",
    "remove_self_loops",
    # GNN layers
    "TensorMessagePassingLayer",
    "LinearMessagePassing",
    "ConvMessagePassing",
    "AttentionMessagePassing",
    "TensorGATLayer",
    "TensorGraphSAGELayer",
    "TensorGINLayer",
    # Vector model-zoo (v0.3.0)
    "GCNConv",
    "GATv2Conv",
    "APPNP",
    "global_mean_pool",
    "global_sum_pool",
    "global_max_pool",
    # Graph builders
    "build_grid_graph",
    "build_grid_graph_3d",
    "build_fully_connected_graph",
    "build_knn_graph",
    "build_radius_graph",
    "build_iou_graph",
    "build_random_graph",
    # Patch helpers
    "patch_grid_shape",
    "image_to_patches",
    "volume_patch_grid_shape",
    "volume_to_patches",
    # Factories
    "make_layer",
    "build_model",
    "build_model_from_config",
    # Model classes
    "CNNEncoder",
    "EdgePredictor",
    "NodeRegressor",
    "GraphRegressor",
    "GraphClassifier",
    "NodeClassifier",
    # Training utilities
    "set_seed",
    "count_parameters",
    "save_checkpoint",
    "load_checkpoint",
    "accuracy",
    "mean_absolute_error",
    "mean_squared_error",
    "train_epoch",
    "evaluate",
    "fit",
    # Metric logging + dashboard metadata writers
    "CSVLogger",
    "TensorBoardLogger",
    "MLflowLogger",
    "write_graph_stats",
    "write_run_metadata",
    "write_dataset_metadata",
    "write_transform_metadata",
    "write_metrics_summary",
    "write_benchmark_results",
    "write_explanation_metadata",
    "write_experiment_config",
    "write_hardware_report",
    "write_sampling_metadata",
    "write_hetero_graph_metadata",
    "write_temporal_metadata",
    # Performance utilities
    "env_report",
    "recommended_device",
    "estimate_message_memory",
    # Experimental hetero / temporal containers and batches
    "HeteroGraph",
    "HeteroGraphBatch",
    "TemporalGraphSequence",
    "TemporalGraphBatch",
    # Sampling
    "induced_subgraph",
    "edge_subgraph",
    "k_hop_subgraph",
    "sample_nodes",
    "sample_edges",
    "neighbor_sample",
    "random_walk_sample",
    "SubgraphDataLoader",
    "NeighborSamplerLoader",
    "hetero_induced_subgraph",
    "hetero_neighbor_sample",
    "temporal_window_sample",
    "temporal_window_sample_batch",
    # Negative sampling (v0.3.2 beta)
    "negative_sampling",
    "structured_negative_sampling",
    "batched_negative_sampling",
    "hard_negative_sampling",
    # v0.5.0 Beta: production loaders + feature store
    "GraphMiniBatch",
    "NeighborLoader",
    "LinkNeighborLoader",
    "GraphLoader",
    "fetch_features_for_subgraph",
    "make_neighbor_loader",
    "make_link_loader",
    "make_graph_loader",
    "map_global_to_local",
    "seed_logits",
    "InMemoryFeatureStore",
    "MemmapFeatureStore",
    "FeatureStoreError",
    # v0.5.0 Beta: GraphSAINT / Cluster-GCN
    "GraphSAINTNodeSampler",
    "GraphSAINTEdgeSampler",
    "GraphSAINTRandomWalkSampler",
    "GraphSAINTLoader",
    "estimate_norm_coefficients",
    "PartitionResult",
    "RandomBalancedPartitioner",
    "BFSPartitioner",
    "ConnectedComponentPartitioner",
    "SpectralPartitioner",
    "ClusterLoader",
    # Graph algorithms (v0.3.2 beta)
    # (imported from tgraphx.algorithms; not re-exported at top level
    #  to avoid namespace pollution)
]

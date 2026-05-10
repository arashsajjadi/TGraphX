"""Tensor-aware Knowledge Graph subsystem for TGraphX.

Provides:
- KnowledgeGraph   — directed multi-relational container with tensor features
- negative samplers — Uniform, Bernoulli, Filtered
- KGEvaluator      — chunked head+tail filtered ranking
- KG model zoo     — TransE, DistMult, ComplEx, RotatE
- KGTrainer        — reproducible, dashboard-aware training
- KG+GNN           — RGCN KG-completion integration
- Temporal KG      — timestamps, chronological split, temporal filtered eval
- KG reasoning     — path extraction, Horn-rule candidate generation
- KG datasets      — synthetic family/academic/multimodal KGs

TGraphX does **not** claim to replace PyKEEN, DGL-KE, or any production
KG platform. This subsystem focuses on tensor-aware KG learning and
TGraphX-native workflow integration.

Stability:
- KnowledgeGraph data model: Beta
- Negative sampling (Uniform/Bernoulli/Filtered): Beta
- Filtered ranking evaluator: Beta
- TransE, DistMult: Beta
- ComplEx, RotatE: Experimental
- KGTrainer: Experimental
- KG+GNN (RGCN): Experimental
- Temporal KG: Experimental
- KG reasoning: Experimental
"""
from __future__ import annotations

from .data import KnowledgeGraph, TemporalKnowledgeGraph
from .projectors import (
    VectorEntityProjector,
    ImageEntityProjector,
    TextEntityProjector,
    UserEntityProjector,
    RelationFeatureProjector,
    TripleFeatureProjector,
    MultimodalEntityFusion,
)
from .multimodal import MultimodalKGModel
from .sampling import (
    UniformNegativeSampler,
    BernoulliNegativeSampler,
    FilteredNegativeSampler,
    TypedNegativeSampler,
)
from .evaluation import KGEvaluator, evaluate_filtered_ranking
from .models import TransEModel, DistMultModel, ComplExModel, RotatEModel
from .losses import MarginRankingLoss, BCEKGLoss, SoftplusKGLoss
from .trainer import KGTrainer, KGTrainingConfig
from .gnn import KGRGCNModel, kg_to_edge_index
from .temporal import TemporalKGNegativeSampler, evaluate_temporal_filtered_ranking
from .reasoning import PathExtractor, HornRuleCandidate, LogicalConstraintChecker
from .datasets import (
    FamilyKG,
    AcademicKG,
    MultimodalKG,
    generate_synthetic_kg,
)
from .reports import (
    write_kg_summary,
    write_kg_evaluation_report,
    write_kg_training_report,
    write_kg_model_report,
    write_kg_gnn_report,
    write_temporal_kg_report,
    write_kg_reasoning_report,
    write_kg_benchmark_report,
    write_kg_multimodal_feature_report,
)

_KG_MODELS: dict = {
    "TransE": "Translation-based embedding (Bordes et al., 2013).",
    "DistMult": "Diagonal bilinear scoring (Yang et al., 2015).",
    "ComplEx": "Complex-valued embeddings (Trouillon et al., 2016).",
    "RotatE": "Rotation-based relational embedding (Sun et al., 2019).",
}


def list_kg_models() -> dict:
    """Return available KG embedding model names and descriptions."""
    return dict(_KG_MODELS)


__all__ = [
    # Data model
    "KnowledgeGraph",   # .from_hrt() and .from_triples() are classmethods
    "TemporalKnowledgeGraph",
    # Multimodal projectors
    "VectorEntityProjector",
    "ImageEntityProjector",
    "TextEntityProjector",
    "UserEntityProjector",
    "RelationFeatureProjector",
    "TripleFeatureProjector",
    "MultimodalEntityFusion",
    "MultimodalKGModel",
    # Samplers
    "UniformNegativeSampler",
    "BernoulliNegativeSampler",
    "FilteredNegativeSampler",
    "TypedNegativeSampler",
    # Evaluation
    "KGEvaluator",
    "evaluate_filtered_ranking",
    # Models
    "TransEModel",
    "DistMultModel",
    "ComplExModel",
    "RotatEModel",
    # Losses
    "MarginRankingLoss",
    "BCEKGLoss",
    "SoftplusKGLoss",
    # Trainer
    "KGTrainer",
    "KGTrainingConfig",
    # GNN integration
    "KGRGCNModel",
    "kg_to_edge_index",
    # Temporal
    "TemporalKGNegativeSampler",
    "evaluate_temporal_filtered_ranking",
    # Reasoning
    "PathExtractor",
    "HornRuleCandidate",
    "LogicalConstraintChecker",
    # Datasets
    "FamilyKG",
    "AcademicKG",
    "MultimodalKG",
    "generate_synthetic_kg",
    # Discovery
    "list_kg_models",
    # Reports
    "write_kg_summary",
    "write_kg_evaluation_report",
    "write_kg_training_report",
    "write_kg_model_report",
    "write_kg_gnn_report",
    "write_temporal_kg_report",
    "write_kg_reasoning_report",
    "write_kg_benchmark_report",
    "write_kg_multimodal_feature_report",
]

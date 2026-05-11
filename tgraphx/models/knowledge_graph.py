"""Compatibility shim that re-exports KG models from ``tgraphx.kg``.

LLM-generated code often guesses ``from tgraphx.models.knowledge_graph import TransEModel``
because that mirrors a common naming convention. The canonical path is
``from tgraphx.kg import TransEModel`` — both work.

Stability: Beta (compatibility re-export; canonical APIs live in ``tgraphx.kg``).
"""
from __future__ import annotations

from tgraphx.kg import (
    KnowledgeGraph,
    TransEModel,
    DistMultModel,
    ComplExModel,
    RotatEModel,
    RESCALModel,
    SimplEModel,
    KGTrainer,
    KGTrainingConfig,
)

__all__ = [
    "KnowledgeGraph",
    "TransEModel",
    "DistMultModel",
    "ComplExModel",
    "RotatEModel",
    "RESCALModel",
    "SimplEModel",
    "KGTrainer",
    "KGTrainingConfig",
]

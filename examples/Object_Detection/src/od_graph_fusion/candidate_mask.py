"""Central candidate-node mask — single source of truth.

Use candidate_node_mask everywhere:
  compute_source_utilities, build_source_slot_labels,
  fuse_v3, metrics, graph oracle, exact node accuracy, source distribution.

Never hard-code NODE_TYPES values as fallback integers.
"""
from __future__ import annotations
from typing import Dict
import torch


_CANDIDATE_KEYS = (
    "proposal",
    "cluster",              # WBF aggregate
    "consensus",            # Union aggregate
    "nms_candidate",
    "soft_nms_candidate",
    "best_proposal_candidate",
    "calibrated_consensus",  # optional, skipped if absent
)


def candidate_node_mask(node_types: torch.Tensor, node_types_dict: Dict[str, int]) -> torch.Tensor:
    """Return a boolean mask of shape [N] that is True for any candidate source node.

    Uses guarded key-existence checks — never a bare .get(..., fallback_int).
    Context nodes and unknown types are always False.

    Args:
        node_types:       [N] long tensor of node type indices
        node_types_dict:  NODE_TYPES dict mapping type-name → int

    Returns:
        [N] bool tensor
    """
    mask = torch.zeros(node_types.shape[0], dtype=torch.bool, device=node_types.device)
    for key in _CANDIDATE_KEYS:
        if key in node_types_dict:
            mask |= (node_types == node_types_dict[key])
    return mask

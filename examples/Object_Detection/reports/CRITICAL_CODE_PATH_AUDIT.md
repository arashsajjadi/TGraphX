# Critical Code-Path Audit

## 1. Trained Class

| Item | Value |
|------|-------|
| type(model).__name__ | `TGraphXSourceRouterV3` |
| module | `od_graph_fusion.source_router_v3` |
| uses_source_logits | **YES** (output key `"source_logits"`) |
| uses_source_slot_loss | **PARTIAL** — inline experiment scripts YES; `training.py` pipeline NO |
| uses_fuse_v3 | YES (inline); NO in `training.py` eval path |
| uses_legacy_fuse_with_model | YES in `training.py` eval |

## 2. Forward Source Slot Visibility — CRITICAL BUG

`TGraphXSourceRouterV3._build_node_source_slots` (source_router_v3.py:224) maps:

| Node Type | Slot | Status |
|-----------|------|--------|
| proposal | detector_name_to_slot(det) | ✓ mapped |
| cluster | wbf (5) | ✓ mapped |
| consensus | union (4) | ✓ mapped |
| **nms_candidate** | — | **MISSING → slot=-1** |
| **soft_nms_candidate** | — | **MISSING → slot=-1** |
| **best_proposal_candidate** | — | **MISSING → slot=-1** |

**Result:** Inside `model.forward()`, NMS/Soft-NMS/BestProposal nodes are invisible to `SourceSlotAggregator`. `slot_mask[c, 6/7/8]` is always False. The router cannot select these sources. `fuse_v3` patches slot_assignments post-hoc, but the forward-time slot_mask already excluded them — so the source logits for slots 6/7/8 are trained on empty embeddings.

## 3. training.py Pipeline — WRONG TRAINING PATH

`training.py:98-119` for V3 models:
- Uses `quality_logits` (not `source_logits`)
- Uses old `source_routing_loss` (per-node, not per-slot)
- `best_src_pc[c]` = **node index**, not slot index
- Loss: old IoU-based node utility, not slot utility

The inline experiment scripts used in ad-hoc runs are correct (source_logits + source_slot_loss + slot indices). The pipeline scripts are broken. These are two different code paths.

## 4. Version Mismatch

| Pattern | Location | Issue |
|---------|----------|-------|
| `source_routing_loss` used | training.py:69,118,211 | Legacy loss on V3 model |
| `best_src_pc = node_index` | training.py:112,115 | Node index ≠ slot index |
| Old `source_routing_loss` | source_router.py:141 | Not deprecated, still called |

## 5. Strict Source Router

Not yet implemented. `training.py` silently falls into legacy path for V3 models without raising an error.

## 6. Root Cause of AP < NMS

Two causes:
1. `_build_node_source_slots` omits NMS/SoftNMS/BestProposal → slot_mask wrong → source logits for those slots untrained
2. Oracle utility = continuous IoU (includes IoU < 0.5) → model optimizes IoU, not AP50
3. Graph construction: multi-detector clustering can destroy yolo_modern's clean detections (verified: raw yolo AP=1.0, cluster oracle AP<1.0)

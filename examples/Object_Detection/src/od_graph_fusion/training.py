"""Training loop for TGraphX fusion / source-router."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_builder import DetectionGraphMeta, NODE_TYPES
from .models import DetectionFusionModel


def _supervised_mask(meta: DetectionGraphMeta) -> torch.Tensor:
    nt = meta.node_types
    if meta.targets is not None and "candidate_mask" in meta.targets:
        return meta.targets["candidate_mask"]
    return ((nt == NODE_TYPES["proposal"])
            | (nt == NODE_TYPES["cluster"])
            | (nt == NODE_TYPES["consensus"]))


def train_fusion_model(
    train_graphs: List[Tuple[Any, DetectionGraphMeta]],
    val_graphs: List[Tuple[Any, DetectionGraphMeta]],
    num_classes: int,
    num_detectors: int,
    crop_size: int,
    crop_channels: int = 16,
    hidden_dim: int = 48,
    num_message_passing: int = 2,
    epochs: int = 3,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    objectness_weight: float = 1.0,
    class_weight: float = 0.5,
    box_weight: float = 0.0,
    device: str = "auto",
    log_every: int = 1,
    use_source_router: bool = True,
    detector_names: Optional[List[str]] = None,
    utility_mode: str = "ap50",
    class_agnostic: bool = True,
    strict_source_router: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Train TGraphXSourceRouter (default) or legacy DetectionFusionModel.

    Args:
      detector_names: REQUIRED for V3. Must be non-empty and a list of strings.
        Passing an empty list with V3 makes proposal nodes invisible to the
        slot aggregator (no slot mapping), which silently collapses training
        to the auxiliary heads only — the source-router fails to train.
      utility_mode: AP-aware utility mode passed to _build_util_and_labels.
        One of "iou" | "ap50" | "ap75" | "deployable" | "anchor_delta_ap50".
      class_agnostic: false for multi-class — utility uses class-aware match.
      strict_source_router: when True (default) and use_source_router=True,
        the legacy objectness/class/box loss branch is disabled — any
        unexpected fall-through raises instead of silently training a
        different objective.
    """
    from .config import resolve_device, device_audit
    device = resolve_device(device)
    audit = device_audit(device, device)
    import warnings
    if audit["cuda_available"] and device == "cpu":
        warnings.warn(
            f"CUDA is available ({audit['gpu_name']}) but training on CPU. "
            "Set device='auto' or device='cuda' in your config.", stacklevel=2
        )
    if not train_graphs:
        raise ValueError("train_fusion_model: no training graphs")
    if use_source_router and (detector_names is None or len(detector_names) == 0):
        raise RuntimeError(
            "train_fusion_model: detector_names is required for V3 source router. "
            "Got None or empty. Pass detector_names=manifest['detector_names'] "
            "from step 03's split_manifest.json. Passing an empty list silently "
            "disables proposal-slot routing and trains the wrong objective."
        )
    g0, _ = train_graphs[0]
    md = g0.metadata.get("node_metadata")
    metadata_dim = md.shape[1] if md is not None else None
    edge_feat_dim = (g0.edge_features.shape[1]
                     if g0.edge_features is not None and g0.edge_features.numel() > 0
                     else 14)

    if use_source_router:
        from .source_router_v3 import TGraphXSourceRouterV3
        model = TGraphXSourceRouterV3(
            num_classes=num_classes, num_detectors=num_detectors,
            crop_size=crop_size, crop_channels=crop_channels,
            hidden_dim=hidden_dim, metadata_dim=metadata_dim,
            edge_feat_dim=edge_feat_dim, num_message_passing=num_message_passing,
        ).to(device)
    else:
        model = DetectionFusionModel(
            num_classes=num_classes, num_detectors=num_detectors,
            crop_size=crop_size, crop_channels=crop_channels,
            hidden_dim=hidden_dim, metadata_dim=metadata_dim,
            edge_feat_dim=edge_feat_dim, num_message_passing=num_message_passing,
        ).to(device)

    from .source_router import source_routing_loss
    from .source_router_v3 import TGraphXSourceRouterV3 as SRV3Class, source_slot_loss
    is_router = isinstance(model, SRV3Class)
    if use_source_router and not is_router:
        raise RuntimeError(
            "train_fusion_model: use_source_router=True but constructed model is not "
            f"TGraphXSourceRouterV3 (got {type(model).__name__}). Refusing to train "
            "the legacy objectness loss against a router model."
        )

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: Dict[str, Any] = {
        "train_loss": [], "val_loss": [], "val_objectness_acc": [],
        "epoch_time_s": [], "device": device,
    }
    best_val = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total = 0.0
        n_graphs = 0
        for graph, meta in train_graphs:
            if meta.targets is None:
                continue
            g = graph.to(device)
            # Use the explicit detector_names argument. Do NOT fall back silently
            # to an empty list — that path was the root cause of v8/v9 routing failure.
            if is_router:
                _det_names = list(detector_names) if detector_names else []
                if not _det_names:
                    raise RuntimeError(
                        "train_fusion_model: empty detector_names reached the train loop. "
                        "Step 03 must record split_manifest['detector_names']."
                    )
                out = model(g, detector_names=_det_names)
            else:
                out = model(g)
            mask = _supervised_mask(meta).to(device)
            if mask.sum() == 0:
                continue

            if is_router:
                # V3 path: source_slot_loss on source_logits (not quality_logits)
                from .multi_seed_v2 import _build_util_and_labels
                src_logits = out.get("source_logits")
                src_mask = out.get("source_mask")
                if src_logits is None:
                    continue
                _det_names_list = list(_det_names) if _det_names else []
                # GT boxes: try graph.metadata (stored by step 03), then meta attribute
                if isinstance(graph.metadata, dict):
                    gt_b = graph.metadata.get("gt_boxes", getattr(meta, "gt_boxes", None))
                    gt_l = graph.metadata.get("gt_labels", getattr(meta, "gt_labels", None))
                else:
                    gt_b = getattr(meta, "gt_boxes", None)
                    gt_l = getattr(meta, "gt_labels", None)
                if gt_b is None or gt_l is None or gt_b.numel() == 0:
                    continue
                util_result = _build_util_and_labels(
                    graph, meta, gt_b.to(device), gt_l.to(device),
                    class_agnostic=class_agnostic, baseline_source="nms_candidate",
                    utility_mode=utility_mode,
                )
                if util_result is None:
                    continue
                _, best_slot, bl_slot, ups, slot_avail = util_result
                valid = (best_slot >= 0)
                if not valid.any():
                    continue
                losses = source_slot_loss(
                    src_logits[valid], src_mask[valid],
                    best_slot[valid].to(device), ups[valid].to(device),
                    baseline_slot=bl_slot[valid].to(device),
                    regret_lambda=2.0,
                )
                loss = losses["total"]
                if not loss.requires_grad:
                    continue
            else:
                if strict_source_router and use_source_router:
                    raise RuntimeError(
                        "Unexpected legacy loss branch under strict_source_router=True. "
                        "The V3 router fell through; this would silently train objectness "
                        "loss instead of source routing. Check is_router and src_logits."
                    )
                obj_target = meta.targets["objectness"].to(device)
                cls_target = meta.targets["class"].to(device)
                box_target = meta.targets["box_reg"].to(device)
                obj_l = out["objectness_logits"][mask]
                cls_l = out["class_logits"][mask]
                box_r = out["box_reg"][mask]
                loss_obj = F.binary_cross_entropy_with_logits(obj_l, obj_target[mask])
                valid_cls = cls_target[mask] >= 0
                loss_cls = (F.cross_entropy(cls_l[valid_cls], cls_target[mask][valid_cls])
                            if valid_cls.any() else torch.tensor(0.0, device=device))
                valid_box = obj_target[mask] > 0.5
                loss_box = (F.smooth_l1_loss(box_r[valid_box], box_target[mask][valid_box])
                            if valid_box.any() else torch.tensor(0.0, device=device))
                loss = (objectness_weight * loss_obj + class_weight * loss_cls
                        + box_weight * loss_box)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.item())
            n_graphs += 1

        avg = total / max(1, n_graphs)
        epoch_t = time.time() - t0
        history["train_loss"].append(avg)
        history["epoch_time_s"].append(epoch_t)

        val_loss, val_obj_acc = _evaluate_loss(
            model, val_graphs, device,
            objectness_weight, class_weight, box_weight,
            is_router=is_router,
        )
        history["val_loss"].append(val_loss)
        history["val_objectness_acc"].append(val_obj_acc)
        if val_loss < best_val:
            best_val = val_loss

        if ep % log_every == 0 or ep == epochs:
            print(f"[fusion] epoch {ep}/{epochs}  "
                  f"train_loss={avg:.4f}  val_loss={val_loss:.4f}  "
                  f"obj_acc={val_obj_acc:.3f}  ({epoch_t:.1f}s)")

    history["best_val_loss"] = best_val
    return model, history


def _evaluate_loss(
    model: nn.Module,
    graphs: List[Tuple[Any, DetectionGraphMeta]],
    device: str,
    w_obj: float,
    w_cls: float,
    w_box: float,
    is_router: bool = False,
) -> Tuple[float, float]:
    from .source_router import TGraphXSourceRouter as SRClass
    from .source_router_v3 import TGraphXSourceRouterV3 as SRV3Class, source_slot_loss
    model.eval()
    total = 0.0
    n = 0
    correct = 0
    seen = 0
    with torch.no_grad():
        for graph, meta in graphs:
            g = graph.to(device)
            _det_names = (meta.detector_names if hasattr(meta, "detector_names") else
                          g.metadata.get("detector_names", []) if isinstance(g.metadata, dict) else [])
            if isinstance(model, SRV3Class):
                out = model(g, detector_names=list(_det_names) if _det_names else [])
            else:
                out = model(g)
            mask = _supervised_mask(meta).to(device)

            if isinstance(model, SRV3Class):
                from .multi_seed_v2 import _build_util_and_labels
                src_logits = out.get("source_logits")
                src_mask_t = out.get("source_mask")
                if src_logits is None: continue
                # GT from graph.metadata (set by step 03)
                if isinstance(graph.metadata, dict):
                    gt_b = graph.metadata.get("gt_boxes", getattr(meta, "gt_boxes", None))
                    gt_l = graph.metadata.get("gt_labels", getattr(meta, "gt_labels", None))
                else:
                    gt_b = getattr(meta, "gt_boxes", None)
                    gt_l = getattr(meta, "gt_labels", None)
                if gt_b is None or gt_l is None or gt_b.numel() == 0: continue
                util_result = _build_util_and_labels(graph, meta, gt_b, gt_l, True, "nms_candidate")
                if util_result is None: continue
                _, best_slot, bl_slot, ups, _ = util_result
                valid = (best_slot >= 0)
                if not valid.any(): continue
                losses = source_slot_loss(src_logits[valid], src_mask_t[valid],
                                          best_slot[valid], ups[valid],
                                          baseline_slot=bl_slot[valid])
                loss = losses["total"]
                total += float(loss.item()); n += 1
                # Source slot accuracy for val
                pred_slot = src_logits[valid].masked_fill(~src_mask_t[valid], float("-inf")).argmax(dim=-1)
                correct += int((pred_slot == best_slot[valid]).sum().item())
                seen += int(valid.sum().item())
                continue
            elif is_router or isinstance(model, SRClass):
                from .source_router import source_routing_loss
                quality = out["quality_logits"]
                iou_t = meta.targets.get("iou", torch.zeros(quality.shape[0])).to(device)
                cluster_of = meta.cluster_of_node.to(device)
                n_clusters = int(cluster_of.max().item() + 1) if cluster_of.numel() > 0 else 0
                best_src_pc = torch.full((n_clusters,), -1, dtype=torch.long, device=device)
                best_s = meta.targets.get("is_best_source", None)
                for c in range(n_clusters):
                    in_c = (cluster_of == c) & mask
                    if not in_c.any(): continue
                    if best_s is not None:
                        best_in_c = in_c & (best_s.to(device) > 0.5)
                        if best_in_c.any():
                            best_src_pc[c] = int(best_in_c.nonzero(as_tuple=False)[0].item()); continue
                    idx_c = in_c.nonzero(as_tuple=False).squeeze(-1)
                    best_src_pc[c] = int(idx_c[iou_t[idx_c].argmax()].item())
                losses = source_routing_loss(quality, iou_t, best_src_pc, cluster_of, mask)
                loss = losses["total"]
                total += float(loss.item()); n += 1
                # Quality accuracy: for each cluster, does highest-quality node match best?
                for c in range(n_clusters):
                    in_c = (cluster_of == c) & mask
                    if not in_c.any() or best_src_pc[c] < 0:
                        continue
                    idx_c = in_c.nonzero(as_tuple=False).squeeze(-1)
                    pred_best = int(idx_c[quality[idx_c].argmax()].item())
                    correct += int(pred_best == int(best_src_pc[c].item()))
                    seen += 1
            else:
                obj_target = meta.targets["objectness"].to(device)
                cls_target = meta.targets["class"].to(device)
                box_target = meta.targets["box_reg"].to(device)
                obj_l = out["objectness_logits"][mask]
                cls_l = out["class_logits"][mask]
                box_r = out["box_reg"][mask]
                loss_obj = F.binary_cross_entropy_with_logits(obj_l, obj_target[mask])
                valid_cls = cls_target[mask] >= 0
                loss_cls = (F.cross_entropy(cls_l[valid_cls], cls_target[mask][valid_cls])
                            if valid_cls.any() else torch.tensor(0.0, device=device))
                valid_box = obj_target[mask] > 0.5
                loss_box = (F.smooth_l1_loss(box_r[valid_box], box_target[mask][valid_box])
                            if valid_box.any() else torch.tensor(0.0, device=device))
                loss = w_obj * loss_obj + w_cls * loss_cls + w_box * loss_box
                total += float(loss.item()); n += 1
                preds = (obj_l > 0.0).long()
                correct += (preds == obj_target[mask].long()).sum().item()
                seen += int(mask.sum().item())

    return total / max(1, n), (correct / max(1, seen))

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
    device: str = "cpu",
    log_every: int = 1,
    use_source_router: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Train TGraphXSourceRouter (default) or legacy DetectionFusionModel."""
    if not train_graphs:
        raise ValueError("train_fusion_model: no training graphs")
    g0, _ = train_graphs[0]
    md = g0.metadata.get("node_metadata")
    metadata_dim = md.shape[1] if md is not None else None
    edge_feat_dim = (g0.edge_features.shape[1]
                     if g0.edge_features is not None and g0.edge_features.numel() > 0
                     else 14)

    if use_source_router:
        from .source_router import TGraphXSourceRouter
        model = TGraphXSourceRouter(
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

    from .source_router import source_routing_loss, TGraphXSourceRouter as SRClass
    is_router = isinstance(model, SRClass)

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
            out = model(g)
            mask = _supervised_mask(meta).to(device)
            if mask.sum() == 0:
                continue

            if is_router:
                quality = out["quality_logits"]
                iou_t = meta.targets.get("iou", torch.zeros(quality.shape[0])).to(device)
                best_s = meta.targets.get("is_best_source", None)
                cluster_of = meta.cluster_of_node.to(device)
                n_clusters = int(cluster_of.max().item() + 1) if cluster_of.numel() > 0 else 0
                best_src_pc = torch.full((n_clusters,), -1, dtype=torch.long, device=device)
                for c in range(n_clusters):
                    in_c = (cluster_of == c) & mask
                    if not in_c.any():
                        continue
                    if best_s is not None:
                        best_in_c = in_c & (best_s.to(device) > 0.5)
                        if best_in_c.any():
                            best_src_pc[c] = int(best_in_c.nonzero(as_tuple=False)[0].item())
                            continue
                    idx_c = in_c.nonzero(as_tuple=False).squeeze(-1)
                    best_src_pc[c] = int(idx_c[iou_t[idx_c].argmax()].item())
                ns = graph.metadata.get("node_score")
                base_s = ns.to(device) if ns is not None else None
                losses = source_routing_loss(quality, iou_t, best_src_pc, cluster_of, mask,
                                              baseline_scores=base_s)
                loss = losses["total"]
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
    from .source_router import source_routing_loss, TGraphXSourceRouter as SRClass
    model.eval()
    total = 0.0
    n = 0
    correct = 0
    seen = 0
    with torch.no_grad():
        for graph, meta in graphs:
            if meta.targets is None:
                continue
            g = graph.to(device)
            out = model(g)
            mask = _supervised_mask(meta).to(device)
            if mask.sum() == 0:
                continue

            if is_router or isinstance(model, SRClass):
                quality = out["quality_logits"]
                iou_t = meta.targets.get("iou", torch.zeros(quality.shape[0])).to(device)
                cluster_of = meta.cluster_of_node.to(device)
                n_clusters = int(cluster_of.max().item() + 1) if cluster_of.numel() > 0 else 0
                best_src_pc = torch.full((n_clusters,), -1, dtype=torch.long, device=device)
                best_s = meta.targets.get("is_best_source", None)
                for c in range(n_clusters):
                    in_c = (cluster_of == c) & mask
                    if not in_c.any():
                        continue
                    if best_s is not None:
                        best_in_c = in_c & (best_s.to(device) > 0.5)
                        if best_in_c.any():
                            best_src_pc[c] = int(best_in_c.nonzero(as_tuple=False)[0].item())
                            continue
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

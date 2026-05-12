"""Training loop for the TGraphX fusion model."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_builder import DetectionGraphMeta, NODE_TYPES
from .models import DetectionFusionModel


def _supervised_mask(meta: DetectionGraphMeta) -> torch.Tensor:
    """Return a boolean mask over nodes selecting candidate nodes
    (proposal + cluster + consensus). v1.1: proposals are now scored too."""
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
    crop_channels: int = 32,
    hidden_dim: int = 64,
    num_message_passing: int = 2,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    objectness_weight: float = 1.0,
    class_weight: float = 0.5,
    box_weight: float = 1.0,
    device: str = "cpu",
    log_every: int = 1,
) -> Tuple[DetectionFusionModel, Dict[str, Any]]:
    """Train the fusion model. Returns model + history dict."""
    if not train_graphs:
        raise ValueError("train_fusion_model: no training graphs")
    # Infer feature dims from first graph
    g0, _ = train_graphs[0]
    md = g0.metadata.get("node_metadata")
    metadata_dim = md.shape[1] if md is not None else None
    edge_feat_dim = g0.edge_features.shape[1] if g0.edge_features is not None and g0.edge_features.numel() > 0 else 14

    model = DetectionFusionModel(
        num_classes=num_classes, num_detectors=num_detectors,
        crop_size=crop_size, crop_channels=crop_channels,
        hidden_dim=hidden_dim, metadata_dim=metadata_dim,
        edge_feat_dim=edge_feat_dim,
        num_message_passing=num_message_passing,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_objectness_acc": [],
               "epoch_time_s": [], "device": device}
    best_val = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total = 0.0; n_graphs = 0
        for graph, meta in train_graphs:
            if meta.targets is None:
                continue
            g = graph.to(device)
            out = model(g)
            mask = _supervised_mask(meta).to(device)
            if mask.sum() == 0:
                continue
            obj_target = meta.targets["objectness"].to(device)
            cls_target = meta.targets["class"].to(device)
            box_target = meta.targets["box_reg"].to(device)

            obj_logits = out["objectness_logits"][mask]
            cls_logits = out["class_logits"][mask]
            box_reg = out["box_reg"][mask]

            loss_obj = F.binary_cross_entropy_with_logits(
                obj_logits, obj_target[mask],
            )

            valid_cls = cls_target[mask] >= 0
            loss_cls = (F.cross_entropy(cls_logits[valid_cls], cls_target[mask][valid_cls])
                        if valid_cls.any() else torch.tensor(0.0, device=device))

            valid_box = obj_target[mask] > 0.5
            loss_box = (F.smooth_l1_loss(box_reg[valid_box], box_target[mask][valid_box])
                        if valid_box.any() else torch.tensor(0.0, device=device))

            loss = (objectness_weight * loss_obj
                    + class_weight * loss_cls
                    + box_weight * loss_box)
            optim.zero_grad(); loss.backward(); optim.step()
            total += float(loss.item()); n_graphs += 1

        avg = total / max(1, n_graphs)
        epoch_t = time.time() - t0
        history["train_loss"].append(avg)
        history["epoch_time_s"].append(epoch_t)

        val_loss, val_obj_acc = _evaluate_loss(model, val_graphs, device,
                                                objectness_weight, class_weight, box_weight)
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
    w_obj: float, w_cls: float, w_box: float,
) -> Tuple[float, float]:
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
            obj_target = meta.targets["objectness"].to(device)
            cls_target = meta.targets["class"].to(device)
            box_target = meta.targets["box_reg"].to(device)

            obj_logits = out["objectness_logits"][mask]
            cls_logits = out["class_logits"][mask]
            box_reg = out["box_reg"][mask]

            loss_obj = F.binary_cross_entropy_with_logits(obj_logits, obj_target[mask])
            valid_cls = cls_target[mask] >= 0
            loss_cls = (F.cross_entropy(cls_logits[valid_cls], cls_target[mask][valid_cls])
                        if valid_cls.any() else torch.tensor(0.0, device=device))
            valid_box = obj_target[mask] > 0.5
            loss_box = (F.smooth_l1_loss(box_reg[valid_box], box_target[mask][valid_box])
                        if valid_box.any() else torch.tensor(0.0, device=device))
            loss = w_obj * loss_obj + w_cls * loss_cls + w_box * loss_box
            total += float(loss.item()); n += 1

            preds = (obj_logits > 0.0).long()
            correct += (preds == obj_target[mask].long()).sum().item()
            seen += int(mask.sum().item())
    return total / max(1, n), (correct / max(1, seen))

# UNIVERSAL CANDIDATE NODE SELECTION — design doc

**Date:** 2026-05-13
**Status:** paper-faithful TGraphX formulation. Replaces source-routing
and learned-box-fusion attempts as the main path.

## Task definition (one paragraph)

For each detection cluster in an image, build a TGraphX graph whose
nodes are **all available candidate detection boxes** — raw detector
proposals (one per detector) plus fusion-method proposals (NMS, Soft-
NMS, WBF, Union, BestProposal). Each node carries the image crop
under its candidate box as a tensor `[3, H, W]`, plus its box,
confidence, label, source id, and edge features over geometric/source
relations. A `TGraphXCandidateNodeSelector` runs a CNN encoder over
each node's crop, performs tensor-aware graph message passing across
candidate nodes, and predicts a per-node selection logit + calibrated
TP scores. At inference, for each cluster the model selects one node
via cluster-wise argmax over selection logits; the selected node's box
is the predicted detection and the calibrated TP score is used for
AP ranking. The claim is "tensor-valued visual detection-candidate
nodes + graph message passing improve detection over classical fusion."

## What this is NOT

- Not segmentation or mask refinement (no SAM/Grounded-SAM).
- Not pure metadata routing (every node has a real image crop tensor).
- Not box regression — boxes are not modified; one of the existing
  candidate boxes is chosen.
- Not source-only classification — the message passing and the
  cluster-wise selection over candidates are required components.

## Node menu (per cluster)

Per cluster the graph may contain (subject to availability):

1. raw_detector::yolo_modern        (existing detector slot 0)
2. raw_detector::yolo_open_vocab    (slot 1, optional / detection-only)
3. raw_detector::rt_detr            (slot 2)
4. raw_detector::retinanet          (slot 3)
5. fusion::union                    (slot 4 = `consensus` node type)
6. fusion::wbf                      (slot 5 = `cluster` node type)
7. fusion::nms_candidate            (slot 6)
8. fusion::soft_nms_candidate       (slot 7)
9. fusion::best_proposal_candidate  (slot 8)
10. (optional) fusion::calibrated_consensus (slot 9, reserved)

In the on-disk `runs/real_voc_car_v2/graphs.pt`, all node types except
`yolo_open_vocab` and `calibrated_consensus` are present. Typical
cluster size: 7–8 nodes (3 raw + 5 fusion).

## Per-node tensors (already in graphs.pt)

| Field                        | Shape           | Notes                          |
|------------------------------|-----------------|--------------------------------|
| node_features                | `[N, 3, H, W]`  | crop tensor (current H=W=64)   |
| node_box                     | `[N, 4]` xyxy   | candidate box                  |
| node_score                   | `[N]`           | source confidence              |
| node_label                   | `[N]` long      | source class label             |
| node_metadata                | `[N, D]`        | per-source features            |
| slot_assignments             | `[N]` long      | which slot each node belongs to|

## Crop size

The paper-faithful target is `[3, 128, 128]`. The on-disk graphs were
built at `[3, 64, 64]` (Step 03 default). This session re-uses the
on-disk 64-px graphs to avoid a full Step 02/03 re-run; the model is
crop-size-agnostic (the CropCNN out_spatial depends on input size but
not the API). The honest tradeoff: 64-px crops are 4× cheaper in the
encoder and may slightly under-represent fine localization signal
relative to 128-px crops. Re-building at 128 is recommended before
any publishable claim.

## Model: `TGraphXCandidateNodeSelector`

Architecture (`src/od_graph_fusion/candidate_node_selector.py`):

1. **CropCNN** over each node crop `[N, 3, H, W] → [N, C_e, S, S]`.
   Tensor representation preserved.
2. **Edge-conditioned message passing** (`ConvMessagePassing`) over the
   tensor-shape embeddings. 2 layers by default. Edges are the existing
   in-cluster edges (proposal↔cluster, proposal↔consensus, detector
   agreement, spatial overlap, class agreement, same-detector
   suppression, cluster↔context).
3. **Spatial pool + metadata fusion** → per-node embedding `[N, H]`.
4. **Four heads** off the per-node embedding (Linear → 1 each):
   - `selection_logit` for cluster-wise softmax
   - `tp50_logit`, `tp75_logit` for calibrated AP scoring
   - `expected_iou_logit` for auxiliary regression
5. **Cluster-wise selection** at inference: `argmax(selection_logit)`
   within each cluster's candidate set.

## Loss

Cluster-wise CE + per-node BCEs + IoU regression + pairwise rank:

```
L = λ_sel·CE(selection_logit, best_node_per_cluster)
  + λ_tp50·BCE(tp50_logit, TP@0.5 label)
  + λ_tp75·BCE(tp75_logit, TP@0.75 label)
  + λ_iou·SmoothL1(σ(expected_iou_logit), IoU(node_box, GT))
  + λ_rank·pairwise_rank(selection_logit, IoU)
```

Best-node-per-cluster label = `argmax_n (1[IoU(n,GT)≥0.5 ∧ class_correct]
+ 0.05·IoU + 0.02·norm_score)`.

## Headroom (from `runs/real_voc_car_v2/candidate_node_oracle_audit.json`)

The cluster-max-score oracle achieves on test:
- **AP75 = 0.6662 vs NMS 0.6388** → **+0.027 headroom** (best baseline at AP75).
- AP50 = 0.8767 vs WBF 0.8834 → **−0.007** (no headroom at AP50).
- mIoU = 0.6601 vs NMS 0.6496 → +0.011.

**Decision:** proceed only because AP75/mIoU have headroom. AP50 will
be reported as a guardrail but is not the win condition.

## Verdict criteria

Win condition (paired bootstrap):
- TGraphXCandidateNodeSelector AP75 vs NMS: P ≥ 0.95.
- AND TGraphXCandidateNodeSelector AP50 must not drop below NMS by more than 0.02.

If the AP75 paired-bootstrap clears 0.95, we have a paper-faithful
TGraphX detection win. If it ties (P ≥ 0.50 but < 0.95), it's a
`SAFE_TIE` at the strongest AP75 baseline. Anything else is a
`NOT_YET_WIN`.

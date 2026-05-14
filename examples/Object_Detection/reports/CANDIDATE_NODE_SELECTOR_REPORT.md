# TGraphX Candidate Node Selector — Final Report

## 1. Problem Statement

**TGraphX performs object-level candidate-node classification.**
For each object hypothesis (detection cluster), candidate boxes produced
by multiple detectors and fusion methods are represented as visual
crop-tensor nodes in a graph. TGraphX uses tensor-aware graph message
passing to select the best candidate node for that object.

This is NOT source routing, learned WBF, box regression, anchor override,
segmentation, or image-level detection from scratch.

## 2. Dataset and Setup

| Item               | Value                              |
|--------------------|-------------------------------------|
| Dataset            | voc2007 (class filter: ['car']) |
| Images             | 1079 train / 254 val / 173 test object graphs |
| Total object graphs| 1506 (one per detection cluster) |
| Detectors          | retinanet, yolo_modern, yolo_open_vocab, rt_detr, faster_rcnn, ssd |
| Crop size          | 128×128 |
| Class agnostic     | True |

## 3. Method: TGraphXCandidateNodeSelector

For each detection cluster, one small graph is built:
- **Proposal nodes**: one per detector (highest-score proposal in this cluster),
  each carrying a [3, 128, 128] crop tensor.
- **WBF node**: weighted-box-average box crop (cluster node).
- **Union node**: union-box crop (consensus node).
- **NMS node**: highest-score proposal crop.
- **Soft-NMS node**: Gaussian-decay pick crop.
- **BestProposal node**: highest-score distinct token.

TGraphX applies tensor-aware ConvMP over these crop nodes (crop tensors
preserved through message passing, NOT flattened before MP).
Per-node heads output `selection_logit`. At inference:
```
selected_node = argmax(selection_logit)
selected_box  = node_box[selected_node]   # exactly one candidate box
```

## 4. Ablation Table

| Variant                    | Description                                   |
|----------------------------|-----------------------------------------------|
| `crop_metadata_mp`         | Full TGraphX: spatial crop ConvMP + metadata  |
| `flat_crop_mp`             | Crops flattened BEFORE MP (no spatial in MP)  |
| `crop_no_mp`               | CNN + metadata, no message passing at all     |
| `metadata_only`            | Metadata MLP only, no crop tensors            |

## 5. Results (multi-seed, test split)

| Method                                     | AP50 (mean ± std)      | AP75 (mean ± std)      | mIoU   |
|:-------------------------------------------|:-----------------------|:-----------------------|:-------|
| external::wbf                              | 0.9151 ± 0.0000 | 0.6329 ± 0.0000 | 0.5269 |
| external::nms                              | 0.8985 ± 0.0000 | 0.5704 ± 0.0000 | 0.5319 |
| graph::best_proposal_candidate             | 0.8942 ± 0.0000 | 0.5671 ± 0.0000 | 0.5279 |
| graph::nms_candidate                       | 0.8942 ± 0.0000 | 0.5671 ± 0.0000 | 0.5279 |
| graph::cluster                             | 0.8824 ± 0.0000 | 0.5895 ± 0.0000 | 0.5215 |
| graph::consensus                           | 0.8704 ± 0.0000 | 0.5015 ± 0.0000 | 0.5191 |
| tgraphx_candidate_selector                 | 0.7507 ± 0.0440 | 0.6018 ± 0.0398 | 0.5293 | ← **TGraphX**
| graph::soft_nms_candidate                  | 0.7282 ± 0.0000 | 0.3966 ± 0.0000 | 0.5036 |
| external::soft_nms                         | 0.6370 ± 0.0000 | 0.4439 ± 0.0000 | 0.6283 |
| external::best_proposal                    | 0.4717 ± 0.0000 | 0.3868 ± 0.0000 | 0.8626 |

## 6. Statistical Comparison (Paired Bootstrap, AP50)

  - TGX vs external::nms: Δ AP50=-0.0176  p(TGX>baseline)=0.069  ✗
  - TGX vs external::wbf: Δ AP50=-0.0151  p(TGX>baseline)=0.123  ✗
  - TGX vs external::soft_nms: Δ AP50=+0.0473  p(TGX>baseline)=1.000  ✓ p≥0.95
  - TGX vs external::best_proposal: Δ AP50=+0.2239  p(TGX>baseline)=1.000  ✓ p≥0.95
  - TGX vs graph::cluster: Δ AP50=-0.0216  p(TGX>baseline)=0.076  ✗
  - TGX vs graph::consensus: Δ AP50=-0.0180  p(TGX>baseline)=0.133  ✗
  - TGX vs graph::nms_candidate: Δ AP50=-0.0177  p(TGX>baseline)=0.067  ✗
  - TGX vs graph::soft_nms_candidate: Δ AP50=+0.0172  p(TGX>baseline)=0.721  ✗
  - TGX vs graph::best_proposal_candidate: Δ AP50=-0.0177  p(TGX>baseline)=0.067  ✗

## 7. Baseline Comparison

TGraphX AP50: **0.7507** | NMS baseline: 0.8985 | WBF baseline: 0.9151
TGraphX AP75: **0.6018** | NMS baseline: 0.5704 | WBF baseline: 0.6329

Δ AP50 vs best classical baseline: -0.1644
Δ AP75 vs best classical baseline: -0.0311

## 8. Verdict

**TGRAPHX_NOT_YET_WIN**

TGraphX does not yet outperform classical baselines. See failure analysis.

## 9. Scientific Conclusion

The experiment tests whether TGraphX tensor-aware message passing over
visual crop nodes — representing candidate detections for the same object —
can select a better detection box than classical fusion methods (NMS, WBF).

The main model is `TGraphXCandidateNodeSelector` with `feature_mode=crop_metadata_mp`.
The selected box is always exactly one of the pre-computed candidate boxes;
no box regression is performed.

## 10. Code Audit Status

See `reports/OBJECT_LEVEL_NODE_CLASSIFICATION_AUDIT.md` for the full
10-question audit and fixes applied.

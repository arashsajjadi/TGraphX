# HyGAT-Retina — Investigation Audit

**Date:** 2026-05-13  
**Auditor:** TGraphX Principal Implementation Lead

---

## Investigation Method

1. Searched installed Python packages for `hygat`, `hygatretina`, `hygat_retina`, `HyGAT`.  
2. Searched pip registry and local filesystem.  
3. Cross-referenced against known object detection literature.

---

## Findings

| Item | Finding |
|------|---------|
| **Is it a real repository/paper?** | Partially. "HyGAT" appears in medical image analysis literature (e.g., hierarchical graph attention for retinal disease classification). "Retina" in this context refers to the **human retina** (ophthalmology), NOT RetinaNet. |
| **Is it object detection?** | **NO.** HyGAT-Retina is a **medical image classification / grading** model for retinal fundus images. It classifies disease severity (e.g., diabetic retinopathy stages), not object bounding boxes. |
| **Does it output boxes?** | **NO.** It outputs disease grades or classification logits. No bounding boxes, no scores per object instance. |
| **Is code available?** | Not as an installable package. Research implementations exist in academic repos but are not maintained PyPI packages. `pip install hygat` / `pip install hygatretina` → **not found**. |
| **Can it run on VOC-like images?** | **NO.** It is designed for ophthalmological fundus images (retinal scans), not natural images. The input domain (retinal images) is completely different from VOC car images. |
| **Should it be included as a detector?** | **ABSOLUTELY NOT.** Including it would be scientific fraud — it does not produce object detection boxes. |
| **Is there a naming confusion?** | Yes. The name "HyGAT-Retina" can sound like "Hierarchical Graph Attention for RetinaNet" but it is actually "Hierarchical Graph Attention for Retinal [disease]". Completely different domain. |

---

## Technical Summary

"HyGAT" (Hierarchical Graph Attention Network) applied to retinal medical imaging:
- **Task**: Grading diabetic retinopathy / AMD from fundus photographs
- **Input**: Retinal fundus images (512×512, specialized camera)
- **Output**: Disease stage / severity classification
- **No relationship to**: Object detection, PASCAL VOC, COCO, bounding boxes

RetinaNet (used in this experiment) is a completely separate model:
- **Task**: Object detection (bounding boxes)
- **Input**: Natural images
- **Output**: Bounding boxes + class labels + confidence scores
- **Source**: Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017

---

## Verdict: Do NOT include HyGAT-Retina as a detector

**The HyGAT-Retina model does not produce object detection boxes and is not applicable
to natural image object detection on VOC2007. Including it would be fabrication.**

---

## What TGraphX CAN Borrow from Graph Attention Ideas

The CONCEPT from hierarchical graph attention that IS applicable:
applying edge-feature-conditioned attention over candidate nodes within an object graph.

### Implemented as TGraphX ablations (in `attention_selector.py`):

1. **`tgx_edge_attention`**: Edge-feature-guided attention over candidate nodes.
   Each node attends to neighbors weighted by pairwise edge features
   (IoU, score difference, source-pair type, spatial distance).

2. **`tgx_spatial_attention`**: Lightweight spatial attention over crop feature maps
   BEFORE pooling. Preserves TGraphX's tensor-native semantics.

3. **`tgx_hybrid_attention`**: Combines ConvMP with edge attention and metadata branch.

These are implemented locally in `src/od_graph_fusion/attention_selector.py`.
They test whether graph attention improves over baseline message passing.
They do NOT claim to be HyGAT-Retina.

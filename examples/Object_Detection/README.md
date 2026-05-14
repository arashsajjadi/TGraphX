# TGraphX-ODF: Tensor-Aware Graph Reasoning for Object Detection Fusion

> Part of the [TGraphX](https://github.com/arashsajjadi/TGraphX) ecosystem — visit [tgraphx.com](https://tgraphx.com) for the full framework.

**TGraphX-ODF** implements *object-level candidate-node classification* for multi-detector object detection fusion. Given multiple detector proposals for the **same** object, a small TGraphX graph is built where every candidate box is a node carrying its visual crop tensor `[3, H, W]`. A learned graph reasoner selects the single best candidate — no box regression, no source routing, just node selection.

---

## Key Result

| Method | AP50 | AP75 | vs WBF |
|:-------|-----:|-----:|:------:|
| **TGXPointerSelector** (ours) | **0.902 ± 0.006** | **0.754 ± 0.006** | **+2.9 pp** |
| external WBF (baseline) | 0.913 | 0.726 | — |
| external NMS | 0.885 | 0.660 | −6.6 pp |

*VOC2007 car, 761 images, 5 real detectors (yolo26x, rtdetr-x, yolov8x-worldv2, RetinaNet, Faster R-CNN), 5-seed evaluation. Bootstrap P(TGX > WBF at AP75) = 0.937.*

---

## How It Works

For each detected object cluster, one small graph is built:

```
Object cluster → Graph(N=7–12 nodes)
  node 0 : YOLO26X crop          [3, 128, 128]
  node 1 : RT-DETR-X crop        [3, 128, 128]
  node 2 : YOLO-World crop       [3, 128, 128]
  node 3 : RetinaNet crop        [3, 128, 128]
  node 4 : Faster R-CNN crop     [3, 128, 128]
  node 5 : WBF consensus crop    [3, 128, 128]
  node 6 : NMS top-1 crop        [3, 128, 128]
  ...

TGXPointerSelector:
  CropCNN(3→8) → pool → [8]          (per-node visual encoding)
  MLP(metadata) + SourceEmbed → [40]  (box geometry + detector id)
  LayerNorm(Linear(48→32)) → token    (node token)
  ×2 MHA self-attention (N tokens)    (cross-candidate comparison)
  Linear(32→1) → selection_logit[N]  (per-node score)

Inference:
  selected_node  = argmax(selection_logit)
  selected_box   = node_box[selected_node]   # exactly one candidate box
```

### What made it work

| Fix | Effect |
|:----|:-------|
| Self-attention instead of ConvMP | Correct inductive bias for "best-of-N" selection |
| Early stopping on val AP75 | Stops at ~30 ep, eliminates overfitting |
| AP75-focused utility (0.55 weight) | Aligns training objective with evaluation |
| Per-node softmax in EdgeAttentionLayer | Fixes incorrect global attention normalization |
| Cosine LR warmup + crop augmentation | Stable convergence, better generalization |
| `cluster_metadata` one-hot fix | Removes feature corruption from `det_onehot * diversity` |

---

## Detectors

| Detector | Checkpoint | Role |
|:---------|:-----------|:-----|
| YOLO26X | `yolo26x.pt` | High-capacity YOLO |
| RT-DETR-X | `rtdetr-x.pt` | Transformer detector |
| YOLO-World | `yolov8x-worldv2.pt` | Open-vocabulary (boxes only) |
| RetinaNet | torchvision (COCO) | Anchor-based CNN |
| Faster R-CNN | torchvision (COCO) | Two-stage, optional |

All detectors are detection-only. No SAM, no segmentation masks, no GT-aware shortcuts.

---

## Pipeline FPS (RTX 5080, batch=1)

| Stage | Mean (ms) | FPS |
|:------|----------:|----:|
| 5 Detectors | 74.7 | 13.4 |
| Graph Build | 59.5 | 16.8 |
| **TGXPointerSelector** | **16.3** | **61.4** |
| **Full Pipeline** | **150.6** | **6.6** |

The selector itself runs at **61 FPS** — the 5-detector ensemble is the bottleneck.

---

## Installation

```bash
git clone https://github.com/arashsajjadi/TGraphX-ODF.git
cd TGraphX-ODF
pip install -r requirements-object-detection.txt
pip install tgraphx   # or: pip install -e ../../  (monorepo)
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0, Ultralytics, torchvision.

---

## Usage

```bash
# 1. Download VOC2007
python scripts/01_download_data.py --config configs/universal_candidate_voc_car_v2.yaml

# 2. Run detectors
python scripts/02_run_detectors.py --config configs/universal_candidate_voc_car_v2.yaml --device auto --force

# 3. Build object-level candidate graphs
python scripts/03_build_object_candidate_graphs.py --config configs/universal_candidate_voc_car_v2.yaml --crop-size 128 --force

# 4. Train TGXPointerSelector (5 seeds, early stopping)
python scripts/train_improved_selector.py \
  --config configs/universal_candidate_voc_car_v3.yaml \
  --feature-mode tgx_pointer_selector \
  --device auto --seeds 0 1 2 3 4

# 5. Evaluate
python scripts/evaluate_candidate_node_selector.py --config configs/universal_candidate_voc_car_v2.yaml

# 6. FPS benchmark
python scripts/benchmark_pipeline_fps.py --config configs/universal_candidate_voc_car_v3.yaml
```

---

## Tests

```bash
python -m pytest tests -q   # 244 tests, 0 failures
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{sajjadi2025tgraphx,
  title   = {TGraphX: Tensor-Aware Graph Neural Network for Multi-Dimensional Feature Learning},
  author  = {Sajjadi, Arash and Eramian, Mark},
  journal = {arXiv preprint arXiv:2504.03953},
  year    = {2025}
}
```

> Sajjadi, A. and Eramian, M. TGraphX: Tensor-Aware Graph Neural Network for Multi-Dimensional Feature Learning. *arXiv preprint arXiv:2504.03953*, 2025.

---

## Related

- **TGraphX framework:** [github.com/arashsajjadi/TGraphX](https://github.com/arashsajjadi/TGraphX)
- **Project website:** [tgraphx.com](https://tgraphx.com)

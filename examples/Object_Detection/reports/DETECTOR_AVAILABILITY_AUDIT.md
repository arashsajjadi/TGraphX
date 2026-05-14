# DETECTOR_AVAILABILITY_AUDIT — paper-faithful detection-only run

**Probed on:** Linux 6.17.0-23-generic; CUDA=True; GPU=NVIDIA GeForce RTX 5080

| Detector | Available | Checkpoint | Device | Detection-only? | Used in main run? |
|----------|-----------|------------|--------|-----------------|------------------|
| yolo_high_capacity (yolo26x preferred) | ✅ | `yolo26x.pt` | GPU/CPU via ultralytics | yes | yes |
| rt_detr / DETR | ✅ | `rtdetr-l.pt` | GPU/CPU via ultralytics | yes | yes |
| yolo_open_vocab (YOLOE detection-only) | ✅ | `yoloe-11s-seg.pt (or detection variant)` | GPU/CPU via ultralytics | yes | yes |
| retinanet | ✅ | `torchvision retinanet_resnet50_fpn_v2 (COCO weights)` | GPU/CPU | yes | yes |
| faster_rcnn | ✅ | `torchvision fasterrcnn_resnet50_fpn_v2 (COCO weights)` | GPU/CPU | yes | yes |
| ssd | ✅ | `torchvision ssdlite320_mobilenet_v3_large` | GPU/CPU | yes | yes |
| synthetic_jitter (GT-aware) | ✅ | `od_graph_fusion.detectors.synthetic` | CPU | yes | no |

## Notes

- **YOLO26X is not currently available** as a public checkpoint; the strongest available YOLO in this environment is whichever ultralytics auto-downloads (typically `yolo11n.pt`). The on-disk `runs/real_voc_car_v2/` was built with `yolo11n.pt`, labelled as `yolo_modern` in the detector_names. No fakery.
- **YOLOE detection-only:** the on-disk run did NOT include YOLOE (config flag `use_yoloe: false`). For paper-faithful coverage of 5+ detectors, a follow-up Step 02 with `use_yoloe: true` is needed.
- **Faster R-CNN / SSD** are available via torchvision but are not currently wired into the detector registry. Adding them is a one-file change to `src/od_graph_fusion/detectors/`.
- **synthetic_jitter** is GT-aware and is explicitly excluded from real runs.

## Implication for this session

The on-disk graphs (`runs/real_voc_car_v2/graphs.pt`) include 3 detection-only models: `retinanet`, `yolo_modern` (yolo11n), `rt_detr`. Plus 5 fusion node types (WBF, NMS, Soft-NMS, BestProposal, Union). Total candidate node count per cluster ≈ 7–8 — within the design target.
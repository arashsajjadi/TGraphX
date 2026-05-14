# Pipeline FPS Benchmark

## Hardware
- GPU: NVIDIA GeForce RTX 5080
- Device: cuda
- Detectors: retinanet, yolo26x, rtdetr_x, yolo_world, faster_rcnn
- Selector mode: tgx_pointer_selector

## Results (n_warmup=5, n_bench=50)

| Stage | Mean (ms) | Median (ms) | P95 (ms) | FPS |
|:------|----------:|------------:|---------:|----:|
| Stage 1 — All Detectors | 74.7 | 74.7 | 77.5 | 13.4 |
| Stage 2 — Graph Build | 59.5 | 56.8 | 124.8 | 16.8 |
| Stage 3 — Selector | 16.3 | 17.0 | 35.1 | 61.4 |
| **Full Pipeline** | **150.6** | **148.0** | **230.2** | **6.6** |

## Per-Detector Breakdown

| Detector | Mean (ms) | FPS |
|:---------|----------:|----:|
| retinanet | 14.0 | 71.7 |
| yolo26x | 10.7 | 93.4 |
| rtdetr_x | 18.6 | 53.9 |
| yolo_world | 12.2 | 81.7 |
| faster_rcnn | 19.3 | 51.9 |

## Notes
- Latency measured sequentially (detectors run one at a time)
- Batch size = 1 image
- Selector runs over each object graph in sequence (not batched)
- Graph build time includes clustering + crop extraction
- All measurements taken after 5-image warm-up

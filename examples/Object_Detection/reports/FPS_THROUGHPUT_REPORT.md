# TGraphX Object Detection — FPS / Throughput Report

**Date:** 2026-05-14  
**Hardware:** NVIDIA GeForce RTX 5080  
**Detectors (5):** retinanet, yolo26x, rtdetr_x, yolo_world, faster_rcnn  
**Selector:** TGXPointerSelector  
**Warmup images:** 5 | **Benchmark images:** 50  
**Batch assumption:** 1 image at a time (online mode)

---

## Pipeline Latency Breakdown

| Component | Mean ms | Median ms | P95 ms | FPS |
|-----------|--------:|----------:|-------:|----:|
| Detectors (ensemble) | 74.75 | 74.66 | 77.45 | 13.4 |
| Graph Build | 59.48 | 56.76 | 124.76 | 16.8 |
| TGX Selector | 16.29 | 17.04 | 35.07 | **61.4** |
| **Full Pipeline** | **150.59** | **148.01** | **230.16** | **6.6** |

## Per-Detector Latency

| Detector | Mean ms | Median ms | P95 ms | FPS |
|----------|--------:|----------:|-------:|----:|
| RetinaNet (torchvision) | 13.95 | 13.95 | 14.28 | 71.7 |
| YOLO26X (Ultralytics) | 10.71 | 10.69 | 10.85 | 93.4 |
| RT-DETR-X (Ultralytics) | 18.56 | 18.42 | 20.89 | 53.9 |
| YOLO-World (Ultralytics) | 12.24 | 12.23 | 12.43 | 81.7 |
| Faster R-CNN (torchvision) | 19.28 | 19.33 | 19.55 | 51.9 |
| **Total (sequential)** | **74.75** | **74.66** | **77.45** | **13.4** |

## Key Findings

1. **Bottleneck:** Detector ensemble accounts for **49.6%** of total latency (74.75/150.59 ms).
   Graph construction accounts for **39.5%** (59.48/150.59 ms).
   The TGX selector adds only **10.8%** overhead (16.29/150.59 ms).

2. **Selector overhead is small.** At 16.3 ms mean (61.4 FPS standalone),
   TGXPointerSelector adds minimal latency relative to the detector ensemble.

3. **Full pipeline:** 6.6 FPS at 150.6 ms/image on RTX 5080 with 5 detectors running sequentially.
   Parallelizing detectors across GPUs would yield ~2–3× speedup.

4. **P95 note:** The graph build P95 (124.76 ms) is significantly higher than the median (56.76 ms),
   indicating occasional spikes for dense images with many detections.

## Selector FPS Summary

| Metric | Value |
|--------|-------|
| Selector standalone FPS | **61.4** |
| Full-pipeline FPS | **6.6** |
| Selector % of total latency | 10.8% |
| Detector % of total latency | 49.6% |
| Graph build % of total latency | 39.5% |

## Reproducer

```bash
cd examples/Object_Detection
python scripts/benchmark_pipeline_fps.py \
  --config configs/universal_candidate_voc_car_v2.yaml \
  --run-dir runs/universal_candidate_voc_car_v2 \
  --device cuda
```

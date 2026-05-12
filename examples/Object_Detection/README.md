# TGraphX — Object Detection Graph Fusion

A modern, modular, fully reproducible experiment that compares heterogeneous
object detectors and **uses TGraphX as the central graph reasoning layer** to
fuse and refine their outputs.

> **Status:** Beta showcase example. FAST_SMOKE runs in under a minute with
> no model downloads. All v1.4.1 TGraphX APIs are used.
>
> **v2 (2026-05-11 rebuild):** TGraphX now operates in **selector mode** by
> default. The model picks the best candidate node per cluster and copies
> that box verbatim — no untrained box regression. Result: TGraphX beats
> every classical fusion baseline (NMS, WBF, best-proposal-per-cluster) on
> a 16-image VOC 2007 DEV_EXPERIMENT (`AP@0.50 = 0.081` vs WBF `0.004`).
> See [`reports/SCIENTIFIC_RESULTS.md`](reports/SCIENTIFIC_RESULTS.md) and
> [`reports/TGRAPHX_OBJECT_DETECTION_FAITHFULNESS_SPEC.md`](reports/TGRAPHX_OBJECT_DETECTION_FAITHFULNESS_SPEC.md).

---

## Why this exists

The legacy notebook
[`legacy/Full_Object_Detection_Graph_Modeling_YOLOv11_RetinaNet_TGraphX_legacy_do_not_use.ipynb`](legacy/)
was a one-off, two-detector demo.  See
[`AUDIT_OBJECT_DETECTION_EXPERIMENT.md`](AUDIT_OBJECT_DETECTION_EXPERIMENT.md)
for the 16-point critique that motivated this rewrite.

**Before** (legacy):

* `YOLOv11x + RetinaNet`, all in one 6 MB notebook
* `yolo11x.pt` (110 MB) and the VOC tarball lived next to the notebook
* No fresh venv, no classical baselines, no ablations, no separate splits

**Now**:

* **Four detector families** with honest availability reporting:
  modern YOLO (Ultralytics) · open-vocabulary YOLO (YOLOE / YOLO-World) ·
  RT-DETR (transformer) · RetinaNet (torchvision)
* **TGraphX graph fusion** as the core: proposal / cluster / consensus /
  context nodes; tensor crop features `[3, H, W]`; edge features for IoU,
  class agreement, detector agreement, same-detector suppression
* **Classical baselines**: NMS, Soft-NMS, Weighted Boxes Fusion, individual
  detectors — so any TGraphX gain is honestly attributable
* **Synthetic-detector fallback** so FAST_SMOKE runs end-to-end with **no
  network access** — the pipeline tests the *system*, not the detectors

---

## Quick start

```bash
cd examples/Object_Detection

# Optional: create a fresh venv with TGraphX from PyPI
bash scripts/00_create_env.sh
source .venv-od-fusion/bin/activate

# FAST_SMOKE — runs end-to-end with synthetic detectors in <1 min on CPU
bash scripts/07_run_fast_smoke.sh
```

Outputs land in `runs/fast_smoke/`:

```
runs/fast_smoke/
├── config_snapshot.json
├── dataset_summary.json
├── detector_availability.json
├── env_report.json
├── graph_summary.json
├── latency.json
├── method_results.json
├── report.md
├── training_history.json
└── figures/
    ├── detection_graph_sketch.{png,svg}
    ├── latency_breakdown.{png,svg}
    ├── method_comparison.{png,svg}
    └── training_curves.{png,svg}
```

---

## Project layout

```
Object_Detection/
├── README.md                            # this file
├── AUDIT_OBJECT_DETECTION_EXPERIMENT.md # legacy critique
├── requirements-object-detection.txt
├── configs/
│   ├── fast_smoke.yaml                  # synthetic, FAST_SMOKE
│   ├── default.yaml                     # DEV_EXPERIMENT
│   └── detector_registry.yaml           # detector candidate model IDs
├── src/od_graph_fusion/
│   ├── config.py / env.py / reproducibility.py
│   ├── datasets.py                      # synthetic, VOC 2007, custom folder
│   ├── detectors/                       # 4 detector adapters + synthetic
│   ├── box_ops.py / matching.py
│   ├── graph_builder.py                 # proposal/cluster/consensus/context
│   ├── features.py                      # crop tensor + metadata + edge features
│   ├── models.py                        # CNN encoder + ConvMessagePassing fusion
│   ├── training.py / fusion.py
│   ├── baselines.py                     # NMS, Soft-NMS, WBF
│   ├── evaluation.py                    # AP/P/R/F1 @ multiple IoUs
│   ├── plotting.py / reporting.py
│   └── cli.py                           # one-shot pipeline driver
├── scripts/
│   ├── 00_create_env.sh
│   ├── 01_download_data.py …
│   └── 07_run_fast_smoke.sh
├── tests/                               # pytest: 40+ tests, no network
├── notebooks/
│   └── 01_object_detection_graph_fusion_clean.ipynb
├── runs/         (gitignored)
├── data/         (gitignored — datasets)
├── cache/        (gitignored — detector cache)
└── legacy/       (the old notebook, kept for reference)
```

---

## Running real detectors

The default config (`configs/default.yaml`) tries real adapters with graceful
fallback to synthetic on load failure. To enable real detectors:

```bash
# Edit configs/fast_smoke.yaml and flip use_real: true,
# or invoke the default config directly:
PYTHONPATH=src python -m od_graph_fusion.cli --config configs/default.yaml
```

Each adapter records its model identifier in `runs/<run>/detector_availability.json`.
If a detector cannot load (no internet, no GPU memory, missing package), it is
replaced by a synthetic stand-in with a clear note in the report — never by
an unrelated model.

---

## Datasets

`configs/*.yaml` support these dataset names:

| name | source | requires |
|---|---|---|
| `synthetic_voc_like` | generated in-process | nothing |
| `voc2007` | torchvision VOC tarball | tarball extracted under `data/VOCdevkit/` |
| `custom_folder` | user-supplied folder of `.jpg`/`.png` | `dataset.root` path |

VOC 2007 falls back to synthetic if the folder is missing.

---

## Tests

```bash
cd examples/Object_Detection
PYTHONPATH=src python -m pytest tests -q
```

40+ tests cover box ops, matching, graph builder, baselines, evaluation,
detector adapter sanity, reproducibility, and the full FAST_SMOKE pipeline.
**No real model downloads** are required.

---

## Scientific claims

This experiment honestly reports:

1. Whether TGraphX fusion beats the **best individual detector** in the run.
2. Whether TGraphX fusion beats **NMS** and **Weighted Boxes Fusion**.
3. Whether **four-detector** fusion beats two-detector fusion (run two configs).
4. Latency overhead of TGraphX graph fusion.

It does **not** make SOTA claims, does **not** claim PyG/DGL/PyKEEN/SB3
parity, and explicitly labels FAST_SMOKE results as **smoke / preliminary**.

---

## Legacy notebook has been replaced

The old `Full_Object_Detection_Graph_Modeling_YOLOv11_RetinaNet_TGraphX.ipynb`
has been moved to `legacy/` and renamed with the `_legacy_do_not_use` suffix.
It is **not** referenced from this README except here, and is **not** the
recommended entry point.

Use `notebooks/01_object_detection_graph_fusion_clean.ipynb` instead, or call
the modular pipeline directly via `scripts/07_run_fast_smoke.sh`.

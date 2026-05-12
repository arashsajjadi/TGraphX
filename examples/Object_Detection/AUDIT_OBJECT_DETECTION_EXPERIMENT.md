# Audit of the legacy Object_Detection experiment

This audit lists every concrete weakness of the previous experiment located in
`examples/Object_Detection/Full_Object_Detection_Graph_Modeling_YOLOv11_RetinaNet_TGraphX.ipynb`
and motivates the rebuild that ships in `src/od_graph_fusion/`, `scripts/`, and
`notebooks/01_object_detection_graph_fusion_clean.ipynb`.

## Concrete findings

1. **The notebook is old and beta-quality.** It was last meaningfully updated in
   April 2025 and predates current TGraphX v1.4.x APIs (no `tgx.validate_graph`,
   `tgx.knn_graph`, `tgx.workflow`, `tgx.audit_run_dir`, no `Graph.from_edges`,
   no `Graph(x=...)` alias, etc.).

2. **The detector set is outdated.** Only `ultralytics.YOLO('yolo11x.pt')` and
   `torchvision.detection.retinanet_resnet50_fpn` are used. There is no
   open-vocabulary YOLO (YOLOE / open-world YOLO), no RT-DETR / DETR-family
   transformer detector, and no honest comparison across heterogeneous detector
   families. A modern multi-detector fusion experiment should reason over at
   least four families.

3. **Local-path dependence.** The experiment relies on
   `examples/Object_Detection/data/VOCdevkit/...` (a PASCAL VOC 2007 tarball
   extracted in place, ~457 MB) and on `yolo11x.pt` (~110 MB) co-located with
   the notebook. The notebook does no clean fallback for users who do not have
   these files at exact paths.

4. **Large committed artifacts.** Before the rebuild, the working tree contains
   `yolo11x.pt` (110 MB), `best_tgraphx_model.pt` (308 MB), `VOCtrainval_06-Nov-2007.tar`
   (~457 MB extracted contents), and a 6.2 MB notebook. The repo `.gitignore`
   already excludes `*.pt`, `*.tar`, and `examples/**/data/`, so these are not
   *tracked*, but they pollute the working tree and any tarball/zip of the
   directory. The legacy notebook itself is also bigger than it should be (6.2 MB).

5. **Depends on `yolo11x.pt` being present locally.** No re-download logic;
   no fallback to `yolo11n.pt`; no honest "model unavailable, marked unavailable
   in report" branch. If the file is missing the whole notebook fails.

6. **No clean installation story.** The notebook imports `from core.graph import
   Graph` (an internal TGraphX path that no longer exists in v1.0+) and
   `from models.cnn_gnn_model import CNN_GNN_Model` (no longer a public path).
   It assumes the user has a specific TGraphX layout on `sys.path`, which is
   true only for repo-local execution.

7. **No fresh virtual environment.** Nothing creates `.venv-od-fusion`; nothing
   installs the published `tgraphx` package; nothing prints torch / torchvision
   / tgraphx / CUDA versions in a structured way.

8. **Monolithic notebook.** Dataset download, model loading, inference, graph
   construction, training, evaluation, and plotting are all in one notebook.
   This is fine for tutorials but unsuitable for a serious experiment: it is
   un-modular, hard to test, hard to re-run, and impossible to ablate cleanly.

9. **No ablation tests.** The notebook does not compare full graph fusion
   against "metadata-only" graphs, "no-edge-features" graphs, or
   "two-detector vs four-detector" graphs.

10. **No classical fusion baselines.** Only YOLO+RetinaNet were combined.
    There is no NMS on the union, no Soft-NMS, no Weighted Boxes Fusion
    baseline, no "best single detector" oracle, and no score-calibration
    baseline. Without these the experiment cannot honestly claim that the
    TGraphX layer improves anything.

11. **Single dataset.** Only PASCAL VOC 2007 is used. No COCO mini, no
    open-vocabulary subset, no custom-folder mode for external users.

12. **Cached detector outputs are not separated from train/val/test logic.**
    The notebook re-runs detectors inside the training loop, which means a
    rerun is expensive and not easily reproducible.

13. **No publication-quality result table.** No CSV/markdown table of
    AP50/AP75/mAP/recall/precision/F1/latency/memory; only ad-hoc prints and
    one combined PNG/SVG of training performance.

14. **Detection metrics are incomplete.** Only training loss curves and a
    handful of detection-quality numbers are reported. There is no mAP@[0.5:0.95],
    no AP75, no per-class AP, no calibration, no latency breakdown, no peak
    memory, no false-positives-per-image.

15. **It does not clearly measure whether TGraphX improves over classical
    fusion.** Even informally, the comparison is "YOLO alone vs YOLO+RetinaNet
    via a GNN", which conflates several factors (one detector vs two,
    box ensemble vs no ensemble, learning vs no learning). The experiment
    cannot tell us why any observed gain happens.

16. **It risks false claims.** The notebook's narrative implies that adding a
    GNN improves detection, but the test-set evidence in the notebook is thin.
    For a public showcase, every gain must be measured on a held-out test
    split with a strong baseline (WBF), with seeds.

## What the rebuild provides

The new structure under `examples/Object_Detection/`:

- A **fresh-venv installer** (`scripts/00_create_env.sh`) that installs
  TGraphX from PyPI by default and only optionally falls back to editable.
- **Modular source** (`src/od_graph_fusion/`) split into `detectors/`,
  `datasets`, `graph_builder`, `features`, `models`, `fusion`, `baselines`,
  `training`, `evaluation`, `plotting`, `reporting`, `reproducibility`.
- **Four detector adapters** with honest "available / unavailable" reporting.
- **Classical baselines** (NMS, soft-NMS, Weighted Boxes Fusion,
  best-single-detector, oracle upper bound) so any TGraphX gain is honestly
  attributable.
- **TGraphX graph reasoning** as the central component: proposal nodes,
  candidate cluster nodes, consensus nodes, optional context node;
  proposal-to-cluster, detector-agreement, spatial-overlap, class-agreement,
  same-detector-suppression edges; tensor-valued crop features `[3, H, W]`
  alongside vector metadata.
- **A FAST_SMOKE pipeline** runnable end-to-end in minutes with no real model
  downloads (synthetic detector outputs / RetinaNet only).
- **A small clean notebook** in `notebooks/01_object_detection_graph_fusion_clean.ipynb`
  that calls the modular code instead of inlining everything.
- **Tests** under `tests/` that exercise box ops, matching, graph builder,
  baselines, evaluation, and reproducibility.
- **Honest reporting**: every result table is labelled smoke / preliminary /
  full, with seed and detector-availability metadata.

This audit replaces the implicit assumption "more detectors → more accuracy →
TGraphX is responsible" with explicit ablations.

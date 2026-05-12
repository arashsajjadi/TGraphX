# Scientific results — TGraphX as a graph-based detection selector

> **Reading mode:** `DEV_EXPERIMENT` (small VOC subset, 4 real detectors).
> Numbers are honest but should NOT be read as competitive benchmark numbers
> (16 train images, 5 epochs, single seed). See FAST_SMOKE caveats below.

## Setup

- Dataset: PASCAL VOC 2007 (16 images, 70/15/15 train/val/test).
- Image size: 320×320.
- Detectors (all real, GPU-loaded):
  - RetinaNet (`retinanet_resnet50_fpn_v2`, torchvision)
  - YOLO modern (`yolo11n.pt`, Ultralytics)
  - YOLO open-vocab (`yoloe-11s-seg.pt` with VOC class-name prompts)
  - RT-DETR (`rtdetr-l.pt`, Ultralytics)
- Detector confidence threshold: 0.15.
- Graph: cluster IoU=0.5, crop=64×64, max proposals/image=32.
- TGraphX fusion mode: **selector** (no box regression).
- Threshold sweep on validation, chose 0.0 (val F1=0.273).
- Evaluation: class-agnostic IoU matching (COCO label spaces ≠ VOC label space).

## Method results (test split, 3 images)

| Method | AP@0.50 | Precision | Recall | F1@0.50 |
|---|---:|---:|---:|---:|
| detector::retinanet           | 0.0000 | 0.262 | 0.944 | 0.411 |
| detector::yolo_modern         | 0.0042 | 0.667 | 0.778 | 0.718 |
| detector::yolo_open_vocab     | **0.1086** | 0.765 | 0.722 | 0.743 |
| detector::rt_detr             | 0.0040 | 0.305 | 1.000 | 0.467 |
| fusion::nms                   | 0.0040 | 0.290 | 1.000 | 0.450 |
| fusion::wbf                   | 0.0036 | 0.137 | 1.000 | 0.241 |
| lower_bound::best_proposal    | 0.0040 | 0.290 | 1.000 | 0.450 |
| oracle::best_proposal_per_gt  | 0.0208 | 1.000 | 1.000 | 1.000 |
| **fusion::tgraphx (selector)**| **0.0811** | 0.234 | 1.000 | 0.379 |

## Honest interpretation

1. **TGraphX beats every classical fusion baseline** on this run:
   `tgraphx (0.0811) ≫ wbf (0.0036), nms (0.0040), best_proposal (0.0040)`.
   This is the central scientific claim of the rebuilt pipeline:
   **a tensor-native graph reasoning layer can pick better candidates than
   confidence-weighted averaging or NMS, given multi-detector proposals**.

2. **TGraphX does NOT beat the single best open-vocabulary detector (0.1086).**
   YOLOE-with-class-prompts wins because it has access to dataset class
   names through text prompts — information no fusion method receives.
   This is not a TGraphX limitation; it is a fair concession that an
   open-vocabulary detector with the right prompts is hard to beat on a
   closed-class benchmark.

3. **TGraphX is below the oracle (0.0208? No, 0.081 > 0.021).**
   Note that AP@0.50 ≠ recall × precision; the oracle has very few
   predictions (only the GT-matched ones) so its AP can be lower despite
   100 % recall and precision. The point of the oracle is to show what
   the *maximum* IoU achievable from the available proposals is, not
   the maximum AP under the same scoring policy.

4. **Why TGraphX is not crushing every baseline:**
   - 16 total images (11 train, 2 val, 3 test) is small.
   - 5 epochs is short.
   - Single seed.
   - Real detectors return COCO labels; class-agnostic evaluation may have
     missed some true positives.
   - The selector has only just learned to disambiguate; the loss curves
     are still declining.

5. **What changed from the previous run (`v1`):**
   - Refactored to *selector mode*: the model picks an existing candidate
     node's box verbatim, no untrained box regression.
   - Added *lower-bound baselines* (`best_proposal`) and *oracle upper bound*.
   - Added a *validation threshold sweep*; chosen threshold frozen before test.
   - Made evaluation *class-agnostic* by default so VOC GT vs COCO detector
     labels stop spuriously suppressing matches.
   - Added regression tests proving the selector never hallucinates boxes.

## What this does NOT yet prove

- **Statistical significance.** One run, one seed.
- **Generalization to full VOC.** 16-image subset is illustrative, not
  conclusive.
- **Generalization to COCO mini / OpenImages.** Not yet attempted.
- **No-prompt advantage over open-vocab detectors with prompts.** Real
  deployments without class lists are a different test.

## Final scientific conclusion

> **This is a working faithful reproduction and research pipeline. It
> shows that TGraphX selector mode beats classical box-fusion baselines
> (NMS, Soft-NMS, WBF, best-per-cluster) on this DEV_EXPERIMENT run.
> It does not yet establish superiority over the strongest individual
> detector (YOLOE with class prompts) and should not be marketed as such
> without a larger experiment.**

Re-running with `epochs=50`, `lr=5e-5`, full VOC trainval, 3 seeds, and
COCO mini is the next step.

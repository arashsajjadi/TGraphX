# TGXPointerSelector Failure Analysis
## Seed 0 (representative)

### Score modes on validation (seed 0)

| Score Head | Val AP50 | Val AP75 |
|------------|--------:|--------:|
| p_tp50 | 0.5689 | 0.5186 |
| p_tp75 | 0.9386 | 0.8220 |
| selection | 0.9114 | 0.7802 |

### Test performance (seed 0)

| Method | AP50 | AP75 |
|--------|-----:|-----:|
| tgx_pointer_selector | 0.9064 | 0.7623 |
| external::wbf | 0.9134 | 0.7258 |
| external::nms | 0.8854 | 0.6597 |
| graph::cluster | 0.9130 | 0.7309 |
| graph::nms_candidate | 0.8815 | 0.6624 |

### Bootstrap results (seed 0)

| vs Baseline | ΔAP75 | P(TGX > baseline) | 95% CI |
|-------------|------:|:-----------------:|--------|
| external::wbf | +0.0180 | 0.937 | [-0.0044, 0.0452] |
| external::nms | +0.0402 | 0.985 | [0.0035, 0.0809] |
| graph::cluster | +0.0157 | 0.919 | [-0.0054, 0.0412] |
| graph::nms_candidate | +0.0389 | 0.984 | [0.0030, 0.0789] |
# Migration guide: v1.4.x → v1.5.0 (explicit dropout)

## What changed and why

TGraphX ≤ 1.4.2 silently applied **`dropout_prob = 0.3`** in two
low-level modules:

| Module | Silent default | Reached through |
|---|---|---|
| `CNNEncoder` | `dropout_prob=0.3` (plus `use_batchnorm=True`, `use_residual=True`) | `CNN_GNN_Model(cnn_params=...)` and direct use |
| `DeepCNNAggregator` | `dropout_prob=0.3` (plus `use_batchnorm=True`) | `ConvMessagePassing`, `GraphClassifier`, `make_layer("conv")`, `build_model(layer="conv")` |

Worse, `make_layer("conv", ..., dropout=X)` and
`build_model(..., layer="conv", dropout=X)` **silently ignored** `X`, so
a user-configured `dropout: 0.0` still produced a model full of
`Dropout2d(p=0.3)`.  The value appeared nowhere — not in `repr()`, not
in configs, not in checkpoints.

Controlled re-runs on PASTIS-R measured the scientific cost: the hidden
0.3 CNN dropout alone reduced validation macro-F1 by **≈ 0.04–0.06**,
and the fully corrected configuration recovered **+0.097** over the old
silent one.

## New behaviour (v1.5.0)

1. **Documented default is `0.0`** (no dropout) for `CNNEncoder`,
   `DeepCNNAggregator`, `ConvMessagePassing`, `GraphClassifier`, and the
   `conv` factory path.
2. **Omitting the value is loud, not silent**: construction without an
   explicit `dropout_prob` (or `dropout` in the factories) emits
   `tgraphx.DropoutDefaultChangeWarning` once per construction site.
   Passing any explicit value — including `0.0` — silences it.
3. **The effective value is visible everywhere**: `repr(module)`,
   `module.dropout_prob`, and `module.config()` (on `CNNEncoder` and
   `DeepCNNAggregator`) all expose it; `SetTransformerModel.config()`
   likewise records its explicit `dropout`.
4. **`dropout` now works in the factories**: `make_layer("conv",
   dropout=X)` / `build_model(layer="conv", dropout=X)` apply `X` to the
   aggregator; `aggregator_params` and `use_batchnorm` are forwarded too.
   Conflicting `dropout_prob` vs `aggregator_params["dropout_prob"]`
   raises a `ValueError` instead of picking one silently.

## Restoring the old behaviour intentionally

```python
from tgraphx import CNNEncoder, LEGACY_CNN_DROPOUT_PROB   # == 0.3
from tgraphx.layers.aggregator import DeepCNNAggregator

enc = CNNEncoder.legacy(13, 32, num_layers=3, hidden_channels=32)   # dropout 0.3, BN, residual
agg = DeepCNNAggregator.legacy(32, 32)                              # dropout 0.3, BN
layer = ConvMessagePassing(in_shape, out_shape, dropout_prob=0.3)   # explicit legacy value
```

`.legacy(...)` constructors emit no warning — they are an explicit
opt-in and exist so pre-1.5 experiments can be reproduced exactly.

## Checkpoints are safe

- Dropout modules hold **no parameters**: `state_dict` layouts are
  identical for any `dropout_prob`, so 1.4.2-era checkpoints load into
  v1.5.0 models unchanged (and vice versa).
- **Evaluation outputs never depended on dropout** (`model.eval()`
  disables it), so inference on loaded checkpoints is bit-identical.
- What *does* change is training-time regularization for code that
  relied on the silent default. If you are fine-tuning a model that was
  trained with the hidden 0.3 and want identical behaviour, pass
  `dropout_prob=0.3` (or use `.legacy(...)`).
- `use_batchnorm` / `use_residual` defaults are **unchanged** (they
  affect the parameter layout of checkpoints); they are now surfaced in
  `repr()` and `config()`. Note that aggregator BatchNorm helps on dense
  graphs but hurts when many nodes have zero incoming edges — choose per
  graph density.

## Silencing the transition warning globally

```python
import warnings, tgraphx
warnings.filterwarnings("ignore", category=tgraphx.DropoutDefaultChangeWarning)
```

Preferred fix: pass `dropout_prob` (or factory `dropout`) explicitly —
the warning marks exactly the construction sites whose training
behaviour changed between 1.4.2 and 1.5.0.

# TGraphXSetAttention reference-configuration parity (1.5.1)

**Question.** Can the released package represent the *exact* set-attention
architecture evaluated in the TGraphX experiment program (the frozen-base
`set_transformer` arm of the revised PASTIS-R study), and reproduce its
results from the completed checkpoints?

**Answer: yes — proven at prediction level on the full validation split,
for all five seeds.**

## What was compared

| side | construction |
|---|---|
| experiment | `SetTransformer` from the evaluation program (`shared/src/models/set_transformer.py`): custom strided channel-doubling CNN encoder (32→64→128, BatchNorm, no residual), 2 × `torch.nn.TransformerEncoderLayer` (post-LN, ReLU, dropout 0.1), single-query PMA, linear head — 189,650 parameters |
| packaged | `tgraphx.TGraphXSetAttention(**TGraphXSetAttention.reference_config(in_shape=(13, 32, 32), num_classes=18))` — `StridedConvEncoder`, 2 × post-LN ReLU `SetAttentionBlock` (dropout 0.1, attention dropout 0.1), `AttentionPooling` (1 seed, no attention-weight dropout), linear head |

Checkpoints: the completed frozen-base best-validation states
(`checkpoints/frozen_base/set_transformer_s{0..4}.pt`, key `best.state`),
loaded into the packaged model via
`TGraphXSetAttention.map_reference_state_dict` (documented key mapping;
**strict** load, 0 missing / 0 unexpected keys, all 54 tensors bitwise
equal after load).

Scripts: `tools/verify_set_attention_reference_parity.py` (stage-by-stage,
seed 0) and `tools/verify_set_attention_reference_parity_allseeds.py`
(full-split, all seeds). Both are read-only with respect to the evidence
repositories and require the external evidence tree; they are maintenance
tools, not CI tests.

## Stage-by-stage numerical parity (seed 0, fixed real validation batch)

One fixed batch of 64 validation graphs (3,904 nodes of shape [13, 32, 32]),
both models in eval mode, CPU, torch 2.12.1:

| stage | max abs error | mean abs error |
|---|---|---|
| encoder output | 0.0 (bitwise) | 0.0 |
| attention-stack output | 1.43e-06 | 1.09e-07 |
| pooled representation | 1.43e-06 | 2.00e-07 |
| logits | 2.38e-06 | 3.99e-07 |

Parameter count identical (189,650 = 189,650). Predicted labels identical on
the fixed batch. The residual ~1e-06 float32 differences come from
`torch.nn.TransformerEncoder`'s nested-tensor fast path evaluating the same
mathematics in a different kernel order; the encoder (evaluated identically
on both sides) is bitwise equal.

## Full-validation parity (all seeds, GPU)

Full natural-label validation split (25,772 samples), both implementations:

| seed | predictions identical | macro-F1 (both implementations) | recorded raw result | exact match |
|---|---|---|---|---|
| 0 | yes | 0.6961913111905216 | 0.6961913111905216 | yes |
| 1 | yes | 0.6869007203236991 | 0.6869007203236991 | yes |
| 2 | yes | 0.6997762812137694 | 0.6997762812137694 | yes |
| 3 | yes | 0.7261988270944028 | 0.7261988270944028 | yes |
| 4 | yes | 0.7022463419414295 | 0.7022463419414295 | yes |

5-seed mean ± SD reproduced through the packaged class: **0.7023 ± 0.0146**
— identical to the recorded frozen-base summary for the `set_transformer`
arm.

## What this does and does not establish

- **Established:** the evaluated set-attention instance is exactly
  representable as an explicit, documented configuration of the released
  `TGraphXSetAttention` class (`reference_config`), and the completed
  checkpoints reconstruct it via a strict mapped state-dict load with
  identical predictions. The published family-level result is therefore
  reconstructable from the released package plus the archived checkpoints.
- **Not claimed:** that the *default* `TGraphXSetAttention` construction
  (pre-LN, GELU, dropout 0.0, `CNNEncoder`) matches the evaluated instance —
  it deliberately does not; the evaluated architecture is selected only by
  the explicit reference configuration.

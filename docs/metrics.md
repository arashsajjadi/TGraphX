# Metrics

`tgraphx.metrics` is a pure-PyTorch metrics library.  Importing the
module is lightweight.

## Classification

| Function | Inputs | Output |
|----------|--------|--------|
| `accuracy(preds, labels)` | `[N]` indices or `[N, C]` logits + `[N]` indices | float |
| `top_k_accuracy(preds, labels, k)` | `[N, C]` logits + `[N]` indices | float |
| `confusion_matrix(preds, labels, num_classes)` | as above | `[C, C]` long tensor |
| `precision_recall_f1(preds, labels, ...)` | as above | dict |
| `classification_report(preds, labels, ...)` | as above | dict (acc + P/R/F1 + CM) |

Convention:

* Inputs accept either raw logits or class indices — the helper
  resolves the prediction with `argmax(dim=-1)`.
* All metrics call `.detach()` before computing — gradients are not
  retained.
* `precision_recall_f1` accepts `zero_division=0.0` to control the
  return value when a class has no predictions.

## Regression

| Function | Description |
|----------|-------------|
| `mae(preds, targets)` | mean absolute error |
| `mse(preds, targets)` | mean squared error |
| `rmse(preds, targets)` | root MSE |
| `r2_score(preds, targets)` | coefficient of determination |
| `regression_report(preds, targets)` | dict of all of the above |

`mean_absolute_error` and `mean_squared_error` are aliases for
backward compatibility with `tgraphx.training`.

## Ranking and link prediction

| Function | Description |
|----------|-------------|
| `hits_at_k(scores, target_idx, k)` | top-k hit fraction |
| `mean_reciprocal_rank(scores, target_idx)` | MRR |
| `ndcg_at_k(scores, targets, k)` | normalised DCG@k |
| `roc_auc(pos_scores, neg_scores)` | binary ROC-AUC (Mann-Whitney U) |
| `average_precision(pos_scores, neg_scores)` | binary AP |
| `link_prediction_report(pos_scores, neg_scores)` | dict |

## Reports

`tgraphx.metrics.reports` provides convenience aggregations for graph
/ node / edge classification and regression — all return JSON-friendly
dicts.

## OGB evaluator

`tgraphx.metrics.OGBEvaluatorWrapper` (re-exported from
`tgraphx.datasets.ogb_wrappers`) wraps the official OGB evaluators
when the user has `pip install "tgraphx[ogb]"`.

## Logits vs probabilities

* Classification metrics treat 1-D inputs as **class indices**
  (LongTensor expected) and 2-D inputs as **logits** (any float
  tensor).  Both produce the same answer for the standard top-1
  metrics.
* `roc_auc` and `average_precision` accept *raw scores*; higher score
  must mean higher confidence in the positive class.

## Autograd hygiene

Every metric calls `.detach()` on its inputs and returns Python floats
(or python lists/dicts of floats).  Storing metric values in a list
will not retain a computation graph.

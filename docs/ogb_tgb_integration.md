# OGB / TGB evaluator integration

TGraphX ships **optional** wrappers around the official OGB and TGB
evaluators in `tgraphx.benchmarks`.  The wrappers never download
datasets; callers must construct datasets themselves (with explicit
`download=True` on the upstream class) and pass predictions to
`.eval(...)`.

```python
from tgraphx.benchmarks import OGBNodeEvaluator

if OGBNodeEvaluator.is_available:
    evaluator = OGBNodeEvaluator(name="ogbn-arxiv")
    metrics = evaluator.eval(y_true=y_true, y_pred=y_pred)
    # {'acc': 0.6712}
```

When the optional `ogb` (resp. `tgb`) package is missing,
instantiation raises `OptionalDependencyError` with a clear install
hint, and `.is_available` is `False`.

## Wrappers

| Class | Upstream evaluator |
|---|---|
| `OGBNodeEvaluator` | `ogb.nodeproppred.Evaluator` |
| `OGBLinkEvaluator` | `ogb.linkproppred.Evaluator` |
| `OGBGraphEvaluator` | `ogb.graphproppred.Evaluator` |
| `TGBLinkEvaluator` | `tgb.linkproppred.evaluate.Evaluator` (or user-supplied) |

## Honesty

These wrappers do not implement the OGB protocol — they forward to the
official package.  TGraphX makes **no SOTA / leaderboard claims**.

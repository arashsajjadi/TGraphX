# Experiment manager

`tgraphx.experiments` (new in v0.3.0) is a lightweight, dashboard-compatible
experiment manager.  Importing it starts no background work; everything
happens inside `Runner.fit()`.

## Install

No extra dependencies beyond the base `tgraphx` install.  YAML configs require
PyYAML, which is already a base dependency.

## Import

```python
from tgraphx.experiments import (
    Runner,
    GridRunner,
    load_config,
    ExperimentConfig,
    expand_grid,
    summarize_runs,
    write_markdown_report,
    EarlyStopping,
    ModelCheckpoint,
    CSVLoggerCallback,
    LearningRateLogger,
)
```

## Config schema

Configs are plain YAML or JSON files.  No `eval`, no `exec`; unknown
top-level keys raise `ValueError`.

```yaml
seed: 0
run_name: my_run
run_dir: runs/my_run       # optional; auto-generated when omitted

dataset:
  name: synthetic:patch_graph   # any key accepted by get_dataset()
  kwargs: {num_graphs: 32, seed: 0}

transforms: []                  # optional list; each item: {name, kwargs}

model:
  task: graph_classification    # graph_classification | graph_regression
                                # node_classification | node_regression
  layer: conv                   # conv | gat | sage | gin | linear | ...
  in_shape: [1, 8, 8]
  hidden_shape: [8, 8, 8]
  num_layers: 2
  num_classes: 6                # for classification
  out_dim: 1                    # for regression (alternative to num_classes)
  pooling: mean                 # mean | sum | max

training:
  epochs: 5
  batch_size: 8
  lr: 0.005
  optimizer: adam               # adam | adamw | sgd
  device: cpu                   # cpu | cuda | mps
  loss: cross_entropy           # cross_entropy | mse | mae | bce_with_logits | auto
  val_ratio: 0.2
  test_ratio: 0.0
  weight_decay: 0.0

callbacks:
  - {name: csv_logger}
  - {name: lr_logger}
  - name: early_stopping
    kwargs: {monitor: train_loss, patience: 3, mode: min}
  - name: model_checkpoint
    kwargs: {monitor: train_loss, mode: min, save_best: true, save_latest: true}
```

### Config dataclasses

| Dataclass | Key fields |
|-----------|-----------|
| `ExperimentConfig` | `seed`, `run_name`, `run_dir`, `dataset`, `transforms`, `model`, `training`, `callbacks` |
| `DatasetConfig` | `name`, `kwargs` |
| `ModelConfig` | `task`, `layer`, `in_shape`, `hidden_shape`, `num_layers`, `num_classes`, `out_dim`, `pooling`, `extra` |
| `TrainingConfig` | `epochs`, `batch_size`, `lr`, `optimizer`, `device`, `loss`, `val_ratio`, `test_ratio`, `weight_decay` |
| `TransformConfig` | `name`, `kwargs` |
| `CallbackConfig` | `name`, `kwargs` |

## Runner

```python
from tgraphx.experiments import Runner, load_config

cfg = load_config("examples/configs/synthetic_patch_graph.yaml")
runner = Runner(cfg)
history = runner.fit()
# history: list of {"epoch": int, "train_loss": float, ...}
```

`Runner.__init__` arguments:

| Argument | Type | Description |
|----------|------|-------------|
| `config` | `ExperimentConfig` | Validated config (from `load_config`) |
| `run_dir` | `str \| Path \| None` | Override run directory; default: `runs/<run_name>/<timestamp>` |
| `callbacks` | `list[Callback] \| None` | Extra programmatic callbacks appended after config-specified ones |

`Runner.fit()` returns `list[dict[str, float]]` — one dict per epoch.

`Runner.resume(checkpoint="checkpoints/latest.pt")` loads model + optimizer
state from a checkpoint inside `run_dir`.

## Run artifacts

Every `Runner.fit()` call writes files under `run_dir` only:

```
runs/<run_name>/<timestamp>/
├── run_metadata.json       # status, device, seed, tgraphx version, timestamps
├── experiment_config.json  # exact copy of the config
├── experiment_summary.json # total epochs, best metric, final train loss
├── metrics.csv             # dashboard-compatible; one row per epoch
└── checkpoints/
    ├── best.pt             # saved when ModelCheckpoint(save_best=True) is active
    └── latest.pt           # saved when ModelCheckpoint(save_latest=True) is active
```

The dashboard reads these files when you run:

```bash
tgraphx-dashboard --logdir runs/<run_name>/<timestamp>
```

## GridRunner

`GridRunner` runs a cartesian product of hyperparameter values across
multiple seeds.  The grid spec lives in the same YAML file:

```yaml
seed: 0
run_name: lr_sweep
dataset: {name: synthetic:patch_graph, kwargs: {num_graphs: 8}}
model: {task: graph_classification, layer: conv, in_shape: [1, 4, 4],
        hidden_shape: [8, 4, 4], num_layers: 2, num_classes: 6}
training: {epochs: 4, lr: 0.001, optimizer: adam}
grid:
  training.lr: [0.001, 0.005]
  training.epochs: [3, 4]
seeds: [0, 1]
```

```python
import yaml
from tgraphx.experiments import GridRunner

raw = yaml.safe_load(open("examples/configs/grid_sweep.yaml"))
gr = GridRunner.from_dict(raw)
results = gr.run()   # list of per-run summary dicts
```

`GridRunner.run()` writes a `grid_summary.json` in the top-level sweep
directory listing all per-run results.

## Built-in callbacks

| Name (YAML) | Class | Key kwargs |
|-------------|-------|-----------|
| `csv_logger` | `CSVLoggerCallback` | `filename="metrics.csv"`, `with_timestamp=True` |
| `early_stopping` | `EarlyStopping` | `monitor`, `patience`, `mode`, `min_delta` |
| `model_checkpoint` | `ModelCheckpoint` | `monitor`, `mode`, `save_best`, `save_latest` |
| `lr_logger` | `LearningRateLogger` | — |

Custom callbacks extend `tgraphx.experiments.Callback` and override
`on_train_begin`, `on_epoch_end`, and/or `on_train_end`.

## CLI

Console scripts registered by the package:

```bash
# Train a single experiment from a YAML/JSON config
tgraphx-train  examples/configs/synthetic_patch_graph.yaml

# Run a grid sweep
tgraphx-grid   examples/configs/grid_sweep.yaml

# Summarise all run directories under a path (prints Markdown)
tgraphx-report runs/
```

## Summarise runs

```python
from tgraphx.experiments import summarize_runs, write_markdown_report

summaries = summarize_runs("runs/")
write_markdown_report(summaries, "runs/summary.md")
```

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: Unknown top-level config keys` | Typo in YAML key | Check allowed keys: `seed`, `run_name`, `run_dir`, `dataset`, `transforms`, `model`, `training`, `callbacks` |
| `ValueError: config.dataset.name is required` | Missing `name` under `dataset:` | Add `dataset: {name: ...}` |
| `ValueError: Unknown callback 'x'` | Callback name not in built-ins | Built-ins: `csv_logger`, `early_stopping`, `model_checkpoint`, `lr_logger` |
| `FileNotFoundError: Config not found` | Wrong path to YAML | Check the path; use absolute paths when calling from outside the repo root |

## Related examples and tests

- `examples/configs/synthetic_patch_graph.yaml` — minimal single-run config
- `examples/configs/grid_sweep.yaml` — grid sweep config
- `examples/experiment_config_quickstart.py` — Python API walkthrough
- `examples/experiment_end_to_end_validation.py` — full validation (train → checkpoint → resume)
- `tests/test_experiments.py` — unit tests covering config validation, runner, early stopping, checkpoints, CLI, grid expansion

---
name: Bug report
about: Report a reproducible bug in TGraphX
title: "[BUG] "
labels: bug
assignees: ''
---

## Description

A clear, concise description of the bug.

## Environment

Paste the output of:
```python
from tgraphx.performance import env_report
import json
print(json.dumps(env_report(include_hardware=True), indent=2))
```

## Minimal reproducible example

```python
# Paste a minimal script that reproduces the bug
import torch
from tgraphx import ...

```

## Expected behaviour

What you expected to happen.

## Actual behaviour

What actually happened. Include the full traceback.

## Additional context

Any other context (dataset size, node/edge counts, custom layer subclasses, etc.).

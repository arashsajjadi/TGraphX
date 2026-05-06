---
name: Feature request
about: Suggest a new feature or improvement
title: "[FEAT] "
labels: enhancement
assignees: ''
---

## Summary

A clear, concise description of the proposed feature.

## Motivation

Why is this feature useful? What problem does it solve?

## Proposed API

```python
# Show what the API would look like from a user's perspective
from tgraphx import ...
```

## Alternatives considered

What alternatives did you consider? Why do you prefer the proposed approach?

## Additional context

Links to papers, other libraries, or reference implementations that informed
this request.

## Scope check

Please confirm:
- [ ] This is not a request for `train_epoch` / `evaluate` / `fit`
  (intentionally not implemented; write your own loop).
- [ ] This is not a request for `TensorBoardLogger` / `MLflowLogger`
  (use upstream tools directly).
- [ ] This does not require breaking changes to the `Graph` or `GraphBatch` API.

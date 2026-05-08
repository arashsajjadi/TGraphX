"""Manual public-dataset validation scripts.

These scripts are **not** run in CI.  They:

* require the user to pass ``--download`` before any network access,
* cap dataset size by default (``--max-samples`` / ``--max-nodes``),
* skip cleanly when an optional upstream package is missing,
* write dashboard-compatible artefacts (`metrics.csv`,
  `run_metadata.json`, `dataset_metadata.json`, etc.) under an
  explicit ``--output-run-dir``,
* never make benchmark or SOTA claims.

See ``docs/public_dataset_validation.md`` for the full policy.
"""

# Runtime Reproduction

Command:
`PYTHONPATH=repo/src python -m release_readiness.cli sample-input.json`

Observed stdout:
```text
# Release Readiness

- ready: True
- services: api, worker
```

Command:
`PYTHONPATH=repo/src python -m release_readiness.cli sample-input.json --output json`

Observed stdout:
```text
"# Release Readiness\n\n- ready: True\n- services: api, worker"
```

These outputs show that the default CLI path is no longer machine-readable JSON,
and the explicit `--output json` path now emits a JSON-encoded markdown string
instead of the historical object payload.

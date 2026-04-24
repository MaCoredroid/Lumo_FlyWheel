# Runtime Checks

## Default CLI Output

Command:
`PYTHONPATH=repo/src python -m release_readiness.cli dummy.json`

Output:

```text
# Release Readiness

- ready: True
- services: api, worker
```

## Explicit JSON Output

Command:
`PYTHONPATH=repo/src python -m release_readiness.cli dummy.json --output json`

Output:

```text
"# Release Readiness\n\n- ready: True\n- services: api, worker"
```

## Export Path for `output="json"`

Command:
`PYTHONPATH=repo/src python - <<'PY' ... export_report(report, output='json') ... PY`

Observed type and value:

```text
str
# Release Readiness

- ready: True
- services: api, worker
```

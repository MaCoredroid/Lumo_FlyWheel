# Runtime Checks

## Default CLI output

Command:
`PYTHONPATH=repo/src python3 -m release_readiness.cli dummy`

Output:
```text
# Release Readiness

- ready: True
- services: api, worker
```

## Explicit JSON export

Command:
`PYTHONPATH=repo/src python3 -m release_readiness.cli dummy --output json`

Output:
```text
"# Release Readiness\n\n- ready: True\n- services: api, worker"
```

## Export API return type

Command:
`PYTHONPATH=repo/src python3 - <<'PY'
from release_readiness.export import export_report
print(type(export_report({'version':1,'ready':True,'services':['api','worker']}, output='json')).__name__)
print(export_report({'version':1,'ready':True,'services':['api','worker']}, output='json'))
PY`

Output:
```text
str
# Release Readiness

- ready: True
- services: api, worker
```

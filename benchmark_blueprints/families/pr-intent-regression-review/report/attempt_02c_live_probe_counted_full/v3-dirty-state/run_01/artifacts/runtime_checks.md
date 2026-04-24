PYTHONPATH=src python - <<'PY'
from release_readiness.export import export_report
report = {'version': 1, 'ready': True, 'services': ['api', 'worker']}
value = export_report(report, output='json')
print(type(value).__name__)
print(value)
PY

Output:
str
# Release Readiness

- ready: True
- services: api, worker

PYTHONPATH=src python -m release_readiness.cli dummy

Output:
# Release Readiness

- ready: True
- services: api, worker

PYTHONPATH=src python -m release_readiness.cli dummy --output json

Output:
"# Release Readiness\n\n- ready: True\n- services: api, worker"

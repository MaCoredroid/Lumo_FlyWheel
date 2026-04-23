python3 - <<'PY'
import json, os
r=json.loads(open(os.environ['RESULT_FILE']).read())
raise SystemExit(0 if r['milestones']['M5_e2e'] else 1)
PY

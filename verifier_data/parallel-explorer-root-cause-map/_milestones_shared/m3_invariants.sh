python3 - <<'PY'
import json, os
r=json.loads(open(os.environ['RESULT_FILE']).read())
raise SystemExit(0 if r['milestones']['M3_invariants'] else 1)
PY

python3 - <<'PY'
import json, os
r=json.loads(open(os.environ['RESULT_FILE']).read())
raise SystemExit(0 if r['milestones']['M1_localization'] else 1)
PY

#!/usr/bin/env bash
python3 - "$RESULT_FILE" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
    raise SystemExit(0 if data.get("milestones", {}).get("M2_primary_fix") else 1)
except Exception:
    raise SystemExit(2)
PY

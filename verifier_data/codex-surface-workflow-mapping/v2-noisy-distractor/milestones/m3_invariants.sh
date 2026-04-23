#!/bin/sh
set -eu
python3 - "$RESULT_FILE" <<'PY'
import json, sys
path = sys.argv[1]
payload = json.load(open(path, "r", encoding="utf-8"))
sys.exit(0 if payload.get("milestones", {}).get("M3_invariants", False) else 1)
PY

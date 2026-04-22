#!/usr/bin/env bash
set -euo pipefail
RESULT_FILE="${RESULT_FILE:-/results/verify_result.json}"
if [[ ! -s "$RESULT_FILE" ]]; then
  exit 2
fi
python3 - "$RESULT_FILE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ok = bool(d.get("milestones", {}).get("M3_invariants", False))
sys.exit(0 if ok else 1)
PY

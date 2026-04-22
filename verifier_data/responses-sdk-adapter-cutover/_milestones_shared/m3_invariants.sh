#!/usr/bin/env bash
set -euo pipefail
RESULT_FILE="${RESULT_FILE:-/results/verify_result.json}"
python3 - "$RESULT_FILE" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
ok = bool(data.get("milestones", {}).get("M3_invariants", False))
if data.get("integrity_flag", 0) == 1 and ok:
    sys.exit(1)
sys.exit(0 if ok else 1)
PY

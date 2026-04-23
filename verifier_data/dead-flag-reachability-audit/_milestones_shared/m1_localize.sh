#!/usr/bin/env bash
set -euo pipefail
RESULT_FILE="${RESULT_FILE:-/results/verify_result.json}"
python3 - "$RESULT_FILE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if bool(d.get("milestones", {}).get("M1_localization", False)) else 1)
PY

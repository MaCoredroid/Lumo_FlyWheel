#!/usr/bin/env bash
set -euo pipefail
python3 - "${RESULT_FILE:-/results/verify_result.json}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("milestones", {}).get("M1_localization", False) else 1)
PY

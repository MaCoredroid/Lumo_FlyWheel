#!/usr/bin/env bash
set -euo pipefail
RESULT_FILE="${RESULT_FILE:-/results/verify_result.json}"
python3 - "$RESULT_FILE" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1]))
mapping = {
    "m1_localize": "M1_localization",
    "m2_primary_fix": "M2_primary_fix",
    "m3_invariants": "M3_invariants",
    "m4_functional": "M4_functional",
    "m5_e2e": "M5_e2e",
}
key = mapping["m4_functional"]
sys.exit(0 if data.get("milestones", {}).get(key, False) else 1)
PY

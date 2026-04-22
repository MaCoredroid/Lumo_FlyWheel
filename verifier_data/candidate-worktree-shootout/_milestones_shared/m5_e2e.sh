#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
sys.exit(0 if data.get("milestones", {}).get("M5_e2e", False) else 1)
PY

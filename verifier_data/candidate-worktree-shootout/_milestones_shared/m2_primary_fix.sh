#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
sys.exit(0 if data.get("milestones", {}).get("M2_primary_fix", False) else 1)
PY

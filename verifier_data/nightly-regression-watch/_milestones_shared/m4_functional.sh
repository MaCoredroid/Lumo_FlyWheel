#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1]))
print("")
sys.exit(0 if result.get("milestones", {}).get("M4_functional", False) else 1)
PY

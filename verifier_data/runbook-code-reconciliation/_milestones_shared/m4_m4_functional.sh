#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
raise SystemExit(0 if data.get("milestones", {}).get("M4_functional", False) else 1)
PY

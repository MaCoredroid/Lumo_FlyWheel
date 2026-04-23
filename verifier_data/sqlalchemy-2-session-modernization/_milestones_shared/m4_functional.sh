#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path))
sys.exit(0 if data["milestones"]["M4_functional"] else 1)
PY

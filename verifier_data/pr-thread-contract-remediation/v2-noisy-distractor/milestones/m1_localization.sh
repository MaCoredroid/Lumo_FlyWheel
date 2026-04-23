#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" "$1" <<'PY'
import json
import sys
path, key = sys.argv[1], sys.argv[2]
data = json.load(open(path))
sys.exit(0 if data.get("milestones", {}).get(key, False) else 1)
PY

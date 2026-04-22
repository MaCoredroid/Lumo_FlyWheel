#!/usr/bin/env bash
set -euo pipefail
RESULT_FILE="${RESULT_FILE:?RESULT_FILE must be set}"
python3 - "$RESULT_FILE" <<'PY'
import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
value = data.get('milestones', {}).get('M4_functional', False)
raise SystemExit(0 if value else 1)
PY

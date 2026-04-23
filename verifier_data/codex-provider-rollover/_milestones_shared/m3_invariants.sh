#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
ok = data.get('integrity_flag') == 0
raise SystemExit(0 if ok else 1)
PY

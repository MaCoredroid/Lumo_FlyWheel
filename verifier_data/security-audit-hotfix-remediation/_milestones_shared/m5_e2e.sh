#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" <<'PY'
import json, sys
result = json.load(open(sys.argv[1], "r", encoding="utf-8"))
ok = bool(result.get("milestones", {}).get("M5_e2e", False))
raise SystemExit(0 if ok else 1)
PY

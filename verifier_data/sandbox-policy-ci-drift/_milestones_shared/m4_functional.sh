
#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)
raise SystemExit(0 if payload.get("milestones", {}).get("M4_functional", False) else 1)
PY

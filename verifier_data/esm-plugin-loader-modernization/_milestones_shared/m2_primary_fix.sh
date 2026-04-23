
#!/usr/bin/env bash
set -euo pipefail

python3 - "$RESULT_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

passed = bool(payload.get("milestones", {}).get("M2_primary_fix", False))
raise SystemExit(0 if passed else 1)
PY

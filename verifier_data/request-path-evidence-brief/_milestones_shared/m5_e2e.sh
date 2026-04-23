
#!/usr/bin/env bash
set -euo pipefail
RESULT_FILE="${RESULT_FILE:-/results/verify_result.json}"
python3 - "$RESULT_FILE" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
sys.exit(0 if data.get("milestones", {}).get("M5_e2e", False) else 1)
PY

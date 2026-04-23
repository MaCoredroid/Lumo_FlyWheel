#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
value = bool(result.get("milestones", {}).get("M2_primary_fix", False))
raise SystemExit(0 if value else 1)
PY

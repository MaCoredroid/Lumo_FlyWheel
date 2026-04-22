#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
sys.exit(0 if result.get("milestones", {}).get("M1_localization", False) else 1)
PY

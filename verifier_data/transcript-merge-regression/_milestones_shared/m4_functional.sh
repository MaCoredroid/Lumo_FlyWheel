#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(Path(os.environ["RESULT_FILE"]).read_text())
key = "M4_functional"
raise SystemExit(0 if result.get("milestones", {}).get(key, False) else 1)
PY

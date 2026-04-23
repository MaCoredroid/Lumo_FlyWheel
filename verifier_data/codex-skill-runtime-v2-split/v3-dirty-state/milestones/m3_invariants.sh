#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json
import os
import sys
from pathlib import Path

result = json.loads(Path(os.environ["RESULT_FILE"]).read_text())
passed = bool(result.get("milestones", {}).get("M3_invariants", False))
sys.exit(0 if passed else 1)
PY

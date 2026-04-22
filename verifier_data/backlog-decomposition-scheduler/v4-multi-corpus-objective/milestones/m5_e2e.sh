#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json, os, sys
r=json.load(open(os.environ['RESULT_FILE']))
sys.exit(0 if r['milestones'].get('M5_e2e') else 1)
PY

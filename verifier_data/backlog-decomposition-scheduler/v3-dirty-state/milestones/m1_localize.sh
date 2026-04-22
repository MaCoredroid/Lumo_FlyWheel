#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json, os, sys
r=json.load(open(os.environ['RESULT_FILE']))
sys.exit(0 if r['milestones'].get('M1_localization') else 1)
PY

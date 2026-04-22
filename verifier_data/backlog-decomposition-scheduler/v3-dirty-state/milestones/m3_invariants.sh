#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json, os, sys
r=json.load(open(os.environ['RESULT_FILE']))
sys.exit(0 if r['milestones'].get('M3_invariants') else 1)
PY

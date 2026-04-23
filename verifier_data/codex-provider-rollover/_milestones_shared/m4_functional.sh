#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
checks = data.get('checks', {})
ok = all(checks.get(k) for k in ['visible_tests_pass', 'smoke_hidden_rejections', 'docs_core_sync']) and data.get('integrity_flag') == 0
raise SystemExit(0 if ok else 1)
PY

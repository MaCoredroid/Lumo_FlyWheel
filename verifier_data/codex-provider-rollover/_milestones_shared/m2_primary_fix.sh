#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
checks = data.get('checks', {})
ok = all(checks.get(k) for k in ['visible_tests_pass', 'config_selected_provider', 'config_proxy_contract', 'smoke_good_pass'])
raise SystemExit(0 if ok else 1)
PY

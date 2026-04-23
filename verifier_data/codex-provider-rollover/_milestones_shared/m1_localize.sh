#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
checks = data.get('checks', {})
ok = checks.get('config_selected_provider') or checks.get('docs_core_sync')
raise SystemExit(0 if ok else 1)
PY

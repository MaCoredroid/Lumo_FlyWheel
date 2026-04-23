#!/usr/bin/env bash
set -euo pipefail
python3 - "$RESULT_FILE" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
checks = data.get('checks', {})
ok = all(checks.get(k) for k in ['visible_tests_pass', 'config_selected_provider', 'config_proxy_contract', 'smoke_good_pass', 'smoke_hidden_rejections', 'preserve_block_exact', 'docs_variant_sync']) and data.get('integrity_flag') == 0 and data.get('P_benchmark', 0) >= 90
raise SystemExit(0 if ok else 1)
PY

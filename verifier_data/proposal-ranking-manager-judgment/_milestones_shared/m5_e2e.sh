#!/usr/bin/env bash
# M5 E2E integration (HLD §7.5 weight 0.30).
# Passes iff the ranking aligns with gold (accepted_match AND kendall_tau above
# the per-variant threshold) AND no partial-credit ceiling ≤ 30 fired. This is
# the top-of-stack "did the agent nail the whole task" signal. Depends on M2.
# Force-failed by H=1 per HLD §7.7.5.
set -euo pipefail
RESULT_FILE="${RESULT_FILE:-/results/verify_result.json}"
if [[ ! -s "$RESULT_FILE" ]]; then
  echo "m5_e2e: missing $RESULT_FILE" >&2
  exit 2
fi
python3 - "$RESULT_FILE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ok = bool(d.get("milestones", {}).get("M5_e2e", False))
sys.exit(0 if ok else 1)
PY

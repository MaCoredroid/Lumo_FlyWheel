#!/usr/bin/env bash
# M4 Functional (HLD §7.5 weight 0.20).
# Passes iff the structural brief gates pass (ranking_length, accepted_valid,
# entry_fields, assumption_ledger) AND the ranking_without_grounding ceiling
# did NOT trigger. Depends on M2 (family.yaml milestone_dependencies).
# Force-failed by H=1 per HLD §7.7.5 (scorer applies this before this script
# runs).
set -euo pipefail
RESULT_FILE="${RESULT_FILE:-/results/verify_result.json}"
if [[ ! -s "$RESULT_FILE" ]]; then
  echo "m4_functional: missing $RESULT_FILE" >&2
  exit 2
fi
python3 - "$RESULT_FILE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ok = bool(d.get("milestones", {}).get("M4_functional", False))
sys.exit(0 if ok else 1)
PY

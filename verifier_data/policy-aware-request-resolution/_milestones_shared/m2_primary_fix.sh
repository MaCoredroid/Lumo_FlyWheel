#!/usr/bin/env bash
# M2 Primary fix (HLD §7.5 weight 0.20).
# Passes if brief_exists AND brief_parses (manager_brief.json present, valid
# schema, parseable JSON). This is the "did the agent produce the artifact"
# floor — required before M4/M5 can possibly pass (family.yaml declares the
# M4→M2, M5→M2 dependency).
set -euo pipefail
RESULT_FILE="${RESULT_FILE:-/results/verify_result.json}"
if [[ ! -s "$RESULT_FILE" ]]; then
  echo "m2_primary_fix: missing $RESULT_FILE" >&2
  exit 2
fi
python3 - "$RESULT_FILE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ok = bool(d.get("milestones", {}).get("M2_primary_fix", False))
sys.exit(0 if ok else 1)
PY

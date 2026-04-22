#!/usr/bin/env bash
# M1 Localization (HLD §7.5 weight 0.10).
# Passes if the brief exists and cites ≥ 3 distinct evidence files across its
# rejection set. Scorer-side proxy for "agent localized the relevant evidence
# before writing the brief". LLD-06 can override with turn-level tool-call data.
set -euo pipefail
RESULT_FILE="${RESULT_FILE:-/results/verify_result.json}"
if [[ ! -s "$RESULT_FILE" ]]; then
  echo "m1_localize: missing $RESULT_FILE" >&2
  exit 2
fi
python3 - "$RESULT_FILE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ok = bool(d.get("milestones", {}).get("M1_localization", False))
sys.exit(0 if ok else 1)
PY

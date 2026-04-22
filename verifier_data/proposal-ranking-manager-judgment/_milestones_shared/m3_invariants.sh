#!/usr/bin/env bash
# M3 Invariants / anti-shortcut (HLD §7.5 weight 0.20).
# Passes iff integrity_flag == 0 AND shortcut_detected == false AND none of the
# integrity-related ceilings fired (tests_modified, pytest_shim,
# readonly_tree_mutated, wrote_outside_brief, network_egress). Per HLD §7.7.5
# this slot is force-failed when integrity_flag == 1 regardless of any other
# signal; the scorer enforces that and so does this script (transitively via
# the milestones dict).
set -euo pipefail
RESULT_FILE="${RESULT_FILE:-/results/verify_result.json}"
if [[ ! -s "$RESULT_FILE" ]]; then
  echo "m3_invariants: missing $RESULT_FILE" >&2
  exit 2
fi
python3 - "$RESULT_FILE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ok = bool(d.get("milestones", {}).get("M3_invariants", False))
if d.get("integrity_flag", 0) == 1 and ok:
    # Defense-in-depth: if integrity fired the milestone MUST be False.
    # If scorer and script disagree, fail closed.
    print("m3_invariants: integrity_flag=1 but milestone reports True (inconsistent)", file=sys.stderr)
    sys.exit(1)
sys.exit(0 if ok else 1)
PY

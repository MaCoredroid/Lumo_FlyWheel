#!/usr/bin/env bash
# E7a step 2: one capture boot per FROZEN pilot prefix, serially, each followed by the provenance inspector.
# Stops at the first failed capture or failed provenance. `--first-only` proves ONE complete real-request payload
# (review 05 item 2) and exits; the multi-prefix loop is only run after that proof.
# Usage: capture_loop.sh <frozen_prefixes.json> <pool.json> <layer_prefix> <out_root> [--first-only]
set -euo pipefail
WT=/home/mark/lumo-paper-v2-20260921
E=$WT/papers/gdn-tree-scan-mlsys/v2/experiments/e7a
FROZEN=$1; POOL=$2; LAYER=$3; ROOT=$4; MODE=${5:-}
mkdir -p "$ROOT"
python3 - "$FROZEN" > "$ROOT/pilot_ids.txt" <<'PY'
import json, sys
f = json.load(open(sys.argv[1]))
assert f["counts"]["pilot"] == len(f["pilot"]), "frozen pilot count mismatch"
for r in f["pilot"]:
    print(r["id"])
PY
cp "$FROZEN" "$ROOT/frozen_prefixes.json"; sha256sum "$FROZEN" "$POOL" > "$ROOT/inputs.sha256"
n=0
while read -r pid; do
  [[ -z "$pid" ]] && continue
  n=$((n+1))
  RUN="$ROOT/capture_$(printf '%02d' $n)_${pid}"
  echo "=== [$n] capture prefix $pid layer $LAYER -> $RUN  ($(date -u +%H:%M:%SZ)) ==="
  if ! bash "$E/serve_drivers.sh" capture "$RUN" "$pid" "$POOL" "$LAYER" > "$RUN.driver.log" 2>&1; then
    echo "CAPTURE FAILED for $pid (see $RUN.driver.log)"; tail -20 "$RUN.driver.log"; exit 10
  fi
  if ! python3 "$E/inspect_capture.py" "$RUN" > "$RUN.provenance.log" 2>&1; then
    echo "PROVENANCE FAILED for $pid (see $RUN.provenance.log)"; tail -30 "$RUN.provenance.log"; exit 11
  fi
  tail -1 "$RUN.provenance.log"
  [[ "$MODE" == "--first-only" ]] && { echo "first-only mode: stopping after one proven capture"; exit 0; }
done < "$ROOT/pilot_ids.txt"
echo "CAPTURE LOOP DONE: $n prefixes"

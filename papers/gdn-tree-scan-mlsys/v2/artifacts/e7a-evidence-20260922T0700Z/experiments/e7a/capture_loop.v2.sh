#!/usr/bin/env bash
# E7a step 2: one capture boot per FROZEN pilot prefix, serially, each followed by the provenance inspector.
# Stops at the first failed capture or failed provenance. `--first-only` proves ONE complete real-request payload
# (review 05 item 2) and exits; the multi-prefix loop is only run after that proof.
# v2: START_INDEX=<k> skips the first k frozen pilot prefixes (already proven in an earlier root). Staged as a new file
# because the v1 loop was executing when this was written (bash reads by byte offset).
# Usage: SCORES=<prefix_scores.json> [START_INDEX=k] capture_loop.v2.sh <frozen_prefixes.json> <pool.json> <layer_prefix> <out_root> [--first-only]
# Validates the frozen set (schema, pool/scores hash chain, exact 8+32 counts, per-row membership/hash/tokens/stratum) BEFORE any boot.
set -euo pipefail
WT=/home/mark/lumo-paper-v2-20260921
E=$WT/papers/gdn-tree-scan-mlsys/v2/experiments/e7a
FROZEN=$1; POOL=$2; LAYER=$3; ROOT=$4; MODE=${5:-}
mkdir -p "$ROOT"
python3 - "$FROZEN" "$POOL" "${SCORES:-}" > "$ROOT/pilot_ids.txt" <<'PY'
import json, sys, hashlib
frozen_p, pool_p, scores_p = sys.argv[1], sys.argv[2], sys.argv[3]
f = json.load(open(frozen_p)); pool = json.load(open(pool_p))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def die(m): print("FROZEN VALIDATION FAILED: " + m, file=sys.stderr); sys.exit(20)
if f.get("schema") != "e7a.frozen_prefixes.v2": die(f"schema {f.get('schema')}")
if f.get("pool_sha256") != sha(pool_p): die("frozen.pool_sha256 != sha256(pool)")
if scores_p and f.get("scores_sha256") != sha(scores_p): die("frozen.scores_sha256 != sha256(scores)")
if f["counts"] != {"pilot": 8, "confirmation": 32} or len(f["pilot"]) != 8 or len(f["confirmation"]) != 32: die(f"counts {f['counts']}")
by_id = {p["id"]: p for p in pool["prefixes"]}
ids = [r["id"] for r in f["pilot"]] + [r["id"] for r in f["confirmation"]]
if len(set(ids)) != len(ids): die("duplicate ids across pilot/confirmation")
for r in f["pilot"] + f["confirmation"]:
    e = by_id.get(r["id"])
    if e is None: die(f"{r['id']} not in pool")
    if e["prefix_sha256"] != r["prefix_sha256"] or hashlib.sha256(e["text"].encode()).hexdigest() != r["prefix_sha256"]: die(f"{r['id']} hash mismatch")
    if e["tokens"] != r["tokens"] or e["stratum"] != r["context_stratum_actual"]: die(f"{r['id']} tokens/stratum mismatch")
for r in f["pilot"]:
    print(r["id"])
print("FROZEN VALIDATION OK", file=sys.stderr)
PY
cp "$FROZEN" "$ROOT/frozen_prefixes.json"; sha256sum "$FROZEN" "$POOL" > "$ROOT/inputs.sha256"
n=0
START_INDEX=${START_INDEX:-0}
echo "start_index=$START_INDEX" > "$ROOT/loop_meta.txt"
while read -r pid; do
  [[ -z "$pid" ]] && continue
  n=$((n+1))
  if (( n <= START_INDEX )); then echo "skip [$n] $pid (proven in an earlier root)"; continue; fi
  RUN="$ROOT/capture_$(printf '%02d' $n)_${pid}"
  echo "=== [$n] capture prefix $pid layer $LAYER -> $RUN  ($(date -u +%H:%M:%SZ)) ==="
  if ! bash "$E/serve_drivers.v2.sh" capture "$RUN" "$pid" "$POOL" "$LAYER" > "$RUN.driver.log" 2>&1; then
    echo "CAPTURE FAILED for $pid (see $RUN.driver.log)"; tail -20 "$RUN.driver.log"; exit 10
  fi
  if ! python3 "$E/inspect_capture.py" "$RUN" > "$RUN.provenance.log" 2>&1; then
    echo "PROVENANCE FAILED for $pid (see $RUN.provenance.log)"; tail -30 "$RUN.provenance.log"; exit 11
  fi
  tail -1 "$RUN.provenance.log"
  [[ "$MODE" == "--first-only" ]] && { echo "first-only mode: stopping after one proven capture"; exit 0; }
done < "$ROOT/pilot_ids.txt"
echo "CAPTURE LOOP DONE: $n prefixes"

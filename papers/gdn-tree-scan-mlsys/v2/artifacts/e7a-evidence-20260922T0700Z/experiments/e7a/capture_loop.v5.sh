#!/usr/bin/env bash
# E7a step 2: one capture boot per FROZEN pilot prefix, serially, each followed by the provenance inspector.
# Stops at the first failed capture or failed provenance. `--first-only` proves ONE complete real-request payload
# (review 05 item 2) and exits; the multi-prefix loop is only run after that proof.
# v5: SET=confirmation runs the 32 frozen confirmation prefixes ONLY if the structured e7a/PILOT_FREEZE.json validates via
#     check_pilot_freeze.py (hash-bound pilot evidence + filled criteria; review 10) AND PILOT_FREEZE.md says FROZEN; both
#     are copied into the run root. SET=pilot (default) unchanged. FREEZE_JSON/FREEZE_MD env overrides exist for gate tests.
# v4 (review 08): driver v4 (records finish_reason/logprobs tokens); hardened inspector (mandatory exact token binding).
# v3 (review 07): driver v3 + the inspector is executed from the run snapshot ($RUN/script_snapshot/inspect_capture.py).
# v2: START_INDEX=<k> skips the first k frozen pilot prefixes (already proven in an earlier root). Staged as a new file
# because the v1 loop was executing when this was written (bash reads by byte offset).
# Usage: SCORES=<prefix_scores.json> [START_INDEX=k] capture_loop.v5.sh <frozen_prefixes.json> <pool.json> <layer_prefix> <out_root> [--first-only]
# Validates the frozen set (schema, pool/scores hash chain, exact 8+32 counts, per-row membership/hash/tokens/stratum) BEFORE any boot.
set -euo pipefail
WT=/home/mark/lumo-paper-v2-20260921
E=$WT/papers/gdn-tree-scan-mlsys/v2/experiments/e7a
FROZEN=$1; POOL=$2; LAYER=$3; ROOT=$4; MODE=${5:-}
mkdir -p "$ROOT"
# confirmation gate FIRST (before any pool/score validation or boot)
SET=${SET:-pilot}
if [[ "$SET" == "confirmation" ]]; then
  # review 10: a status line is NOT evidence. The structured record must validate (eight hash-bound passing pilot
  # captures, hash-bound harness summary covering them, filled criteria); the .md must ALSO say FROZEN.
  FREEZE_JSON=${FREEZE_JSON:-$E/PILOT_FREEZE.json}; FREEZE_MD=${FREEZE_MD:-$E/PILOT_FREEZE.md}
  if ! python3 "$E/check_pilot_freeze.py" "$FREEZE_JSON" "$FROZEN" --json "$ROOT/PILOT_FREEZE.check.json"; then
    echo "REFUSING: SET=confirmation but $FREEZE_JSON does not validate (see $ROOT/PILOT_FREEZE.check.json)" >&2; exit 30
  fi
  if ! grep -qE '^Status: FROZEN [0-9]{4}-[0-9]{2}-[0-9]{2}T' "$FREEZE_MD"; then
    echo "REFUSING: $FREEZE_MD status line is not 'Status: FROZEN <utc>'" >&2; exit 32
  fi
  cp "$FREEZE_MD" "$ROOT/PILOT_FREEZE.frozen.md"; cp "$FREEZE_JSON" "$ROOT/PILOT_FREEZE.frozen.json"; sha256sum "$FREEZE_MD" "$FREEZE_JSON" > "$ROOT/PILOT_FREEZE.sha256"
elif [[ "$SET" != "pilot" ]]; then echo "REFUSING: unknown SET=$SET" >&2; exit 31; fi
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
import os
for r in f["confirmation" if os.environ.get("SET") == "confirmation" else "pilot"]:
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
  if ! bash "$E/serve_drivers.v4.sh" capture "$RUN" "$pid" "$POOL" "$LAYER" > "$RUN.driver.log" 2>&1; then
    echo "CAPTURE FAILED for $pid (see $RUN.driver.log)"; tail -20 "$RUN.driver.log"; exit 10
  fi
  if ! python3 "$RUN/script_snapshot/inspect_capture.py" "$RUN" > "$RUN.provenance.log" 2>&1; then
    echo "PROVENANCE FAILED for $pid (see $RUN.provenance.log)"; tail -30 "$RUN.provenance.log"; exit 11
  fi
  tail -1 "$RUN.provenance.log"
  [[ "$MODE" == "--first-only" ]] && { echo "first-only mode: stopping after one proven capture"; exit 0; }
done < "$ROOT/pilot_ids.txt"
echo "CAPTURE LOOP DONE: $n prefixes"

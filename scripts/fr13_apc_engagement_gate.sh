#!/usr/bin/env bash
# FR13 APC — UNIFIED ENGAGEMENT GATE (draft; design-only artifact, not yet wired).
#
# For EVERY active APC fix flag, assert BEFORE any SWE/lossless verdict is trusted:
#   (a) the flag is 1 in the WORKER (bridge marker / sidecar), AND
#   (b) the fix's ENGAGEMENT COUNTER incremented >0 (the code body actually executed
#       on a cache-hit / num_accepted>1 event).
# If either is 0/absent -> FAIL LOUD, verdict UNTRUSTED (exit non-zero).
#
# This generalizes the existing per-fix checks (FR13_CONV_REDIRECT_FIRED, ES_REDIRECT_USED,
# the FR13_APC_ENV_BRIDGE_LOADED marker) into one manifest-driven gate.
#
# Catches all THREE vacuity modes:
#   (1) flag not passed via docker -e  -> bridge field ABSENT  (part a fails)
#   (2) flag dropped by worker env curation / sidecar -> bridge field ABSENT (part a fails)
#   (3) code-path predicate never matches -> counter stays 0   (part b fails)
#
# USAGE (call AFTER boot + the first cache-hit turn, BEFORE recording any verdict):
#   CONTAINER=<name> FR13_APC_REQUIRE="FR13_APC_SNAP_FIX FR13_APC_CONV_FIX FR13_APC_SHADOW" \
#     bash scripts/fr13_apc_engagement_gate.sh
# FR13_APC_REQUIRE = space-separated list of flags this arm REQUIRES active. Empty -> the
# whole gate is a no-op (locked non-APC path), byte-identical behaviour.
set -u

MANIFEST="${FR13_APC_MANIFEST:-scripts/fr13_apc_engagement_manifest.json}"
CONTAINER="${CONTAINER:?CONTAINER required}"
BRIDGE_MARKER="${FR13_APC_BRIDGE_MARKER_FILE:-/logs/fr13_apc_bridge_loaded.flag}"
ENGAGE_FILE="${FR13_APC_ENGAGEMENT_FILE:-/logs/fr13_apc_engagement.flag}"
BRIDGE_ERR="${FR13_APC_BRIDGE_ERR_FILE:-/logs/fr13_apc_bridge_error.flag}"
OUTDIR="${OUTDIR:-.}"
REQUIRE="${FR13_APC_REQUIRE:-}"

[[ -z "$REQUIRE" ]] && { echo "[apc-gate] no FR13_APC_REQUIRE -> non-APC path, gate skipped"; exit 0; }

command -v jq >/dev/null 2>&1 || { echo "[apc-gate] FAIL: jq required to read manifest"; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "[apc-gate] FAIL: manifest $MANIFEST not found"; exit 2; }

# Pull the worker-written marker + engagement counters out of the container's /logs.
docker exec "$CONTAINER" sh -c "cat '$BRIDGE_MARKER' 2>/dev/null" > "$OUTDIR/apc_bridge_marker.txt" 2>/dev/null
docker exec "$CONTAINER" sh -c "cat '$ENGAGE_FILE'   2>/dev/null" > "$OUTDIR/apc_engagement.txt"    2>/dev/null
MARKER_TXT="$(cat "$OUTDIR/apc_bridge_marker.txt" 2>/dev/null)"
ENGAGE_TXT="$(cat "$OUTDIR/apc_engagement.txt"    2>/dev/null)"

# A swallowed bridge read must never look like engagement.
if docker exec "$CONTAINER" sh -c "test -f '$BRIDGE_ERR'" 2>/dev/null; then
  echo "[apc-gate] FAIL: bridge error flag present in worker:"; docker exec "$CONTAINER" cat "$BRIDGE_ERR" 2>/dev/null
  exit 4
fi
[[ -n "$MARKER_TXT" ]] || { echo "[apc-gate] FAIL: bridge marker $BRIDGE_MARKER absent/empty (gdn bridge did not run)"; exit 4; }

FAIL=0
for FLAG in $REQUIRE; do
  ENTRY="$(jq -c --arg f "$FLAG" '.fixes[] | select(.flag==$f)' "$MANIFEST")"
  [[ -n "$ENTRY" ]] || { echo "[apc-gate] FAIL: flag $FLAG not in manifest $MANIFEST"; FAIL=1; continue; }
  REQVAL="$(jq -r '.required_value' <<<"$ENTRY")"
  FIELD="$(jq -r '.bridge_field' <<<"$ENTRY")"
  CKEY="$(jq -r '.counter_key' <<<"$ENTRY")"

  # part (a): flag value in the worker bridge marker
  if grep -qE "(^|[[:space:]])${FIELD}=${REQVAL}([[:space:]]|\$)" "$OUTDIR/apc_bridge_marker.txt"; then
    echo "[apc-gate] (a) OK   $FLAG: worker ${FIELD}=${REQVAL}"
  else
    echo "[apc-gate] (a) FAIL $FLAG: worker ${FIELD} != ${REQVAL} (vacuity mode 1/2: flag absent in worker). marker: $MARKER_TXT"
    FAIL=1; continue
  fi

  # part (b): engagement counter > 0
  if [[ "$CKEY" == "null" || -z "$CKEY" ]]; then
    echo "[apc-gate] (b) FAIL $FLAG: NO ENGAGEMENT COUNTER in manifest (counter_key=null). Add one before trusting any verdict for this fix."
    FAIL=1; continue
  fi
  CVAL="$(grep -oE "(^|[[:space:]])${CKEY}=[0-9]+" "$OUTDIR/apc_engagement.txt" 2>/dev/null | grep -oE '[0-9]+$' | tail -1)"
  if [[ -n "$CVAL" && "$CVAL" -gt 0 ]]; then
    echo "[apc-gate] (b) OK   $FLAG: ${CKEY}=${CVAL} (>0, fix body executed)"
  else
    echo "[apc-gate] (b) FAIL $FLAG: ${CKEY}=${CVAL:-0} (vacuity mode 3: flag set but code body never ran). engagement: $ENGAGE_TXT"
    FAIL=1
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "[apc-gate] VERDICT: UNTRUSTED — at least one required APC fix did not fully engage. DO NOT record the SWE/lossless number."
  exit 4
fi
echo "[apc-gate] VERDICT: all required APC fixes ENGAGED (flag=1 in worker AND counter>0). Verdict trustworthy."
exit 0

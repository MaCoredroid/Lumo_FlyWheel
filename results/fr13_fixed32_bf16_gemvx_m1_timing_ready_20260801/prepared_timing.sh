#!/usr/bin/env bash
set -euo pipefail

: "${LIVE_RUNROOT:?set the completed M1 live-gate runroot}"
: "${LIVE_ARM:?set the live-gate arm directory name}"
: "${STOCK_FA2_SO:?set the absolute canonical FA2 binary path}"

REPO=$(git rev-parse --show-toplevel)
cd "$REPO"
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
ARTIFACT=results/fr13_fixed32_bf16_gemvx_m1_b1_ready_20260801
LIVE_ARM_DIR=$(realpath "$LIVE_RUNROOT/$LIVE_ARM")
LIVE_PASS_JSON="$LIVE_ARM_DIR/logs/fr13_draft_head_m1.live.json"
LIVE_FINAL_FLUSH_JSON="$LIVE_ARM_DIR/fixed32_final_flush.json"
LIVE_CHAT_TRAFFIC_AUDIT_JSON="$LIVE_ARM_DIR/fixed32_chat_traffic_audit.json"

for path in \
  "$LIVE_PASS_JSON" \
  "$LIVE_FINAL_FLUSH_JSON" \
  "$LIVE_CHAT_TRAFFIC_AUDIT_JSON"; do
  [[ -f "$path" && ! -L "$path" ]] \
    || { echo "missing regular live-gate evidence: $path" >&2; exit 2; }
done
FLUSH_GENERATION=$(
  "$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["ack"]["generation"])' \
    "$LIVE_FINAL_FLUSH_JSON"
)
LIVE_BOUNDARY_SNAPSHOT_JSON="$LIVE_ARM_DIR/logs/fr13_fixed32_boundary_snapshot.${FLUSH_GENERATION}.json"
[[ -f "$LIVE_BOUNDARY_SNAPSHOT_JSON" \
   && ! -L "$LIVE_BOUNDARY_SNAPSHOT_JSON" ]] \
  || { echo "missing regular live-gate boundary snapshot" >&2; exit 2; }

UTC=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT="output/fr13_bf16_gemvx_m1_exact4_timing_${UTC}" \
TAG="bf16_gemvx_m1_exact4_${UTC}" \
PYTHON_BIN="$PYTHON_BIN" \
STOCK_FA2_SO="$STOCK_FA2_SO" \
FR13_DRAFT_HEAD_M1_SO="$REPO/$ARTIFACT/fr13_bf16_gemvx_m1.abi3.so" \
FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION="$REPO/$ARTIFACT/build_attestation.json" \
LIVE_PASS_JSON="$LIVE_PASS_JSON" \
LIVE_PASS_SHA256="$(sha256sum "$LIVE_PASS_JSON" | cut -d' ' -f1)" \
LIVE_FINAL_FLUSH_JSON="$LIVE_FINAL_FLUSH_JSON" \
LIVE_BOUNDARY_SNAPSHOT_JSON="$LIVE_BOUNDARY_SNAPSHOT_JSON" \
LIVE_CHAT_TRAFFIC_AUDIT_JSON="$LIVE_CHAT_TRAFFIC_AUDIT_JSON" \
bash scripts/fr13_run_b1_draft_head_m1_timing.sh

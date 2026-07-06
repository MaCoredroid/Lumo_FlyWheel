#!/usr/bin/env bash
# FR13 APC DRIFT FIX-TEST: re-measure the block=1024 align drift WITH the preprocess
# SSM redirect (FR13_APC_PRE_SNAP_FIX=1) vs the SAME reused cfg@8192 continuous reference.
# Baseline (no fix) = 77.96 state / 75% argmax flips. If the fix works: drift -> ~fp (0.0078)
# + flips -> 0. NON-VACUITY canary: a background grep captures the SSM redirect FIRE counter
# while the container is alive (the conv-redirect-fired=0 trap detector).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
SRC=output/fr13_apc_rategate/run_20260625T084654Z/rg_OFF_r1/proxy_pair_dumps
OLD_REF=output/fr13_apc_drift_curve/run_20260629T000358Z/logs   # reuse the 193-capture cfg@8192 REF
SEQ=49; PORT=9953; GPU_UTIL=0.82; REPLAY_TEMP=0.6; K_DECODE=8; GDN_LIMIT=80; B=1024
TS=$(date -u +%Y%m%dT%H%M%SZ)
RD=output/fr13_apc_drift_fix/run_${TS}; mkdir -p "$RD/logs"
echo "$RD" > /home/mark/.claude/jobs/22c39bb9/tmp/drift_fix_root.txt
SEQPAD=$(printf '%06d' "$SEQ"); SEQDIR="$RD/seq${SEQ}_dumps"; mkdir -p "$SEQDIR"
cp "$SRC"/pair_*_${SEQPAD}_*.json "$SEQDIR"/ 2>/dev/null
cp "$OLD_REF"/ref_gdn*.pt "$RD/logs/" 2>/dev/null
cp "$OLD_REF"/ref_logit*.pt "$RD/logs/" 2>/dev/null
echo "=== FIX TEST  rundir=$RD  REF reused=$(ls "$RD/logs"/ref_gdn*.pt 2>/dev/null|wc -l) gdn  block=$B + PRE_SNAP_FIX=1 ==="
PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
for p in fr13 swe-agent swe-codex codex; do docker ps -aq --filter "name=$p" | xargs -r docker rm -f >/dev/null 2>&1; done

# background non-vacuity canary: grep the worker fire counters while the container is alive
( CF="$RD/logs/fire_canary.txt"; for i in $(seq 1 240); do
    docker logs fr13-apc-multiturn 2>&1 | grep -aoE "FR13_SNAP_FIX_FIRED=[0-9]+|_FR13_SNAP_FIX_FIRED=[0-9]+|SNAP_FIX_FIRED=[0-9]+|FR13_CONV_REDIRECT_FIRED=[0-9]+|PRE_SNAP_FIX=[A-Za-z0-9]+" 2>/dev/null | sort -u > "$CF" 2>/dev/null
    sleep 20
  done ) >/dev/null 2>&1 &
CANARY=$!

echo "[FIX] booting b=$B with FR13_APC_PRE_SNAP_FIX=1 (preprocess SSM redirect)"
ARM=on RUNDIR="$RD" DUMPS="$SRC" LIMIT_TURNS=$((SEQ+1)) PORT="$PORT" GPU_UTIL="$GPU_UTIL" \
  ENFORCE_EAGER=1 MAMBA_BLOCK_SIZE="$B" FR13_APC_PRE_SNAP_FIX=1 \
  FR13_REPLAY_TEMP="$REPLAY_TEMP" \
  FR13_PREFILL_GDN_CAPTURE="/logs/on_b${B}_gdn.pt" \
  FR13_PREFILL_GDN_CAPTURE_LAYER_PREFIX='*' \
  FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX="$GDN_LIMIT" \
  FR13_FINAL_LOGIT_CAPTURE="/logs/on_b${B}_logit.pt" \
  FR13_FINAL_LOGIT_CAPTURE_NUM_TOKENS= \
  FR13_FINAL_LOGIT_CAPTURE_SKIP=0 \
  FR13_FINAL_LOGIT_CAPTURE_LIMIT="$K_DECODE" \
  bash scripts/fr13_apc_multiturn_one_arm.sh >> "$RD/logs/boot_fix.log" 2>&1
echo "boot rc=$?"
kill "$CANARY" 2>/dev/null
echo "=== NON-VACUITY canary (did the SSM redirect FIRE? the conv-redirect=0 trap detector) ==="
cat "$RD/logs/fire_canary.txt" 2>/dev/null | sed 's/^/  /' || echo "  (canary empty)"
echo "=== bridge marker (PRE_SNAP_FIX reached worker?) ==="
grep -aoE "PRE_SNAP_FIX=[A-Za-z0-9]+|FR13_APC_ENV_BRIDGE_LOADED.*" "$RD/logs/launch_on.log" 2>/dev/null | tail -1 || true
cat "$RD/logs/fr13_apc_env.flag" 2>/dev/null | tr '\n' ' ' | grep -oE "PRE_SNAP_FIX=[01]" || echo "  (check pid-1 env)"
echo
echo "=== DRIFT vs reused REF (compare to baseline 77.96 / 75% flips) ==="
.venv/bin/python scripts/fr13_apc_drift_curve_reduce.py \
  --rundir "$RD" --ref-gdn "$RD/logs/ref_gdn" --ref-logit "$RD/logs/ref_logit" \
  --blocks "$B" --k-decode "$K_DECODE" --out "$RD/drift_fix.jsonl" 2>&1 | tail -22
echo "=== FIX TEST DONE -> $RD/drift_fix.jsonl ==="

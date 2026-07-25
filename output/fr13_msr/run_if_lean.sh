#!/usr/bin/env bash
# ATTRIBUTION arm (two-kinds: instrument-full, DIAGNOSTIC ONLY): LEAN stack
# at B=4 with span timers + torch-profiler window — names the verify-forward
# soup/norms ops POST-recomposition/deletion for the trio build (#62/R3).
# Window numbers carry profiler overhead: site NAMING + relative ranking
# only; never mixed into the clean-speed ledger.
# Waits for the GPU (R4 local gate runs first), then boots.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
while docker ps --format '{{.Names}}' | grep -q fr13; do sleep 60; done
sleep 30

ARM=if_lean
RUNROOT=output/fr13_msr
mkdir -p "$RUNROOT" output/fr13_sfwd_sidecar
SEQF="$RUNROOT/seq_${ARM}.sh"
cat > "$SEQF" <<'EOF'
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
export FR13_PARENT_GATHER=1
export FR13_CONV_PREGATHER=1
export FR13_FLAGS_INKERNEL=1
export FR13_HC_INTERNAL=0
export FR13_CONV_WB_BATCHED=0
export FR13_CONV_NODEBANK=0
export FR13_SPEC_BLOCKS_CAP=0
export FR13_SUBTREE_PARALLEL=1
export FR13_SUBTREE_PARALLEL_SELFCHECK=0
run_variant if_lean tail6 21 1
EOF

export FR13_TORCHPROF=1
RUNROOT="$RUNROOT" TAG=msr WALL=0 HEALTH_TIMEOUT_S=3600 \
  SUBSET=output/fr13_b1_gold_swe/subset_b4_four.json \
  BSIZE=1 CONC=1 GPU_UTIL=0.70 DEPLOY_FORCE_TEMP=0.6 \
  SEQUENCE_FILE="$PWD/$SEQF" \
  bash scripts/fr13_b4_campaign_driver.sh &
DRIVER_PID=$!
echo "$DRIVER_PID" > "$RUNROOT/${ARM}_driver.pid"

PORT="${PORT:-9950}"
for _i in $(seq 1 360); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  sleep 10
done
echo "[torchprof] health OK; warmup margin 600s"
sleep 600
date -Is > "$RUNROOT/${ARM}_window_open.ts"
curl -s "http://127.0.0.1:${PORT}/metrics" > "$RUNROOT/${ARM}_metrics_open.txt"
curl -sf -X POST "http://127.0.0.1:${PORT}/start_profile" && echo "[torchprof] window OPEN (B=1, 120s — proven 2g recipe; B=4 stop-crash structural)"
sleep 120
curl -sf -X POST "http://127.0.0.1:${PORT}/stop_profile" && echo "[torchprof] window CLOSE"
curl -s "http://127.0.0.1:${PORT}/metrics" > "$RUNROOT/${ARM}_metrics_close.txt"
date -Is > "$RUNROOT/${ARM}_window_close.ts"

TRACE_DIR="$(ls -dt output/fr13_bigdenom_swe/*/logs/torchprof 2>/dev/null | head -1)"
if [[ -z "$TRACE_DIR" ]]; then TRACE_DIR="$(find output -type d -name torchprof -mmin -180 2>/dev/null | head -1)"; fi
if [[ -n "$TRACE_DIR" ]]; then
  prev=-1
  for _i in $(seq 1 60); do
    cur=$(du -sb "$TRACE_DIR" 2>/dev/null | cut -f1)
    [[ "$cur" == "$prev" && "$cur" != "0" ]] && break
    prev="$cur"; sleep 5
  done
  echo "[torchprof] trace stable at $TRACE_DIR ($cur bytes)"
  echo "$TRACE_DIR" > "$RUNROOT/${ARM}_trace_dir.txt"
fi
echo "[torchprof] leaving driver to finish; reduce with reduce_torchprof_stacks.py"
wait "$DRIVER_PID"
echo "IF_LEAN_DONE"

#!/usr/bin/env bash
# FR13 TORCH-PROFILER-WITH-STACKS residual-naming arm (DIAGNOSTIC ONLY).
#
# Purpose: NAME the python/host sites behind the measured verifier residuals
# (host gaps +13.1/step, index-soup +9.3, norms +8.5, sampler +4.2) so the
# next build round attacks named sites, not guesses (user: measurement-first).
#
# Mechanism: vLLM's BUILT-IN torch profiler (with_stack=True by default),
# armed via serve CLI config (FR13_TORCHPROF=1 -> launcher appends
# --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=
# /logs/torchprof). Window control = POST /start_profile + /stop_profile.
# No patcher changes; engine config sidesteps the worker-env-drop class.
#
# OBSERVER-EFFECT LABELING (standing rule): every number captured inside the
# profile window carries profiler overhead. Outputs are for SITE NAMING and
# relative ranking only — never compare them as clean speed, never mix them
# into the fr13_measure ledger.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
if docker ps --format '{{.Names}}' | grep -q fr13; then echo "REFUSING: fr13 container running"; exit 2; fi

ARM=torchprof_070
RUNROOT=output/fr13_msr
mkdir -p "$RUNROOT" output/fr13_sfwd_sidecar
SEQF="$RUNROOT/seq_${ARM}.sh"
cat > "$SEQF" <<'EOF'
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
run_variant torchprof_070 tail6 21 1
EOF

# Solo task (deepest-running of the four) so the window is single-request
# decode — B~1-labeled like the nsys captures it complements.
export FR13_TORCHPROF=1
RUNROOT="$RUNROOT" TAG=msr WALL=0 HEALTH_TIMEOUT_S=3600 \
  SUBSET=output/fr13_b1_gold_swe/subset_b4_four.json \
  BSIZE=1 CONC=1 GPU_UTIL=0.70 DEPLOY_FORCE_TEMP=0.6 \
  SEQUENCE_FILE="$PWD/$SEQF" \
  bash scripts/fr13_b4_campaign_driver.sh &
DRIVER_PID=$!
echo "$DRIVER_PID" > "$RUNROOT/${ARM}_driver.pid"

PORT="${PORT:-9950}"
# Wait for health, then warmup margin, then a 120s profile window mid-decode.
for _i in $(seq 1 360); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  sleep 10
done
echo "[torchprof] health OK; warmup margin 600s (past compile/autotune inflation)"
sleep 600
date -Is > "$RUNROOT/${ARM}_window_open.ts"
curl -s "http://127.0.0.1:${PORT}/metrics" > "$RUNROOT/${ARM}_metrics_open.txt"
curl -sf -X POST "http://127.0.0.1:${PORT}/start_profile" && echo "[torchprof] window OPEN"
sleep 120
curl -sf -X POST "http://127.0.0.1:${PORT}/stop_profile" && echo "[torchprof] window CLOSE"
curl -s "http://127.0.0.1:${PORT}/metrics" > "$RUNROOT/${ARM}_metrics_close.txt"
date -Is > "$RUNROOT/${ARM}_window_close.ts"

# Teardown-race freeze: hold the variant container until the trace file under
# LOG_DIR/torchprof is size-stable (tensorboard_trace_handler writes on stop).
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
else
  echo "[torchprof] WARNING: no torchprof trace dir found yet (check /logs mount)"
fi
echo "[torchprof] leaving driver to finish the task; reduce with:"
echo "  python3 output/fr13_msr/reduce_torchprof_stacks.py <trace.json[.gz]>"
wait "$DRIVER_PID"

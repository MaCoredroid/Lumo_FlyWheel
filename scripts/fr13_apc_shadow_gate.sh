#!/usr/bin/env bash
# FR13 APC SHADOW GATE driver — boot ONE clean-flags cat6root server (FR13_ENABLE_APC=1
# => only SNAP_FIX + conv + block-align; NO experimental flags after the ba6b1496 cleanup),
# cuda-graph (deploy, to reproduce the deploy cache-ON failure), then replay one real
# captured /v1/responses turn cache-OFF (reset+send) vs cache-ON (send-again-hit) and
# localize the first divergence. The honest "cache off == cache on?" instrument on the
# REAL failing path with the clean deployment config.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel

REQ=${REQ:?path to a captured /v1/responses request json}
CONTAINER=${CONTAINER:-fr13-apc-shadow}
PORT=${PORT:-9953}
GPU_UTIL=${GPU_UTIL:-0.82}
MAX_OUT=${MAX_OUT:-2048}
REPS=${REPS:-2}
MODE=${MODE:-shadow}
ENABLE_APC=${ENABLE_APC:-1}   # set 0 to boot cache fully OFF (determinism control)
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNDIR=output/fr13_apc_shadow/run_${TS}
mkdir -p "$RUNDIR/logs"
CAT6ROOT_TREE="[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]"

echo "=== FR13 APC SHADOW GATE  container=$CONTAINER port=$PORT rundir=$RUNDIR ==="
echo "    request=$REQ  max_out=$MAX_OUT reps=$REPS  (clean flags: SNAP_FIX+conv only)"

_hygiene() {
  pgrep -af 'oom_protect_watchdog\.sh' | grep -qv pgrep || { setsid bash scripts/oom_protect_watchdog.sh >/dev/null 2>&1 </dev/null & disown; }
  bash scripts/oom_protect_session.sh >/dev/null 2>&1 || true
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  echo "  [hygiene] avail=$(free -g|awk '/Mem/{print $7}')GiB watchdog=$(pgrep -f oom_protect_watchdog|head -1||echo DEAD) claude_oom=$(cat /proc/$(pgrep -x claude|head -1)/oom_score_adj 2>/dev/null)"
}

teardown() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap teardown EXIT

_hygiene
docker ps -a --format '{{.Names}}'|grep -i fr13|xargs -r docker rm -f >/dev/null 2>&1 || true

echo "[boot] forked cat6root, FR13_ENABLE_APC=1 (clean), cuda-graph, GPU_UTIL=$GPU_UTIL"
CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=$GPU_UTIL MAX_NUM_SEQS=1 \
  TREE="$CAT6ROOT_TREE" FR10_METRICS=0 BATCH_INVARIANT=${BATCH_INVARIANT:-0} \
  FR13_BI_TREE_ATTN=${FR13_BI_TREE_ATTN:-0} \
  LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
  FR13_ENABLE_APC=$ENABLE_APC FR13_APC_CONFIG_ONLY=${FR13_APC_CONFIG_ONLY:-0} \
  ENFORCE_EAGER=${ENFORCE_EAGER:-0} \
  MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 \
  FR13_RUN_DIR="$PWD/$RUNDIR" LOG_DIR="$PWD/$RUNDIR/logs" \
  setsid bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$RUNDIR/launch.log" 2>&1 &
LPID=$!

echo "[boot] waiting for /health (cuda-graph boot ~5min)..."
T0=$SECONDS; HEALTHY=0
while [ $((SECONDS-T0)) -lt 900 ]; do
  if curl -fsS -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
  if grep -qiE "CUDA out of memory|NVRM_NO_MEMORY" "$RUNDIR/launch.log" 2>/dev/null; then echo "[boot] OOM in launch log"; break; fi
  if grep -qi "docker kill" output/gpu_oom_guard.log 2>/dev/null; then echo "[boot] gpu guard killed container"; break; fi
  sleep 10
done
if [ "$HEALTHY" != "1" ]; then echo "FAIL: server not healthy"; tail -25 "$RUNDIR/launch.log"; exit 2; fi
echo "[boot] healthy at $((SECONDS-T0))s; cached_tokens sanity + shadow replay"

# sanity: confirm prefix caching is actually ON in this boot
curl -fsS -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && echo "[boot] models endpoint OK"

.venv/bin/python scripts/fr13_apc_shadow_gate.py \
  --port "$PORT" --request-json "$REQ" --reps "$REPS" --mode "$MODE" \
  --max-output-tokens "$MAX_OUT" --out "$RUNDIR/shadow_verdict.json" 2>&1 | tee "$RUNDIR/shadow_run.log"

echo "=== shadow gate done -> $RUNDIR/shadow_verdict.json ==="

#!/bin/bash
# FR13 dedicated cat8+cache server (NO agent loop) — for the geodetic reproducer /
# wa_capture / drift-lever probes. Mirrors EXACTLY the cat8 launch env that
# fr13_bigdenom_swe_serve_variant.sh (L245-249) + fr13_garble_fix_run.sh set, minus the
# SWE agent. Boots the forked FA2 tree server on 127.0.0.1:$PORT and waits for health.
#   env overrides: PORT (9950) CONTAINER (fr13-cat8-repro) EXTRA_FLAGS ("K=V K=V")
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
PORT=${PORT:-9950}
CONTAINER=${CONTAINER:-fr13-cat8-repro}
RUNROOT=${RUNROOT:-output/fr13_garble_fix_v1/cat8_dedicated}
mkdir -p "$RUNROOT/logs"

# cat8 tree shape (variant L60) — spine + rank-2 siblings at first 3 depths, 8 nodes.
CAT8_TREE="[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0)]"

# cat8+cache baked config (fr13_garble_fix_run.sh) — stateless-tree trio + APC.
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1
# caller-supplied drift-lever flags (e.g. a fix under test) appended verbatim
for kv in ${EXTRA_FLAGS:-}; do export "$kv"; done

echo "=== cat8 dedicated server PORT=$PORT CONTAINER=$CONTAINER EXTRA_FLAGS=[${EXTRA_FLAGS:-none}] ==="
echo "$(git rev-parse HEAD)"
# recover host memory first (assert free) — same hygiene as the variant.
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python -c \
  'from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()' || true

# launch env mirrors variant L245-249 (cat8/reshape branch): TREE + BATCH_INVARIANT + the
# BAKED in_proj_ba pad (LUMO_FB_KERNEL_ROWS=1, PROJ_PAD_ROWS=16). The forked launcher
# defaults FIX-1/2/3, REPLAY_ROUTE, FA2_TREE_BIAS, CONV_COMMITTED_PATH ON.
CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL="${GPU_UTIL:-0.78}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}" \
  TREE="$CAT8_TREE" FR10_METRICS=0 BATCH_INVARIANT="${BATCH_INVARIANT:-0}" \
  LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
  FR13_RUN_DIR="$PWD/$RUNROOT" LOG_DIR="$PWD/$RUNROOT/logs" \
  scripts/fr13_launch_forked_fa2_tree_server.sh > "$RUNROOT/launch.log" 2>&1
RC=$?
if (( RC != 0 )); then echo "FAIL launcher rc=$RC"; tail -30 "$RUNROOT/launch.log"; exit 2; fi

# wait for health
T0=$(date +%s); HEALTHY=0
while (( $(date +%s) < T0 + 1200 )); do
  if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
  if [[ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" != "running" ]]; then
    echo "FAIL container died before health"; docker logs "$CONTAINER" 2>&1 | tail -40; exit 2
  fi
  sleep 5
done
(( HEALTHY == 1 )) || { echo "FAIL health not up in 1200s"; docker logs "$CONTAINER" 2>&1 | tail -40; exit 2; }
echo "=== cat8 dedicated server HEALTHY on :$PORT after $(( $(date +%s) - T0 ))s ==="
docker exec "$CONTAINER" env 2>/dev/null | grep -E '^(SPEC_CONFIG|LUMO_FB_KERNEL_ROWS|FR13_ENABLE_APC|FR10_DECODE_MODE_DEFAULT)=' | sed 's/^/  /'
echo "  model(s):"; curl -s "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | python3 -c 'import sys,json;[print("   ",m["id"]) for m in json.load(sys.stdin).get("data",[])]' 2>/dev/null || true
echo "SERVER_READY"

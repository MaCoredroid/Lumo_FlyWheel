#!/usr/bin/env bash
# FR13 APC MULTI-TURN ORACLE GATE — single-arm runner.
# Boots ONE arm into an EXISTING run dir and replays the trajectory. Lets us add the
# config-matched CFG control (chunked+1024, NO cache) to a run whose oracle/off/on
# arms already completed, without disturbing the live 3-arm orchestrator.
#
# Usage: ARM=cfg RUNDIR=output/fr13_apc_multiturn/run_XXXX bash scripts/fr13_apc_multiturn_one_arm.sh
#   ARM in {oracle, off, on, cfg}
#     oracle = no-spec recurrent decode (ground truth)
#     off    = base tree, APC off (loose floor)
#     cfg    = tree, FR13_APC_CONFIG_ONLY=1 (chunked+1024, NO cache) = CONFIG-MATCHED control
#     on     = tree, full APC (chunked+1024 + fp32-SSM + prefix-restore)
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true

ARM=${ARM:?ARM required (oracle|off|on|cfg)}
RUNDIR=${RUNDIR:?RUNDIR required (existing run dir)}
DUMPS=${DUMPS:-output/fr13_apc_rategate/run_20260625T084654Z/rg_OFF_r1/proxy_pair_dumps}
PORT=${PORT:-9953}
GPU_UTIL=${GPU_UTIL:-0.82}
MAX_OUT=${MAX_OUT:-384}
LIMIT_TURNS=${LIMIT_TURNS:-0}
CAT6ROOT_TREE="[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]"
CONTAINER="fr13-apc-multiturn"
mkdir -p "$RUNDIR/logs"
echo "=== single-arm: $ARM into $RUNDIR (max_out=$MAX_OUT) ==="

_hygiene() {
  pgrep -af 'oom_protect_watchdog\.sh' | grep -qv pgrep || { setsid bash scripts/oom_protect_watchdog.sh >/dev/null 2>&1 </dev/null & disown; }
  bash scripts/oom_protect_session.sh >/dev/null 2>&1 || true
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  echo "  [hygiene] avail=$(free -g|awk '/Mem/{print $7}')GiB watchdog=$(pgrep -f oom_protect_watchdog|head -1||echo DEAD)"
}
teardown() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap teardown EXIT

_hygiene
docker ps -a --format '{{.Names}}'|grep -i fr13|xargs -r docker rm -f >/dev/null 2>&1 || true
sleep 2
LAUNCH="$RUNDIR/logs/launch_${ARM}.log"
echo "[$ARM] booting..."
case "$ARM" in
  oracle)
    CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=$GPU_UTIL MAX_NUM_SEQS=1 \
      FR12_NO_SPECULATIVE_CONFIG=1 ATTENTION_BACKEND=FLASH_ATTN \
      FR13_ENABLE_APC=0 ENFORCE_EAGER=1 BATCH_INVARIANT=1 FR10_METRICS=0 \
      LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
      FR13_RUN_DIR="$PWD/$RUNDIR" LOG_DIR="$PWD/$RUNDIR/logs" \
      setsid bash scripts/fr10_launch_speed_server.sh > "$LAUNCH" 2>&1 & ;;
  off|on|cfg)
    APC=0; CONFIG_ONLY=0
    [ "$ARM" = "on" ] && APC=1
    [ "$ARM" = "cfg" ] && { APC=1; CONFIG_ONLY=1; }
    CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=$GPU_UTIL MAX_NUM_SEQS=1 \
      TREE="$CAT6ROOT_TREE" FR10_METRICS=0 ENFORCE_EAGER=1 BATCH_INVARIANT=1 FR13_BI_TREE_ATTN=1 \
      LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
      FR13_ENABLE_APC=$APC FR13_APC_CONFIG_ONLY=$CONFIG_ONLY \
      MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 \
      FR13_RUN_DIR="$PWD/$RUNDIR" LOG_DIR="$PWD/$RUNDIR/logs" \
      setsid bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$LAUNCH" 2>&1 & ;;
  *) echo "unknown ARM $ARM"; exit 2;;
esac

T0=$SECONDS; HEALTHY=0
while [ $((SECONDS-T0)) -lt 1200 ]; do
  if curl -fsS -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
  if grep -qiE "CUDA out of memory|NVRM_NO_MEMORY" "$LAUNCH" 2>/dev/null; then echo "[$ARM] OOM"; tail -15 "$LAUNCH"; exit 3; fi
  sleep 10
done
[ "$HEALTHY" = 1 ] || { echo "[$ARM] FAIL not healthy"; tail -25 "$LAUNCH"; exit 4; }
echo "[$ARM] healthy at $((SECONDS-T0))s; replaying..."
.venv/bin/python scripts/fr13_apc_multiturn_replay.py \
  --port "$PORT" --dumps-dir "$DUMPS" --arm "$ARM" \
  --out "$RUNDIR/replay_${ARM}.json" --max-output-tokens "$MAX_OUT" \
  ${LIMIT_TURNS:+--limit-turns "$LIMIT_TURNS"} 2>&1 | tee "$RUNDIR/logs/replay_${ARM}.log"
teardown
echo "=== single-arm $ARM DONE -> $RUNDIR/replay_${ARM}.json ==="

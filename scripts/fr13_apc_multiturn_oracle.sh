#!/usr/bin/env bash
# FR13 APC MULTI-TURN ORACLE GATE driver.
#
# Boots THREE arms SERIALLY (<=1 fr13 container ever) and replays a FIXED reference
# trajectory through each at temp=0 (greedy==argmax), BI=1, eager:
#   oracle = no-spec recurrent decode (fr10_launch_speed_server.sh + FR12_NO_SPECULATIVE_CONFIG=1,
#            FLASH_ATTN) — the GROUND TRUTH reference (standing rule: US vs no-spec oracle).
#   off    = tree spec-decode, APC OFF (forked cat6root) — the calibrated FLOOR.
#   on     = tree spec-decode, APC ON  (forked cat6root) — replayed IN SEQUENCE, NO reset,
#            so the cache genuinely populates and each turn's prefix is a real APC restore.
# Then reduces: per turn, first-divergence vs oracle for off (floor) and on (test); does ON
# diverge EARLIER than the floor and does the excess GROW with turn index (compounding)?
#
# Diagnostics regime = eager + BI=1 + temp0 (clean argmax). If ON is lossless even multi-turn
# here, the temp-0.6 rate-gate failure is cuda-graph/temp-0.6 specific -> escalate to that.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true

DUMPS=${DUMPS:-output/fr13_apc_rategate/run_20260625T084654Z/rg_OFF_r1/proxy_pair_dumps}
PORT=${PORT:-9953}
GPU_UTIL=${GPU_UTIL:-0.82}
MAX_OUT=${MAX_OUT:-384}
LIMIT_TURNS=${LIMIT_TURNS:-0}
ARMS=${ARMS:-"oracle off on"}
CAT6ROOT_TREE="[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNDIR=${RUNDIR:-output/fr13_apc_multiturn/run_${TS}}
mkdir -p "$RUNDIR/logs"

echo "=== FR13 APC MULTI-TURN ORACLE GATE  rundir=$RUNDIR ==="
echo "    dumps=$DUMPS  arms=[$ARMS]  max_out=$MAX_OUT  limit_turns=${LIMIT_TURNS:-all}  (temp0/BI/eager)"

_hygiene() {
  pgrep -af 'oom_protect_watchdog\.sh' | grep -qv pgrep || { setsid bash scripts/oom_protect_watchdog.sh >/dev/null 2>&1 </dev/null & disown; }
  bash scripts/oom_protect_session.sh >/dev/null 2>&1 || true
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  echo "  [hygiene] avail=$(free -g|awk '/Mem/{print $7}')GiB watchdog=$(pgrep -f oom_protect_watchdog|head -1||echo DEAD)"
}

CONTAINER="fr13-apc-multiturn"
teardown() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap teardown EXIT

boot_arm() {
  local arm="$1" launch_log="$RUNDIR/logs/launch_${1}.log"
  _hygiene
  docker ps -a --format '{{.Names}}'|grep -i fr13|xargs -r docker rm -f >/dev/null 2>&1 || true
  sleep 2
  echo "[$arm] booting..."
  case "$arm" in
    oracle)
      CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=$GPU_UTIL MAX_NUM_SEQS=1 \
        FR12_NO_SPECULATIVE_CONFIG=1 ATTENTION_BACKEND=FLASH_ATTN \
        FR13_ENABLE_APC=0 ENFORCE_EAGER=1 BATCH_INVARIANT=1 FR10_METRICS=0 \
        LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
        FR13_RUN_DIR="$PWD/$RUNDIR" LOG_DIR="$PWD/$RUNDIR/logs" \
        setsid bash scripts/fr10_launch_speed_server.sh > "$launch_log" 2>&1 &
      ;;
    off|on)
      local apc=0; [ "$arm" = "on" ] && apc=1
      CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=$GPU_UTIL MAX_NUM_SEQS=1 \
        TREE="$CAT6ROOT_TREE" FR10_METRICS=0 ENFORCE_EAGER=1 BATCH_INVARIANT=1 FR13_BI_TREE_ATTN=1 \
        LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
        FR13_ENABLE_APC=$apc FR13_APC_CONFIG_ONLY=0 \
        MAMBA_BLOCK_SIZE=${MAMBA_BLOCK_SIZE:-1024} MAMBA_SSM_CACHE_DTYPE=${MAMBA_SSM_CACHE_DTYPE:-float32} \
        FR13_RUN_DIR="$PWD/$RUNDIR" LOG_DIR="$PWD/$RUNDIR/logs" \
        setsid bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$launch_log" 2>&1 &
      ;;
    *) echo "unknown arm $arm"; return 2;;
  esac
  local T0=$SECONDS
  while [ $((SECONDS-T0)) -lt 1200 ]; do
    if curl -fsS -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "[$arm] healthy at $((SECONDS-T0))s"; return 0; fi
    if grep -qiE "CUDA out of memory|NVRM_NO_MEMORY" "$launch_log" 2>/dev/null; then echo "[$arm] OOM"; tail -15 "$launch_log"; return 3; fi
    sleep 10
  done
  echo "[$arm] FAIL not healthy in 1200s"; tail -25 "$launch_log"; return 4
}

for arm in $ARMS; do
  if ! boot_arm "$arm"; then echo "FATAL: $arm boot failed"; exit 2; fi
  echo "[$arm] replaying trajectory (temp0, no reset-each)..."
  .venv/bin/python scripts/fr13_apc_multiturn_replay.py \
    --port "$PORT" --dumps-dir "$DUMPS" --arm "$arm" \
    --out "$RUNDIR/replay_${arm}.json" --max-output-tokens "$MAX_OUT" \
    ${LIMIT_TURNS:+--limit-turns "$LIMIT_TURNS"} 2>&1 | tee "$RUNDIR/logs/replay_${arm}.log"
  teardown; sleep 3
done

if echo "$ARMS" | grep -q oracle && echo "$ARMS" | grep -q off && echo "$ARMS" | grep -q on; then
  echo "[reduce] all 3 arms captured -> reducing"
  .venv/bin/python scripts/fr13_apc_multiturn_reduce.py \
    --oracle "$RUNDIR/replay_oracle.json" --off "$RUNDIR/replay_off.json" --on "$RUNDIR/replay_on.json" \
    --out "$RUNDIR/multiturn_verdict.json" 2>&1 | tee "$RUNDIR/logs/reduce.log"
fi
echo "=== MULTI-TURN ORACLE GATE DONE -> $RUNDIR ==="

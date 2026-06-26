#!/usr/bin/env bash
# FR13 APC MULTI-TURN FLOOR-BRACKETED gate driver (5 boots, SERIAL, <=1 container).
#
# Removes the cross-boot autotune-fork confound by booting each config TWICE:
#   oracle  = no-spec recurrent decode (ground truth)
#   on, on2 = full APC (incremental sequential replay), two boots => ON autotune floor
#   cfg,cfg2= config-only (chunked+1024, NO cache), two boots => CFG autotune floor
# Each arm = one boot+replay of the same rg_OFF_r1 55-turn trajectory at temp0/BI=1/eager,
# via fr13_apc_multiturn_one_arm.sh (which boots, replays, tears down). Then the
# floor-bracketed reducer scores ON-vs-CFG against the same-config cross-boot floors +
# the (confound-immune) cold-restart metric.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true

DUMPS=${DUMPS:-output/fr13_apc_rategate/run_20260625T084654Z/rg_OFF_r1/proxy_pair_dumps}
PORT=${PORT:-9953}
GPU_UTIL=${GPU_UTIL:-0.82}
MAX_OUT=${MAX_OUT:-384}
LIMIT_TURNS=${LIMIT_TURNS:-0}
ARMS=${ARMS:-"oracle on on2 cfg cfg2"}
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNDIR=${RUNDIR:-output/fr13_apc_multiturn_fb/run_${TS}}
mkdir -p "$RUNDIR/logs"
echo "=== FR13 APC MULTI-TURN FLOOR-BRACKETED  rundir=$RUNDIR  arms=[$ARMS]  max_out=$MAX_OUT ===" | tee "$RUNDIR/driver.log"

for arm in $ARMS; do
  echo "[driver] === arm $arm ===" | tee -a "$RUNDIR/driver.log"
  ARM="$arm" RUNDIR="$RUNDIR" DUMPS="$DUMPS" PORT="$PORT" GPU_UTIL="$GPU_UTIL" \
    MAX_OUT="$MAX_OUT" LIMIT_TURNS="$LIMIT_TURNS" \
    bash scripts/fr13_apc_multiturn_one_arm.sh >> "$RUNDIR/driver.log" 2>&1
  rc=$?
  if [ ! -f "$RUNDIR/replay_${arm}.json" ]; then
    echo "[driver] FATAL: arm $arm produced no replay (rc=$rc) — STOP" | tee -a "$RUNDIR/driver.log"
    exit 2
  fi
  echo "[driver] arm $arm done -> replay_${arm}.json" | tee -a "$RUNDIR/driver.log"
done

# reduce when all 5 present
ok=1
for arm in oracle on on2 cfg cfg2; do [ -f "$RUNDIR/replay_${arm}.json" ] || ok=0; done
if [ "$ok" = 1 ]; then
  echo "[driver] all 5 arms captured -> floor-bracketed reduce" | tee -a "$RUNDIR/driver.log"
  .venv/bin/python scripts/fr13_apc_multiturn_reduce_fb.py \
    --oracle "$RUNDIR/replay_oracle.json" --on "$RUNDIR/replay_on.json" --on2 "$RUNDIR/replay_on2.json" \
    --cfg "$RUNDIR/replay_cfg.json" --cfg2 "$RUNDIR/replay_cfg2.json" \
    --out "$RUNDIR/verdict_fb.json" 2>&1 | tee -a "$RUNDIR/driver.log"
fi
echo "=== FLOOR-BRACKETED GATE DONE -> $RUNDIR ===" | tee -a "$RUNDIR/driver.log"

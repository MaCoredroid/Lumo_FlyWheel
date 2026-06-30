#!/usr/bin/env bash
# FR13 APC CONV-redirect ENGAGEMENT + AXIS-A SELF-DIFF gate (single boot, replay harness).
#
# Answers two questions the confounded e2e SWE gate cannot:
#   (1) ENGAGEMENT (vacuity mode 1/2/3): does the conv-leaf redirect actually FIRE?
#       The deployed conv redirect was VACUOUS (CONV_REDIRECT_FIRED=0) because it was
#       gated to phase=="postprocess" while the num_accepted>1 conv copy lives in
#       PREPROCESS. FR13_APC_CONV_PRE_REDIRECT=1 lets the same committed-leaf redirect
#       fire at preprocess. Proof = bridge marker CONV_PRE_REDIRECT=1 (mode 1/2) AND
#       FR13_CONV_REDIRECT_FIRED>0 in the worker log (mode 3).
#   (2) AXIS-A SELF-DIFF (Axis-B-isolating, temp-independent state probe): for every
#       fired event, does leaf_row (redirect target) DIFFER from stock_row (what stock
#       get_conv_copy_spec would commit)? FR13_APC_CACHEROW_DUMP captures both rows.
#         - stock==leaf everywhere -> redirect INERT -> conv NOT the Axis-A carrier
#           (stock copy already correct; deviation lives elsewhere = SSM leaf/packed-decode).
#         - stock!=leaf -> redirect does real work -> follow up with a conv ground-truth
#           reference to judge WHICH is correct.
#
# This is a STATE-FAITHFULNESS probe (cache tensor rows), so the temp-0 replay harness is
# VALID here (the committed conv-window bytes are temp-independent) -- it is NOT an
# output/solve-rate comparison (those must run temp 0.6). cat6root SPEC + full APC, so
# num_accepted>1 cache-hit conv copies genuinely occur.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true

SRC=${SRC:-output/fr13_apc_rategate/run_20260625T084654Z/rg_OFF_r1/proxy_pair_dumps}
SEQ=${SEQ:-49}
LIMIT_TURNS=${LIMIT_TURNS:-50}
PORT=${PORT:-9953}
GPU_UTIL=${GPU_UTIL:-0.82}
TS=$(date -u +%Y%m%dT%H%M%SZ)
RD=${RD:-output/fr13_apc_convengage/run_${TS}}
mkdir -p "$RD/logs"
echo "$RD" > /tmp/claude-1000/-home-mark-shared/46f03809-5059-4e30-936d-1adda7f44337/scratchpad/convengage_root.txt
echo "=== FR13 APC conv-engage gate  rundir=$RD  src=$SRC  turns=$LIMIT_TURNS ==="

[ -d "$SRC" ] || { echo "FATAL: SRC dumps dir missing: $SRC"; exit 2; }

echo "[BOOT] ARM=on (cat6root spec + full APC) + CONV_SNAP_FIX=1 + CONV_PRE_REDIRECT=1 + CACHEROW_DUMP"
ARM=on RUNDIR="$RD" DUMPS="$SRC" LIMIT_TURNS="$LIMIT_TURNS" PORT="$PORT" GPU_UTIL="$GPU_UTIL" \
  FR13_APC_CONV_SNAP_FIX=1 FR13_APC_CONV_PRE_REDIRECT=1 \
  FR13_APC_LEAF_CROSSCHECK=1 \
  FR13_APC_CACHEROW_DUMP=/logs/cacherows \
  FR13_APC_CACHEROW_DUMP_LIMIT=2000 \
  bash scripts/fr13_apc_multiturn_one_arm.sh >> "$RD/logs/boot_on.log" 2>&1
RC=$?
echo "boot rc=$RC"

echo "=== (1) ENGAGEMENT ==="
echo "-- bridge marker (mode 1/2: did CONV_PRE_REDIRECT reach the worker env?) --"
grep -rhoE "FR13_APC_ENV_BRIDGE_LOADED[^\"]*" "$RD" 2>/dev/null | tail -1 | sed 's/^/  /' || echo "  (no bridge marker)"
F=$(grep -rhoE "FR13_CONV_REDIRECT_FIRED=[0-9]+" "$RD" 2>/dev/null | grep -oE "[0-9]+$" | sort -n | tail -1)
echo "-- mode 3: max FR13_CONV_REDIRECT_FIRED = ${F:-0} --"
if [ "${F:-0}" -gt 0 ] 2>/dev/null; then echo "  -> conv redirect FIRED (wiring fix works, NON-vacuous)"; else echo "  -> !! CONV_REDIRECT_FIRED=0 STILL VACUOUS"; fi

echo "=== (2) AXIS-A SELF-DIFF (stock_row vs leaf_row, no reference boot) ==="
DUMPDIR="$RD/logs/cacherows"
if [ -n "$(ls "$DUMPDIR"/cacherow_*.pt 2>/dev/null)" ]; then
  .venv/bin/python scripts/fr13_apc_cacherow_selfdiff.py --cacherow "$DUMPDIR" 2>&1 | tee "$RD/logs/selfdiff.log"
else
  echo "  !! no cacherow dumps in $DUMPDIR -> CACHEROW_DUMP vacuous (flag did not reach worker?)"
fi
echo "[conv-engage] DONE -> $RD"

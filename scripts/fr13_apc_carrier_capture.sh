#!/usr/bin/env bash
# FR13 APC carrier capture — tensor-level localization of the multi-turn cache-ON
# corruption on ONE high-occupancy cold-restart turn (seq49, cached ~28560).
#
# 2 SERIAL eager boots into one rundir, then an offline per-GDN-layer diff:
#   BOOT A (CFG, chunked+1024, NO cache): replay ONLY seq49 (1-pair dumps) =>
#       fresh full prefill => no-cache ground-truth final_state per GDN layer
#       (FR13_PREFILL_GDN_CAPTURE, latest-per-layer = high occupancy).
#   BOOT B (ON, full APC): full incremental replay to seq49 => the cache snapshots;
#       FR13_APC_CACHEROW_DUMP captures stock_row (stock get_temporal_copy_spec
#       source) + leaf_row (SNAP_FIX redirect target) + wraparound fields at the
#       postprocess block-boundary snapshots (latest snap_fired = high occupancy).
#   DIFF: stock_row vs gt-final (stale-write?), leaf_row vs gt-final (is SNAP_FIX's
#       committed-leaf itself unfaithful at high occupancy = the live residual?),
#       wraparound OOB (#27264-class). First divergent layer + axis = the carrier.
# restore_read axis omitted in pass 1 (needs a separate seq49-targeted ON capture).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true

SRC=${SRC:-output/fr13_apc_rategate/run_20260625T084654Z/rg_OFF_r1/proxy_pair_dumps}
SEQ=${SEQ:-49}
PORT=${PORT:-9953}
GPU_UTIL=${GPU_UTIL:-0.82}
TS=$(date -u +%Y%m%dT%H%M%SZ)
RD=${RD:-output/fr13_apc_carrier/run_${TS}_capture}
mkdir -p "$RD/logs"
SEQDIR="$RD/seq${SEQ}_dumps"; mkdir -p "$SEQDIR"
SEQPAD=$(printf '%06d' "$SEQ")
cp "$SRC"/pair_*_${SEQPAD}_*.json "$SEQDIR"/ 2>/dev/null
NP=$(ls "$SEQDIR"/*.json 2>/dev/null | wc -l)
echo "=== FR13 APC carrier capture  rundir=$RD  seq=$SEQ (one-pair dumps n=$NP) ==="
[ "$NP" -eq 1 ] || { echo "FATAL: expected exactly 1 seq$SEQ pair, got $NP"; exit 2; }

echo "[BOOT A] CFG (no cache) single-seq$SEQ -> no-cache ground-truth final_state"
ARM=cfg RUNDIR="$RD" DUMPS="$SEQDIR" PORT="$PORT" GPU_UTIL="$GPU_UTIL" \
  FR13_PREFILL_GDN_CAPTURE=/logs/nocache_gdn.pt \
  FR13_PREFILL_GDN_CAPTURE_LAYER_PREFIX='*' \
  FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX=80 \
  bash scripts/fr13_apc_multiturn_one_arm.sh >> "$RD/logs/bootA_cfg.log" 2>&1
[ -n "$(ls "$RD"/logs/nocache_gdn*.pt 2>/dev/null)" ] || { echo "FATAL: no nocache capture produced"; tail -20 "$RD/logs/bootA_cfg.log"; exit 3; }

echo "[BOOT B] ON (full APC) full replay to seq$SEQ -> cache snapshots (stock/leaf/wraparound)"
ARM=on RUNDIR="$RD" DUMPS="$SRC" LIMIT_TURNS=$((SEQ+1)) PORT="$PORT" GPU_UTIL="$GPU_UTIL" \
  FR13_APC_CACHEROW_DUMP=/logs/cacherows \
  FR13_APC_CACHEROW_DUMP_LIMIT=2000 \
  bash scripts/fr13_apc_multiturn_one_arm.sh >> "$RD/logs/bootB_on.log" 2>&1
[ -n "$(ls "$RD"/logs/cacherows/cacherow_*.pt 2>/dev/null)" ] || { echo "FATAL: no cacherow dumps produced"; tail -20 "$RD/logs/bootB_on.log"; exit 4; }

echo "[DIFF] per-GDN-layer carrier localization"
.venv/bin/python scripts/fr13_apc_cacherow_diff.py \
  --nocache  "$RD/logs" \
  --cacherow "$RD/logs/cacherows" \
  --out "$RD/cacherow_diff.jsonl" --thresh 1e-2 2>&1 | tee "$RD/logs/diff.log"
echo "=== carrier capture DONE -> $RD/cacherow_diff.jsonl ==="

#!/usr/bin/env bash
# FR13 SPEED GATE — decode-TPS on the DEPLOYED config (cache-ON: EXACT_SEED + block 1024 + spec +
# FULL GRAPH baked default), cat6root + cat10, on the 4-task live SWE set (subset_b4_four:
# astropy 12907/13033/13236/13398), B=1, temp 0.6. CACHE-ON ONLY (no OFF arm, per user).
# Comparable to the prior cat6root ship gate (fr13_apc_shipgate_cat6root.sh): same serve_variant +
# fr13_measure.py deploy-speed --basis decode_seconds. Prior cat6root: realized ~10-13.6 TPS.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_b4_four.json}
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT=output/fr13_apc_speedgate/run_$TS; mkdir -p "$RUNROOT"; export RUNROOT
echo "$RUNROOT" > /home/mark/.claude/jobs/22c39bb9/tmp/speedgate_root.txt
echo "=== FR13 SPEED GATE  cat6root+cat10  cache-ON only  4-task(b4_four)  B=1 temp0.6  FULL GRAPH+EXACT_SEED -> $RUNROOT ==="
# B=1 + temp 0.6 deployment regime; metrics-OFF clean speed read; full graph is the launcher default now
# GPU_UTIL 0.76 (down from serve_variant's 0.82) + guard floor 3000: the 4-task run died exit-137
# (unified-mem spike on GB10) at 0.82/floor-4000; lower util gives prefill-spike headroom.
export MAX_NUM_SEQS_OVR=1 OFFLOAD_CODEX=1 DEPLOY_FORCE_TEMP=0.6 DOCKER_MEM_CAP=105g \
  GPU_UTIL="${GPU_UTIL:-0.76}" GPU_GUARD_FLOOR_MIB="${GPU_GUARD_FLOOR_MIB:-3000}" FR10_METRICS=0

run_arm() {
  local KIND=$1 ETPD=$2
  local ARM="sg_${KIND}_ON"
  echo "--- [$ARM] boot @ $(date -u +%H:%M:%S) ---"
  docker ps -aq --filter "name=fr13-bigdenom-$ARM" | xargs -r docker rm -f >/dev/null 2>&1 || true
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  FR13_ENABLE_APC=1 FR13_APC_CONFIG_ONLY=0 FR13_APC_EXACT_SEED=1 \
    MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" "$KIND" "$SUBSET" > "$RUNROOT/${ARM}.log" 2>&1 </dev/null
  echo "  [$ARM] serve_variant rc=$? @ $(date -u +%H:%M:%S)"
  docker ps -aq --filter "name=fr13-bigdenom-$ARM" | xargs -r docker rm -f >/dev/null 2>&1 || true
  if [ -d "$RUNROOT/$ARM/swe_out" ]; then
    .venv/bin/python scripts/fr13_measure.py deploy-speed \
      --arm "$ARM" --out-root "$RUNROOT/$ARM/swe_out" \
      --expected-tok-per-draft "$ETPD" --batch-size 1 --basis decode_seconds \
      --out "$RUNROOT/${ARM}_speed.json" > "$RUNROOT/${ARM}_speed.log" 2>&1 \
      && echo "  [$ARM] SPEED: $(cat "$RUNROOT/${ARM}_speed.json")" \
      || { echo "  [$ARM] measure failed (see ${ARM}_speed.log):"; tail -8 "$RUNROOT/${ARM}_speed.log" | sed 's/^/    /'; }
  else
    echo "  [$ARM] WARN no swe_out/ (boot or run failed) — tail of log:"; tail -12 "$RUNROOT/${ARM}.log" | sed 's/^/    /'
  fi
}

run_arm cat6root 6
run_arm cat10    10

echo ""
echo "=== SPEED GATE SUMMARY (decode TPS, cache-ON, full graph) ==="
for f in "$RUNROOT"/sg_*_speed.json; do
  [ -f "$f" ] && echo "  $(basename "$f" .json): $(cat "$f")"
done
echo "  prior cat6root ship-gate: realized ~10-13.6 TPS (pre-EXACT_SEED/full-graph era)"
echo "=== SPEED GATE DONE @ $(date -u +%H:%M:%S) -> $RUNROOT ==="

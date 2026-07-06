#!/bin/bash
# FR13 REPLICA SELF-NOISE producer — run each config K times at temp 0.6 with DIFFERENT seeds,
# into one runroot, so fr13_replica_selfnoise_gate.py can adjudicate carrier-vs-seed statistically.
# Mirrors fr13_tree_cache_matrix.sh's env + serve_variant call, but loops (config x seed).
# Each replica = a fresh server boot with --seed=$SEED (varied) -> independent temp-0.6 sampling.
#
# Usage:  K=5 SUBSET=output/fr13_b1_gold_swe/subset_char8_localize.json \
#         CONFIGS="native:nativemtp5 chain5:chain5" bash scripts/fr13_replica_selfnoise_run.sh
# Then:   .venv/bin/python scripts/fr13_replica_selfnoise_gate.py <RUNROOT> --ref native --test chain5
set -u
cd "$(dirname "$0")/.." || exit 2
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT=${RUNROOT:-output/fr13_replica_selfnoise/run_$TS}; mkdir -p "$RUNROOT"; export RUNROOT
SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_char8_localize.json}
K=${K:-5}
CONFIGS=${CONFIGS:-"native:nativemtp5 chain5:chain5"}   # space-sep name:kind pairs (both cache-OFF)
# same deployment regime as the matrix (offload codex, forced temp 0.6). cache OFF (no APC env).
export MAX_NUM_SEQS_OVR=1 OFFLOAD_AGENT=1 DEPLOY_FORCE_TEMP=0.6 DOCKER_MEM_CAP=105g
export GPU_UTIL=${GPU_UTIL:-0.6} LUMO_PROXY_THINK_BUDGET=${LUMO_PROXY_THINK_BUDGET:-500}

echo "=== FR13 REPLICA SELF-NOISE  K=$K  CONFIGS='$CONFIGS'  SUBSET=$SUBSET -> $RUNROOT ==="
for seed in $(seq 0 $((K-1))); do
  for cfg in $CONFIGS; do
    name=${cfg%%:*}; kind=${cfg##*:}
    ARM="${name}_s${seed}"                    # serve_variant prefixes m_ -> m_${name}_s${seed}
    echo "--- [$ARM] kind=$kind SEED=$seed @ $(date -u +%H:%M:%S) ---"
    SEED=$seed bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" "$kind" "$SUBSET" \
        > "$RUNROOT/${ARM}.log" 2>&1 </dev/null
    echo "  [$ARM] serve_variant rc=$? @ $(date -u +%H:%M:%S)"
  done
done
echo "=== REPLICA RUN DONE @ $(date -u) -> $RUNROOT ==="
echo "GATE: .venv/bin/python scripts/fr13_replica_selfnoise_gate.py $RUNROOT --ref native --test chain5"

#!/usr/bin/env bash
# FR13 B-SWEEP v2 (stack recomposition, user 2026-07-25): after native_b4
# (running under the old master when this launches), run the LEAN tree arm
# at B=4, then pick the WINNING tree flavor (lean vs sealed/g3 24.64) for
# the remaining B8/B1 tree arms. All arms offloaded, cache-ON, decode-only
# metric, B-aware floor. Reads lean cache-hit context but gates on speed.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
SUB4=output/fr13_b1_gold_swe/subset_b4_four.json
SUB8=output/fr13_b1_gold_swe/subset_b8_eight.json
G3_TPS=24.64

wait_gpu_free() {
  while docker ps --format '{{.Names}}' | grep -q fr13; do sleep 60; done
  sleep 30
}

run_arm() {  # name seqfile bsize conc subset
  local name=$1 seqf=$2 bs=$3 conc=$4 subset=$5
  wait_gpu_free
  echo "===== BSWEEP2 ARM $name (B=$bs CONC=$conc seq=$seqf) $(date -u +%H:%M:%SZ) ====="
  BSWEEP_ARM=$name \
  RUNROOT=output/fr13_msr TAG=msr WALL=0 HEALTH_TIMEOUT_S=3600 \
    SUBSET=$subset BSIZE=$bs CONC=$conc GPU_UTIL=0.70 DEPLOY_FORCE_TEMP=0.6 \
    SEQUENCE_FILE="$PWD/output/fr13_msr/$seqf" \
    bash scripts/fr13_b4_campaign_driver.sh
  echo "===== BSWEEP2 ARM $name done rc=$? $(date -u +%H:%M:%SZ) ====="
}

run_arm tree_lean_b4 seq_tree_lean_only.sh 4 4 "$SUB4"

LEAN_TPS=$(python3 -c "
import json
try: print(json.load(open('output/fr13_msr/tree_lean_b4/deploy_speed_msr.json')).get('measured_tps_fullstep_wall') or 0)
except Exception: print(0)")
TREE_SEQ=$(python3 -c "print('seq_tree_lean_only.sh' if float('$LEAN_TPS') > $G3_TPS else 'seq_tree_stack_only.sh')")
echo "===== FLAVOR DECISION: lean=$LEAN_TPS vs sealed(g3)=$G3_TPS -> $TREE_SEQ ====="

run_arm tree_b8   "$TREE_SEQ"          8 8 "$SUB8"
run_arm native_b8 seq_native5_only.sh  8 8 "$SUB8"
run_arm tree_b1   "$TREE_SEQ"          1 1 "$SUB4"
run_arm native_b1 seq_native5_only.sh  1 1 "$SUB4"

echo "===== BSWEEP2 COMPLETE ====="
python3 - <<'EOF'
import json
for arm in ("g3_stack4","native_b4","tree_lean_b4","tree_b8","native_b8","tree_b1","native_b1"):
    f = f"output/fr13_msr/{arm}/deploy_speed_msr.json"
    try: d = json.load(open(f))
    except Exception: print(f"{arm:14s} MISSING"); continue
    print("%-14s tps=%-6.2f eps=%-5.2f step=%-6.0fms floor=%-5.0fms ratio=%-5.2f comb=%.2f" % (
        arm, d.get("measured_tps_fullstep_wall") or -1, d.get("events_per_step") or -1,
        d.get("step_wall_ms") or -1, d.get("floor_ms") or d.get("weight_floor_ms") or -1,
        d.get("floor_ratio") or -1, d.get("committed_per_event") or -1))
EOF

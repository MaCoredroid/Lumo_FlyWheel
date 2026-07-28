#!/usr/bin/env bash
# FR13 B-SWEEP (user 2026-07-25): co-residency sweep with REAL SWE tasks,
# agent OFFLOADED to alienware (never local), cache-ON, DECODE-ONLY metric
# (measured_tps_fullstep_wall) + B-aware floor (floor_ms = max(weight,
# compute x rows)). Tree = the byte-sealed stack; native MTP-5 = control.
# tree-B4 point = g3_stack4 (already run). Arms run SERIALLY (GPU is
# serialized); each waits for the previous teardown.
#
# Arm plan (order: control-first at matched B, then the B8 regime, then B1):
#   native_b4  BSIZE=4 CONC=4  4-task   (direct control for g3)
#   tree_b8    BSIZE=8 CONC=8  8-task   (new regime: near compute-bound)
#   native_b8  BSIZE=8 CONC=8  8-task
#   tree_b1    BSIZE=1 CONC=1  4-task
#   native_b1  BSIZE=1 CONC=1  4-task
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
SUB4=output/fr13_b1_gold_swe/subset_b4_four.json
SUB8=output/fr13_b1_gold_swe/subset_b8_eight.json

wait_gpu_free() {
  while docker ps --format '{{.Names}}' | grep -q fr13; do sleep 60; done
  sleep 30
}

run_arm() {  # name seqfile bsize conc subset
  local name=$1 seqf=$2 bs=$3 conc=$4 subset=$5
  wait_gpu_free
  echo "===== BSWEEP ARM $name (B=$bs CONC=$conc subset=$(basename $subset)) $(date -u +%H:%M:%SZ) ====="
  BSWEEP_ARM=$name \
  RUNROOT=output/fr13_msr TAG=msr WALL=0 HEALTH_TIMEOUT_S=3600 \
    SUBSET=$subset BSIZE=$bs CONC=$conc GPU_UTIL=0.70 DEPLOY_FORCE_TEMP=0.6 \
    SEQUENCE_FILE="$PWD/output/fr13_msr/$seqf" \
    bash scripts/fr13_b4_campaign_driver.sh
  echo "===== BSWEEP ARM $name done rc=$? $(date -u +%H:%M:%SZ) ====="
}

run_arm native_b4 seq_native5_only.sh    4 4 "$SUB4"
run_arm tree_b8   seq_tree_stack_only.sh 8 8 "$SUB8"
run_arm native_b8 seq_native5_only.sh    8 8 "$SUB8"
run_arm tree_b1   seq_tree_stack_only.sh 1 1 "$SUB4"
run_arm native_b1 seq_native5_only.sh    1 1 "$SUB4"

echo "===== BSWEEP COMPLETE ====="
python3 - <<'EOF'
import json, glob
rows = []
for arm in ("g3_stack4","native_b4","tree_b8","native_b8","tree_b1","native_b1"):
    f = f"output/fr13_msr/{arm}/deploy_speed_msr.json"
    try: d = json.load(open(f))
    except Exception: rows.append((arm,"MISSING")); continue
    rows.append((arm, "tps=%.2f eps=%.2f step=%.0fms floor=%.0fms ratio=%.2f comb=%.2f" % (
        d.get("measured_tps_fullstep_wall") or -1, d.get("events_per_step") or -1,
        d.get("step_wall_ms") or -1, d.get("floor_ms") or d.get("weight_floor_ms") or -1,
        d.get("floor_ratio") or -1, d.get("committed_per_event") or -1)))
for a, s in rows: print(f"{a:12s} {s}")
EOF

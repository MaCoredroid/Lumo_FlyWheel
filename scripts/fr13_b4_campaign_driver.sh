#!/usr/bin/env bash
# FR13 deployment campaign driver (batch-parameterized): run the depth-matched arms
# SERIALLY (GPU is serialized) and reduce deploy-speed after each. native bars use the
# native serve (SPEC_N); tree arms use the variant vehicle (KIND) + the device multidraft
# committer. Each arm: fresh boot, B=$BSIZE co-residency, OFFLOAD_AGENT per arg, timer ON.
#
# Env: BSIZE (vLLM max_num_seqs, default 4), CONC (codex concurrency, default 4),
#      TAG (arm-name + sidecar + deploy-json suffix, default b4), RUNROOT, SUBSET, WALL.
#
#   B=1 CLEAN run (user 2026-06-16): B=1 => eff_conc 1, ONE codex task at a time, so
#   s/fwd_gpu has NO co-residency confound and is directly comparable across shapes:
#     BSIZE=1 CONC=1 TAG=b1 RUNROOT=output/fr13_b1_swe SUBSET=subset_b4_four.json \
#       bash scripts/fr13_b4_campaign_driver.sh
#
# Usage line layout: <arm_dir> <vehicle:native|variant> <KIND_or_SPEC_N> <expect> <offload>
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
RUNROOT=${RUNROOT:-output/fr13_bigdenom_swe}
SUBSET=${SUBSET:-subset_b4_sixteen.json}
WALL=${WALL:-1800}   # codex wall per task = deployment-faithful 30min (retries still ~2x)
# WALL=0 => NO wall: emit EMPTY AGENT_WALL_S so the runner omits --agent-wall-s (a "0" would
# pass --agent-wall-s 0 = 0s = instant timeout). Hang protection = the 600s stall-watchdog.
# Matches the cachefirst header's "NO wall (WALL=0)" intent + the no-AGENT_WALL_S gate policy.
WALL_ENV="$WALL"; [[ "$WALL" == "0" ]] && WALL_ENV=""
BSIZE=${BSIZE:-4}    # vLLM max_num_seqs (B). B=1 = single-stream, no co-residency.
CONC=${CONC:-4}      # codex task concurrency. B=1 clean => CONC=1 (one task at a time).
TAG=${TAG:-b4}       # arm-name / sidecar / deploy-json suffix (keeps B=1 + B=4 separate).
mkdir -p "$RUNROOT" output/fr13_sfwd_sidecar

run_native() {  # arm spec_n expect offload
  local arm=$1 spec_n=$2 expect=$3 offload=$4
  echo "===== NATIVE ARM $arm (E$spec_n) B=$BSIZE offload=$offload ====="
  OFFLOAD_AGENT=$offload SPEC_N=$spec_n MAX_NUM_SEQS_OVR=$BSIZE SWE_CONCURRENCY=$CONC AGENT_WALL_S=$WALL_ENV \
    FR13_SFWD_GPU_TIMER=1 \
    FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}.json \
    FR13_DFWD_GPU_TIMER=1 \
    FR13_DFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json \
    FR13_CFWD_GPU_TIMER=1 \
    FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json \
    bash scripts/fr13_bigdenom_swe_serve.sh "$arm" native "$SUBSET" \
    > "$RUNROOT/$arm.runlog" 2>&1
  local rc=$?
  echo "[$arm] serve rc=$rc"
  reduce "$arm" native_e$spec_n "$expect" "$BSIZE"
}

run_variant() {  # arm kind expect offload
  local arm=$1 kind=$2 expect=$3 offload=$4
  echo "===== VARIANT ARM $arm (kind=$kind) B=$BSIZE offload=$offload ====="
  # FR13_DEVICE_MULTIDRAFT=1 -> the device-side temp>0 multidraft committer (the real
  # t0.6 speed lever; lossless-within-floor; user 2026-06-16 accepted). Engages on tree
  # arms at temp>0 only; no-op for the native bars.
  OFFLOAD_AGENT=$offload MAX_NUM_SEQS_OVR=$BSIZE SWE_CONCURRENCY=$CONC AGENT_WALL_S=$WALL_ENV \
    FR13_DEVICE_MULTIDRAFT=${FR13_DEVICE_MULTIDRAFT:-1} \
    FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
    FR13_SFWD_GPU_TIMER=1 \
    FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}.json \
    FR13_DFWD_GPU_TIMER=1 \
    FR13_DFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json \
    FR13_CFWD_GPU_TIMER=1 \
    FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh "$arm" "$kind" "$SUBSET" \
    > "$RUNROOT/$arm.runlog" 2>&1
  local rc=$?
  echo "[$arm] serve rc=$rc"
  reduce "$arm" "$arm" "$expect" "$BSIZE"
}

reduce() {  # arm label expect bsize
  local arm=$1 label=$2 expect=$3 bsize=$4
  local np
  np=$(find "$RUNROOT/$arm/swe_out" -name vllm_metrics_post.txt 2>/dev/null | wc -l)
  echo "[$arm] post-brackets=$np"
  if [ "$np" -ge 1 ]; then
    .venv/bin/python scripts/fr13_measure.py deploy-speed \
      --arm "$label" --out-root "$RUNROOT/$arm/swe_out" \
      --expected-tok-per-draft "$expect" --batch-size "$bsize" \
      --out "$RUNROOT/$arm/deploy_speed_${TAG}.json" 2>&1 | tail -14 \
      || echo "[$arm] deploy-speed reduce FAILED"
  else
    echo "[$arm] NO post-brackets — deploy-speed VACUOUS"
  fi
  docker ps -q 2>/dev/null | wc -l | sed "s/^/[$arm] docker containers after: /"
}

# ---- sequence: FIVE arms (user 2026-06-16) — depth-5 trio (E5 bar + cat9 + cat6) then
# depth-3 pair (E3 bar + 333). cat10 + cat9_contam DROPPED. temp 0.6 forced via the
# offload proxy (DEPLOY_FORCE_TEMP). Depth-matched compares: cat9/cat6 -> E5; 333 -> E3.
# Arm sequence: default = the depth-5 trio + depth-3 pair. Override with
# SEQUENCE_FILE=<path> (a bash snippet that calls run_native/run_variant, which
# are in scope here) to run a custom set of arms while reusing the SAME tested
# helpers + env wiring (no duplication).
if [[ -n "${SEQUENCE_FILE:-}" ]]; then
  echo "===== CUSTOM SEQUENCE_FILE=$SEQUENCE_FILE ====="
  source "$SEQUENCE_FILE"
else
  run_native  nativeE5_${TAG}    5 5 1
  run_variant cat9_${TAG}        cat9     9  1
  run_variant cat6root_${TAG}    cat6root 6  1
  run_native  nativeE3_${TAG}    3 3 1
  run_variant threethree_${TAG}  333      9  1
fi

echo "===== CAMPAIGN DRIVER DONE ====="

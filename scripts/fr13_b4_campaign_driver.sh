#!/usr/bin/env bash
# FR13 B=4 campaign driver: run the remaining depth-matched arms SERIALLY (GPU is
# serialized) and reduce deploy-speed after each. native bars use the native serve
# (SPEC_N); tree arms use the variant vehicle (KIND). Each arm: fresh boot, B=4
# co-residency (MAX_NUM_SEQS_OVR=4, SWE_CONCURRENCY=4), OFFLOAD_CODEX per arg.
#
# Usage: fr13_b4_campaign_driver.sh   (runs the whole remaining sequence)
#   Each line: <arm_dir> <vehicle:native|variant> <KIND_or_SPEC_N> <expect> <offload>
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
RUNROOT=output/fr13_bigdenom_swe
SUBSET=subset_b4_four.json
WALL=${WALL:-600}   # bound the codex wall per task (retries still ~2x)

run_native() {  # arm spec_n expect offload
  local arm=$1 spec_n=$2 expect=$3 offload=$4
  echo "===== NATIVE ARM $arm (E$spec_n) offload=$offload ====="
  OFFLOAD_CODEX=$offload SPEC_N=$spec_n MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S=$WALL \
    bash scripts/fr13_bigdenom_swe_serve.sh "$arm" native "$SUBSET" \
    > "$RUNROOT/$arm.runlog" 2>&1
  local rc=$?
  echo "[$arm] serve rc=$rc"
  reduce "$arm" native_e$spec_n "$expect" 4
}

run_variant() {  # arm kind expect offload
  local arm=$1 kind=$2 expect=$3 offload=$4
  echo "===== VARIANT ARM $arm (kind=$kind) offload=$offload ====="
  OFFLOAD_CODEX=$offload MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S=$WALL \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh "$arm" "$kind" "$SUBSET" \
    > "$RUNROOT/$arm.runlog" 2>&1
  local rc=$?
  echo "[$arm] serve rc=$rc"
  reduce "$arm" "$arm" "$expect" 4
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
      --out "$RUNROOT/$arm/deploy_speed_b4.json" 2>&1 | tail -14 \
      || echo "[$arm] deploy-speed reduce FAILED"
  else
    echo "[$arm] NO post-brackets — deploy-speed VACUOUS"
  fi
  docker ps -q 2>/dev/null | wc -l | sed "s/^/[$arm] docker containers after: /"
}

# ---- sequence (native E5 already running separately) ----
# bars first: E3, E4 ; then candidates: cat9, OPT-1, cat6root, cat10, 3-3-3 ;
# then the contaminated cat9 contrast (offload=0).
run_native  nativeE3_b4 3 3 1
run_native  nativeE4_b4 4 4 1
run_variant cat9_b4      cat9     9  1
run_variant opt1_b4c     cat9-opt1 9 1
run_variant cat6root_b4  cat6root 6  1
run_variant cat10_b4     cat10    10 1
run_variant threethree_b4 333     9  1
# contaminated contrast (codex co-located on GB10)
run_variant cat9_contam_b4 cat9   9  0

echo "===== CAMPAIGN DRIVER DONE ====="

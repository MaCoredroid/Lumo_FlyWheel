#!/usr/bin/env bash
# FR13 deployment campaign driver (batch-parameterized): run the depth-matched arms
# SERIALLY (GPU is serialized) and reduce deploy-speed after each. native bars use the
# native serve (SPEC_N); tree arms use the variant vehicle (KIND) + the device multidraft
# committer. Each arm: fresh boot, B=$BSIZE co-residency, OFFLOAD_AGENT per arg, timer ON.
#
# Env: BSIZE (vLLM max_num_seqs, default 4), CONC (codex concurrency, default 4),
#      TAG (arm-name + sidecar + deploy-json suffix, default b4), RUNROOT, SUBSET, WALL.
#      FR13_{S,D,C}FWD_GPU_TIMER may be set to 0 by clean measurement-off
#      sequences; each retains the historical default of 1 when unset.
#
#   B=1 CLEAN run (user 2026-06-16): B=1 => eff_conc 1, ONE codex task at a time, so
#   s/fwd_gpu has NO co-residency confound and is directly comparable across shapes:
#     BSIZE=1 CONC=1 TAG=b1 RUNROOT=output/fr13_b1_swe \
#       SUBSET=config/fr13_fixed32/subset_b4_four.json \
#       bash scripts/fr13_b4_campaign_driver.sh
#
# Usage line layout: <arm_dir> <vehicle:native|variant> <KIND_or_SPEC_N> <expect> <offload>
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
REPO=$(cd "$REPO" && pwd)
cd "$REPO"
RUNROOT=${RUNROOT:-output/fr13_bigdenom_swe}
SUBSET=${SUBSET:-config/fr13_fixed32/subset_b4_sixteen.json}
# ALL baked flags + era-fix protections live in the canonical registry —
# never here, never in seq files (two lost-protection regressions taught
# this: ATTN_KV_REMAP near-loss, SSE_HEARTBEAT emit-wedge return).
source "$SCRIPT_DIR/fr13_canonical_env.sh"
WALL=${WALL:-1800}   # codex wall per task = deployment-faithful 30min (retries still ~2x)
# WALL=0 => NO wall: emit EMPTY AGENT_WALL_S so the runner omits --agent-wall-s (a "0" would
# pass --agent-wall-s 0 = 0s = instant timeout). Hang protection = the 600s stall-watchdog.
# Matches the cachefirst header's "NO wall (WALL=0)" intent + the no-AGENT_WALL_S gate policy.
WALL_ENV="$WALL"; [[ "$WALL" == "0" ]] && WALL_ENV=""
# FR13_CAMPAIGN_TASK_BUDGET_S: per-task wall budget, DEFAULT OFF (empty => the
# runner sees no env and caps nothing). This is NOT the WALL above: WALL is the
# legacy per-attempt agent wall whose timeout is provenance-FATAL, and this is a
# declared budget whose terminal is accounted (verdict "capped", counted in the
# campaign summary and in the qwen completion algebra's abort class).
#
# OFF for quality/QC arms -- capping changes what the agent does, so a capped
# arm is not a behavioural observation and Mark's exact16 QC must see the agent
# uncapped. ON for timing and gate arms, where wall determinism is the point and
# the resolve verdict is not:
#   FR13_CAMPAIGN_TASK_BUDGET_S=5400 bash scripts/fr13_b4_campaign_driver.sh
# 5400 s is the recommended value; see run_swe_bench_q36_a.py for the floor and
# the commit that derived it from the banked per-task wall distribution.
FR13_CAMPAIGN_TASK_BUDGET_S=${FR13_CAMPAIGN_TASK_BUDGET_S:-}
export FR13_CAMPAIGN_TASK_BUDGET_S
BSIZE=${BSIZE:-4}    # vLLM max_num_seqs (B). B=1 = single-stream, no co-residency.
CONC=${CONC:-4}      # codex task concurrency. B=1 clean => CONC=1 (one task at a time).
TAG=${TAG:-b4}       # arm-name / sidecar / deploy-json suffix (keeps B=1 + B=4 separate).
# FR13_B4_TASK_REFILL=1 keeps CONC tasks in flight from a pool bigger than CONC
# (admit on completion) instead of one decaying wave. DEFAULT 0 = OFF = every
# banked contract's admission. Needs SUBSET larger than CONC, and its output is
# NOT exact4-citable without a contract update.
FR13_B4_TASK_REFILL=${FR13_B4_TASK_REFILL:-0}
[[ "$FR13_B4_TASK_REFILL" == "0" || "$FR13_B4_TASK_REFILL" == "1" ]] \
  || { echo "FAIL: FR13_B4_TASK_REFILL must be exactly 0 or 1" >&2; exit 2; }
mkdir -p "$RUNROOT" output/fr13_sfwd_sidecar

FIXED32_MANIFEST_ACTIVE=0
FIXED32_MANIFEST_FINALIZED=0
FIXED32_TASK_COUNT=""
FIXED32_SEQUENCE="scripts/fr13_fixed32_floor_timers_seq.sh"
if [[ -n "${SEQUENCE_FILE:-}" ]] \
   && [[ "$(realpath "$SEQUENCE_FILE")" == "$PWD/$FIXED32_SEQUENCE" ]]; then
  FIXED32_MANIFEST_ACTIVE=1
  if ! FIXED32_TASK_COUNT=$(
    .venv/bin/python - "$SUBSET" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from fr13_floor_gate import validate_canonical_subset


binding = validate_canonical_subset(Path(sys.argv[1]))
print(binding["task_count"])
PY
  ); then
    echo "FAIL: fixed32 requires the canonical real SWE-Verified 4- or 16-task subset" >&2
    exit 2
  fi
  [[ "$FIXED32_TASK_COUNT" == "4" || "$FIXED32_TASK_COUNT" == "16" ]] \
    || {
      echo "FAIL: fixed32 canonical task count is invalid" >&2
      exit 2
    }
  .venv/bin/python scripts/fr13_runtime_manifest.py \
    --repo "$PWD" \
    --profile fixed32 \
    --sequence "$FIXED32_SEQUENCE" \
    --output "$RUNROOT/runtime_manifest.at_launch.json" \
    || { echo "FAIL: fixed32 launch manifest"; exit 2; }
  .venv/bin/python scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" \
    --output "$RUNROOT/external_manifest.at_launch.json" \
    || { echo "FAIL: fixed32 external launch manifest"; exit 2; }
fi

finalize_fixed32_manifest() {
  (( FIXED32_MANIFEST_ACTIVE == 1 )) || return 0
  (( FIXED32_MANIFEST_FINALIZED == 0 )) || return 0
  FIXED32_MANIFEST_FINALIZED=1
  .venv/bin/python scripts/fr13_runtime_manifest.py \
    --repo "$PWD" \
    --profile fixed32 \
    --sequence "$FIXED32_SEQUENCE" \
    --output "$RUNROOT/runtime_manifest.at_end.json" \
    || return 1
  cmp -s \
    "$RUNROOT/runtime_manifest.at_launch.json" \
    "$RUNROOT/runtime_manifest.at_end.json" \
    || {
      echo "FAIL: fixed32 runtime/source manifest changed during campaign" >&2
      return 1
    }
  .venv/bin/python scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" \
    --output "$RUNROOT/external_manifest.at_end.json" \
    || return 1
  cmp -s \
    "$RUNROOT/external_manifest.at_launch.json" \
    "$RUNROOT/external_manifest.at_end.json" \
    || {
      echo "FAIL: fixed32 external artifacts changed during campaign" >&2
      return 1
    }
  echo "fixed32 runtime/source manifest byte-equal at campaign end"
  echo "fixed32 image/model/FA2 manifest byte-equal at campaign end"
}

campaign_exit() {
  local rc=$?
  trap - EXIT
  if (( FIXED32_MANIFEST_ACTIVE == 1 && FIXED32_MANIFEST_FINALIZED == 0 )); then
    finalize_fixed32_manifest
    local manifest_rc=$?
    (( rc == 0 && manifest_rc != 0 )) && rc=14
  fi
  exit "$rc"
}
trap campaign_exit EXIT

run_native() {  # arm spec_n expect offload
  local arm=$1 spec_n=$2 expect=$3 offload=$4
  echo "===== NATIVE ARM $arm (E$spec_n) B=$BSIZE offload=$offload ====="
  OFFLOAD_AGENT=$offload SPEC_N=$spec_n MAX_NUM_SEQS_OVR=$BSIZE SWE_CONCURRENCY=$CONC AGENT_WALL_S=$WALL_ENV \
    FR13_B4_TASK_REFILL="$FR13_B4_TASK_REFILL" \
    FR13_SFWD_GPU_TIMER="${FR13_SFWD_GPU_TIMER:-1}" \
    FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}.json \
    FR13_DFWD_GPU_TIMER="${FR13_DFWD_GPU_TIMER:-1}" \
    FR13_DFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json \
    FR13_CFWD_GPU_TIMER="${FR13_CFWD_GPU_TIMER:-1}" \
    FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json \
    bash scripts/fr13_bigdenom_swe_serve.sh "$arm" native "$SUBSET" \
    > "$RUNROOT/$arm.runlog" 2>&1
  local rc=$?
  echo "[$arm] serve rc=$rc"
  reduce "$arm" native_e$spec_n "$expect" "$BSIZE"
  if (( rc != 0 )); then
    echo "[$arm] FAIL: serving/gate rc=$rc; terminating campaign"
    exit "$rc"
  fi
}

run_variant() {  # arm kind expect offload
  local arm=$1 kind=$2 expect=$3 offload=$4
  if [[ "$kind" == "tail6_fixed32" || "$kind" == "hydra27_fixed32" ]] \
     && (( FIXED32_MANIFEST_ACTIVE != 1 )); then
    echo "FAIL: fixed32 arms require the exact canonical fixed32 sequence" >&2
    exit 2
  fi
  echo "===== VARIANT ARM $arm (kind=$kind) B=$BSIZE offload=$offload ====="
  # FR13_DEVICE_MULTIDRAFT=1 -> the device-side temp>0 multidraft committer (the real
  # t0.6 speed lever; lossless-within-floor; user 2026-06-16 accepted). Engages on tree
  # arms at temp>0 only; no-op for the native bars.
  OFFLOAD_AGENT=$offload MAX_NUM_SEQS_OVR=$BSIZE SWE_CONCURRENCY=$CONC AGENT_WALL_S=$WALL_ENV \
    FR13_B4_TASK_REFILL="$FR13_B4_TASK_REFILL" \
    FR13_DEVICE_MULTIDRAFT=${FR13_DEVICE_MULTIDRAFT:-1} \
    FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
    FR13_SFWD_GPU_TIMER="${FR13_SFWD_GPU_TIMER:-1}" \
    FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}.json \
    FR13_DFWD_GPU_TIMER="${FR13_DFWD_GPU_TIMER:-1}" \
    FR13_DFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json \
    FR13_CFWD_GPU_TIMER="${FR13_CFWD_GPU_TIMER:-1}" \
    FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh "$arm" "$kind" "$SUBSET" \
    > "$RUNROOT/$arm.runlog" 2>&1
  local rc=$?
  echo "[$arm] serve rc=$rc"
  reduce "$arm" "$arm" "$expect" "$BSIZE"
  if (( rc != 0 )); then
    echo "[$arm] FAIL: serving/gate rc=$rc; terminating campaign"
    exit "$rc"
  fi
}

reduce() {  # arm label expect bsize
  local arm=$1 label=$2 expect=$3 bsize=$4
  local np
  np=$(find "$RUNROOT/$arm/swe_out" -name vllm_metrics_post.txt 2>/dev/null | wc -l)
  echo "[$arm] post-brackets=$np"
  # The work census, when the arm wrote one. MANDATORY for a STAGGERED bracket
  # topology (a refilled pool): the envelope max(post)-min(pre) keeps no
  # independent per-task check, so fr13_measure fail-louds without it -- which is
  # why every pool16 arm's in-pass reduce died at fr13_measure.py:1744 while its
  # evidence was perfectly good. Passing it makes this artifact CENSUS-GATED for
  # nested arms too, which is strictly better than the ungated one it replaces.
  # Absent census: unchanged behaviour, so no existing campaign shape moves.
  local census_args=()
  local census_path="$RUNROOT/$arm/logs/fr13_fixed32_work_census.jsonl"
  if [ -f "$census_path" ] && [ ! -L "$census_path" ]; then
    census_args=(--work-census "$census_path")
    echo "[$arm] work census: $census_path"
  else
    echo "[$arm] NO work census — reduce stays ungated"
  fi
  if [ "$np" -ge 1 ]; then
    .venv/bin/python scripts/fr13_measure.py deploy-speed \
      --arm "$label" --out-root "$RUNROOT/$arm/swe_out" \
      --expected-tok-per-draft "$expect" --batch-size "$bsize" \
      "${census_args[@]}" \
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

finalize_fixed32_manifest || exit 14
if (( FIXED32_MANIFEST_ACTIVE == 1 )); then
  if [[ "${FR13_FIXED32_ATTRIBUTION_ONLY:-0}" == "1" ]]; then
    echo "fixed32 attribution-only campaign cannot enter the acceptance gate" >&2
    exit 86
  fi
  .venv/bin/python scripts/fr13_floor_gate.py \
    --repo "$PWD" \
    --runroot "$RUNROOT" \
    --tag "$TAG" \
    --task-count "$FIXED32_TASK_COUNT" \
    --expect-concurrency "$CONC" \
    --draft-vocab-root "$FR13_DRAFT_VOCAB_ROOT" \
    > "$RUNROOT/fixed32_floor_gate.json" \
    || {
      rc=$?
      echo "FAIL: fixed32 floor gate rc=$rc" >&2
      exit "$rc"
    }
fi
echo "===== CAMPAIGN DRIVER DONE ====="

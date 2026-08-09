#!/usr/bin/env bash
# Real SWE-Verified diagnostic for the fixed32 BV64/4-warp committer geometry.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

case "${FR13_RUN_COMMITTER_BV64_REAL:-0}" in
  1) ;;
  0)
    echo "BV64 committer real diagnostic is disabled; set FR13_RUN_COMMITTER_BV64_REAL=1" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_COMMITTER_BV64_REAL must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned stock FA2 binary}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
BATCH=${FR13_COMMITTER_BV64_REAL_BATCH:-1}
MODE=hydra27_fixed32
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
FORKED_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
FORKED_FA2_BYTES=299183936
MANDATORY_WEIGHT_BYTES=32666638208
MANDATORY_WEIGHT_FLOOR_MS=119.658015414
ONE_SIDED_U95_CAP_MS=137.6067177261
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
REDUCER=scripts/fr13_committer_bv64_real_result.py
SOURCE_COMMIT=$(git rev-parse --verify HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
REDUCER_SHA256=$(sha256sum "$REDUCER" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}
KV_CACHE_MEMORY_BYTES=

case "$BATCH" in
  1)
    SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
    SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
    B1_DIAGNOSTIC=1
    CLASS=one_real_swe_verified_b1_committer_bv64_diagnostic
    ;;
  4)
    SUBSET=config/fr13_fixed32/subset_b4_four.json
    SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
    B1_DIAGNOSTIC=0
    CLASS=real_swe_verified_exact4_b4_committer_bv64_diagnostic
    KV_CACHE_MEMORY_BYTES=49392123904
    ;;
  *)
    echo "FR13_COMMITTER_BV64_REAL_BATCH must be exactly 1 or 4" >&2
    exit 2
    ;;
esac

ARM="hydra27_fixed32_k64_committer_bv64_b${BATCH}_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* \
   && ! -e "$RUNROOT_ABS" \
   && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be a new path below $REPO/output" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ \
   && -z "$(git status --porcelain=v1 --untracked-files=no)" \
   && "$(git rev-parse '@{upstream}')" == "$SOURCE_COMMIT" ]] \
  || { echo "BV64 real diagnostic requires a clean source commit pushed upstream" >&2; exit 2; }
for binding in \
  "$SUBSET:$SUBSET_SHA256" \
  "$BLOCK_MAP:$BLOCK_MAP_SHA256"; do
  path=${binding%%:*}
  expected=${binding#*:}
  [[ -f "$path" && ! -L "$path" \
     && "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] \
    || { echo "BV64 real diagnostic input identity drifted: $path" >&2; exit 2; }
done
unset binding path expected
[[ "$FORKED_FA2_SO" == /* \
   && -f "$FORKED_FA2_SO" \
   && ! -L "$FORKED_FA2_SO" \
   && "$(stat -c '%s' "$FORKED_FA2_SO")" == "$FORKED_FA2_BYTES" \
   && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" == "$FORKED_FA2_SHA256" ]] \
  || { echo "FORKED_FA2_SO is not the pinned stock FA2 binary" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the real diagnostic" >&2; exit 2; }

export BSIZE="$BATCH"
export CONC="$BATCH"
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER"
export FR13_NEEDS_ALLOW=
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_DRAFT_VOCAB_ROOT" == "1" \
   && "$FR13_DRAFT_VOCAB_K" == "65536" \
   && "$FR13_DRAFT_VOCAB_BLOCKS" == "$BLOCK_MAP_CONTAINER" \
   && "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" \
   && "$LUMO_SWE_AUTOCOMMIT" == "0" ]] \
  || { echo "K64/root1 fixed32 floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS/sidecars"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"
printf 'classification=%s\nacceptance_valid=0\ntiming_eligible=0\nfloor_acceptance_eligible=0\nmeasurement_scope=informative_real_task_diagnostic\nmode=%s\ntask_count=%s\nbatch_size=%s\nconcurrency=%s\nphysical_rows=32\nlogical_active_nodes=27\ndraft_vocab_root=1\ndraft_vocab_k=65536\ncommitter_layer_batch=1\ncommitter_bv64_warp4=1\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nsource=%s\nrunner_sha256=%s\nreducer_sha256=%s\nsubset_sha256=%s\nblock_map_sha256=%s\nfa2_sha256=%s\nstarted=%s\n' \
  "$CLASS" "$MODE" "$BATCH" "$BATCH" "$BATCH" \
  "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$SOURCE_COMMIT" "$RUNNER_SHA256" \
  "$REDUCER_SHA256" "$SUBSET_SHA256" "$BLOCK_MAP_SHA256" \
  "$FORKED_FA2_SHA256" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

MANIFEST_FINALIZED=0
finalize_manifests() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
    --source-commit "$SOURCE_COMMIT" \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json" || return $?
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json" || return $?
  cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || { echo "runtime/source manifest changed during BV64 diagnostic" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during BV64 diagnostic" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" \
     && "$(sha256sum "$REDUCER" | awk '{print $1}')" == "$REDUCER_SHA256" ]] \
    || { echo "BV64 runner/reducer changed during execution" >&2; return 14; }
  MANIFEST_FINALIZED=1
}
runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    if finalize_manifests; then :; else
      local manifest_rc=$?
      (( rc == 0 )) && rc=$manifest_rc
    fi
  fi
  exit "$rc"
}
trap runner_exit EXIT

if env \
    RUNROOT="$RUNROOT_ABS" \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR="$BATCH" SWE_CONCURRENCY="$BATCH" \
    AGENT_WALL_S= KV_CACHE_MEMORY_BYTES="$KV_CACHE_MEMORY_BYTES" \
    LUMO_SWE_AUTOCOMMIT=0 GPU_UTIL=0.70 \
    FR13_FIXED32_B1_DIAGNOSTIC="$B1_DIAGNOSTIC" \
    FR13_B1_DIAGNOSTIC_TASK_PROFILE=astropy12907 \
    FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
    FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER" FR13_NEEDS_ALLOW= \
    FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES" \
    FR13_WEIGHT_FLOOR_MS="$MANDATORY_WEIGHT_FLOOR_MS" \
    FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
    FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
    FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${ARM}.json" \
    FR13_DFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${ARM}_dfwd.json" \
    FR13_CFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${ARM}_cfwd.json" \
    FR13_DEVICE_MULTIDRAFT=1 \
    FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
    FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_DRAFT_HEAD_M32_TIMING_ARM=0 \
    FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB=0 \
    FR13_DRAFT_HEAD_M1_R64_U8_QUALITY_GATE=0 \
    FR13_DRAFT_HEAD_M1_R64_U8_TAW_QUALITY_GATE=0 \
    FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION=0 \
    FR13_DRAFT_HEAD_M4_R64_U8_LIVE_AB=0 \
    FR13_DRAFT_HEAD_M4_R64_U8_QUALITY_GATE=0 \
    FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION=0 \
    FR13_DRAFT_HEAD_FP8=0 FR13_DRAFT_HEAD_FP8_STATIC_IO=0 \
    FR13_DRAFT_HEAD_FP8_ARM= FR13_DFWD_K64_TOP3=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_SO= \
    FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FA2_QROW32_LIVE_PAGED_AB=0 FR13_FA2_QROW32_PRODUCTION=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_TIMING_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
    FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0 \
    FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=0 \
    FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB=0 \
    FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0 FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0 \
    FR13_CFWD_PACKED_WALK_NODE_TRUST_BYTE_AB=0 \
    FR13_CFWD_PACKED_WALK_NODE_TRUST_PRODUCTION=0 \
    FR13_CFWD_PACKED_WALK_ACTIVE_DEPTH_BYTE_AB=0 \
    FR13_FIXED32_COMMITTER_LAYER_BATCH=1 \
    FR13_FIXED32_COMMITTER_BV64_WARP4=1 \
    FR13_FIXED32_COMMITTER_LAYER_BATCH_QUALIFICATION=1 \
    FR13_FIXED32_COMMITTER_METADATA_FUSION=0 \
    FR13_FIXED32_COMMITTER_DIRECT_METADATA=0 \
    FR13_FIXED32_COMMITTER_STICKY_GUARD=0 \
    FR13_FIXED32_COMMITTER_KNORM_RING=0 \
    FR13_FIXED32_COMMITTER_GATE_RING=0 \
    FR13_FIXED32_COMMITTER_DECAY_RING=0 \
    FR13_FIXED32_CONV_COMMIT_ZERO_TAIL=0 \
    FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB=0 \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 \
    FORKED_FA2_SO="$FORKED_FA2_SO" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" "$MODE" "$SUBSET" \
      > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi
printf 'serve_rc=%s ended=%s\n' "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
(( serve_rc == 0 )) || exit "$serve_rc"

[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after the BV64 real diagnostic" >&2; exit 2; }
CONTAINER_ENV="$ARMDIR/container_env.txt"
RUNTIME_LOG="$ARMDIR/docker_after_tasks.log"
for evidence in "$CONTAINER_ENV" "$RUNTIME_LOG"; do
  [[ -f "$evidence" && ! -L "$evidence" ]] \
    || { echo "BV64 real diagnostic evidence is missing: $evidence" >&2; exit 4; }
done
unset evidence

MEASUREMENT="$ARMDIR/deploy_speed_fullwall.json"
FR13_DRAFT_HEAD_FP8=0 FR13_DRAFT_HEAD_FP8_STATIC_IO=0 \
FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES" \
FR13_WEIGHT_FLOOR_MS="$MANDATORY_WEIGHT_FLOOR_MS" \
"$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
  --arm "$ARM" \
  --out-root "$ARMDIR/swe_out" \
  --expected-tok-per-draft 31 \
  --batch-size "$BATCH" \
  --out "$MEASUREMENT"

SUMMARY="$ARMDIR/committer_bv64_real_summary.json"
"$PYTHON_BIN" "$REDUCER" \
  --arm-dir "$ARMDIR" \
  --batch-size "$BATCH" \
  --subset "$SUBSET" \
  --measurement "$MEASUREMENT" \
  --runtime-log "$RUNTIME_LOG" \
  --container-env "$CONTAINER_ENV" \
  --source-commit "$SOURCE_COMMIT" \
  --runner-sha256 "$RUNNER_SHA256" \
  --output "$SUMMARY"

finalize_manifests
printf 'summary=%s completed=%s\n' "$SUMMARY" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
trap - EXIT

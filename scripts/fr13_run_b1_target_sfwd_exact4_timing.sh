#!/usr/bin/env bash
# Exact4 Qrow16 reference vs historical-target/current-SFWD real B1 timing pair.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

case "${FR13_RUN_B1_TARGET_SFWD_EXACT4_TIMING:-0}" in
  1) ;;
  0)
    echo "exact4 target/SFWD timing is disabled; set FR13_RUN_B1_TARGET_SFWD_EXACT4_TIMING=1" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_B1_TARGET_SFWD_EXACT4_TIMING must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

: "${RUNROOT:?set RUNROOT to a new path below output/}"
: "${TAG:?set TAG to a unique tag}"
: "${QROW16_FA2_SO:?set QROW16_FA2_SO to the pinned Qrow16 binary}"
: "${CUTLASS_TARGET_SO:?set CUTLASS_TARGET_SO to the pinned cooperative target}"
: "${SFWD_CONV_POSTPREP_PASS:?set the fresh SFWD live PASS}"
: "${SFWD_CONV_POSTPREP_PASS_SHA256:?set its raw SHA-256}"
: "${SFWD_CONV_POSTPREP_SOURCE_MANIFEST:?set the fresh SFWD source manifest}"
: "${SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256:?set its raw SHA-256}"
: "${SFWD_CONV_POSTPREP_GATE_SUMMARY:?set the fresh standalone SFWD gate summary}"
: "${SFWD_CONV_POSTPREP_GATE_SUMMARY_SHA256:?set its raw SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
TASK_SET=exact4
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
EXPECTED_TASKS=4
EXPECTED_TASK_IDS=(
  astropy__astropy-12907
  astropy__astropy-13033
  astropy__astropy-13236
  astropy__astropy-13398
)

BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
QROW16_PASS=results/fr13_fixed32_qrow16_num_splits0_live_pass_20260731T173608Z/fr13_fa2_qrow16_live_paged_ab.json
QROW16_SHA256=1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86
QROW16_BYTES=299507792
QROW16_PASS_SHA256=36940fd43d11399529d1bfe7e11baa9961907193267f3bb43d41057328737b77
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
TARGET_SELECTOR=identity_wide256_fullgrid_b1
TARGET_SHA256=d8c6502e7a166e6d2124576a9e36814401d6dbc215516adfffa7ac436f93ba0f
TARGET_BYTES=119704312
TARGET_PASS="$REPO/results/fr13_b1_m128_cooperative_target_sfwd_real_gate_a8a904ed6_20260805/target_combined_pass.json"
TARGET_PASS_SHA256=169704fac7c544600437e7785f5d810c9df8ffaf5f9ce70d96d83b21de46236d
TARGET_QUALIFICATION_SOURCE_COMMIT=a8a904ed6c27a6338d43151038c155ebb76e3656
TARGET_QUALIFICATION_PATCH_SHA256=ae9591a0c255c54bd8b5fed8576105013fce7f5f0834dbfb51ca1d455441f976
SFWD_PROFILE_PRESEED_COMMIT=ff067115c547a39bad706c10f91552896a87d264
WEIGHT_FLOOR_MS=119.658015414
ONE_SIDED_U95_CAP_MS=137.6067177261
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}
STOCK_ARM="hydra27_fixed32_qrow16_reference_${TASK_SET}_${TAG}"
CANDIDATE_ARM="hydra27_fixed32_qrow16_target_sfwd_${TASK_SET}_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* && ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be a new path below $REPO/output" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }

INPUT_BINDINGS=(
  "$TARGET_PASS:$TARGET_PASS_SHA256"
  "$SFWD_CONV_POSTPREP_PASS:$SFWD_CONV_POSTPREP_PASS_SHA256"
  "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST:$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256"
  "$SFWD_CONV_POSTPREP_GATE_SUMMARY:$SFWD_CONV_POSTPREP_GATE_SUMMARY_SHA256"
)
for binding in "${INPUT_BINDINGS[@]}"; do
  path=${binding%:*}
  digest=${binding##*:}
  [[ "$path" == /* && -f "$path" && ! -L "$path" \
     && "$(stat -c '%h' "$path")" == "1" \
     && "$digest" =~ ^[0-9a-f]{64}$ \
     && "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] \
    || { echo "credential or evidence identity drifted: $path" >&2; exit 2; }
done
unset binding path digest
for binary in "$QROW16_FA2_SO" "$CUTLASS_TARGET_SO"; do
  [[ "$binary" == /* && -f "$binary" && ! -L "$binary" \
     && "$(stat -c '%h' "$binary")" == "1" ]] \
    || { echo "candidate binary must be an absolute regular file: $binary" >&2; exit 2; }
done
unset binary
[[ "$(stat -c '%s' "$QROW16_FA2_SO")" == "$QROW16_BYTES" \
   && "$(sha256sum "$QROW16_FA2_SO" | awk '{print $1}')" == "$QROW16_SHA256" \
   && "$(stat -c '%s' "$CUTLASS_TARGET_SO")" == "$TARGET_BYTES" \
   && "$(sha256sum "$CUTLASS_TARGET_SO" | awk '{print $1}')" == "$TARGET_SHA256" \
   && "$(sha256sum "$QROW16_PASS" | awk '{print $1}')" == "$QROW16_PASS_SHA256" \
   && "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" \
   && "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "Qrow16/target binary or committed evidence identity drifted" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" \
   && "$(git rev-parse '@{upstream}')" == "$SOURCE_COMMIT" ]] \
  || { echo "full-stack timing requires a clean source commit pushed to upstream" >&2; exit 2; }
git merge-base --is-ancestor "$SFWD_PROFILE_PRESEED_COMMIT" "$SOURCE_COMMIT" \
  || { echo "runtime source lacks the merged SFWD profile-preseed fix" >&2; exit 2; }
[[ "$(grep -Fc '_fr13_fixed32_preseed_sfwd_conv_postprep_profile_capture(num_reqs)' scripts/fr10_phase4_patch_vllm_tree_gdn.py)" -eq 1 \
   && "$(grep -Fc 'FR13 SFWD conv/post-prep profile output preseed is incomplete' scripts/fr10_phase4_patch_vllm_tree_gdn.py)" -eq 1 ]] \
  || { echo "runtime SFWD profile-preseed source contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS/runtime_inputs" "$RUNROOT_ABS/sidecars"

"$PYTHON_BIN" - "$QROW16_PASS" "$QROW16_SHA256" <<'PY'
import sys
from pathlib import Path

from scripts import fr13_qrow16_pass_sidecar as qrow

payload, _ = qrow.load_json(Path(sys.argv[1]))
qrow.validate_live_result(payload, candidate_sha256=sys.argv[2])
PY

"$PYTHON_BIN" scripts/fr13_cutlass_streamk_pass.py validate \
  --live-result "$TARGET_PASS" \
  --expected-live-sha256 "$TARGET_PASS_SHA256" \
  --candidate-so "$CUTLASS_TARGET_SO" \
  --patch-source scripts/fr13_patch_cutlass_fixed32_wave.py \
  --expected-source-commit "$TARGET_QUALIFICATION_SOURCE_COMMIT" \
  --runtime-source-commit "$SOURCE_COMMIT" \
  --candidate-selector "$TARGET_SELECTOR" \
  --qualification-profile k64_root \
  --diagnostic-task-profile astropy12907 \
  --draft-vocab-blocks "$BLOCK_MAP" >/dev/null
"$PYTHON_BIN" scripts/fr13_sfwd_conv_postprep_gate.py validate-pass \
  --repo "$REPO" \
  --live-pass "$SFWD_CONV_POSTPREP_PASS" \
  --expected-live-pass-sha256 "$SFWD_CONV_POSTPREP_PASS_SHA256" \
  --source-manifest "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST" \
  --expected-source-manifest-sha256 "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" \
  --source-commit "$SOURCE_COMMIT" >/dev/null
"$PYTHON_BIN" - \
  "$SFWD_CONV_POSTPREP_GATE_SUMMARY" "$SOURCE_COMMIT" \
  "$SFWD_CONV_POSTPREP_PASS_SHA256" \
  "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
required = {
    "schema": "fr13.fixed32.sfwd_conv_postprep.k64_root_b1_gate.v1",
    "status": "pass",
    "source_commit": sys.argv[2],
    "live_pass_sha256": sys.argv[3],
    "source_manifest_sha256": sys.argv[4],
    "task_id": "astropy__astropy-12907",
    "task_count": 1,
    "fixed32_mode": "hydra27_fixed32",
    "batch_size": 1,
    "physical_rows_per_request": 32,
    "draft_vocab_root": 1,
    "draft_vocab_k": 65536,
    "layer_count": 48,
    "decision_exact": True,
    "no_fallback": True,
    "production_enabled": False,
    "timing_eligible": False,
}
if any(summary.get(key) != value for key, value in required.items()):
    raise SystemExit("fresh SFWD real-task gate summary drifted")
PY

cp -- "$SFWD_CONV_POSTPREP_PASS" "$RUNROOT_ABS/runtime_inputs/sfwd_pass.json"
cp -- "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST" "$RUNROOT_ABS/runtime_inputs/sfwd_source_manifest.json"
chmod 0400 "$RUNROOT_ABS/runtime_inputs/sfwd_pass.json" \
  "$RUNROOT_ABS/runtime_inputs/sfwd_source_manifest.json"
SFWD_PASS_CONTAINER="/workspace/$RUNROOT_REL/runtime_inputs/sfwd_pass.json"
SFWD_MANIFEST_CONTAINER="/workspace/$RUNROOT_REL/runtime_inputs/sfwd_source_manifest.json"
[[ "$(sha256sum "$RUNROOT_ABS/runtime_inputs/sfwd_pass.json" | awk '{print $1}')" == "$SFWD_CONV_POSTPREP_PASS_SHA256" \
   && "$(sha256sum "$RUNROOT_ABS/runtime_inputs/sfwd_source_manifest.json" | awk '{print $1}')" == "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" ]] \
  || { echo "runtime SFWD credential copy drifted" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

export BSIZE=1 CONC=1 WALL=0
export FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "32666638208" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$WEIGHT_FLOOR_MS" \
   && "$LUMO_SWE_AUTOCOMMIT" == "0" ]] \
  || { echo "fixed K64/root1 B1 floor contract drifted" >&2; exit 2; }

printf 'classification=real_swe_verified_b1_target_sfwd_exact4_timing_pair\ntask_set=%s\ntask_count=%s\nbatch_size=1\nconcurrency=1\ntiming_eligible=1\nfloor_acceptance_eligible=0\nproduction_default_enabled=0\nmode=hydra27_fixed32\nphysical_rows=32\ndraft_vocab_root=1\ndraft_vocab_k=65536\nruntime=FULL_graph_exact_geometry\nruntime_source_commit=%s\nrunner_sha256=%s\nsubset_sha256=%s\nqrow16_fa2_sha256=%s\nqrow16_pass_sha256=%s\ntarget_selector=%s\ntarget_so_sha256=%s\ntarget_qualification_source_commit=%s\ntarget_qualification_patch_sha256=%s\ntarget_pass_sha256=%s\nsfwd_pass_sha256=%s\nsfwd_source_manifest_sha256=%s\nsfwd_gate_summary_sha256=%s\nstock_arm=%s\ncandidate_arm=%s\nstarted=%s\n' \
  "$TASK_SET" "$EXPECTED_TASKS" "$SOURCE_COMMIT" "$RUNNER_SHA256" \
  "$SUBSET_SHA256" "$QROW16_SHA256" "$QROW16_PASS_SHA256" \
  "$TARGET_SELECTOR" "$TARGET_SHA256" \
  "$TARGET_QUALIFICATION_SOURCE_COMMIT" "$TARGET_QUALIFICATION_PATCH_SHA256" \
  "$TARGET_PASS_SHA256" \
  "$SFWD_CONV_POSTPREP_PASS_SHA256" \
  "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" \
  "$SFWD_CONV_POSTPREP_GATE_SUMMARY_SHA256" \
  "$STOCK_ARM" "$CANDIDATE_ARM" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$REPO" --profile fixed32 --sequence "$SEQUENCE" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$REPO" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

MANIFEST_FINALIZED=0
finalize_manifests() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$REPO" --profile fixed32 --sequence "$SEQUENCE" \
    --source-commit "$SOURCE_COMMIT" \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$REPO" --output "$RUNROOT_ABS/external_manifest.at_end.json"
  cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || { echo "runtime/source manifest changed during timing" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" \
     && -f "$QROW16_FA2_SO" && ! -L "$QROW16_FA2_SO" \
     && "$(stat -c '%h' "$QROW16_FA2_SO")" == "1" \
     && "$(stat -c '%s' "$QROW16_FA2_SO")" == "$QROW16_BYTES" \
     && "$(sha256sum "$QROW16_FA2_SO" | awk '{print $1}')" == "$QROW16_SHA256" \
     && -f "$CUTLASS_TARGET_SO" && ! -L "$CUTLASS_TARGET_SO" \
     && "$(stat -c '%h' "$CUTLASS_TARGET_SO")" == "1" \
     && "$(stat -c '%s' "$CUTLASS_TARGET_SO")" == "$TARGET_BYTES" \
     && "$(sha256sum "$CUTLASS_TARGET_SO" | awk '{print $1}')" == "$TARGET_SHA256" ]] \
    || { echo "Qrow16 target/SFWD runner or binary changed during timing" >&2; return 14; }
  for binding in "${INPUT_BINDINGS[@]}"; do
    local path=${binding%:*}
    local digest=${binding##*:}
    [[ -f "$path" && ! -L "$path" && "$(stat -c '%h' "$path")" == "1" \
       && "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] \
      || { echo "credential changed during timing: $path" >&2; return 14; }
  done
  MANIFEST_FINALIZED=1
}
runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    finalize_manifests || { local manifest_rc=$?; (( rc == 0 )) && rc=$manifest_rc; }
  fi
  exit "$rc"
}
trap runner_exit EXIT

run_arm() {
  local arm=$1
  local production=$2
  local device_kernel=/workspace/scripts/fr13_device_multidraft_kernel.py
  local target_selector=stock target_so="" target_pass="" target_pass_sha=""
  local target_qualification_source_commit=""
  local sfwd_fusion=0 sfwd_pass="" sfwd_pass_sha="" sfwd_manifest=""
  local sfwd_manifest_sha="" sfwd_commit="" conv_wb_batched=1
  if [[ "$production" == "1" ]]; then
    target_selector=$TARGET_SELECTOR
    target_so=$CUTLASS_TARGET_SO
    target_pass=$TARGET_PASS
    target_pass_sha=$TARGET_PASS_SHA256
    target_qualification_source_commit=$TARGET_QUALIFICATION_SOURCE_COMMIT
    sfwd_fusion=1
    sfwd_pass=$SFWD_PASS_CONTAINER
    sfwd_pass_sha=$SFWD_CONV_POSTPREP_PASS_SHA256
    sfwd_manifest=$SFWD_MANIFEST_CONTAINER
    sfwd_manifest_sha=$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256
    sfwd_commit=$SOURCE_COMMIT
  fi

  echo "===== $arm: B1 task_set=$TASK_SET target_sfwd_production=$production ====="
  if env \
      RUNROOT="$RUNROOT_ABS" \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
      LUMO_SWE_AUTOCOMMIT=0 \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
      FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
      FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json \
      FR13_DEVICE_MULTIDRAFT=1 FR13_DEVICE_MULTIDRAFT_KERNEL="$device_kernel" \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_cfwd.json" \
      FR13_B1_U8_CFWD_SFWD_STACK_TIMING=0 \
      FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
      FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB=0 \
      FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION=0 \
      FR13_DRAFT_HEAD_FP8=0 FR13_DFWD_K64_TOP3=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON= \
      FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0 \
      FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0 \
      FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_JSON= \
      FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_SHA256= \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 \
      FR13_FA2_QROW16_SO_SHA256="$QROW16_SHA256" \
      FR13_FA2_QROW16_PRODUCTION=1 \
      FR13_FA2_QROW16_LIVE_PASS_JSON="$QROW16_PASS" \
      FR13_FA2_QROW16_LIVE_PASS_SHA256="$QROW16_PASS_SHA256" \
      FR13_FA2_QROW32_LIVE_PAGED_AB=0 \
      FR13_FA2_QROW32_B1_LIVE_AB_ARM= FR13_FA2_QROW32_B1_PRODUCTION_ARM= \
      FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=0 \
      FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH= \
      FR13_FIXED32_GDN_PATH_BV_CANDIDATE= FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
      FR13_FIXED32_BATCH_GDN_BYTE_AB=0 FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_PRODUCTION=0 FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_FIXED32_CUTLASS_WAVE="$target_selector" \
      FR13_FIXED32_CUTLASS_WAVE_SO="$target_so" \
      FR13_FIXED32_CUTLASS_WAVE_PRODUCTION="$production" \
      FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE=k64_root \
      FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_SOURCE_COMMIT="$target_qualification_source_commit" \
      FR13_FIXED32_CUTLASS_WAVE_DIAGNOSTIC_TASK_PROFILE=astropy12907 \
      FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_JSON="$target_pass" \
      FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_SHA256="$target_pass_sha" \
      FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
      FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
      FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB=0 \
      FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION="$sfwd_fusion" \
      FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=0 \
      FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_JSON="$sfwd_pass" \
      FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_SHA256="$sfwd_pass_sha" \
      FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_PATH="$sfwd_manifest" \
      FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256="$sfwd_manifest_sha" \
      FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_COMMIT="$sfwd_commit" \
      FR13_CONV_WB_BATCHED="$conv_wb_batched" \
      FR13_TREE_CONV_FUSED=1 FR13_FIXED32_CONV_SOURCE_BATCH=0 \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$QROW16_FA2_SO" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh \
        "$arm" hydra27_fixed32 "$SUBSET" \
        > "$RUNROOT_ABS/$arm.runlog" 2>&1; then
    :
  else
    local serve_rc=$?
    printf 'arm=%s serve_rc=%s ended=%s\n' \
      "$arm" "$serve_rc" "$(date -u +%FT%TZ)" \
      >> "$RUNROOT_ABS/launcher_meta.txt"
    return "$serve_rc"
  fi

  local env_path="$RUNROOT_ABS/$arm/container_env.txt"
  [[ -f "$env_path" && ! -L "$env_path" ]] \
    || { echo "$arm lacks container_env.txt" >&2; return 4; }
  for expected in \
      'FR13_FIXED32_MODE=hydra27_fixed32' \
      'FR13_FIXED32_B1_DIAGNOSTIC=0' \
      'FR13_DRAFT_VOCAB_ROOT=1' \
      'FR13_DRAFT_VOCAB_K=65536' \
      'MAX_NUM_SEQS=1' \
      'SWE_CONCURRENCY=1' \
      'ENFORCE_EAGER=0' \
      'CUDAGRAPH_MODE=FULL_AND_PIECEWISE' \
      "FR13_DEVICE_MULTIDRAFT_KERNEL=$device_kernel" \
      'FR13_FA2_QROW16_PRODUCTION=1' \
      'FR13_FA2_QROW32_B1_PRODUCTION_ARM=' \
      'FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=0' \
      'FR13_DFWD_K64_TOP3=0' \
      'FR13_B1_U8_CFWD_SFWD_STACK_TIMING=0' \
      'FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION=0' \
      'FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0' \
      'FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0' \
      'FR13_CONV_WB_BATCHED=1' \
      "FR13_FIXED32_CUTLASS_WAVE=$target_selector" \
      "FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=$production" \
      "FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_SOURCE_COMMIT=$target_qualification_source_commit" \
      "FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=$sfwd_fusion"; do
    [[ "$(grep -Fxc "$expected" "$env_path")" -eq 1 ]] \
      || { echo "$arm lacks exact environment pin: $expected" >&2; return 4; }
  done
  unset expected
  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$arm" \
    --out-root "$RUNROOT_ABS/$arm/swe_out" \
    --expected-tok-per-draft 31 \
    --batch-size 1 \
    --out "$RUNROOT_ABS/$arm/deploy_speed_fullwall.json"
  printf 'arm=%s serve_rc=0 container_env_sha256=%s ended=%s\n' \
    "$arm" "$(sha256sum "$env_path" | awk '{print $1}')" \
    "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
}

run_arm "$STOCK_ARM" 0
for absent in \
  "$RUNROOT_ABS/$STOCK_ARM/logs/fr13_dfwd_k64_m1_r64_u8.production_engagement.json" \
  "$RUNROOT_ABS/$STOCK_ARM/logs/fr13_cfwd_logit_direct.production_engagement.json" \
  "$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_taw_native_precompute.production_pass.json" \
  "$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_cutlass_streamk.production_pass.json" \
  "$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_sfwd_conv_postprep.production_pass.json"; do
  [[ ! -e "$absent" && ! -L "$absent" ]] \
    || { echo "stock arm emitted production evidence: $absent" >&2; exit 4; }
done
unset absent
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after stock arm" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" 1
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after candidate arm" >&2; exit 2; }

CANDIDATE_DIR="$RUNROOT_ABS/$CANDIDATE_ARM"
TARGET_PRODUCTION_SIDECAR="$CANDIDATE_DIR/logs/fr13_fixed32_cutlass_streamk.production_pass.json"
TARGET_BINARY_RECORD="$CANDIDATE_DIR/logs/fr13_fixed32_cutlass_streamk_binary.json"
TARGET_SELECTOR_RECORD="$CANDIDATE_DIR/logs/fr13_fixed32_cutlass_wave.selector"
SFWD_PRODUCTION_PASS="$CANDIDATE_DIR/logs/fr13_fixed32_sfwd_conv_postprep.production_pass.json"
SFWD_PRODUCTION_MANIFEST="$CANDIDATE_DIR/logs/fr13_fixed32_sfwd_conv_postprep.source_manifest.json"
DOCKER_LOG="$CANDIDATE_DIR/docker_after_tasks.log"
for artifact in \
  "$TARGET_PRODUCTION_SIDECAR" "$TARGET_BINARY_RECORD" \
  "$TARGET_SELECTOR_RECORD" "$SFWD_PRODUCTION_PASS" \
  "$SFWD_PRODUCTION_MANIFEST" "$DOCKER_LOG"; do
  [[ -f "$artifact" && ! -L "$artifact" && "$(stat -c '%h' "$artifact")" == "1" ]] \
    || { echo "candidate production evidence is missing or unsafe: $artifact" >&2; exit 4; }
done
unset artifact
[[ "$(sha256sum "$SFWD_PRODUCTION_PASS" | awk '{print $1}')" == "$SFWD_CONV_POSTPREP_PASS_SHA256" \
   && "$(sha256sum "$SFWD_PRODUCTION_MANIFEST" | awk '{print $1}')" == "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" \
   && "$(cat "$TARGET_SELECTOR_RECORD")" == "$TARGET_SELECTOR" ]] \
  || { echo "candidate production credential or selector drifted" >&2; exit 4; }

for arm in "$STOCK_ARM" "$CANDIDATE_ARM"; do
  qrow_sidecar="$RUNROOT_ABS/$arm/logs/fr13_fa2_qrow16_production_pass.json"
  [[ -f "$qrow_sidecar" && ! -L "$qrow_sidecar" && "$(stat -c '%h' "$qrow_sidecar")" == "1" ]] \
    || { echo "$arm lacks Qrow16 production evidence" >&2; exit 4; }
  "$PYTHON_BIN" scripts/fr13_qrow16_pass_sidecar.py verify \
    --sidecar "$qrow_sidecar" \
    --expected-sidecar-sha256 "$(sha256sum "$qrow_sidecar" | awk '{print $1}')" \
    --candidate-so "$QROW16_FA2_SO" \
    --expected-candidate-sha256 "$QROW16_SHA256" >/dev/null
done
unset arm qrow_sidecar
for absent in \
  "$CANDIDATE_DIR/logs/fr13_dfwd_k64_m1_r64_u8.production_credential.json" \
  "$CANDIDATE_DIR/logs/fr13_dfwd_k64_m1_r64_u8.production_engagement.json" \
  "$CANDIDATE_DIR/logs/fr13_cfwd_logit_direct.production_pass.json" \
  "$CANDIDATE_DIR/logs/fr13_cfwd_logit_direct.production_engagement.json" \
  "$CANDIDATE_DIR/logs/fr13_fixed32_taw_native_precompute.production_pass.json" \
  "$CANDIDATE_DIR/logs/fr13_fixed32_taw_native_precompute_production.arm"; do
  [[ ! -e "$absent" && ! -L "$absent" ]] \
    || { echo "U8/TAW/CFWD production unexpectedly engaged: $absent" >&2; exit 4; }
done
unset absent
TARGET_PRODUCTION_SIDECAR_SHA256=$(sha256sum "$TARGET_PRODUCTION_SIDECAR" | awk '{print $1}')
"$PYTHON_BIN" scripts/fr13_cutlass_streamk_pass.py verify \
  --sidecar "$TARGET_PRODUCTION_SIDECAR" \
  --expected-sidecar-sha256 "$TARGET_PRODUCTION_SIDECAR_SHA256" \
  --candidate-so "$CUTLASS_TARGET_SO" \
  --patch-source scripts/fr13_patch_cutlass_fixed32_wave.py \
  --candidate-selector "$TARGET_SELECTOR" \
  --qualification-profile k64_root \
  --diagnostic-task-profile astropy12907 \
  --draft-vocab-blocks "$BLOCK_MAP" >/dev/null

ENGAGEMENT_VALIDATION="$RUNROOT_ABS/fullstack_engagement_validation.json"
"$PYTHON_BIN" - \
  "$TARGET_BINARY_RECORD" "$TARGET_PRODUCTION_SIDECAR" \
  "$DOCKER_LOG" "$ENGAGEMENT_VALIDATION" \
  "$SOURCE_COMMIT" "$TARGET_SELECTOR" "$TARGET_SHA256" \
  "$TARGET_QUALIFICATION_SOURCE_COMMIT" \
  "$TARGET_QUALIFICATION_PATCH_SHA256" "$TARGET_PASS_SHA256" <<'PY'
import hashlib
import json
import re
import stat
import sys
from pathlib import Path

def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise SystemExit(f"runtime artifact is not one regular file: {path}")
    raw = path.read_bytes()
    payload = json.loads(
        raw.decode("ascii"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise SystemExit(f"runtime artifact is not one object: {path}")
    return payload, raw


binary_path, sidecar_path, docker_path, out_path = map(Path, sys.argv[1:5])
(
    source_commit,
    target_selector,
    target_sha,
    qualification_commit,
    qualification_patch_sha,
    target_pass_sha,
) = sys.argv[5:11]
binary, binary_raw = load(binary_path)
sidecar, sidecar_raw = load(sidecar_path)
docker_info = docker_path.lstat()
if docker_path.is_symlink() or not stat.S_ISREG(docker_info.st_mode):
    raise SystemExit("Docker log is not a regular file")
docker_raw = docker_path.read_bytes()
docker_text = docker_raw.decode("utf-8", errors="replace")
qualification = binary.get("qualification")
if (
    binary.get("schema") != "fr13.fixed32.cutlass_streamk_binary.v2"
    or binary.get("selector") != target_selector
    or binary.get("production_enabled") is not True
    or binary.get("qualification_profile") != "k64_root"
    or not isinstance(binary.get("source"), dict)
    or binary["source"].get("sha256") != target_sha
    or not isinstance(binary.get("destination"), dict)
    or binary["destination"].get("sha256") != target_sha
    or binary.get("installed_mode") != "0555"
    or not isinstance(qualification, dict)
    or qualification.get("qualification_source_mode") != "historical_pinned"
    or qualification.get("qualification_source_commit") != qualification_commit
    or qualification.get("runtime_source_commit") != source_commit
    or qualification.get("patch_source_sha256") != qualification_patch_sha
    or sidecar.get("candidate_selector") != target_selector
    or sidecar.get("candidate_sha256") != target_sha
    or sidecar.get("live_result_sha256") != target_pass_sha
    or sidecar.get("qualification_source_mode") != "historical_pinned"
    or sidecar.get("qualification_source_commit") != qualification_commit
    or sidecar.get("runtime_source_commit") != source_commit
    or sidecar.get("patch_source_sha256") != qualification_patch_sha
):
    raise SystemExit("current target installation evidence drifted")
layers = set(
    re.findall(
        r"\[FR13_SFWD_CONV_POSTPREP\] production engaged "
        r"layer=([^ ]+) B=1 rows=32",
        docker_text,
    )
)
marker = "[FR13_SFWD_CONV_POSTPREP] production engaged layer="
if len(layers) != 48 or docker_text.count(marker) != 48:
    raise SystemExit("SFWD conv/post-prep did not engage exactly 48 layers")
for forbidden in (
    "linear fallback engaged",
    "FR13 SFWD conv/post-prep capture lacks preseeded output bindings",
    "FR13 SFWD conv/post-prep output preseed ran during capture",
    "FR13 SFWD conv/post-prep profile output preseed ran during capture",
    "FR13 SFWD conv/post-prep profile output preseed is incomplete",
    "FR13 SFWD conv/post-prep graph output preseed is incomplete",
    "[FR13_STEP_GRAPH] S1 DISABLED",
):
    if forbidden in docker_text:
        raise SystemExit(f"candidate runtime emitted forbidden fallback: {forbidden}")
payload = {
    "schema": "fr13.fixed32.b1_target_sfwd_exact4.engagement_validation.v1",
    "status": "PASS",
    "source_commit": source_commit,
    "target_binary_record_sha256": hashlib.sha256(binary_raw).hexdigest(),
    "target_production_sidecar_sha256": hashlib.sha256(sidecar_raw).hexdigest(),
    "target_selector_engaged": True,
    "target_production_attested": True,
    "target_qualification_source_commit": qualification_commit,
    "docker_log_sha256": hashlib.sha256(docker_raw).hexdigest(),
    "sfwd_engaged_layers": 48,
    "sfwd_no_fallback": True,
    "u8_production_enabled": False,
    "taw_production_enabled": False,
    "cfwd_production_enabled": False,
}
out_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
PY

finalize_manifests

"$PYTHON_BIN" - \
  "$SUBSET" \
  "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  "$ENGAGEMENT_VALIDATION" \
  "$RUNROOT_ABS/timing_summary.json" \
  "$TASK_SET" "$EXPECTED_TASKS" "$STOCK_ARM" "$CANDIDATE_ARM" \
  "$SOURCE_COMMIT" "$RUNNER_SHA256" "$SUBSET_SHA256" \
  "$QROW16_SHA256" "$QROW16_PASS_SHA256" \
  "$TARGET_SELECTOR" "$TARGET_SHA256" "$TARGET_PASS_SHA256" \
  "$TARGET_QUALIFICATION_SOURCE_COMMIT" "$TARGET_QUALIFICATION_PATCH_SHA256" \
  "$SFWD_CONV_POSTPREP_PASS_SHA256" \
  "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" \
  "$SFWD_CONV_POSTPREP_GATE_SUMMARY_SHA256" \
  "$WEIGHT_FLOOR_MS" "$ONE_SIDED_U95_CAP_MS" <<'PY'
import hashlib
import json
import math
import stat
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "scripts"))
from fr13_b4_timing_math import phase_breakdown


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise SystemExit(f"timing artifact is not one regular file: {path}")
    raw = path.read_bytes()
    payload = json.loads(
        raw.decode("ascii"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise SystemExit(f"timing artifact is not one object: {path}")
    return payload, raw


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def positive(payload, key, label):
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{label} lacks numeric {key}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"{label} lacks positive finite {key}")
    return value


paths = list(map(Path, sys.argv[1:6]))
subset_path, stock_path, candidate_path, stack_path, out_path = paths
task_set, expected_tasks, stock_arm, candidate_arm = sys.argv[6:10]
source_commit, runner_sha, subset_sha = sys.argv[10:13]
qrow16_sha, qrow16_pass_sha = sys.argv[13:15]
target_selector, target_sha, target_pass_sha = sys.argv[15:18]
qualification_commit, qualification_patch_sha = sys.argv[18:20]
sfwd_pass_sha, sfwd_manifest_sha, sfwd_gate_sha = sys.argv[20:23]
floor_ms, cap_ms = map(float, sys.argv[23:25])
subset, subset_raw = load(subset_path)
stock, stock_raw = load(stock_path)
candidate, candidate_raw = load(candidate_path)
stack_validation, stack_raw = load(stack_path)
task_ids = subset.get("instance_ids")
expected_task_ids = [
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
]
if (
    digest(subset_raw) != subset_sha
    or not isinstance(task_ids, list)
    or len(task_ids) != int(expected_tasks)
    or task_ids != expected_task_ids
    or stack_validation.get("status") != "PASS"
    or stack_validation.get("source_commit") != source_commit
    or stack_validation.get("sfwd_engaged_layers") != 48
    or stack_validation.get("sfwd_no_fallback") is not True
    or stack_validation.get("target_selector_engaged") is not True
    or stack_validation.get("target_production_attested") is not True
    or stack_validation.get("target_qualification_source_commit")
    != qualification_commit
):
    raise SystemExit("full-stack timing provenance drifted")


def validate_measure(payload, raw, arm):
    if (
        payload.get("schema") != "fr13.measure.deploy_speed.v1"
        or payload.get("kind") != "speed"
        or payload.get("instrument") != "OFF"
        or payload.get("regime") != "deployment"
        or payload.get("arm") != arm
        or payload.get("batch_size") != 1
        or payload.get("n_tasks") != int(expected_tasks)
        or payload.get("task_instance_ids") != task_ids
        or payload.get("draft_vocab_root") != 1
        or payload.get("draft_vocab_k") != 65536
        or payload.get("mandatory_weight_bytes") != 32666638208
        or payload.get("weight_floor_ms") != floor_ms
        or payload.get("floor_ms") != floor_ms
        or payload.get("floor_is_full_step_hardware_floor") is not False
    ):
        raise SystemExit(f"{arm} is not canonical {task_set} K64/root1 B1 timing")
    per_task = payload.get("per_task")
    if (
        not isinstance(per_task, list)
        or len(per_task) != int(expected_tasks)
        or [row.get("instance_id") for row in per_task] != task_ids
        or any(positive(row, "wall_steps", arm + ":task") < 1 for row in per_task)
    ):
        raise SystemExit(f"{arm} lacks complete per-task timing windows")
    phases = phase_breakdown(payload, arm)
    rows_per_step = positive(payload, "rows_per_step", arm)
    if not math.isclose(rows_per_step, 32.0, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(f"{arm} physical row count drifted")
    return {
        "deploy_speed_sha256": digest(raw),
        "step_wall_ms": positive(payload, "step_wall_ms", arm),
        "measured_tps_fullstep_wall": positive(
            payload, "measured_tps_fullstep_wall", arm
        ),
        "accepted_drafts_per_event": positive(payload, "accept_per_event", arm),
        "committed_tokens_per_event": positive(payload, "committed_per_event", arm),
        "events_per_step": positive(payload, "events_per_step", arm),
        "rows_per_step": rows_per_step,
        "sfwd_gpu_ms_per_step": phases["sfwd_gpu_ms_per_step"],
        "dfwd_gpu_ms_per_step": phases["dfwd_gpu_ms_per_step"],
        "cfwd_gpu_ms_per_step": phases["cfwd_gpu_ms_per_step"],
        "gpu_component_ms_per_step": phases["gpu_component_ms_per_step"],
        "other_wall_ms_per_step": phases["other_wall_ms_per_step"],
    }


s = validate_measure(stock, stock_raw, stock_arm)
c = validate_measure(candidate, candidate_raw, candidate_arm)
summary = {
    "schema": "fr13.fixed32.b1_target_sfwd_exact4_timing.v1",
    "status": "MEASURED",
    "classification": "real_swe_verified_b1_target_sfwd_exact4_timing_pair",
    "task_set": task_set,
    "task_count": int(expected_tasks),
    "task_ids": task_ids,
    "batch_size": 1,
    "concurrency": 1,
    "mode": "hydra27_fixed32",
    "physical_rows": 32,
    "draft_vocab_root": 1,
    "draft_vocab_k": 65536,
    "runtime": "FULL_AND_PIECEWISE",
    "timing_eligible": True,
    "floor_acceptance_eligible": False,
    "production_default_enabled": False,
    "performance_claim": False,
    "source_commit": source_commit,
    "runner_sha256": runner_sha,
    "subset_sha256": subset_sha,
    "component_credentials": {
        "qrow16_fa2_sha256": qrow16_sha,
        "qrow16_live_pass_sha256": qrow16_pass_sha,
        "target_sfwd_engagement_validation_sha256": digest(stack_raw),
        "target_selector": target_selector,
        "target_so_sha256": target_sha,
        "target_live_pass_sha256": target_pass_sha,
        "target_qualification_source_commit": qualification_commit,
        "target_qualification_patch_sha256": qualification_patch_sha,
        "sfwd_live_pass_sha256": sfwd_pass_sha,
        "sfwd_source_manifest_sha256": sfwd_manifest_sha,
        "sfwd_gate_summary_sha256": sfwd_gate_sha,
    },
    "production_selectors": {
        "qrow16": True,
        "target": True,
        "sfwd_conv_postprep": True,
        "u8": False,
        "taw": False,
        "cfwd": False,
    },
    "stock": {"arm": stock_arm, **s},
    "candidate": {"arm": candidate_arm, **c},
    "delta": {
        "step_wall_ms": c["step_wall_ms"] - s["step_wall_ms"],
        "full_wall_tps": (
            c["measured_tps_fullstep_wall"] - s["measured_tps_fullstep_wall"]
        ),
        "accepted_drafts_per_event": (
            c["accepted_drafts_per_event"] - s["accepted_drafts_per_event"]
        ),
    },
    "ratios": {
        "candidate_to_stock_step_wall": c["step_wall_ms"] / s["step_wall_ms"],
        "candidate_to_stock_full_wall_tps": (
            c["measured_tps_fullstep_wall"]
            / s["measured_tps_fullstep_wall"]
        ),
        "candidate_to_weight_floor": c["step_wall_ms"] / floor_ms,
    },
    "one_sided_u95_cap_ms": cap_ms,
    "candidate_gap_to_cap_ms": c["step_wall_ms"] - cap_ms,
}
out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY

printf 'summary=%s completed=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
trap - EXIT
echo "B1 Qrow16/target/SFWD timing completed: $RUNROOT_ABS/timing_summary.json"

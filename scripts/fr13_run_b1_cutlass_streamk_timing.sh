#!/usr/bin/env bash
# Real SWE-Verified Hydra27/Qrow16 B1 timing: stock CUTLASS vs Stream-K.
# Exact4 is the default timing arm. The explicit one-task mode is diagnostic only.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${QROW16_FA2_SO:?set QROW16_FA2_SO to the pinned Qrow16 candidate binary}"
: "${CUTLASS_STREAMK_SO:?set CUTLASS_STREAMK_SO to the pinned Stream-K binary}"
: "${STREAMK_PASS_JSON:?set STREAMK_PASS_JSON to a completed authenticated B1 byte-gate PASS}"
: "${STREAMK_PASS_SHA256:?set STREAMK_PASS_SHA256 to its raw SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
TIMING_CANDIDATE=${FR13_STREAMK_TIMING_CANDIDATE:-streamk_coop128}
TIMING_TASK_SET=${FR13_STREAMK_TIMING_TASK_SET:-exact4}
case "$TIMING_CANDIDATE" in
  streamk_coop128)
    STREAMK_SHA256=f9bbbb8dc4ffc2227a71d2bc7b260e586ffbdc0fd946749e4f69e322c46a362d
    STREAMK_BYTES=111417328
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_streamk_live_gate.v3
    K64_ROOT_LIVE_SCHEMA=
    CANDIDATE_ARM_LABEL=cutlass_streamk
    ;;
  streamk_force_wide256)
    STREAMK_SHA256=503277a2dca6784502b709007adfe45f42d0f1a1851107e7b913e1e85a00de5a
    STREAMK_BYTES=113079680
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_streamk_wide256_live_gate.v1
    K64_ROOT_LIVE_SCHEMA=fr13.fixed32.cutlass_streamk_wide256_k64_root_live_gate.v1
    CANDIDATE_ARM_LABEL=cutlass_streamk_force_wide256
    ;;
  identity_onen_b1)
    STREAMK_SHA256=17af1975b1e26cd3d4c3e614bfcab8aa1b0dc031ea5107004b0cc25890fc2b15
    STREAMK_BYTES=118166088
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_onen_b1_live_gate.v1
    K64_ROOT_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_onen_b1_k64_root_live_gate.v1
    CANDIDATE_ARM_LABEL=cutlass_identity_onen_b1
    ;;
  *)
    echo "unsupported Stream-K timing candidate: $TIMING_CANDIDATE" >&2
    exit 2
    ;;
esac
TIMING_PROFILE_EXPLICIT=0
if [[ -v FR13_STREAMK_TIMING_PROFILE ]]; then
  TIMING_PROFILE_EXPLICIT=1
fi
TIMING_PROFILE=${FR13_STREAMK_TIMING_PROFILE:-full_vocab}
if [[ "$TIMING_CANDIDATE" == "identity_onen_b1" \
      && ( "$TIMING_PROFILE_EXPLICIT" != "1" \
           || "$TIMING_PROFILE" != "k64_root" ) ]]; then
  echo "identity_onen_b1 timing requires explicit k64_root qualification" >&2
  exit 2
fi
case "$TIMING_TASK_SET" in
  exact4)
    SUBSET=config/fr13_fixed32/subset_b4_four.json
    SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
    TASK_COUNT=4
    B1_DIAGNOSTIC=0
    TIMING_ELIGIBLE=1
    RUN_CLASSIFICATION_BASE=real_swe_verified_exact4_b1_hydra27_qrow16_streamk_timing
    ;;
  one)
    SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
    SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
    TASK_COUNT=1
    B1_DIAGNOSTIC=1
    TIMING_ELIGIBLE=0
    RUN_CLASSIFICATION_BASE=one_real_swe_verified_b1_hydra27_qrow16_streamk_timing_diagnostic
    ;;
  *)
    echo "FR13_STREAMK_TIMING_TASK_SET must be exact4 or one" >&2
    exit 2
    ;;
esac
DRAFT_VOCAB_BLOCKS_HOST=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
case "$TIMING_PROFILE" in
  full_vocab)
    [[ -n "$FULL_VOCAB_LIVE_SCHEMA" ]] \
      || { echo "$TIMING_CANDIDATE does not support full_vocab timing" >&2; exit 2; }
    STREAMK_LIVE_SCHEMA=$FULL_VOCAB_LIVE_SCHEMA
    DRAFT_VOCAB_ROOT=0
    DRAFT_VOCAB_K=0
    NEEDS_ALLOW=FR13_DRAFT_VOCAB_K=0
    MANDATORY_WEIGHT_BYTES=42025179008
    MANDATORY_WEIGHT_FLOOR_MS=153.9383846446886
    ONE_SIDED_U95_CAP_MS=177.0291423413919
    ;;
  k64_root)
    [[ -n "$K64_ROOT_LIVE_SCHEMA" ]] \
      || { echo "$TIMING_CANDIDATE does not support k64_root timing" >&2; exit 2; }
    STREAMK_LIVE_SCHEMA=$K64_ROOT_LIVE_SCHEMA
    DRAFT_VOCAB_ROOT=1
    DRAFT_VOCAB_K=65536
    NEEDS_ALLOW=
    MANDATORY_WEIGHT_BYTES=32666638208
    MANDATORY_WEIGHT_FLOOR_MS=119.658015414
    ONE_SIDED_U95_CAP_MS=137.6067177261
    ;;
  *)
    echo "FR13_STREAMK_TIMING_PROFILE must be full_vocab or k64_root" >&2
    exit 2
    ;;
esac
if [[ "$TIMING_PROFILE" == "full_vocab" ]]; then
  RUN_CLASSIFICATION="${RUN_CLASSIFICATION_BASE}_candidate"
else
  RUN_CLASSIFICATION="${RUN_CLASSIFICATION_BASE}_${TIMING_PROFILE}_candidate"
fi
QROW16_FA2_SHA256=1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86
QROW16_FA2_BYTES=299507792
QROW16_LIVE_PASS_JSON="$REPO/results/fr13_fixed32_qrow16_num_splits0_live_pass_20260731T173608Z/fr13_fa2_qrow16_live_paged_ab.json"
QROW16_LIVE_PASS_SHA256=36940fd43d11399529d1bfe7e11baa9961907193267f3bb43d41057328737b77
PATCH_SOURCE=scripts/fr13_patch_cutlass_fixed32_wave.py
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
STOCK_ARM="hydra27_fixed32_qrow16_cutlass_stock_${TAG}"
CANDIDATE_ARM="hydra27_fixed32_qrow16_${CANDIDATE_ARM_LABEL}_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for _fr13_streamk_file in \
    "$QROW16_FA2_SO" "$QROW16_LIVE_PASS_JSON" \
    "$CUTLASS_STREAMK_SO" "$STREAMK_PASS_JSON"; do
  [[ "$_fr13_streamk_file" == /* \
     && -f "$_fr13_streamk_file" \
     && ! -L "$_fr13_streamk_file" ]] \
    || { echo "required input must be an absolute regular non-symlink file: $_fr13_streamk_file" >&2; exit 2; }
done
unset _fr13_streamk_file
[[ "$(stat -c '%s' "$QROW16_FA2_SO")" == "$QROW16_FA2_BYTES" \
   && "$(sha256sum "$QROW16_FA2_SO" | awk '{print $1}')" == "$QROW16_FA2_SHA256" ]] \
  || { echo "QROW16_FA2_SO is not the pinned production candidate" >&2; exit 2; }
[[ "$(sha256sum "$QROW16_LIVE_PASS_JSON" | awk '{print $1}')" == "$QROW16_LIVE_PASS_SHA256" ]] \
  || { echo "canonical Qrow16 live PASS SHA-256 drift" >&2; exit 2; }
[[ "$(stat -c '%s' "$CUTLASS_STREAMK_SO")" == "$STREAMK_BYTES" \
   && "$(sha256sum "$CUTLASS_STREAMK_SO" | awk '{print $1}')" == "$STREAMK_SHA256" ]] \
  || { echo "CUTLASS_STREAMK_SO is not the pinned current candidate" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical timing subset SHA-256 drift" >&2; exit 2; }
if [[ "$TIMING_PROFILE" == "k64_root" ]]; then
  [[ -f "$DRAFT_VOCAB_BLOCKS_HOST" \
     && ! -L "$DRAFT_VOCAB_BLOCKS_HOST" \
     && "$(sha256sum "$DRAFT_VOCAB_BLOCKS_HOST" | awk '{print $1}')" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
    || { echo "pinned root-64K draft-vocabulary block map drifted" >&2; exit 2; }
fi
[[ "$STREAMK_PASS_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "STREAMK_PASS_SHA256 must be lowercase SHA-256" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "current source identity is invalid" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }

# Parse and validate the pinned Qrow16 PASS before the first Docker query.
"$PYTHON_BIN" - "$QROW16_LIVE_PASS_JSON" "$QROW16_FA2_SHA256" <<'PY'
import sys
from pathlib import Path

from scripts import fr13_qrow16_pass_sidecar as qrow

payload, _ = qrow.load_json(Path(sys.argv[1]))
qrow.validate_live_result(payload, candidate_sha256=sys.argv[2])
PY

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT"
export FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K"
export FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER"
export FR13_NEEDS_ALLOW="$NEEDS_ALLOW"
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_DRAFT_VOCAB_ROOT" == "$DRAFT_VOCAB_ROOT" \
   && "$FR13_DRAFT_VOCAB_K" == "$DRAFT_VOCAB_K" \
   && "$FR13_NEEDS_ALLOW" == "$NEEDS_ALLOW" \
   && "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" ]] \
  || { echo "$TIMING_PROFILE B1 timing contract drifted" >&2; exit 2; }

# Validate the authenticated real-task comparator credential, including exact
# source identity, before the first Docker query. The comparator contributes no
# timing samples; only the two paired timing arms below do.
"$PYTHON_BIN" scripts/fr13_cutlass_streamk_pass.py validate \
  --live-result "$STREAMK_PASS_JSON" \
  --expected-live-sha256 "$STREAMK_PASS_SHA256" \
  --candidate-so "$CUTLASS_STREAMK_SO" \
  --patch-source "$PATCH_SOURCE" \
  --expected-source-commit "$SOURCE_COMMIT" \
  --candidate-selector "$TIMING_CANDIDATE" \
  --qualification-profile "$TIMING_PROFILE" \
  --draft-vocab-blocks "$DRAFT_VOCAB_BLOCKS_HOST" \
  >/dev/null
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

printf 'classification=%s\ntask_set=%s\ntask_count=%s\ntiming_eligible=%s\ncomparator_gate_timing_eligible=0\nfloor_acceptance_eligible=0\nproduction_default_enabled=0\ntopology=hydra27_fixed32\nlineage=successor_to_legacy_hydra23_not_same_topology\ncommon_fa2_selector=qrow16_production\nonly_arm_delta=CUTLASS_stock_to_%s\ncandidate_selector=%s\ncandidate_live_pass_schema=%s\nqualification_profile=%s\nphysical_rows=32\ndraft_vocab_root=%s\ndraft_vocab_k=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nlauncher_pid=%s\nrunroot=%s\nstock_arm=%s\ncandidate_arm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nqrow16_fa2_sha256=%s\nqrow16_fa2_bytes=%s\nqrow16_live_pass_sha256=%s\nstreamk_sha256=%s\nstreamk_bytes=%s\nstreamk_pass_sha256=%s\nstarted=%s\n' \
  "$RUN_CLASSIFICATION" "$TIMING_TASK_SET" "$TASK_COUNT" "$TIMING_ELIGIBLE" \
  "$TIMING_CANDIDATE" "$TIMING_CANDIDATE" "$STREAMK_LIVE_SCHEMA" \
  "$TIMING_PROFILE" "$DRAFT_VOCAB_ROOT" "$DRAFT_VOCAB_K" \
  "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" "$ONE_SIDED_U95_CAP_MS" \
  "$$" "$RUNROOT_ABS" "$STOCK_ARM" "$CANDIDATE_ARM" \
  "$SOURCE_COMMIT" "$RUNNER_SHA256" "$SUBSET_SHA256" \
  "$QROW16_FA2_SHA256" "$QROW16_FA2_BYTES" "$QROW16_LIVE_PASS_SHA256" \
  "$STREAMK_SHA256" "$STREAMK_BYTES" "$STREAMK_PASS_SHA256" \
  "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"

MANIFEST_FINALIZED=0
finalize_manifests() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || return $?
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json" \
    || return $?
  cmp -s \
    "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || { echo "runtime/source manifest changed during timing" >&2; return 14; }
  cmp -s \
    "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
    || { echo "Stream-K timing runner changed during execution" >&2; return 14; }
  MANIFEST_FINALIZED=1
}

runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    if finalize_manifests; then
      :
    else
      local manifest_rc=$?
      (( rc == 0 )) && rc=$manifest_rc
    fi
  fi
  exit "$rc"
}
trap runner_exit EXIT

run_arm() {
  local arm=$1
  local production=$2
  local selector=stock
  local candidate_so=""
  local pass_json=""
  local pass_sha=""
  if [[ "$production" == "1" ]]; then
    selector=$TIMING_CANDIDATE
    candidate_so=$CUTLASS_STREAMK_SO
    pass_json=$STREAMK_PASS_JSON
    pass_sha=$STREAMK_PASS_SHA256
  fi
  echo "===== $arm: real $TIMING_TASK_SET B1 Hydra27/Qrow16 Stream-K production=$production ====="
  if env \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
      FR13_FIXED32_B1_DIAGNOSTIC="$B1_DIAGNOSTIC" \
      FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT" \
      FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K" \
      FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER" \
      FR13_NEEDS_ALLOW="$NEEDS_ALLOW" \
      FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES" \
      FR13_WEIGHT_FLOOR_MS="$MANDATORY_WEIGHT_FLOOR_MS" \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json" \
      FR13_FIXED32_CUTLASS_WAVE="$selector" \
      FR13_FIXED32_CUTLASS_WAVE_SO="$candidate_so" \
      FR13_FIXED32_CUTLASS_WAVE_PRODUCTION="$production" \
      FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE="$TIMING_PROFILE" \
      FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_JSON="$pass_json" \
      FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_SHA256="$pass_sha" \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 \
      FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_INSTANCE_ID= \
      FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 \
      FR13_FA2_QROW16_SO_SHA256="$QROW16_FA2_SHA256" \
      FR13_FA2_QROW16_PRODUCTION=1 \
      FR13_FA2_QROW16_LIVE_PASS_JSON="$QROW16_LIVE_PASS_JSON" \
      FR13_FA2_QROW16_LIVE_PASS_SHA256="$QROW16_LIVE_PASS_SHA256" \
      FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
      FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
      FR13_FIXED32_BATCH_GDN_GRAPH_LIVE_PASS_JSON= \
      FR13_FIXED32_BATCH_GDN_GRAPH_LIVE_PASS_SHA256= \
      FR13_FIXED32_BATCH_GDN_GRAPH_GATE_VERDICT_JSON= \
      FR13_FIXED32_BATCH_GDN_GRAPH_GATE_VERDICT_SHA256= \
      FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
      FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$QROW16_FA2_SO" \
      RUNROOT="$RUNROOT_ABS" \
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
  local container_env="$RUNROOT_ABS/$arm/container_env.txt"
  [[ -f "$container_env" && ! -L "$container_env" ]] \
    || { echo "$arm lacks a regular container environment artifact" >&2; return 4; }
  [[ "$(grep -Fxc 'FR13_FIXED32_MODE=hydra27_fixed32' "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_DRAFT_VOCAB_ROOT=$DRAFT_VOCAB_ROOT" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_DRAFT_VOCAB_K=$DRAFT_VOCAB_K" "$container_env")" -eq 1 \
     && "$(grep -Fxc 'FR13_FA2_QROW16_LIVE_PAGED_AB=0' "$container_env")" -eq 1 \
     && "$(grep -Fxc 'FR13_FA2_QROW16_PRODUCTION=1' "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW16_SO_SHA256=$QROW16_FA2_SHA256" "$container_env")" -eq 1 ]] \
    || { echo "$arm did not run Hydra27/$TIMING_PROFILE/Qrow16 production" >&2; return 4; }
  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$arm" \
    --out-root "$RUNROOT_ABS/$arm/swe_out" \
    --expected-tok-per-draft 31 \
    --batch-size 1 \
    --out "$RUNROOT_ABS/$arm/deploy_speed_fullwall.json"
  local qrow16_sidecar="$RUNROOT_ABS/$arm/logs/fr13_fa2_qrow16_production_pass.json"
  local qrow16_capture="$RUNROOT_ABS/$arm/logs/fr13_fa2_qrow16_production_capture.json"
  [[ -f "$qrow16_sidecar" && ! -L "$qrow16_sidecar" \
     && -f "$qrow16_capture" && ! -L "$qrow16_capture" ]] \
    || { echo "$arm lacks regular Qrow16 production evidence" >&2; return 4; }
  local qrow16_sidecar_sha256
  qrow16_sidecar_sha256=$(sha256sum "$qrow16_sidecar" | awk '{print $1}')
  "$PYTHON_BIN" scripts/fr13_qrow16_pass_sidecar.py verify \
    --sidecar "$qrow16_sidecar" \
    --expected-sidecar-sha256 "$qrow16_sidecar_sha256" \
    --candidate-so "$QROW16_FA2_SO" \
    --expected-candidate-sha256 "$QROW16_FA2_SHA256" \
    >/dev/null
  printf 'arm=%s serve_rc=0 container_env_sha256=%s qrow16_sidecar_sha256=%s qrow16_capture_sha256=%s ended=%s\n' \
    "$arm" "$(sha256sum "$container_env" | awk '{print $1}')" \
    "$qrow16_sidecar_sha256" \
    "$(sha256sum "$qrow16_capture" | awk '{print $1}')" \
    "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
}

run_arm "$STOCK_ARM" 0
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after the stock reference" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" 1

STOCK_ATTESTATION="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_cutlass_streamk_binary.json"
CANDIDATE_ATTESTATION="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_cutlass_streamk_binary.json"
CANDIDATE_SIDECAR="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_cutlass_streamk.production_pass.json"
CANDIDATE_SELECTOR="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_cutlass_wave.selector"
STOCK_QROW16_SIDECAR="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fa2_qrow16_production_pass.json"
CANDIDATE_QROW16_SIDECAR="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fa2_qrow16_production_pass.json"
STOCK_QROW16_CAPTURE="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fa2_qrow16_production_capture.json"
CANDIDATE_QROW16_CAPTURE="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fa2_qrow16_production_capture.json"
[[ ! -e "$STOCK_ATTESTATION" && ! -L "$STOCK_ATTESTATION" ]] \
  || { echo "stock arm emitted a Stream-K binary attestation" >&2; exit 4; }
[[ -f "$CANDIDATE_ATTESTATION" && ! -L "$CANDIDATE_ATTESTATION" \
   && -f "$CANDIDATE_SIDECAR" && ! -L "$CANDIDATE_SIDECAR" \
   && -f "$CANDIDATE_SELECTOR" && ! -L "$CANDIDATE_SELECTOR" ]] \
  || { echo "candidate arm lacks regular Stream-K identity artifacts" >&2; exit 4; }
[[ "$(<"$CANDIDATE_SELECTOR")" == "$TIMING_CANDIDATE" ]] \
  || { echo "candidate selector sidecar is missing or wrong" >&2; exit 4; }
CANDIDATE_SIDECAR_SHA256=$(sha256sum "$CANDIDATE_SIDECAR" | awk '{print $1}')
"$PYTHON_BIN" scripts/fr13_cutlass_streamk_pass.py verify \
  --sidecar "$CANDIDATE_SIDECAR" \
  --expected-sidecar-sha256 "$CANDIDATE_SIDECAR_SHA256" \
  --candidate-so "$CUTLASS_STREAMK_SO" \
  --patch-source "$PATCH_SOURCE" \
  --candidate-selector "$TIMING_CANDIDATE" \
  --qualification-profile "$TIMING_PROFILE" \
  --draft-vocab-blocks "$DRAFT_VOCAB_BLOCKS_HOST" \
  >/dev/null
"$PYTHON_BIN" scripts/fr13_cutlass_streamk_pass.py attestation \
  --attestation "$CANDIDATE_ATTESTATION" \
  --expected-sidecar-sha256 "$CANDIDATE_SIDECAR_SHA256" \
  --qualification-profile "$TIMING_PROFILE" \
  --draft-vocab-blocks "$DRAFT_VOCAB_BLOCKS_HOST" \
  > "$RUNROOT_ABS/$CANDIDATE_ARM/streamk_production_binding.json"

finalize_manifests

"$PYTHON_BIN" scripts/fr13_cutlass_streamk_timing.py \
  --subset "$SUBSET" \
  --stock-measure "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  --candidate-measure "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  --stock-container-env "$RUNROOT_ABS/$STOCK_ARM/container_env.txt" \
  --candidate-container-env "$RUNROOT_ABS/$CANDIDATE_ARM/container_env.txt" \
  --stock-qrow16-sidecar "$STOCK_QROW16_SIDECAR" \
  --candidate-qrow16-sidecar "$CANDIDATE_QROW16_SIDECAR" \
  --stock-qrow16-capture "$STOCK_QROW16_CAPTURE" \
  --candidate-qrow16-capture "$CANDIDATE_QROW16_CAPTURE" \
  --qrow16-so "$QROW16_FA2_SO" \
  --production-binding "$RUNROOT_ABS/$CANDIDATE_ARM/streamk_production_binding.json" \
  --candidate-so "$CUTLASS_STREAMK_SO" \
  --source-commit "$SOURCE_COMMIT" \
  --candidate-selector "$TIMING_CANDIDATE" \
  --qualification-profile "$TIMING_PROFILE" \
  --task-set "$TIMING_TASK_SET" \
  --out "$RUNROOT_ABS/timing_summary.json"

printf 'timing_summary=%s completed=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"

#!/usr/bin/env bash
# Exact4 real SWE-Verified B1 full-wall timing pair: stock CUTLASS vs Stream-K.
# The one-task comparator gate only authorizes the candidate; it is not timing.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${STOCK_FA2_SO:?set STOCK_FA2_SO to the exact-safe stock FA2 binary}"
: "${CUTLASS_STREAMK_SO:?set CUTLASS_STREAMK_SO to the pinned Stream-K binary}"
: "${STREAMK_PASS_JSON:?set STREAMK_PASS_JSON to a completed authenticated B1 byte-gate PASS}"
: "${STREAMK_PASS_SHA256:?set STREAMK_PASS_SHA256 to its raw SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STREAMK_SHA256=f9bbbb8dc4ffc2227a71d2bc7b260e586ffbdc0fd946749e4f69e322c46a362d
FULL_VOCAB_WEIGHT_BYTES=42025179008
FULL_VOCAB_FLOOR_MS=153.9383846446886
FULL_VOCAB_CAP_MS=177.0291423413919
PATCH_SOURCE=scripts/fr13_patch_cutlass_fixed32_wave.py
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
STOCK_ARM="tail6_fixed32_cutlass_stock_${TAG}"
CANDIDATE_ARM="tail6_fixed32_cutlass_streamk_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for _fr13_streamk_file in \
    "$STOCK_FA2_SO" "$CUTLASS_STREAMK_SO" "$STREAMK_PASS_JSON"; do
  [[ "$_fr13_streamk_file" == /* \
     && -f "$_fr13_streamk_file" \
     && ! -L "$_fr13_streamk_file" ]] \
    || { echo "required input must be an absolute regular non-symlink file: $_fr13_streamk_file" >&2; exit 2; }
done
unset _fr13_streamk_file
[[ "$(sha256sum "$STOCK_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "STOCK_FA2_SO is not the exact-safe stock reference" >&2; exit 2; }
[[ "$(sha256sum "$CUTLASS_STREAMK_SO" | awk '{print $1}')" == "$STREAMK_SHA256" ]] \
  || { echo "CUTLASS_STREAMK_SO is not the pinned current candidate" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical exact4 subset SHA-256 drift" >&2; exit 2; }
[[ "$STREAMK_PASS_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "STREAMK_PASS_SHA256 must be lowercase SHA-256" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "current source identity is invalid" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=0
export FR13_DRAFT_VOCAB_K=0
export FR13_NEEDS_ALLOW='FR13_DRAFT_VOCAB_K=0'
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_DRAFT_VOCAB_ROOT" == "0" \
   && "$FR13_DRAFT_VOCAB_K" == "0" \
   && "$FR13_NEEDS_ALLOW" == "FR13_DRAFT_VOCAB_K=0" \
   && "$FR13_MANDATORY_WEIGHT_BYTES" == "$FULL_VOCAB_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$FULL_VOCAB_FLOOR_MS" ]] \
  || { echo "full-vocabulary B1 timing contract drifted" >&2; exit 2; }

# Validate the authenticated real-task comparator credential, including exact
# source identity, before the first Docker query. The comparator contributes no
# timing samples; only the two exact4 arms below do.
"$PYTHON_BIN" scripts/fr13_cutlass_streamk_pass.py validate \
  --live-result "$STREAMK_PASS_JSON" \
  --expected-live-sha256 "$STREAMK_PASS_SHA256" \
  --candidate-so "$CUTLASS_STREAMK_SO" \
  --patch-source "$PATCH_SOURCE" \
  --expected-source-commit "$SOURCE_COMMIT" \
  >/dev/null
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

printf 'classification=real_swe_verified_exact4_b1_timing_candidate\ntiming_eligible=1\ncomparator_gate_timing_eligible=0\nfloor_acceptance_eligible=0\nproduction_default_enabled=0\nphysical_rows=32\ndraft_vocab_root=0\ndraft_vocab_k=0\nfr13_needs_allow=FR13_DRAFT_VOCAB_K=0\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nlauncher_pid=%s\nrunroot=%s\nstock_arm=%s\ncandidate_arm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nstock_fa2_sha256=%s\nstreamk_sha256=%s\nstreamk_pass_sha256=%s\nstarted=%s\n' \
  "$FULL_VOCAB_WEIGHT_BYTES" "$FULL_VOCAB_FLOOR_MS" "$FULL_VOCAB_CAP_MS" \
  "$$" "$RUNROOT_ABS" "$STOCK_ARM" "$CANDIDATE_ARM" \
  "$SOURCE_COMMIT" "$RUNNER_SHA256" "$SUBSET_SHA256" \
  "$STOCK_FA2_SHA256" "$STREAMK_SHA256" "$STREAMK_PASS_SHA256" \
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
    selector=streamk_coop128
    candidate_so=$CUTLASS_STREAMK_SO
    pass_json=$STREAMK_PASS_JSON
    pass_sha=$STREAMK_PASS_SHA256
  fi
  echo "===== $arm: real exact4 B1 Stream-K production=$production ====="
  if env \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR13_DRAFT_VOCAB_ROOT=0 FR13_DRAFT_VOCAB_K=0 \
      FR13_NEEDS_ALLOW='FR13_DRAFT_VOCAB_K=0' \
      FR13_MANDATORY_WEIGHT_BYTES="$FULL_VOCAB_WEIGHT_BYTES" \
      FR13_WEIGHT_FLOOR_MS="$FULL_VOCAB_FLOOR_MS" \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json" \
      FR13_FIXED32_CUTLASS_WAVE="$selector" \
      FR13_FIXED32_CUTLASS_WAVE_SO="$candidate_so" \
      FR13_FIXED32_CUTLASS_WAVE_PRODUCTION="$production" \
      FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_JSON="$pass_json" \
      FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_SHA256="$pass_sha" \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 \
      FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_INSTANCE_ID= \
      FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
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
      FORKED_FA2_SO="$STOCK_FA2_SO" \
      RUNROOT="$RUNROOT_ABS" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh \
        "$arm" tail6_fixed32 "$SUBSET" \
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
  [[ "$(grep -Fxc 'FR13_DRAFT_VOCAB_ROOT=0' "$container_env")" -eq 1 \
     && "$(grep -Fxc 'FR13_DRAFT_VOCAB_K=0' "$container_env")" -eq 1 ]] \
    || { echo "$arm did not run the exact full-vocabulary contract" >&2; return 4; }
  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$arm" \
    --out-root "$RUNROOT_ABS/$arm/swe_out" \
    --expected-tok-per-draft 31 \
    --batch-size 1 \
    --out "$RUNROOT_ABS/$arm/deploy_speed_fullwall.json"
  printf 'arm=%s serve_rc=0 container_env_sha256=%s ended=%s\n' \
    "$arm" "$(sha256sum "$container_env" | awk '{print $1}')" \
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
[[ ! -e "$STOCK_ATTESTATION" && ! -L "$STOCK_ATTESTATION" ]] \
  || { echo "stock arm emitted a Stream-K binary attestation" >&2; exit 4; }
[[ -f "$CANDIDATE_ATTESTATION" && ! -L "$CANDIDATE_ATTESTATION" \
   && -f "$CANDIDATE_SIDECAR" && ! -L "$CANDIDATE_SIDECAR" \
   && -f "$CANDIDATE_SELECTOR" && ! -L "$CANDIDATE_SELECTOR" ]] \
  || { echo "candidate arm lacks regular Stream-K identity artifacts" >&2; exit 4; }
[[ "$(<"$CANDIDATE_SELECTOR")" == "streamk_coop128" ]] \
  || { echo "candidate selector sidecar is missing or wrong" >&2; exit 4; }
CANDIDATE_SIDECAR_SHA256=$(sha256sum "$CANDIDATE_SIDECAR" | awk '{print $1}')
"$PYTHON_BIN" scripts/fr13_cutlass_streamk_pass.py verify \
  --sidecar "$CANDIDATE_SIDECAR" \
  --expected-sidecar-sha256 "$CANDIDATE_SIDECAR_SHA256" \
  --candidate-so "$CUTLASS_STREAMK_SO" \
  --patch-source "$PATCH_SOURCE" \
  >/dev/null
"$PYTHON_BIN" scripts/fr13_cutlass_streamk_pass.py attestation \
  --attestation "$CANDIDATE_ATTESTATION" \
  --expected-sidecar-sha256 "$CANDIDATE_SIDECAR_SHA256" \
  > "$RUNROOT_ABS/$CANDIDATE_ARM/streamk_production_binding.json"

finalize_manifests

"$PYTHON_BIN" scripts/fr13_cutlass_streamk_timing.py \
  --subset "$SUBSET" \
  --stock-measure "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  --candidate-measure "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  --stock-container-env "$RUNROOT_ABS/$STOCK_ARM/container_env.txt" \
  --candidate-container-env "$RUNROOT_ABS/$CANDIDATE_ARM/container_env.txt" \
  --production-binding "$RUNROOT_ABS/$CANDIDATE_ARM/streamk_production_binding.json" \
  --candidate-so "$CUTLASS_STREAMK_SO" \
  --source-commit "$SOURCE_COMMIT" \
  --out "$RUNROOT_ABS/timing_summary.json"

printf 'timing_summary=%s completed=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"

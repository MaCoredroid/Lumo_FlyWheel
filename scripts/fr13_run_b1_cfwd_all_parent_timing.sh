#!/usr/bin/env bash
# Paired one-real-task B1 diagnostic: stock CFWD vs source-gated all-parent CFWD.
# This runner and both of its arms are timing/floor-acceptance ineligible.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${FR13_RUN_CFWD_ALL_PARENT_B1_TIMING:?set FR13_RUN_CFWD_ALL_PARENT_B1_TIMING=1}"
[[ "$FR13_RUN_CFWD_ALL_PARENT_B1_TIMING" == "1" ]] \
  || { echo "FR13_RUN_CFWD_ALL_PARENT_B1_TIMING must be exactly 1" >&2; exit 2; }

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
LIVE_PASS=results/fr13_fixed32_cfwd_all_parent_b1_live_pass_20260801/live_pass.json
LIVE_PASS_SHA256=b7c8f4e7f8cf3e2619d458b3ec3e5e1ffdcb5a15a2938aa18c6dda936b3c45e3
CANDIDATE_BASE_COMMIT=f19e90053cfe414cafc76a2ffa3326a589da5e1e
CANDIDATE=fixed32_all_parent_commit_v2
SOURCE_CONTRACT_SHA256=51541928c3a758fdac34a70fe46b97753ffc1b6e9f3e5fe470c4b34a96515dc4
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
STOCK_FA2_RELATIVE=output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so
STOCK_FA2_SO="$REPO/$STOCK_FA2_RELATIVE"
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
FULL_VOCAB_WEIGHT_BYTES=42025179008
FULL_VOCAB_FLOOR_MS=153.938384645
FULL_VOCAB_CAP_MS=177.029142341
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
STOCK_ARM="hydra27_fixed32_cfwd_stock_${TAG}"
CANDIDATE_ARM="hydra27_fixed32_cfwd_all_parent_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for required_file in "$SUBSET" "$LIVE_PASS" "$STOCK_FA2_SO"; do
  [[ -f "$required_file" && ! -L "$required_file" ]] \
    || { echo "required input must be a regular non-symlink file: $required_file" >&2; exit 2; }
done
unset required_file
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical one-task B1 subset SHA-256 drifted" >&2; exit 2; }
[[ "$(sha256sum "$LIVE_PASS" | awk '{print $1}')" == "$LIVE_PASS_SHA256" ]] \
  || { echo "curated CFWD live PASS SHA-256 drifted" >&2; exit 2; }
[[ "$(stat -c '%s' "$STOCK_FA2_SO")" == "$STOCK_FA2_BYTES" \
   && "$(sha256sum "$STOCK_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "fixed32 stock FA2 identity drifted" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "current source identity is invalid" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
git merge-base --is-ancestor "$CANDIDATE_BASE_COMMIT" HEAD \
  || { echo "CFWD candidate base commit is not an ancestor of HEAD" >&2; exit 2; }
git diff --quiet "$CANDIDATE_BASE_COMMIT" -- \
  scripts/fr13_device_multidraft_kernel.py \
  scripts/fr13_fixed32_work_census.py \
  scripts/fr13_launch_forked_fa2_tree_server.sh \
  || { echo "CFWD candidate source differs from the qualified base" >&2; exit 2; }

# Validate the exact curated credential before the first Docker query. The
# byte-gate run contributes no timing samples to this diagnostic.
"$PYTHON_BIN" scripts/fr13_cfwd_all_parent_timing.py validate-pass \
  --path "$LIVE_PASS" >/dev/null
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

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

mkdir -p "$RUNROOT_ABS"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

printf 'classification=one_real_swe_verified_b1_cfwd_all_parent_timing_diagnostic\ntask_count=1\ntask_id=astropy__astropy-12907\nbatch_size=1\nconcurrency=1\ntiming_eligible=0\nfloor_acceptance_eligible=0\nformal_floor_acceptance_eligible=0\nproduction_default_enabled=0\nstock_first=1\nonly_arm_delta=FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION_0_to_1\ncandidate=%s\ncandidate_base_commit=%s\nsource_contract_sha256=%s\nphysical_drafts=31\nphysical_rows=32\ndraft_vocab_root=0\ndraft_vocab_k=0\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_1_15x_cap_ms=%s\nlauncher_pid=%s\nrunroot=%s\nstock_arm=%s\ncandidate_arm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nlive_pass_sha256=%s\nstock_fa2_sha256=%s\nstarted=%s\n' \
  "$CANDIDATE" "$CANDIDATE_BASE_COMMIT" "$SOURCE_CONTRACT_SHA256" \
  "$FULL_VOCAB_WEIGHT_BYTES" "$FULL_VOCAB_FLOOR_MS" "$FULL_VOCAB_CAP_MS" \
  "$$" "$RUNROOT_ABS" "$STOCK_ARM" "$CANDIDATE_ARM" "$SOURCE_COMMIT" \
  "$RUNNER_SHA256" "$SUBSET_SHA256" "$LIVE_PASS_SHA256" \
  "$STOCK_FA2_SHA256" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

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
    || { echo "CFWD timing runner changed during execution" >&2; return 14; }
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
  local pass_json=""
  if [[ "$production" == "1" ]]; then
    pass_json="$REPO/$LIVE_PASS"
  fi
  echo "===== $arm: one real SWE-Verified B1 CFWD production=$production ====="
  if env \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
      FR13_FIXED32_B1_DIAGNOSTIC=1 \
      FR13_DRAFT_VOCAB_ROOT=0 FR13_DRAFT_VOCAB_K=0 \
      FR13_NEEDS_ALLOW='FR13_DRAFT_VOCAB_K=0' \
      FR13_MANDATORY_WEIGHT_BYTES="$FULL_VOCAB_WEIGHT_BYTES" \
      FR13_WEIGHT_FLOOR_MS="$FULL_VOCAB_FLOOR_MS" \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json" \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION="$production" \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON="$pass_json" \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 \
      FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_INSTANCE_ID= \
      FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
      FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
      FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
      FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
      FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_SO= \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$STOCK_FA2_SO" \
      RUNROOT="$RUNROOT_ABS" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh \
        "$arm" hydra27_fixed32 "$SUBSET" \
        > "$RUNROOT_ABS/$arm.runlog" 2>&1; then
    :
  else
    local serve_rc=$?
    printf 'arm=%s production=%s serve_rc=%s ended=%s\n' \
      "$arm" "$production" "$serve_rc" "$(date -u +%FT%TZ)" \
      >> "$RUNROOT_ABS/launcher_meta.txt"
    return "$serve_rc"
  fi

  local arm_dir="$RUNROOT_ABS/$arm"
  local container_env="$arm_dir/container_env.txt"
  local eval_report="$arm_dir/swe_out/verified/per_task/astropy__astropy-12907/eval/eval_report.json"
  [[ -f "$container_env" && ! -L "$container_env" ]] \
    || { echo "$arm lacks a regular container environment artifact" >&2; return 4; }
  [[ "$(grep -Fxc 'FR13_FIXED32_MODE=hydra27_fixed32' "$container_env")" -eq 1 \
     && "$(grep -Fxc 'FR13_DRAFT_VOCAB_ROOT=0' "$container_env")" -eq 1 \
     && "$(grep -Fxc 'FR13_DRAFT_VOCAB_K=0' "$container_env")" -eq 1 \
     && "$(grep -Fxc 'FR13_SFWD_GPU_TIMER=1' "$container_env")" -eq 1 \
     && "$(grep -Fxc 'FR13_DFWD_GPU_TIMER=1' "$container_env")" -eq 1 \
     && "$(grep -Fxc 'FR13_CFWD_GPU_TIMER=1' "$container_env")" -eq 1 \
     && "$(grep -Fxc 'FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0' "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=$production" "$container_env")" -eq 1 ]] \
    || { echo "$arm did not run the pinned B1/full-vocab/CFWD timer contract" >&2; return 4; }
  "$PYTHON_BIN" scripts/fr13_cfwd_all_parent_timing.py validate-eval \
    --path "$eval_report" --label "$arm" >/dev/null
  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$arm" \
    --out-root "$arm_dir/swe_out" \
    --expected-tok-per-draft 31 \
    --batch-size 1 \
    --out "$arm_dir/deploy_speed_fullwall.json"
  printf 'arm=%s production=%s serve_rc=0 container_env_sha256=%s ended=%s\n' \
    "$arm" "$production" "$(sha256sum "$container_env" | awk '{print $1}')" \
    "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
}

# The stock arm is always first; no candidate state can warm or contaminate it.
run_arm "$STOCK_ARM" 0
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after the stock reference" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" 1
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after the candidate" >&2; exit 2; }

STOCK_SELECTOR="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_taw_native_precompute_production.arm"
STOCK_PASS="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_taw_native_precompute.production_pass.json"
CANDIDATE_SELECTOR="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_taw_native_precompute_production.arm"
CANDIDATE_PASS="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_taw_native_precompute.production_pass.json"
[[ ! -e "$STOCK_SELECTOR" && ! -L "$STOCK_SELECTOR" \
   && ! -e "$STOCK_PASS" && ! -L "$STOCK_PASS" ]] \
  || { echo "stock arm emitted CFWD production state" >&2; exit 4; }
[[ -f "$CANDIDATE_SELECTOR" && ! -L "$CANDIDATE_SELECTOR" \
   && -f "$CANDIDATE_PASS" && ! -L "$CANDIDATE_PASS" ]] \
  || { echo "candidate arm lacks regular CFWD production state" >&2; exit 4; }
[[ "$(<"$CANDIDATE_SELECTOR")" == "1" ]] \
  || { echo "candidate CFWD production selector is wrong" >&2; exit 4; }
cmp -s "$LIVE_PASS" "$CANDIDATE_PASS" \
  || { echo "candidate copied credential differs from curated PASS" >&2; exit 4; }

finalize_manifests

"$PYTHON_BIN" scripts/fr13_cfwd_all_parent_timing.py reduce \
  --subset "$SUBSET" \
  --curated-pass "$LIVE_PASS" \
  --stock-measure "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  --candidate-measure "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  --stock-container-env "$RUNROOT_ABS/$STOCK_ARM/container_env.txt" \
  --candidate-container-env "$RUNROOT_ABS/$CANDIDATE_ARM/container_env.txt" \
  --stock-census "$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_work_census.jsonl" \
  --candidate-census "$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_work_census.jsonl" \
  --stock-eval-report "$RUNROOT_ABS/$STOCK_ARM/swe_out/verified/per_task/astropy__astropy-12907/eval/eval_report.json" \
  --candidate-eval-report "$RUNROOT_ABS/$CANDIDATE_ARM/swe_out/verified/per_task/astropy__astropy-12907/eval/eval_report.json" \
  --stock-selector "$STOCK_SELECTOR" \
  --stock-production-pass "$STOCK_PASS" \
  --candidate-selector "$CANDIDATE_SELECTOR" \
  --candidate-production-pass "$CANDIDATE_PASS" \
  --source-commit "$SOURCE_COMMIT" \
  --out "$RUNROOT_ABS/timing_summary.json"

printf 'timing_summary=%s completed=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"

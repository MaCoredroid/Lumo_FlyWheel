#!/usr/bin/env bash
# Exact4 real SWE-Verified B1 full-wall timing: the qrow16 incumbent kernel vs
# the byte-gate-qualified qrow32 GQA-pair kernel.
#
# READ THIS BEFORE QUOTING A NUMBER FROM THIS RUNNER.
#
# This pair is NOT single-variable in the sense the B4 GQA-pair pair is. There,
# both arms load one identical .so and the injected sentinel is the only delta.
# Here TWO things differ together, and they cannot be separated:
#
#   stock arm     : FORKED_FA2_SO is the pinned qrow16 incumbent binary
#                   (1649fbe9..., 299507792 bytes) running its own production
#                   selector, FR13_FA2_QROW16_PRODUCTION=1. No B1 selector of
#                   any kind is set, so the tree_attn call site is untouched.
#   candidate arm : FORKED_FA2_SO is the pinned GQA-pair B1 binary
#                   (3560cdc0..., 299815552 bytes) with
#                   FR13_FA2_QROW32_B1_PRODUCTION_ARM=gqa_pair, which retags the
#                   bias operand's batch stride to 0x46523136 so the forked
#                   flash_api dispatches fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_b1.
#
# So the delta is {binary} x {dispatch}, jointly. That is unavoidable: the two
# kernels do not live in one binary, and the sentinel is meaningless in the
# qrow16 .so. What this runner CAN do, and does, is pin everything else
# identical and say so in the artifact -- see arm_delta_disclosure. Both arms
# run the same exact4 subset, the same canonical sampling, batch 1 /
# concurrency 1, ENFORCE_EAGER=0 with CUDAGRAPH_MODE=FULL_AND_PIECEWISE, the
# same vocabulary and block map, the same topology, and the same floor
# contract; every other candidate selector in the stack is explicitly pinned
# off on both sides.
#
# The residual confound is therefore whole-binary: any difference in code
# layout, alignment, or unrelated kernels between the two .so files rides along
# with the dispatch change. At an expected effect of ~17 ms on ~233 ms (~7%)
# against a historical B1 step CV near 2% that confound is small relative to the
# signal, but it is real and it is not measured here.
#
# Unlike B4, the candidate arm adds NO retag copy: at batch 1 the sentinel
# stride is never dereferenced, so the selector is a pure metadata as_strided
# view aliasing the incumbent operand's own bytes. There is no per-step
# device-to-device copy charged to either arm, which is why this pair does not
# carry B4's "conservative against candidate" overhead disclaimer -- it has no
# such overhead to charge.
#
# At B1 per-request and aggregate throughput are the same quantity
# (events_per_step == 1), so promotion_verdict's two conditions collapse and
# the verdict is exactly the step_wall_ms delta, reported alongside the
# ratio-to-floor.
#
# This paired screen is not the formal statistical hardware-floor acceptance
# gate. Promotion here means only fr13_b4_timing_math.promotion_verdict.
set -euo pipefail

case "${FR13_RUN_B1_QROW32_GQA_PAIR_TIMING:-0}" in
  1) ;;
  0)
    echo "B1 qrow32 GQA-pair timing pair is disabled" >&2
    echo "set FR13_RUN_B1_QROW32_GQA_PAIR_TIMING=1 to run it" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_B1_QROW32_GQA_PAIR_TIMING must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${QROW16_FA2_SO:?set QROW16_FA2_SO to the pinned qrow16 incumbent binary}"
: "${QROW32_GQA_PAIR_B1_FA2_SO:?set it to the pinned GQA-pair B1 binary}"
: "${QROW32_GQA_PAIR_B1_GATE_JSON:?set it to the sealed B1 byte gate PASS produced at HEAD}"
: "${QROW32_GQA_PAIR_B1_GATE_SHA256:?set it to that gate artifact SHA-256}"
: "${QROW32_GQA_PAIR_B1_LIVE_RESULT_JSON:?set it to the live A/B result the gate binds}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
FIXED32_MODE=hydra27_fixed32
LOGICAL_TOPOLOGY=Hydra27
ACTIVE_DRAFTS=27
VALID_MASK=0x7abdffff
SIDECAR=scripts/fr13_qrow32_b1_pass_sidecar.py
PATCH_SOURCE=scripts/fr13_patch_fa2_tree_bias.py
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
TASK_IDS=astropy__astropy-12907,astropy__astropy-13033,astropy__astropy-13236,astropy__astropy-13398
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
# The incumbent reference: the qrow16 production path and the live PASS that
# credentials it. Both are already pinned by the qrow16 B1 stack runner.
QROW16_SHA256=1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86
QROW16_BYTES=299507792
QROW16_PASS=$REPO/results/fr13_fixed32_qrow16_num_splits0_live_pass_20260731T173608Z/fr13_fa2_qrow16_live_paged_ab.json
QROW16_PASS_SHA256=36940fd43d11399529d1bfe7e11baa9961907193267f3bb43d41057328737b77
# The candidate: the GQA-pair B1 unit, one translation unit with its own closure.
CANDIDATE_SHA256=3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae
CANDIDATE_BYTES=299815552
FA2_HEAD=29210221863736a08f71a866459e368ad1ac4a95
SOURCE_CLOSURE_SHA256=172b5e7131841ce45650bb8eea35f0b427ca660ce8f145bd39b55b00a336ebf4
SELECTOR_SENTINEL=1179791670
QROW16_REFERENCE_SENTINEL=1179791667
DRAFT_VOCAB_ROOT=1
DRAFT_VOCAB_K=65536
MANDATORY_WEIGHT_BYTES=32666638208
MANDATORY_WEIGHT_FLOOR_MS=119.658015414
ONE_SIDED_U95_CAP_MS=137.6067177261
SUMMARY_SCHEMA=fr13.fixed32.fa2_qrow32_gqa_pair_b1.full_wall_timing_pair.v1
RUN_CLASSIFICATION=real_swe_verified_exact4_b1_fa2_qrow32_gqa_pair_timing
LAUNCH_CLASSIFICATION=real_swe_verified_exact4_b1_fa2_qrow32_gqa_pair_timing_candidate
ENGAGEMENT_SCHEMA=fr13.fixed32.fa2_qrow32_b1_production_engagement.v2
CREDENTIAL_SCHEMA=fr13.fixed32.fa2_qrow32_b1_gqa_pair_production_pass.v1
ONLY_ARM_DELTA=FA2_qrow16_incumbent_binary_and_dispatch_to_qrow32_gqa_pair_b1
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
PATCH_SOURCE_SHA256=$(sha256sum "$PATCH_SOURCE" | awk '{print $1}')
SIDECAR_SHA256=$(sha256sum "$SIDECAR" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
STOCK_ARM="${FIXED32_MODE}_fa2_qrow16_stock_b1_${TAG}"
CANDIDATE_ARM="${FIXED32_MODE}_fa2_qrow32_gqa_pair_b1_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for input in \
    "$QROW16_FA2_SO" "$QROW32_GQA_PAIR_B1_FA2_SO" \
    "$QROW32_GQA_PAIR_B1_GATE_JSON" "$QROW32_GQA_PAIR_B1_LIVE_RESULT_JSON" \
    "$QROW16_PASS"; do
  [[ "$input" == /* && -f "$input" && ! -L "$input" ]] \
    || { echo "timing input must be an absolute regular non-symlink file: $input" >&2; exit 2; }
done
unset input
[[ "$(stat -c '%s' "$QROW16_FA2_SO")" == "$QROW16_BYTES" \
   && "$(sha256sum "$QROW16_FA2_SO" | awk '{print $1}')" == "$QROW16_SHA256" ]] \
  || { echo "QROW16_FA2_SO is not the pinned incumbent binary" >&2; exit 2; }
[[ "$(stat -c '%s' "$QROW32_GQA_PAIR_B1_FA2_SO")" == "$CANDIDATE_BYTES" \
   && "$(sha256sum "$QROW32_GQA_PAIR_B1_FA2_SO" | awk '{print $1}')" == "$CANDIDATE_SHA256" ]] \
  || { echo "QROW32_GQA_PAIR_B1_FA2_SO is not the pinned byte-gate candidate" >&2; exit 2; }
# The two arms MUST load different binaries; if they were the same file the
# pair would be measuring nothing.
[[ "$QROW16_SHA256" != "$CANDIDATE_SHA256" ]] \
  || { echo "stock and candidate binaries are identical" >&2; exit 2; }
[[ "$QROW32_GQA_PAIR_B1_GATE_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$QROW32_GQA_PAIR_B1_GATE_JSON" | awk '{print $1}')" == "$QROW32_GQA_PAIR_B1_GATE_SHA256" ]] \
  || { echo "sealed B1 byte gate PASS identity mismatch" >&2; exit 2; }
[[ "$(sha256sum "$QROW16_PASS" | awk '{print $1}')" == "$QROW16_PASS_SHA256" ]] \
  || { echo "qrow16 incumbent live PASS identity drifted" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical exact4 subset SHA-256 drift" >&2; exit 2; }
[[ -f "$BLOCK_MAP" && ! -L "$BLOCK_MAP" \
   && "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" ]] \
  || { echo "pinned root-64K draft-vocabulary block map drifted" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
# The credential this run will serve on. It must bind to THIS commit, so a gate
# produced before the production plumbing existed cannot be reused: the sealed
# af85792ff gate does NOT satisfy this and the byte gate must be re-run at the
# plumbing commit before this runner can proceed.
"$PYTHON_BIN" "$SIDECAR" validate-gqa-pair \
  --gate "$QROW32_GQA_PAIR_B1_GATE_JSON" \
  --expected-gate-sha256 "$QROW32_GQA_PAIR_B1_GATE_SHA256" \
  --live-result "$QROW32_GQA_PAIR_B1_LIVE_RESULT_JSON" \
  --candidate-so "$QROW32_GQA_PAIR_B1_FA2_SO" \
  --expected-candidate-sha256 "$CANDIDATE_SHA256" \
  --arm gqa_pair \
  --patch-source "$PATCH_SOURCE" \
  --expected-source-commit "$SOURCE_COMMIT" \
  > "$RUNROOT_ABS/gate_binding.at_launch.json"
GATE_BINDING_SHA256=$(
  sha256sum "$RUNROOT_ABS/gate_binding.at_launch.json" | awk '{print $1}'
)

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT"
export FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K"
export FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER"
export FR13_NEEDS_ALLOW=
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" ]] \
  || { echo "canonical B1 qualification floor contract drifted" >&2; exit 2; }

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

printf 'classification=%s\ntiming_eligible=1\nformal_floor_acceptance_eligible=0\nproduction_default_enabled=0\nonly_arm_delta=%s\nsingle_variable=0\narm_delta_spans_two_binaries=1\ncandidate_arm_selector=gqa_pair\nselector_sentinel=%s\nreference_selector_sentinel=%s\ntopology=%s\nlogical_topology=%s\nactive_drafts=%s\nvalid_mask=%s\nphysical_drafts=31\nphysical_rows_root_inclusive=32\nbatch_size=1\nconcurrency=1\ntask_count=4\ntask_ids=%s\nsubset_sha256=%s\ndraft_vocab_root=%s\ndraft_vocab_k=%s\ndraft_vocab_blocks=%s\ndraft_vocab_blocks_sha256=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nlauncher_pid=%s\nrunroot=%s\nstock_arm=%s\ncandidate_arm=%s\nsource_commit=%s\nstock_so_sha256=%s\nstock_so_size=%s\ncandidate_so_sha256=%s\ncandidate_so_size=%s\nfa2_head=%s\nfa2_source_closure_sha256=%s\ngate_sha256=%s\ngate_binding_sha256=%s\nrunner_sha256=%s\nsidecar_sha256=%s\npatch_source_sha256=%s\nenforce_eager=0\ncudagraph_mode=FULL_AND_PIECEWISE\nstarted=%s\n' \
  "$LAUNCH_CLASSIFICATION" "$ONLY_ARM_DELTA" "$SELECTOR_SENTINEL" \
  "$QROW16_REFERENCE_SENTINEL" "$FIXED32_MODE" "$LOGICAL_TOPOLOGY" \
  "$ACTIVE_DRAFTS" "$VALID_MASK" "$TASK_IDS" "$SUBSET_SHA256" \
  "$DRAFT_VOCAB_ROOT" "$DRAFT_VOCAB_K" "$BLOCK_MAP_CONTAINER" \
  "$BLOCK_MAP_SHA256" "$FR13_MANDATORY_WEIGHT_BYTES" "$FR13_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$$" "$RUNROOT_ABS" "$STOCK_ARM" "$CANDIDATE_ARM" \
  "$SOURCE_COMMIT" "$QROW16_SHA256" "$QROW16_BYTES" "$CANDIDATE_SHA256" \
  "$CANDIDATE_BYTES" "$FA2_HEAD" "$SOURCE_CLOSURE_SHA256" \
  "$QROW32_GQA_PAIR_B1_GATE_SHA256" "$GATE_BINDING_SHA256" "$RUNNER_SHA256" \
  "$SIDECAR_SHA256" "$PATCH_SOURCE_SHA256" \
  "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"

MANIFEST_FINALIZED=0
finalize_manifests() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json" || return $?
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json" || return $?
  cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || { echo "runtime/source manifest changed during timing" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during timing" >&2; return 14; }
  [[ "$(sha256sum "$QROW16_FA2_SO" | awk '{print $1}')" == "$QROW16_SHA256" \
     && "$(sha256sum "$QROW32_GQA_PAIR_B1_FA2_SO" | awk '{print $1}')" == "$CANDIDATE_SHA256" ]] \
    || { echo "an FA2 binary changed during timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" \
     && "$(sha256sum "$SIDECAR" | awk '{print $1}')" == "$SIDECAR_SHA256" \
     && "$(sha256sum "$PATCH_SOURCE" | awk '{print $1}')" == "$PATCH_SOURCE_SHA256" ]] \
    || { echo "B1 GQA-pair timing source changed during execution" >&2; return 14; }
  [[ "$(sha256sum "$QROW32_GQA_PAIR_B1_GATE_JSON" | awk '{print $1}')" == "$QROW32_GQA_PAIR_B1_GATE_SHA256" ]] \
    || { echo "sealed byte gate PASS changed during timing" >&2; return 14; }
  [[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" \
     && -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
    || { echo "frozen source changed during timing" >&2; return 14; }
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

# Everything below is identical between the two invocations EXCEPT the four
# variables that carry the arm: the FA2 binary, the qrow16 production switch,
# the B1 timing arm, and the B1 production arm. The subset, the sampling, the
# topology, the vocabulary, the block map, the graph mode, the floor contract
# and every other candidate selector are byte-for-byte the same.
run_arm() {
  local arm=$1
  local timing_arm=$2
  local fa2_so=$3
  local fa2_sha=$4
  local qrow16_production=$5
  local production_arm=$6
  local qrow16_pass=""
  local qrow16_pass_sha=""
  local qrow16_so_sha=""
  local gate_json=""
  local gate_sha=""
  local live_result=""
  if [[ "$qrow16_production" == "1" ]]; then
    qrow16_pass=$QROW16_PASS
    qrow16_pass_sha=$QROW16_PASS_SHA256
    qrow16_so_sha=$fa2_sha
  fi
  if [[ -n "$production_arm" ]]; then
    gate_json=$QROW32_GQA_PAIR_B1_GATE_JSON
    gate_sha=$QROW32_GQA_PAIR_B1_GATE_SHA256
    live_result=$QROW32_GQA_PAIR_B1_LIVE_RESULT_JSON
  fi
  echo "===== $arm: exact4 B1 FA2 timing_arm=$timing_arm production_arm=${production_arm:-none} ====="
  if env \
      RUNROOT="$RUNROOT_ABS" \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S=5400 \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT" \
      FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K" \
      FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER" \
      FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES" \
      FR13_WEIGHT_FLOOR_MS="$MANDATORY_WEIGHT_FLOOR_MS" \
      FR13_NEEDS_ALLOW= \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
      FR13_SCAN_ALIGN=0 FR13_NPAD_INVARIANT=0 \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_cfwd.json" \
      FR13_FA2_QROW32_B1_TIMING_ARM="$timing_arm" \
      FR13_FA2_QROW32_B1_PRODUCTION_ARM="$production_arm" \
      FR13_FA2_QROW32_B1_GQA_PAIR_GATE_JSON="$gate_json" \
      FR13_FA2_QROW32_B1_GQA_PAIR_GATE_SHA256="$gate_sha" \
      FR13_FA2_QROW32_B1_GQA_PAIR_LIVE_RESULT_JSON="$live_result" \
      FR13_FA2_QROW32_B1_EXACT4_TASK_IDS="$TASK_IDS" \
      FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256="$SUBSET_SHA256" \
      FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256="$PATCH_SOURCE_SHA256" \
      FR13_FA2_QROW32_B1_SO_SHA256="$fa2_sha" \
      FR13_FA2_QROW32_B1_SO_SIZE="$(stat -c '%s' "$fa2_so")" \
      FR13_FA2_QROW32_B1_FA2_HEAD="$FA2_HEAD" \
      FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256="$SOURCE_CLOSURE_SHA256" \
      FR13_FA2_QROW32_B1_SOURCE_COMMIT="$SOURCE_COMMIT" \
      FR13_FA2_QROW32_B1_LIVE_AB_ARM= \
      FR13_FA2_QROW32_LIVE_PAGED_AB=0 \
      FR13_FA2_QROW32_LIVE_PAGED_AB_ARM= \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 \
      FR13_FA2_QROW16_PRODUCTION="$qrow16_production" \
      FR13_FA2_QROW16_SO_SHA256="$qrow16_so_sha" \
      FR13_FA2_QROW16_LIVE_PASS_JSON="$qrow16_pass" \
      FR13_FA2_QROW16_LIVE_PASS_SHA256="$qrow16_pass_sha" \
      FR13_FA2_QROW32_B4_TIMING_ARM= FR13_FA2_QROW32_B4_PRODUCTION_ARM= \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_INSTANCE_ID= \
      FR13_DFWD_K64_TOP3=0 \
      FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
      FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
      FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
      FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
      FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
      FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
      FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0 \
      FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=0 \
      FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0 \
      FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0 \
      FR13_FIXED32_CUTLASS_WAVE=stock \
      FR13_FIXED32_CUTLASS_WAVE_SO= \
      FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
      FR13_FIXED32_CONV_SOURCE_BATCH=0 \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$fa2_so" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh \
        "$arm" "$FIXED32_MODE" "$SUBSET" \
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
  # Pin the invariants on BOTH arms, then the arm-carrying variables.
  local expected
  for expected in \
      "FR13_FIXED32_MODE=$FIXED32_MODE" \
      'FR13_FIXED32_B1_DIAGNOSTIC=0' \
      "FR13_DRAFT_VOCAB_ROOT=$DRAFT_VOCAB_ROOT" \
      "FR13_DRAFT_VOCAB_K=$DRAFT_VOCAB_K" \
      "FR13_DRAFT_VOCAB_BLOCKS=$BLOCK_MAP_CONTAINER" \
      'MAX_NUM_SEQS=1' \
      'SWE_CONCURRENCY=1' \
      'ENFORCE_EAGER=0' \
      'CUDAGRAPH_MODE=FULL_AND_PIECEWISE' \
      'FR13_FA2_QROW32_LIVE_PAGED_AB=0' \
      'FR13_FA2_QROW16_LIVE_PAGED_AB=0' \
      'FR13_FA2_QROW32_B1_LIVE_AB_ARM=' \
      'FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0' \
      'FR13_FIXED32_CUTLASS_WAVE=stock' \
      'FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0' \
      "FR13_FA2_QROW32_B1_TIMING_ARM=$timing_arm" \
      "FR13_FA2_QROW32_B1_PRODUCTION_ARM=$production_arm" \
      "FR13_FA2_QROW16_PRODUCTION=$qrow16_production"; do
    [[ "$(grep -Fxc "$expected" "$container_env")" -eq 1 ]] \
      || { echo "$arm lacks exact B1 timing pin: $expected" >&2; return 4; }
  done
  local engagement="$RUNROOT_ABS/$arm/logs/fr13_fa2_qrow32_b1_production_engagement.json"
  if [[ -n "$production_arm" ]]; then
    [[ -f "$engagement" && ! -L "$engagement" ]] \
      || { echo "$arm lacks the GQA-pair production engagement artifact" >&2; return 4; }
    local sidecar="$RUNROOT_ABS/$arm/logs/fr13_fa2_qrow32_b1_production_pass.json"
    [[ -f "$sidecar" && ! -L "$sidecar" ]] \
      || { echo "$arm lacks the GQA-pair production credential" >&2; return 4; }
  else
    # An engagement record on the stock arm would mean the sentinel leaked
    # across the pair, which would invalidate the comparison outright.
    [[ ! -e "$engagement" && ! -L "$engagement" ]] \
      || { echo "$arm emitted a GQA-pair engagement on the stock arm" >&2; return 4; }
  fi
  # Batch 1 brackets are disjoint, but the reduction is still cross-gated
  # against the arm's own topology-blind engine work census: an ungated
  # aggregate is exactly the artifact the alignment study invalidated.
  local deploy_census="$RUNROOT_ABS/$arm/logs/fr13_fixed32_work_census.jsonl"
  [[ -f "$deploy_census" && ! -L "$deploy_census" ]] \
    || { echo "$arm lacks the work census the bracket reduction is gated on" >&2; return 4; }
  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$arm" --out-root "$RUNROOT_ABS/$arm/swe_out" \
    --expected-tok-per-draft 31 --batch-size 1 \
    --work-census "$deploy_census" \
    --out "$RUNROOT_ABS/$arm/deploy_speed_fullwall.json"
  printf 'arm=%s serve_rc=0 fa2_so_sha256=%s container_env_sha256=%s ended=%s\n' \
    "$arm" "$fa2_sha" "$(sha256sum "$container_env" | awk '{print $1}')" \
    "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
}

run_arm "$STOCK_ARM" qrow16_stock "$QROW16_FA2_SO" "$QROW16_SHA256" 1 ""
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after the stock reference" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" gqa_pair "$QROW32_GQA_PAIR_B1_FA2_SO" "$CANDIDATE_SHA256" 0 gqa_pair

CANDIDATE_ENGAGEMENT="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fa2_qrow32_b1_production_engagement.json"
CANDIDATE_SIDECAR="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fa2_qrow32_b1_production_pass.json"
CANDIDATE_SIDECAR_SHA256=$(sha256sum "$CANDIDATE_SIDECAR" | awk '{print $1}')
"$PYTHON_BIN" "$SIDECAR" verify-gqa-pair \
  --sidecar "$CANDIDATE_SIDECAR" \
  --expected-sidecar-sha256 "$CANDIDATE_SIDECAR_SHA256" \
  --candidate-so "$QROW32_GQA_PAIR_B1_FA2_SO" \
  --expected-candidate-sha256 "$CANDIDATE_SHA256" \
  --arm gqa_pair \
  --patch-source "$PATCH_SOURCE" \
  --expected-source-commit "$SOURCE_COMMIT" \
  --expected-patch-source-sha256 "$PATCH_SOURCE_SHA256" >/dev/null

finalize_manifests

"$PYTHON_BIN" - \
  "$SUBSET" "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  "$CANDIDATE_ENGAGEMENT" "$CANDIDATE_SIDECAR" \
  "$RUNROOT_ABS/timing_summary.json" \
  "$CANDIDATE_SIDECAR_SHA256" "$QROW32_GQA_PAIR_B1_GATE_SHA256" \
  "$GATE_BINDING_SHA256" "$CANDIDATE_SHA256" "$CANDIDATE_BYTES" \
  "$QROW16_SHA256" "$QROW16_BYTES" "$QROW16_PASS_SHA256" \
  "$FA2_HEAD" "$SOURCE_CLOSURE_SHA256" "$SOURCE_COMMIT" \
  "$PATCH_SOURCE_SHA256" "$SUMMARY_SCHEMA" "$RUN_CLASSIFICATION" \
  "$ENGAGEMENT_SCHEMA" "$CREDENTIAL_SCHEMA" "$SELECTOR_SENTINEL" \
  "$QROW16_REFERENCE_SENTINEL" "$DRAFT_VOCAB_ROOT" "$DRAFT_VOCAB_K" \
  "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$BLOCK_MAP_CONTAINER" "$BLOCK_MAP_SHA256" \
  "$FIXED32_MODE" "$LOGICAL_TOPOLOGY" "$ACTIVE_DRAFTS" "$VALID_MASK" \
  "$ONLY_ARM_DELTA" \
  "$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_engine_ingress.jsonl" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_engine_ingress.jsonl" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from fr13_b4_timing_math import phase_breakdown, positive, promotion_verdict
from lumo_flywheel_serving.inference_proxy import (
    fixed32_canonical_task_set_sha256,
    fixed32_task_key_id,
    verify_fixed32_ingress_ledger,
)

subset_path, stock_path, candidate_path = map(Path, sys.argv[1:4])
engagement_path, sidecar_path, out_path = map(Path, sys.argv[4:7])
sidecar_sha256, gate_sha256, gate_binding_sha256 = sys.argv[7:10]
candidate_sha256 = sys.argv[10]
candidate_bytes = int(sys.argv[11])
stock_so_sha256 = sys.argv[12]
stock_so_bytes = int(sys.argv[13])
qrow16_pass_sha256 = sys.argv[14]
fa2_head, source_closure_sha256, source_commit = sys.argv[15:18]
patch_source_sha256, summary_schema, run_classification = sys.argv[18:21]
engagement_schema, credential_schema = sys.argv[21:23]
selector_sentinel = int(sys.argv[23])
reference_selector_sentinel = int(sys.argv[24])
draft_vocab_root, draft_vocab_k, mandatory_weight_bytes = map(int, sys.argv[25:28])
mandatory_weight_floor_ms = float(sys.argv[28])
one_sided_u95_cap_ms = float(sys.argv[29])
draft_vocab_blocks, draft_vocab_blocks_sha256 = sys.argv[30:32]
fixed32_mode, logical_topology = sys.argv[32:34]
active_drafts = int(sys.argv[34])
valid_mask = int(sys.argv[35], 0)
only_arm_delta = sys.argv[36]
stock_ingress_path, candidate_ingress_path = map(Path, sys.argv[37:39])

task_ids = sorted(json.loads(subset_path.read_text(encoding="ascii"))["instance_ids"])
stock = json.loads(stock_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
expected_task_keys = sorted(fixed32_task_key_id(task_id) for task_id in task_ids)
expected_task_set_sha256 = fixed32_canonical_task_set_sha256(tuple(task_ids))


def validate_ingress(path, label):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"{label} engine ingress ledger is not a regular file")
    verification = verify_fixed32_ingress_ledger(
        path, expected_role="engine", require_finalized=True
    )
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("ascii").splitlines()]
    accepted = sorted(
        {
            row.get("task_key_id")
            for row in rows
            if row.get("event") == "request_accepted"
        }
    )
    completed = sorted(
        {
            row.get("task_key_id")
            for row in rows
            if row.get("event") == "request_complete"
            and row.get("outcome") == "completed"
        }
    )
    if accepted != expected_task_keys or completed != expected_task_keys:
        raise SystemExit(f"{label} engine ingress is not canonical exact4")
    if not any(
        row.get("event") == "campaign_begin"
        and row.get("evidence_sha256") == expected_task_set_sha256
        for row in rows
    ):
        raise SystemExit(f"{label} engine ingress lacks the exact4 campaign binding")
    return {
        "ledger_sha256": hashlib.sha256(raw).hexdigest(),
        "chain_head_sha256": verification["chain_head_sha256"],
        "accepted_task_key_ids": accepted,
        "completed_task_key_ids": completed,
    }


stock_ingress = validate_ingress(stock_ingress_path, "stock")
candidate_ingress = validate_ingress(candidate_ingress_path, "candidate")


def validate(record, label):
    if (
        record.get("schema") != "fr13.measure.deploy_speed.v1"
        or record.get("regime") != "deployment"
        or record.get("instrument") != "OFF"
        or record.get("batch_size") != 1
        or record.get("n_tasks") != 4
        or sorted(record.get("task_instance_ids", [])) != task_ids
        or record.get("draft_vocab_root") != draft_vocab_root
        or record.get("draft_vocab_k") != draft_vocab_k
        or record.get("mandatory_weight_bytes") != mandatory_weight_bytes
        or not math.isclose(
            float(record.get("weight_floor_ms", math.nan)),
            mandatory_weight_floor_ms,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or record.get("floor_is_full_step_hardware_floor") is not False
    ):
        raise SystemExit(f"{label} deploy-speed provenance is not exact4 B1")
    reduction = record.get("bracket_reduction")
    if not isinstance(reduction, dict) or reduction.get("topology") not in {
        "nested",
        "disjoint",
    }:
        raise SystemExit(f"{label} deploy-speed carries no bracket-topology provenance")
    if (reduction.get("work_census_gate") or {}).get("status") != "pass":
        raise SystemExit(
            f"{label} deploy-speed bracket reduction was not work-census gated"
        )
    for key in (
        "measured_tps_fullstep_wall", "step_wall_ms", "accept_per_event",
        "committed_per_event", "wall_steps_measured", "events_per_step",
        "s_per_fwd_gpu", "s_per_fwd_gpu_per_forward",
        "wall_s_per_event", "drafter_gpu_ms_per_step", "committer_gpu_ms_per_step",
        "weight_floor_ms", "floor_ms", "floor_ratio",
    ):
        positive(record, key)


validate(stock, "stock")
validate(candidate, "candidate")

engagement = json.loads(engagement_path.read_text(encoding="ascii"))
sidecar = json.loads(sidecar_path.read_text(encoding="ascii"))
expected_layers = sorted(
    f"language_model.model.layers.{index}.self_attn.attn"
    for index in range(3, 64, 4)
)
if (
    engagement.get("schema") != engagement_schema
    or engagement.get("status") != "ENGAGED"
    or engagement.get("runtime_mode") != "FULL"
    or engagement.get("arm") != "gqa_pair"
    or engagement.get("batch_size") != 1
    or engagement.get("physical_rows") != 32
    or engagement.get("selector_sentinel") != selector_sentinel
    or engagement.get("num_splits") != 0
    or engagement.get("layer_count") != 16
    or sorted(engagement.get("layers", [])) != expected_layers
    or engagement.get("candidate_so_sha256") != candidate_sha256
    or engagement.get("candidate_so_size") != candidate_bytes
    or engagement.get("fa2_head") != fa2_head
    or engagement.get("fa2_source_closure_sha256") != source_closure_sha256
    or engagement.get("source_commit") != source_commit
    or engagement.get("patch_source_sha256") != patch_source_sha256
    or engagement.get("pass_sidecar_sha256") != sidecar_sha256
    or engagement.get("task_ids") != task_ids
    or engagement.get("candidate_served") is not True
    or engagement.get("fallback_allowed") is not False
):
    raise SystemExit("candidate arm did not serve the GQA-pair kernel on every layer")
if (
    sidecar.get("schema") != credential_schema
    or sidecar.get("status") != "PASS"
    or sidecar.get("arm") != "gqa_pair"
    or sidecar.get("selector_sentinel") != selector_sentinel
    or sidecar.get("reference_selector_sentinel") != reference_selector_sentinel
    or sidecar.get("candidate_so_sha256") != candidate_sha256
    or sidecar.get("source_commit") != source_commit
    or sidecar.get("patch_source_sha256") != patch_source_sha256
    or sidecar.get("gate_sha256") != gate_sha256
    or sidecar.get("output_raw_byte_mismatches") != 0
    or sidecar.get("lse_raw_byte_mismatches") != 0
    or sidecar.get("gate_performance_measurement") is not False
):
    raise SystemExit("candidate arm credential is not the bound byte-gate credential")

stock_wall = positive(stock, "step_wall_ms")
candidate_wall = positive(candidate, "step_wall_ms")
stock_tps = positive(stock, "measured_tps_fullstep_wall")
candidate_tps = positive(candidate, "measured_tps_fullstep_wall")
stock_floor = positive(stock, "floor_ms")
candidate_floor = positive(candidate, "floor_ms")
if not math.isclose(stock_floor, candidate_floor, rel_tol=0.0, abs_tol=1e-9):
    raise SystemExit("stock and candidate floor values differ")

stock_phases = phase_breakdown(stock, "stock")
candidate_phases = phase_breakdown(candidate, "candidate")
# At batch 1 there is exactly one resident request, so the aggregate rate and
# the per-request rate are the same number. Assert it rather than assume it:
# if this ever fails, the run was not really B1 and the step_wall_ms verdict
# below would be reading a co-residency artifact.
for label, phases in (("stock", stock_phases), ("candidate", candidate_phases)):
    if not math.isclose(
        phases["events_per_step"], 1.0, rel_tol=0.0, abs_tol=1e-9
    ) or not math.isclose(
        phases["per_request_step_tps"],
        phases["measured_tps_fullstep_wall"],
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise SystemExit(f"{label} arm is not batch-1: per-request != aggregate")

summary = {
    "schema": summary_schema,
    "status": "complete",
    "run_classification": run_classification,
    "only_arm_delta": only_arm_delta,
    # The honest statement of what differs. Unlike the B4 pair, this one is NOT
    # single-variable: the two kernels live in different binaries, so the .so
    # and the dispatch change together and cannot be separated.
    "arm_delta_disclosure": {
        "single_variable": False,
        "served_dispatch": "qrow16_incumbent -> qrow32_gqa_pair_b1",
        "served_binary": "qrow16_incumbent_so -> qrow32_gqa_pair_b1_so",
        "stock_so_sha256": stock_so_sha256,
        "candidate_so_sha256": candidate_sha256,
        "why_not_single_variable": (
            "the GQA-pair B1 kernel exists only in its own translation unit, "
            "and its batch-stride sentinel is inert in the qrow16 binary, so "
            "no configuration serves the candidate kernel from the stock .so"
        ),
        "residual_confound": (
            "whole-binary differences (code layout, alignment, unrelated "
            "kernels) ride along with the dispatch change and are not measured"
        ),
        "candidate_only_overhead": "none",
        "retag_cost": (
            "zero-copy: at batch 1 the sentinel batch stride is never "
            "dereferenced, so the selector is a pure as_strided metadata view "
            "aliasing the incumbent operand's own bytes"
        ),
        "pinned_identical": [
            "exact4 subset and canonical sampling",
            "batch_size=1, concurrency=1, MAX_NUM_SEQS=1",
            "ENFORCE_EAGER=0, CUDAGRAPH_MODE=FULL_AND_PIECEWISE",
            "draft vocabulary root/K and block map",
            "topology, active drafts, valid mask",
            "mandatory weight floor contract",
            "every other candidate selector pinned off",
        ],
        "candidate_scope": "final_fixed32_b1_full_graph_only",
    },
    "task_count": 4,
    "batch_size": 1,
    "concurrency": 1,
    "arm": fixed32_mode,
    "logical_topology": logical_topology,
    "active_drafts": active_drafts,
    "valid_mask": hex(valid_mask),
    "physical_drafts": 31,
    "physical_rows_root_inclusive": 32,
    "authenticated_task_count": 4,
    "authenticated_task_ids": task_ids,
    "authenticated_task_set_sha256": expected_task_set_sha256,
    "authenticated_task_provenance": {
        "stock_reference": stock_ingress,
        "candidate": candidate_ingress,
    },
    "task_ids": task_ids,
    "subset_sha256": hashlib.sha256(subset_path.read_bytes()).hexdigest(),
    # At B1 per-request == aggregate, so the promotion verdict IS the
    # step_wall_ms verdict; both are reported so neither can be quoted alone.
    "decision_metric": "step_wall_ms",
    "promotion": promotion_verdict(stock_phases, candidate_phases),
    "step_wall_ms_delta": candidate_wall - stock_wall,
    "step_wall_ms_delta_frac": (candidate_wall - stock_wall) / stock_wall,
    "draft_vocab_root": draft_vocab_root,
    "draft_vocab_k": draft_vocab_k,
    "draft_vocab_blocks": draft_vocab_blocks,
    "draft_vocab_blocks_sha256": draft_vocab_blocks_sha256,
    "target_verifier_vocabulary": "full",
    "mandatory_weight_bytes": mandatory_weight_bytes,
    "mandatory_weight_floor_ms": mandatory_weight_floor_ms,
    "one_sided_u95_cap_ms": one_sided_u95_cap_ms,
    "source_commit": source_commit,
    "patch_source_sha256": patch_source_sha256,
    "fa2_head": fa2_head,
    "fa2_source_closure_sha256": source_closure_sha256,
    "gate_sha256": gate_sha256,
    "gate_binding_sha256": gate_binding_sha256,
    "stock_reference": {
        "selector": "qrow16_production",
        "selector_sentinel": reference_selector_sentinel,
        "fa2_so_sha256": stock_so_sha256,
        "fa2_so_size": stock_so_bytes,
        "live_pass_sha256": qrow16_pass_sha256,
        "step_wall_ms": stock_wall,
        "measured_tps_fullstep_wall": stock_tps,
        "accepted_drafts_per_event": float(stock["accept_per_event"]),
        "committed_tokens_per_event": float(stock["committed_per_event"]),
        **stock_phases,
        "step_wall_to_optimistic_floor_ratio": float(stock["floor_ratio"]),
    },
    "candidate": {
        "selector": "gqa_pair",
        "selector_sentinel": selector_sentinel,
        "fa2_so_sha256": candidate_sha256,
        "fa2_so_size": candidate_bytes,
        "production_sidecar_sha256": sidecar_sha256,
        "production_engagement_layer_count": engagement.get("layer_count"),
        "step_wall_ms": candidate_wall,
        "measured_tps_fullstep_wall": candidate_tps,
        "accepted_drafts_per_event": float(candidate["accept_per_event"]),
        "committed_tokens_per_event": float(candidate["committed_per_event"]),
        **candidate_phases,
        "step_wall_to_optimistic_floor_ratio": float(candidate["floor_ratio"]),
    },
    "optimistic_floor_ms": stock_floor,
    "optimistic_floor_is_full_step_hardware_floor": False,
    "candidate_to_stock_full_wall_tps_ratio": candidate_tps / stock_tps,
    "stock_to_candidate_step_wall_ratio": stock_wall / candidate_wall,
    "formal_floor_acceptance_eligible": False,
    "formal_floor_acceptance_reason": (
        "paired exact4 timing candidate only; run the canonical statistical "
        "Hydra floor gate after a positive screen"
    ),
    "production_default_enabled": False,
}
temporary = out_path.with_name(out_path.name + ".tmp")
temporary.write_text(
    json.dumps(summary, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n",
    encoding="ascii",
)
temporary.replace(out_path)
print(json.dumps(summary, sort_keys=True))
PY

printf 'timing_summary=%s completed=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"

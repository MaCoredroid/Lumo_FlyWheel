#!/usr/bin/env bash
# Exact4 real SWE-Verified B4 full-wall timing: stock FA2 dispatch vs the
# dual-byte-gate-qualified qrow32 GQA-pair kernel.
#
# Both arms load the identical pinned GQA-pair FA2 binary and run the identical
# canonical exact4 subset at the identical B4 sampling, so no wall-time
# difference can come from a binary, subset, or sampling difference. The arm
# delta is the tree_attn decode call site on the 16 target layers of the final
# fixed32 B4 FULL graph, and it is NOT a bare kernel swap. State it exactly:
#
#   stock arm     : the 2-D (32,32) census bias goes straight into
#                   flash_attn_varlen_func; set_params_tree_bias leaves
#                   tree_bias_batch_stride == 0, so all 4 batch slots read one
#                   shared 4 KB tile and the stock dispatch runs.
#   candidate arm : the patched call site first materialises the tagged operand
#                   the dual byte gate qualified -- empty_strided((4,32,32),
#                   (0x20014,32,1)) + copy_, recorded INTO the captured graph
#                   and therefore replayed every step, once per target layer --
#                   and the 0x20014 batch stride is the C++ dispatch predicate,
#                   so the GQA-pair kernel runs and reads 4 distinct planes.
#
# The retag copy and the 4-plane operand layout are inseparable from the
# dispatch: the batch stride IS the selector, so there is no way to serve the
# candidate kernel with the stock operand. Both extra costs are charged to the
# CANDIDATE arm, so the pair is conservative: a PROMOTABLE verdict is sound,
# while a no-gain/regression verdict is confounded by harness overhead and must
# not be read as a kernel result. The dual raw-byte gate proved the two
# operands produce identical BYTES; it says nothing about their identical cost.
#
# Scope: the candidate is served only where it is qualified -- the final fixed32
# B4 FULL graph. The mandatory sub-B4 FULL captures, the memory-profile
# bootstrap graph, and any piecewise/eager step keep the stock dispatch on both
# arms and are counted in the engagement record's bypass_counts.
#
# This paired screen is not the formal statistical hardware-floor acceptance
# gate. Promotion here means only fr13_b4_timing_math.promotion_verdict.
set -euo pipefail

case "${FR13_RUN_B4_QROW32_GQA_PAIR_TIMING:-0}" in
  1) ;;
  0)
    echo "B4 qrow32 GQA-pair timing pair is disabled" >&2
    echo "set FR13_RUN_B4_QROW32_GQA_PAIR_TIMING=1 to run it" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_B4_QROW32_GQA_PAIR_TIMING must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${QROW32_GQA_PAIR_FA2_SO:?set QROW32_GQA_PAIR_FA2_SO to the pinned candidate binary}"
: "${QROW32_GQA_PAIR_FA2_SOURCE:?set QROW32_GQA_PAIR_FA2_SOURCE to the regenerated FA2 source}"
: "${QROW32_GQA_PAIR_DUAL_GATE_JSON:?set it to the dual raw-byte gate PASS produced at HEAD}"
: "${QROW32_GQA_PAIR_DUAL_GATE_SHA256:?set it to that PASS artifact SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
FIXED32_MODE=${QROW32_GQA_PAIR_FIXED32_MODE:-hydra27_fixed32}
GATE=scripts/fr13_fa2_qrow32_gqa_pair_gate.py
SIDECAR=scripts/fr13_qrow32_b4_pass_sidecar.py
PATCH_SOURCE=scripts/fr13_patch_fa2_tree_bias.py
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
TASK_IDS=astropy__astropy-12907,astropy__astropy-13033,astropy__astropy-13236,astropy__astropy-13398
DRAFT_VOCAB_BLOCKS_HOST=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
CANDIDATE_SHA256=af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb
CANDIDATE_BYTES=299813360
FA2_HEAD=29210221863736a08f71a866459e368ad1ac4a95
SOURCE_CLOSURE_SHA256=9c3f9e751da7b783e9d07d8e40d5bc2234b99e719a1048668bd6c82244ed2d81
SELECTOR_SENTINEL=131092
B4_KV_CACHE_MEMORY_BYTES=49392123904
DRAFT_VOCAB_ROOT=1
DRAFT_VOCAB_K=65536
MANDATORY_WEIGHT_BYTES=27977022848
MANDATORY_WEIGHT_FLOOR_MS=102.479937172
ONE_SIDED_U95_CAP_MS=117.8519277478
SUMMARY_SCHEMA=fr13.fixed32.fa2_qrow32_gqa_pair_b4.full_wall_timing_pair.v1
RUN_CLASSIFICATION=real_swe_verified_exact4_b4_fa2_qrow32_gqa_pair_timing
LAUNCH_CLASSIFICATION=real_swe_verified_exact4_b4_fa2_qrow32_gqa_pair_timing_candidate
ENGAGEMENT_SCHEMA=fr13.fixed32.fa2_qrow32_b4_production_engagement.v1
ONLY_ARM_DELTA=FA2_stock_dispatch_to_qrow32_gqa_pair_with_candidate_side_bias_retag
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
GATE_SHA256=$(sha256sum "$GATE" | awk '{print $1}')
PATCH_SOURCE_SHA256=$(sha256sum "$PATCH_SOURCE" | awk '{print $1}')
SIDECAR_SHA256=$(sha256sum "$SIDECAR" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")

case "$FIXED32_MODE" in
  tail6_fixed32)
    LOGICAL_TOPOLOGY=Tail23
    ACTIVE_DRAFTS=23
    VALID_MASK=0x7a9ce7ff
    ;;
  hydra27_fixed32)
    LOGICAL_TOPOLOGY=Hydra27
    ACTIVE_DRAFTS=27
    VALID_MASK=0x7abdffff
    ;;
  *)
    echo "QROW32_GQA_PAIR_FIXED32_MODE must be tail6_fixed32 or hydra27_fixed32" >&2
    exit 2
    ;;
esac
STOCK_ARM="${FIXED32_MODE}_fa2_qrow32_stock_dispatch_b4_${TAG}"
CANDIDATE_ARM="${FIXED32_MODE}_fa2_qrow32_gqa_pair_b4_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for input in "$QROW32_GQA_PAIR_FA2_SO" "$QROW32_GQA_PAIR_DUAL_GATE_JSON"; do
  [[ "$input" == /* && -f "$input" && ! -L "$input" ]] \
    || { echo "timing input must be an absolute regular non-symlink file: $input" >&2; exit 2; }
done
[[ "$QROW32_GQA_PAIR_FA2_SOURCE" == /* \
   && -d "$QROW32_GQA_PAIR_FA2_SOURCE" \
   && ! -L "$QROW32_GQA_PAIR_FA2_SOURCE" ]] \
  || { echo "FA2 source must be an absolute non-symlink directory" >&2; exit 2; }
[[ "$(stat -c '%s' "$QROW32_GQA_PAIR_FA2_SO")" == "$CANDIDATE_BYTES" \
   && "$(sha256sum "$QROW32_GQA_PAIR_FA2_SO" | awk '{print $1}')" == "$CANDIDATE_SHA256" ]] \
  || { echo "QROW32_GQA_PAIR_FA2_SO is not the pinned dual-gate candidate" >&2; exit 2; }
[[ "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$QROW32_GQA_PAIR_DUAL_GATE_JSON" | awk '{print $1}')" == "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" ]] \
  || { echo "dual raw-byte gate PASS identity mismatch" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical exact4 subset SHA-256 drift" >&2; exit 2; }
[[ -f "$DRAFT_VOCAB_BLOCKS_HOST" && ! -L "$DRAFT_VOCAB_BLOCKS_HOST" \
   && "$(sha256sum "$DRAFT_VOCAB_BLOCKS_HOST" | awk '{print $1}')" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
  || { echo "pinned root-64K draft-vocabulary block map drifted" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
# The candidate binary and its regenerated source closure, independently of
# anything the byte gate wrote.
"$PYTHON_BIN" "$GATE" validate-candidate \
  --candidate-so "$QROW32_GQA_PAIR_FA2_SO" \
  --fa2-source "$QROW32_GQA_PAIR_FA2_SOURCE" \
  > "$RUNROOT_ABS/candidate_identity.at_launch.json"
# The dual-gate PASS this run will serve on: it must be bound to THIS commit,
# so a gate produced before the production plumbing existed cannot be reused.
"$PYTHON_BIN" "$SIDECAR" validate \
  --dual-gate "$QROW32_GQA_PAIR_DUAL_GATE_JSON" \
  --expected-dual-gate-sha256 "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" \
  --candidate-so "$QROW32_GQA_PAIR_FA2_SO" \
  --expected-candidate-sha256 "$CANDIDATE_SHA256" \
  --arm gqa_pair \
  --patch-source "$PATCH_SOURCE" \
  --expected-source-commit "$SOURCE_COMMIT" \
  > "$RUNROOT_ABS/dual_gate_binding.at_launch.json"
DUAL_GATE_BINDING_SHA256=$(
  sha256sum "$RUNROOT_ABS/dual_gate_binding.at_launch.json" | awk '{print $1}'
)

export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT"
export FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K"
export FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER"
export FR13_NEEDS_ALLOW=
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" ]] \
  || { echo "canonical B4 qualification floor contract drifted" >&2; exit 2; }

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

printf 'classification=%s\ntiming_eligible=1\nformal_floor_acceptance_eligible=0\nonly_arm_delta=%s\ncandidate_arm_selector=gqa_pair\nselector_sentinel=%s\ntopology=%s\nlogical_topology=%s\nactive_drafts=%s\nvalid_mask=%s\nphysical_drafts=31\nphysical_rows_root_inclusive=32\nbatch_size=4\nconcurrency=4\nfixed_rows=128\ntask_ids=%s\nsubset_sha256=%s\ndraft_vocab_root=%s\ndraft_vocab_k=%s\ndraft_vocab_blocks=%s\ndraft_vocab_blocks_sha256=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nlauncher_pid=%s\nrunroot=%s\nstock_arm=%s\ncandidate_arm=%s\nsource_commit=%s\ncandidate_so_sha256=%s\ncandidate_so_size=%s\nfa2_head=%s\nfa2_source_closure_sha256=%s\ndual_gate_sha256=%s\ndual_gate_binding_sha256=%s\nrunner_sha256=%s\ngate_sha256=%s\nsidecar_sha256=%s\npatch_source_sha256=%s\nenforce_eager=0\ncudagraph_mode=FULL_AND_PIECEWISE\nkv_cache_memory_bytes=%s\nstarted=%s\n' \
  "$LAUNCH_CLASSIFICATION" "$ONLY_ARM_DELTA" "$SELECTOR_SENTINEL" \
  "$FIXED32_MODE" "$LOGICAL_TOPOLOGY" "$ACTIVE_DRAFTS" "$VALID_MASK" \
  "$TASK_IDS" "$SUBSET_SHA256" "$DRAFT_VOCAB_ROOT" "$DRAFT_VOCAB_K" \
  "$DRAFT_VOCAB_BLOCKS_CONTAINER" "$DRAFT_VOCAB_BLOCKS_SHA256" \
  "$FR13_MANDATORY_WEIGHT_BYTES" "$FR13_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$$" "$RUNROOT_ABS" "$STOCK_ARM" "$CANDIDATE_ARM" \
  "$SOURCE_COMMIT" "$CANDIDATE_SHA256" "$CANDIDATE_BYTES" "$FA2_HEAD" \
  "$SOURCE_CLOSURE_SHA256" "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" \
  "$DUAL_GATE_BINDING_SHA256" "$RUNNER_SHA256" "$GATE_SHA256" \
  "$SIDECAR_SHA256" "$PATCH_SOURCE_SHA256" "$B4_KV_CACHE_MEMORY_BYTES" \
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
  "$PYTHON_BIN" "$GATE" validate-candidate \
    --candidate-so "$QROW32_GQA_PAIR_FA2_SO" \
    --fa2-source "$QROW32_GQA_PAIR_FA2_SOURCE" \
    > "$RUNROOT_ABS/candidate_identity.at_end.json" || return $?
  cmp -s "$RUNROOT_ABS/candidate_identity.at_launch.json" \
    "$RUNROOT_ABS/candidate_identity.at_end.json" \
    || { echo "candidate binary/source identity changed during timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" \
     && "$(sha256sum "$GATE" | awk '{print $1}')" == "$GATE_SHA256" \
     && "$(sha256sum "$SIDECAR" | awk '{print $1}')" == "$SIDECAR_SHA256" \
     && "$(sha256sum "$PATCH_SOURCE" | awk '{print $1}')" == "$PATCH_SOURCE_SHA256" ]] \
    || { echo "B4 GQA-pair timing source changed during execution" >&2; return 14; }
  [[ "$(sha256sum "$QROW32_GQA_PAIR_DUAL_GATE_JSON" | awk '{print $1}')" == "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" ]] \
    || { echo "dual raw-byte gate PASS changed during timing" >&2; return 14; }
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

# The ONLY difference between the two invocations below is
# FR13_FA2_QROW32_B4_PRODUCTION_ARM. Everything else -- the FA2 binary, the
# subset, the sampling, the topology, the vocabulary, the graph mode -- is
# byte-for-byte the same.
run_arm() {
  local arm=$1
  local timing_arm=$2
  local production_arm=$3
  local dual_gate_json=""
  local dual_gate_sha=""
  if [[ -n "$production_arm" ]]; then
    dual_gate_json=$QROW32_GQA_PAIR_DUAL_GATE_JSON
    dual_gate_sha=$QROW32_GQA_PAIR_DUAL_GATE_SHA256
  fi
  echo "===== $arm: exact4 B4 FA2 timing_arm=$timing_arm production_arm=${production_arm:-none} ====="
  if env \
      RUNROOT="$RUNROOT_ABS" \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S=5400 \
      KV_CACHE_MEMORY_BYTES="$B4_KV_CACHE_MEMORY_BYTES" \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT" \
      FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K" \
      FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER" \
      FR13_NEEDS_ALLOW= \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
      FR13_SCAN_ALIGN=0 FR13_NPAD_INVARIANT=0 \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json" \
      FR13_FA2_QROW32_B4_TIMING_ARM="$timing_arm" \
      FR13_FA2_QROW32_B4_PRODUCTION_ARM="$production_arm" \
      FR13_FA2_QROW32_B4_DUAL_GATE_JSON="$dual_gate_json" \
      FR13_FA2_QROW32_B4_DUAL_GATE_SHA256="$dual_gate_sha" \
      FR13_FA2_QROW32_B4_EXACT4_TASK_IDS="$TASK_IDS" \
      FR13_FA2_QROW32_B4_EXACT4_SUBSET_SHA256="$SUBSET_SHA256" \
      FR13_FA2_QROW32_B4_PATCH_SOURCE_SHA256="$PATCH_SOURCE_SHA256" \
      FR13_FA2_QROW32_SO_SHA256="$CANDIDATE_SHA256" \
      FR13_FA2_QROW32_SO_SIZE="$CANDIDATE_BYTES" \
      FR13_FA2_QROW32_FA2_HEAD="$FA2_HEAD" \
      FR13_FA2_QROW32_SOURCE_CLOSURE_SHA256="$SOURCE_CLOSURE_SHA256" \
      FR13_FA2_QROW32_SOURCE_COMMIT="$SOURCE_COMMIT" \
      FR13_FA2_QROW32_LIVE_PAGED_AB=0 \
      FR13_FA2_QROW32_LIVE_PAGED_AB_ARM= \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
      FR13_FA2_QROW32_B1_LIVE_AB_ARM= FR13_FA2_QROW32_B1_PRODUCTION_ARM= \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_INSTANCE_ID= \
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
      FR13_FIXED32_CUTLASS_WAVE=stock \
      FR13_FIXED32_CUTLASS_WAVE_SO= \
      FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$QROW32_GQA_PAIR_FA2_SO" \
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
  [[ "$(grep -Fxc "FR13_FIXED32_MODE=$FIXED32_MODE" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_DRAFT_VOCAB_ROOT=$DRAFT_VOCAB_ROOT" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_DRAFT_VOCAB_K=$DRAFT_VOCAB_K" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_DRAFT_VOCAB_BLOCKS=$DRAFT_VOCAB_BLOCKS_CONTAINER" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW32_B4_TIMING_ARM=$timing_arm" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW32_B4_PRODUCTION_ARM=$production_arm" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW32_SO_SHA256=$CANDIDATE_SHA256" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW32_LIVE_PAGED_AB=0" "$container_env")" -eq 1 ]] \
    || { echo "$arm did not run the declared single-variable B4 selector" >&2; return 4; }
  local engagement="$RUNROOT_ABS/$arm/logs/fr13_fa2_qrow32_b4_production_engagement.json"
  if [[ -n "$production_arm" ]]; then
    [[ -f "$engagement" && ! -L "$engagement" ]] \
      || { echo "$arm lacks the GQA-pair production engagement artifact" >&2; return 4; }
    local sidecar="$RUNROOT_ABS/$arm/logs/fr13_fa2_qrow32_b4_production_pass.json"
    [[ -f "$sidecar" && ! -L "$sidecar" ]] \
      || { echo "$arm lacks the GQA-pair production credential" >&2; return 4; }
  else
    # A stock-dispatch arm that emitted an engagement record would mean the
    # sentinel leaked across the pair, which would invalidate the comparison.
    [[ ! -e "$engagement" && ! -L "$engagement" ]] \
      || { echo "$arm emitted a GQA-pair engagement on the stock-dispatch arm" >&2; return 4; }
  fi
  # B=4 co-residency nests the per-task /metrics brackets, so a naive sum
  # multiply-counts the shared prefix. fr13_measure.py reduces on the
  # arm-authoritative bracket and cross-gates that against this arm's own
  # topology-blind engine work census. An ungated B4 aggregate is exactly the
  # artifact the alignment study invalidated, so the census is mandatory.
  local deploy_census="$RUNROOT_ABS/$arm/logs/fr13_fixed32_work_census.jsonl"
  [[ -f "$deploy_census" && ! -L "$deploy_census" ]] \
    || { echo "$arm lacks the work census the B4 bracket reduction is gated on" >&2; return 4; }
  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$arm" --out-root "$RUNROOT_ABS/$arm/swe_out" \
    --expected-tok-per-draft 31 --batch-size 4 \
    --work-census "$deploy_census" \
    --out "$RUNROOT_ABS/$arm/deploy_speed_fullwall.json"
  printf 'arm=%s serve_rc=0 container_env_sha256=%s ended=%s\n' \
    "$arm" "$(sha256sum "$container_env" | awk '{print $1}')" \
    "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
}

run_arm "$STOCK_ARM" stock_dispatch ""
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after the stock reference" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" gqa_pair gqa_pair

CANDIDATE_ENGAGEMENT="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fa2_qrow32_b4_production_engagement.json"
CANDIDATE_SIDECAR="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fa2_qrow32_b4_production_pass.json"
CANDIDATE_SIDECAR_SHA256=$(sha256sum "$CANDIDATE_SIDECAR" | awk '{print $1}')
"$PYTHON_BIN" "$SIDECAR" verify \
  --sidecar "$CANDIDATE_SIDECAR" \
  --expected-sidecar-sha256 "$CANDIDATE_SIDECAR_SHA256" \
  --candidate-so "$QROW32_GQA_PAIR_FA2_SO" \
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
  "$CANDIDATE_SIDECAR_SHA256" "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" \
  "$DUAL_GATE_BINDING_SHA256" "$CANDIDATE_SHA256" "$CANDIDATE_BYTES" \
  "$FA2_HEAD" "$SOURCE_CLOSURE_SHA256" "$SOURCE_COMMIT" \
  "$PATCH_SOURCE_SHA256" "$SUMMARY_SCHEMA" "$RUN_CLASSIFICATION" \
  "$ENGAGEMENT_SCHEMA" "$SELECTOR_SENTINEL" \
  "$DRAFT_VOCAB_ROOT" "$DRAFT_VOCAB_K" \
  "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$DRAFT_VOCAB_BLOCKS_CONTAINER" \
  "$DRAFT_VOCAB_BLOCKS_SHA256" "$FIXED32_MODE" "$LOGICAL_TOPOLOGY" \
  "$ACTIVE_DRAFTS" "$VALID_MASK" "$ONLY_ARM_DELTA" \
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
sidecar_sha256, dual_gate_sha256, dual_gate_binding_sha256 = sys.argv[7:10]
candidate_sha256 = sys.argv[10]
candidate_bytes = int(sys.argv[11])
fa2_head, source_closure_sha256, source_commit = sys.argv[12:15]
patch_source_sha256, summary_schema, run_classification = sys.argv[15:18]
engagement_schema = sys.argv[18]
selector_sentinel = int(sys.argv[19])
draft_vocab_root, draft_vocab_k, mandatory_weight_bytes = map(int, sys.argv[20:23])
mandatory_weight_floor_ms = float(sys.argv[23])
one_sided_u95_cap_ms = float(sys.argv[24])
draft_vocab_blocks, draft_vocab_blocks_sha256 = sys.argv[25:27]
fixed32_mode, logical_topology = sys.argv[27:29]
active_drafts = int(sys.argv[29])
valid_mask = int(sys.argv[30], 0)
only_arm_delta = sys.argv[31]
stock_ingress_path, candidate_ingress_path = map(Path, sys.argv[32:34])

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
        or record.get("batch_size") != 4
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
        raise SystemExit(f"{label} deploy-speed provenance is not exact4 B4")
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
    or engagement.get("fixed32_mode") != fixed32_mode
    or engagement.get("batch_size") != 4
    or engagement.get("total_query_rows") != 128
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
    or engagement.get("dual_gate_sha256") != dual_gate_sha256
    or engagement.get("task_ids") != task_ids
    or engagement.get("candidate_served") is not True
    or engagement.get("fallback_allowed") is not False
    # The candidate is qualified for the final FULL graph at the widths named
    # by candidate_scope; every other decode the runtime must execute keeps
    # stock and is counted there. MARK'S RULING 2026-08-13 moved that token
    # from final_fixed32_b4_full_graph_only to the b3-inclusive
    # final_fixed32_b34_full_graph_only. BOTH are accepted on read so the
    # sealed +29.50 ms/step width-4 pair stays re-verifiable; the batch_size
    # and total_query_rows clauses above still pin THIS run to width 4.
    or engagement.get("candidate_scope") not in (
        "final_fixed32_b4_full_graph_only",
        "final_fixed32_b34_full_graph_only",
    )
    or not isinstance(engagement.get("bypass_counts"), dict)
):
    raise SystemExit("candidate arm did not serve the GQA-pair kernel on every layer")
if (
    sidecar.get("schema") != "fr13.fixed32.fa2_qrow32_b4_production_pass.v1"
    or sidecar.get("status") != "PASS"
    or sidecar.get("arm") != "gqa_pair"
    or sidecar.get("selector_sentinel") != selector_sentinel
    or sidecar.get("candidate_so_sha256") != candidate_sha256
    or sidecar.get("source_commit") != source_commit
    or sidecar.get("patch_source_sha256") != patch_source_sha256
    or sidecar.get("dual_gate_sha256") != dual_gate_sha256
    or sidecar.get("qualified_topologies") != ["Tail23", "Hydra27"]
):
    raise SystemExit("candidate arm credential is not the bound dual-gate credential")

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
summary = {
    "schema": summary_schema,
    "status": "complete",
    "run_classification": run_classification,
    "only_arm_delta": only_arm_delta,
    # The delta is the served dispatch plus the operand retag that selects it.
    # Both extra costs land on the candidate, so the pair can only understate a
    # gain -- it cannot manufacture one.
    "arm_delta_disclosure": {
        "served_dispatch": "stock_fa2 -> qrow32_gqa_pair",
        "candidate_only_overhead": (
            "per-target-layer empty_strided((4,32,32),(131092,32,1)) + copy_ "
            "recorded into the captured graph and replayed every step"
        ),
        "bias_operand": "2d_broadcast_tile -> 4_plane_batch_strided",
        "overhead_charged_to": "candidate",
        "bias_direction": "conservative_against_candidate",
        "candidate_scope": engagement.get("candidate_scope"),
        "regression_verdict_is_confounded_by_harness": True,
    },
    "task_count": 4,
    "batch_size": 4,
    "concurrency": 4,
    "arm": fixed32_mode,
    "logical_topology": logical_topology,
    "active_drafts": active_drafts,
    "valid_mask": hex(valid_mask),
    "physical_drafts": 31,
    "physical_rows_root_inclusive": 32,
    "sfwd_projection_rows": 128,
    "authenticated_task_count": 4,
    "authenticated_task_ids": task_ids,
    "authenticated_task_set_sha256": expected_task_set_sha256,
    "authenticated_task_provenance": {
        "stock_reference": stock_ingress,
        "candidate": candidate_ingress,
    },
    "task_ids": task_ids,
    "subset_sha256": hashlib.sha256(subset_path.read_bytes()).hexdigest(),
    "decision_metric": "measured_tps_fullstep_wall",
    "promotion": promotion_verdict(stock_phases, candidate_phases),
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
    "fa2_so_sha256": candidate_sha256,
    "fa2_so_size": candidate_bytes,
    "fa2_head": fa2_head,
    "fa2_source_closure_sha256": source_closure_sha256,
    "dual_gate_sha256": dual_gate_sha256,
    "dual_gate_binding_sha256": dual_gate_binding_sha256,
    "stock_reference": {
        "selector": "stock_dispatch",
        "selector_sentinel": None,
        "fa2_so_sha256": candidate_sha256,
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
        "Tail/Hydra floor gate after a positive screen"
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

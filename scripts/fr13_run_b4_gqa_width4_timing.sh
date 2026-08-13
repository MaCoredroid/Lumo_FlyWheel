#!/usr/bin/env bash
# WIDTH-4 real SWE-Verified B4 timing pair: stock FA2 dispatch vs the
# dual-byte-gate-qualified qrow32 GQA-pair kernel, measured AT THE WIDTH-4
# OPERATING POINT rather than on the exact4 wave.
#
# WHY THIS RUNNER EXISTS (the whole point -- read this before touching anything)
# ----------------------------------------------------------------------------
# The exact4 GQA-pair timing pair (2026-08-11, runroot
# output/fr13_fa2_gqa_pair_b4_timing_20260811T015257Z @ 8940cd1ba) returned a
# NULL: after decomposing away an acceptance-drift and a co-residency confound,
# the kernel-attributable residual was a +1.92 ms/step COST, statistically
# indistinguishable from the disclosed candidate-side sentinel-retag overhead.
#
# That null was measured against a kernel a THIRD of its operating size. The
# width-4 Nsight attribution (results/fr13_b4_width4_nsys_20260813/) showed that
# at exact4 the batch is full width for only ~36% of the arm and events/step ran
# 1.15-1.22, whereas at the width-4 operating point FA2 is 3.26x LARGER -- 69.75
# ms of a 411 ms batch-conditioned step wall, carrying ~52 ms of structural
# headroom above its own bandwidth floor. The GQA-pair kernel's mechanism (halve
# the KV re-staging by pairing GQA heads) scales with exactly the quantity that
# grew. A null against a 21 ms kernel says very little about a 70 ms one.
#
# So this is a RE-TEST at the regime where the lever is supposed to bite, and it
# is designed to be able to REVERSE the prior verdict -- or to close the lever
# honestly if it nulls again.
#
# WHAT IS THE SAME AS THE EXACT4 PAIR, AND WHAT IS DIFFERENT
# ----------------------------------------------------------
# IDENTICAL, by construction:
#   * the pinned candidate binary (both arms load it -- the delta is the served
#     dispatch selector, never a binary swap),
#   * the served GEOMETRY: MAX_NUM_SEQS=4, SWE_CONCURRENCY=4, 4 slots x 32
#     physical rows = 128 query rows, 16 target layers, FULL_AND_PIECEWISE,
#     K64/root1 vocabulary -- i.e. exactly the shape the dual raw-byte gate
#     qualified,
#   * the single-variable arm delta and its honest disclosure: the candidate arm
#     ALSO pays the per-target-layer bias retag that selects the kernel, and that
#     cost stays charged to the candidate, so the pair remains conservative.
#
# DIFFERENT, deliberately:
#   * the ADMISSION POOL. Arms run EVIDENCE_SETS[16] (config/fr13_fixed32/
#     subset_b4_sixteen.json) behind the same 4 slots with FR13_B4_TASK_REFILL=1,
#     so a finishing task is replaced instead of letting the width decay. exact4
#     is the degenerate pool where pool == slots.
#   * the VERDICT INSTRUMENT. The arm-level number is NOT the verdict here: a
#     pool16 arm is a wall-blended MIXTURE of a full-width phase and an exact4-
#     shaped drain tail, and its events/step is the mixture weight, not a rate
#     (sealed finding, 2026-08-13). The verdict is the WIDTH-4 DEPTH WINDOW of
#     each arm, reduced by scripts/fr13_b4_gqa_width4_pair_reduce.py on top of
#     the sealed windowing math in scripts/fr13_b4_width4_window_reduce.py.
#   * NO AGENT WALL. The width-4 baseline this pair is judged against was
#     measured with WALL=0, and a wall would truncate tasks and deform the very
#     admission ledger that DEFINES the window. AGENT_WALL_S is passed empty.
#
# WHY THE CANDIDATE IS STILL SERVED ONLY WHERE IT IS QUALIFIED
# ------------------------------------------------------------
# The dual raw-byte gate qualified a GEOMETRY, not a task list. Pool16 at 4 slots
# serves that identical geometry; only which tasks occupy the slots changes. The
# launcher and the in-container patcher both admit exactly the two byte-pinned
# campaign evidence sets (4 and 16) and nothing else, so an unpinned task list
# still cannot reach the candidate dispatch. The candidate scope is unchanged:
# final fixed32 B4 FULL graph only, every other capture counted as a bypass.
#
# This paired screen is not the formal statistical hardware-floor acceptance
# gate, and the width-4 window class is an INSTRUMENT, not a citable seal.
set -euo pipefail

case "${FR13_RUN_B4_GQA_WIDTH4_TIMING:-0}" in
  1) ;;
  0)
    echo "B4 GQA-pair width-4 timing pair is disabled" >&2
    echo "set FR13_RUN_B4_GQA_WIDTH4_TIMING=1 to run it" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_B4_GQA_WIDTH4_TIMING must be exactly 0 or 1" >&2
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
PAIR_REDUCER=scripts/fr13_b4_gqa_width4_pair_reduce.py
WINDOW_REDUCER=scripts/fr13_b4_width4_window_reduce.py
# EVIDENCE_SETS[16] -- byte-pinned identically to fr13_floor_gate.EVIDENCE_SETS.
SUBSET=config/fr13_fixed32/subset_b4_sixteen.json
SUBSET_SHA256=47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c
TASK_IDS=astropy__astropy-12907,astropy__astropy-13033,astropy__astropy-13236,astropy__astropy-13398,astropy__astropy-13453,astropy__astropy-13579,astropy__astropy-13977,astropy__astropy-14096,astropy__astropy-14182,astropy__astropy-14309,astropy__astropy-14365,astropy__astropy-14369,astropy__astropy-14508,astropy__astropy-14539,astropy__astropy-14598,astropy__astropy-14995
TASK_COUNT=16
SLOTS=4
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
MANDATORY_WEIGHT_BYTES=32666638208
MANDATORY_WEIGHT_FLOOR_MS=119.658015414
ONE_SIDED_U95_CAP_MS=137.6067177261
RUN_CLASSIFICATION=real_swe_verified_pool16_b4_fa2_qrow32_gqa_pair_width4_timing
LAUNCH_CLASSIFICATION=real_swe_verified_pool16_b4_fa2_qrow32_gqa_pair_width4_timing_candidate
ENGAGEMENT_SCHEMA=fr13.fixed32.fa2_qrow32_b4_production_engagement.v1
ONLY_ARM_DELTA=FA2_stock_dispatch_to_qrow32_gqa_pair_with_candidate_side_bias_retag
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
GATE_SHA256=$(sha256sum "$GATE" | awk '{print $1}')
PATCH_SOURCE_SHA256=$(sha256sum "$PATCH_SOURCE" | awk '{print $1}')
SIDECAR_SHA256=$(sha256sum "$SIDECAR" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
# The sealed width-4 window reducer discovers arms as <root>/pass_NN/<mode>_*, so
# serving both arms into pass_00 lets that reducer run over this pair UNMODIFIED
# as an independent second read of the same bytes.
PASS_DIR="$RUNROOT_ABS/pass_00"

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
STOCK_ARM="${FIXED32_MODE}_gqa_w4_stock_dispatch_b4_${TAG}"
CANDIDATE_ARM="${FIXED32_MODE}_gqa_w4_gqa_pair_b4_${TAG}"

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
  || { echo "canonical 16-task pool subset SHA-256 drift" >&2; exit 2; }
[[ -f "$DRAFT_VOCAB_BLOCKS_HOST" && ! -L "$DRAFT_VOCAB_BLOCKS_HOST" \
   && "$(sha256sum "$DRAFT_VOCAB_BLOCKS_HOST" | awk '{print $1}')" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
  || { echo "pinned root-64K draft-vocabulary block map drifted" >&2; exit 2; }
[[ -f "$PAIR_REDUCER" && ! -L "$PAIR_REDUCER" ]] \
  || { echo "missing the width-4 pair reducer: $PAIR_REDUCER" >&2; exit 2; }
[[ -f "$WINDOW_REDUCER" && ! -L "$WINDOW_REDUCER" ]] \
  || { echo "missing the sealed width-4 window reducer: $WINDOW_REDUCER" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

# EVERY PRECONDITION MUST BE ONE THE RUN CAN SATISFY. Six separate campaign
# fossils were runners bound to an artifact nothing ever wrote, so the reducer
# and the window class are resolved BEFORE any GPU time is spent.
"$PYTHON_BIN" "$PAIR_REDUCER" --self-check \
  || { echo "width-4 pair reducer failed its own self-check" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS" "$PASS_DIR"
# The candidate binary and its regenerated source closure, independently of
# anything the byte gate wrote.
"$PYTHON_BIN" "$GATE" validate-candidate \
  --candidate-so "$QROW32_GQA_PAIR_FA2_SO" \
  --fa2-source "$QROW32_GQA_PAIR_FA2_SOURCE" \
  > "$RUNROOT_ABS/candidate_identity.at_launch.json"
# The dual-gate PASS this run will serve on: it must be bound to THIS commit, so
# a gate produced before this runner existed cannot be reused. Adapting the
# runner moves HEAD, so the gate is re-run at the final HEAD before this pair.
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

printf 'classification=%s\ntiming_eligible=1\nformal_floor_acceptance_eligible=0\nonly_arm_delta=%s\ncandidate_arm_selector=gqa_pair\nselector_sentinel=%s\ntopology=%s\nlogical_topology=%s\nactive_drafts=%s\nvalid_mask=%s\nphysical_drafts=31\nphysical_rows_root_inclusive=32\nbatch_size=4\nconcurrency=4\nslots=%s\ntask_pool=%s\ntask_refill=1\nagent_wall=none\nfixed_rows=128\ntask_ids=%s\nsubset=%s\nsubset_sha256=%s\ndraft_vocab_root=%s\ndraft_vocab_k=%s\ndraft_vocab_blocks=%s\ndraft_vocab_blocks_sha256=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nlauncher_pid=%s\nrunroot=%s\npass_dir=%s\nstock_arm=%s\ncandidate_arm=%s\nsource_commit=%s\ncandidate_so_sha256=%s\ncandidate_so_size=%s\nfa2_head=%s\nfa2_source_closure_sha256=%s\ndual_gate_sha256=%s\ndual_gate_binding_sha256=%s\nrunner_sha256=%s\ngate_sha256=%s\nsidecar_sha256=%s\npatch_source_sha256=%s\nenforce_eager=0\ncudagraph_mode=FULL_AND_PIECEWISE\nkv_cache_memory_bytes=%s\nstarted=%s\n' \
  "$LAUNCH_CLASSIFICATION" "$ONLY_ARM_DELTA" "$SELECTOR_SENTINEL" \
  "$FIXED32_MODE" "$LOGICAL_TOPOLOGY" "$ACTIVE_DRAFTS" "$VALID_MASK" \
  "$SLOTS" "$TASK_COUNT" \
  "$TASK_IDS" "$SUBSET" "$SUBSET_SHA256" "$DRAFT_VOCAB_ROOT" "$DRAFT_VOCAB_K" \
  "$DRAFT_VOCAB_BLOCKS_CONTAINER" "$DRAFT_VOCAB_BLOCKS_SHA256" \
  "$FR13_MANDATORY_WEIGHT_BYTES" "$FR13_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$$" "$RUNROOT_ABS" "$PASS_DIR" \
  "$STOCK_ARM" "$CANDIDATE_ARM" \
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
    || { echo "B4 GQA-pair width-4 timing source changed during execution" >&2; return 14; }
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
# subset, the pool depth, the refill flag, the sampling, the topology, the
# vocabulary, the graph mode -- is byte-for-byte the same.
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
  echo "===== $arm: pool16 B4 FA2 timing_arm=$timing_arm production_arm=${production_arm:-none} ====="
  # AGENT_WALL_S is passed EMPTY on purpose: the width-4 baseline was measured
  # with no wall, and a wall would truncate tasks and deform the admission
  # ledger that defines the window.
  if env \
      RUNROOT="$PASS_DIR" \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S= \
      FR13_B4_TASK_REFILL=1 \
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
  local arm_dir="$PASS_DIR/$arm"
  local container_env="$arm_dir/container_env.txt"
  [[ -f "$container_env" && ! -L "$container_env" ]] \
    || { echo "$arm lacks a regular container environment artifact" >&2; return 4; }
  [[ "$(grep -Fxc "FR13_FIXED32_MODE=$FIXED32_MODE" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_DRAFT_VOCAB_ROOT=$DRAFT_VOCAB_ROOT" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_DRAFT_VOCAB_K=$DRAFT_VOCAB_K" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_DRAFT_VOCAB_BLOCKS=$DRAFT_VOCAB_BLOCKS_CONTAINER" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_B4_TASK_REFILL=1" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW32_B4_TIMING_ARM=$timing_arm" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW32_B4_PRODUCTION_ARM=$production_arm" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW32_SO_SHA256=$CANDIDATE_SHA256" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW32_LIVE_PAGED_AB=0" "$container_env")" -eq 1 ]] \
    || { echo "$arm did not run the declared single-variable B4 pool16 selector" >&2; return 4; }
  local engagement="$arm_dir/logs/fr13_fa2_qrow32_b4_production_engagement.json"
  if [[ -n "$production_arm" ]]; then
    [[ -f "$engagement" && ! -L "$engagement" ]] \
      || { echo "$arm lacks the GQA-pair production engagement artifact" >&2; return 4; }
    local sidecar="$arm_dir/logs/fr13_fa2_qrow32_b4_production_pass.json"
    [[ -f "$sidecar" && ! -L "$sidecar" ]] \
      || { echo "$arm lacks the GQA-pair production credential" >&2; return 4; }
  else
    # A stock-dispatch arm that emitted an engagement record would mean the
    # sentinel leaked across the pair, which would invalidate the comparison.
    [[ ! -e "$engagement" && ! -L "$engagement" ]] \
      || { echo "$arm emitted a GQA-pair engagement on the stock-dispatch arm" >&2; return 4; }
  fi
  # THE WINDOW IS DEFINED BY THIS LEDGER. Without it there is no width-4 phase
  # to reduce and the whole verdict is vacuous, so it is required here rather
  # than discovered missing hours later at reduce time.
  local ledger="$arm_dir/swe_out/verified/fr13_task_refill_ledger.jsonl"
  local ledger_summary="$arm_dir/swe_out/verified/fr13_task_refill_summary.json"
  [[ -f "$ledger" && ! -L "$ledger" && -f "$ledger_summary" && ! -L "$ledger_summary" ]] \
    || { echo "$arm lacks the admission ledger the width-4 window is DEFINED by" >&2; return 4; }
  # A pool arm's per-task brackets are STAGGERED, and fr13_measure REFUSES a
  # staggered reduction without the engine work census, so it is mandatory.
  local deploy_census="$arm_dir/logs/fr13_fixed32_work_census.jsonl"
  [[ -f "$deploy_census" && ! -L "$deploy_census" ]] \
    || { echo "$arm lacks the work census the B4 bracket reduction is gated on" >&2; return 4; }
  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$arm" --out-root "$arm_dir/swe_out" \
    --expected-tok-per-draft 31 --batch-size 4 \
    --work-census "$deploy_census" \
    --out "$arm_dir/deploy_speed_fullwall.json"
  printf 'arm=%s serve_rc=0 container_env_sha256=%s ended=%s\n' \
    "$arm" "$(sha256sum "$container_env" | awk '{print $1}')" \
    "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
}

run_arm "$STOCK_ARM" stock_dispatch ""
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after the stock reference" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" gqa_pair gqa_pair

CANDIDATE_SIDECAR="$PASS_DIR/$CANDIDATE_ARM/logs/fr13_fa2_qrow32_b4_production_pass.json"
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

# ------------------------------------------------------------------ verdict ---
# ALL timing math lives in the reducer, over the sealed window class. The runner
# serves and attests; it does not decide.
"$PYTHON_BIN" "$PAIR_REDUCER" \
  --runroot "$RUNROOT_ABS" \
  --stock-arm "$PASS_DIR/$STOCK_ARM" \
  --candidate-arm "$PASS_DIR/$CANDIDATE_ARM" \
  --mode "$FIXED32_MODE" \
  --source-commit "$SOURCE_COMMIT" \
  --subset "$SUBSET" \
  --dual-gate-sha256 "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" \
  --dual-gate-binding-sha256 "$DUAL_GATE_BINDING_SHA256" \
  --candidate-sidecar-sha256 "$CANDIDATE_SIDECAR_SHA256" \
  --patch-source-sha256 "$PATCH_SOURCE_SHA256" \
  --out "$RUNROOT_ABS/width4_timing_pair.json"
REDUCE_RC=$?

# An INDEPENDENT second read of the same bytes by the sealed window reducer,
# unmodified. It cannot return a verdict for a 2-arm pair (it wants 4 passes on
# both topologies), and it is not asked to -- it is run for its per-arm windowed
# records, which must agree with the pair reducer's.
"$PYTHON_BIN" "$WINDOW_REDUCER" \
  --gate-root "$RUNROOT_ABS" \
  --source-commit "$SOURCE_COMMIT" \
  --out "$RUNROOT_ABS/width4_window_independent_read.json" || true

printf 'width4_timing_pair=%s reduce_rc=%s completed=%s\n' \
  "$RUNROOT_ABS/width4_timing_pair.json" "$REDUCE_RC" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
exit "$REDUCE_RC"

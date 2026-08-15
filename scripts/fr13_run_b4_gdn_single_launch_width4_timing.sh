#!/usr/bin/env bash
# WIDTH-4 real SWE-Verified B4 timing pair for the folded GDN scan kernel:
# the deployed two-launch reference vs fixed32_gdn_single_launch_tree_v2,
# measured AT THE WIDTH-4 OPERATING POINT.
#
# WHY THIS RUNNER EXISTS
# ----------------------------------------------------------------------------
# Phase 0 priced the fold directly at b=4 -- two_launch 861.504 us per
# layer-batch against single_launch 674.336, a saving of 8.984 ms/step over 48
# layers, 2.14x the 4.20 ms sealed MDE. That is a MICROBENCHMARK. Phase 1 proved
# the kernel byte-identical on real traffic at width 4 (1,584 records across
# seven surfaces, raw_byte_equal). Neither says what the lever is worth on the
# served wall, which is the only number that decides whether a sealing campaign
# is justified. This is that screen.
#
# THE PAIR IS SINGLE-VARIABLE, AND THE VARIABLE IS THE SERVED KERNEL
# ------------------------------------------------------------------
# Both arms are the SAME commit, source closure and served geometry. The delta
# is FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION alone:
#   control    named 0  -- the deliberate opt-out the launcher obeys verbatim.
#                         NAMED, not unset: an unset selector would let the
#                         registry default pick the control, and the control
#                         must be the incumbent regardless of what ships.
#   candidate  named 1 + a HEAD-bound credential presented.
#
# WHY THE SCREEN RUNS THROUGH THE PRODUCTION ARM
# -----------------------------------------------
# Because it is the only route that serves this kernel. The credential-free bool
# FR13_FIXED32_GDN_SINGLE_LAUNCH is structurally unreachable -- its sidecar is
# only ever removed by the launcher and never written, and the variable is never
# exported into the container. The live byte gate does not serve it either: it
# routes through FR13_FIXED32_GDN_PATH_BV_CANDIDATE to a separate capture route
# and serves the REFERENCE while shadowing the candidate, which is exactly why
# its credential records reference_served=true.
#
# That is the better instrument anyway. The production arm carries an ENGAGEMENT
# NEEDLE that proves per decode call that the fold replaced the incumbent launch
# -- one physical launch, grid z equal to the batch, zero state export writes and
# zero parent reads. A silent fallback cannot produce those zeros, so this pair
# cannot be a measurement of the incumbent wearing the candidate's name.
#
# DISCLOSED CANDIDATE-SIDE COST -- THE PAIR IS CONSERVATIVE
# ----------------------------------------------------------
# The candidate arm also PAYS for that needle, and both arms run FR10_METRICS=1
# because the production contract requires the invocation counter. Neither cost
# is subtracted. As with the FA2 pair's bias retag, the overhead that selects and
# proves the candidate stays charged to the candidate, so this pair can only
# understate a gain.
#
# NOTE ON COMPARABILITY: the FA2 width-4 baseline ran FR10_METRICS=0. These
# absolute walls are therefore NOT comparable to that baseline. Only the
# within-pair contrast is, which is all the verdict uses.
#
# THE PLACEBO IS FREE AND WITHIN-ARM
# -----------------------------------
# The selector folds only when the engine batch is exactly 4. Inside a pool16 arm
# the batch varies 1-4 step to step, so the same arm contains treated (width 4)
# and untreated (widths 1-3) steps, sharing host, page cache and task mix.
#
# This paired screen is NOT the formal statistical hardware-floor acceptance
# gate, and the width-4 window class is an INSTRUMENT, not a citable seal. It can
# halt the lever; only phase 4 can seal it.
set -euo pipefail

case "${FR13_RUN_B4_GDN_SINGLE_LAUNCH_WIDTH4_TIMING:-0}" in
  1) ;;
  0)
    echo "B4 GDN single-launch width-4 timing pair is disabled" >&2
    echo "set FR13_RUN_B4_GDN_SINGLE_LAUNCH_WIDTH4_TIMING=1 to run it" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_B4_GDN_SINGLE_LAUNCH_WIDTH4_TIMING must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${GDN_SL_CREDENTIAL:?set GDN_SL_CREDENTIAL to the live-gate PASS credential produced at HEAD}"
# NO APOSTROPHES IN A ${VAR:?word} MESSAGE. Bash still tracks quoting inside the
# expansion, so a lone ' opens a quote and the parse breaks tens of lines later
# at whatever token follows -- a syntax error that points nowhere near its cause.
: "${GDN_SL_CREDENTIAL_SHA256:?set GDN_SL_CREDENTIAL_SHA256 to the SHA-256 of that credential}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the forked FA2 binary the tree serves on}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
FIXED32_MODE=${GDN_SL_FIXED32_MODE:-hydra27_fixed32}
PRODUCTION_BATCH=${GDN_SL_PRODUCTION_BATCH:-4}

PAIR_REDUCER=scripts/fr13_b4_gdn_single_launch_width4_pair_reduce.py
WINDOW_REDUCER=scripts/fr13_b4_width4_window_reduce.py
CREDENTIAL_VALIDATOR=scripts/fr13_gdn_single_launch_production_credential.py
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
PATCH_SOURCE=scripts/fr10_phase4_patch_vllm_tree_gdn.py

SUBSET=config/fr13_fixed32/subset_b4_sixteen.json
SUBSET_SHA256=47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c
TASK_IDS=astropy__astropy-12907,astropy__astropy-13033,astropy__astropy-13236,astropy__astropy-13398,astropy__astropy-13453,astropy__astropy-13579,astropy__astropy-13977,astropy__astropy-14096,astropy__astropy-14182,astropy__astropy-14309,astropy__astropy-14365,astropy__astropy-14369,astropy__astropy-14508,astropy__astropy-14539,astropy__astropy-14598,astropy__astropy-14995
TASK_COUNT=16
SLOTS=4
DRAFT_VOCAB_BLOCKS_HOST=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
B4_KV_CACHE_MEMORY_BYTES=49392123904
DRAFT_VOCAB_ROOT=1
DRAFT_VOCAB_K=65536
MANDATORY_WEIGHT_BYTES=32666638208
MANDATORY_WEIGHT_FLOOR_MS=119.658015414
RUN_CLASSIFICATION=real_swe_verified_pool16_b4_gdn_single_launch_width4_timing
LAUNCH_CLASSIFICATION=real_swe_verified_pool16_b4_gdn_single_launch_width4_timing_candidate
ONLY_ARM_DELTA=GDN_two_launch_reference_to_single_launch_fold_with_candidate_side_engagement_needle

SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
PATCH_SOURCE_SHA256=$(sha256sum "$PATCH_SOURCE" | awk '{print $1}')
REDUCER_SHA256=$(sha256sum "$PAIR_REDUCER" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
# Validated HERE, before PASS_ROOT, because PASS_ROOT DEFAULTS to RUNROOT_ABS: a
# bad RUNROOT would otherwise surface as a confusing PASS_ROOT error.
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }

# The sealed width-4 window reducer discovers arms as <root>/pass_NN/<mode>_*, so
# serving both arms into pass_00 lets that reducer run over this pair UNMODIFIED
# as an independent second read of the same bytes. A later multi-pass campaign
# passes its own PASS_ROOT and per-pass index and lands in one root as
# pass_00..pass_NN, which is the shape that reducer wanted in the first place.
PASS_INDEX=${PASS_INDEX:-0}
PASS_ROOT=${PASS_ROOT:-$RUNROOT_ABS}
[[ "$PASS_INDEX" =~ ^[0-9]+$ ]] \
  || { echo "PASS_INDEX must be a non-negative integer" >&2; exit 2; }
PASS_ROOT=$(realpath -m "$PASS_ROOT")
[[ "$PASS_ROOT" == "$REPO/output/"* ]] \
  || { echo "PASS_ROOT must resolve below $REPO/output" >&2; exit 2; }
PASS_DIR="$PASS_ROOT/pass_$(printf '%02d' "$PASS_INDEX")"

# ARM ORDER. The second arm of a pass inherits a warmer page cache and a
# differently-aged host. Here the two arms are CONTROL and CANDIDATE, so arm
# position aliases directly into the contrast being measured: serving the
# control first every time would hand the candidate the warmer host every time.
# A campaign alternates on pass parity; a lone pair keeps control-first.
ARM_ORDER=${ARM_ORDER:-TC}
case "$ARM_ORDER" in
  TC|CT) ;;
  *) echo "ARM_ORDER must be TC (control first) or CT (candidate first)" >&2; exit 2 ;;
esac

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
    echo "GDN_SL_FIXED32_MODE must be tail6_fixed32 or hydra27_fixed32" >&2
    exit 2
    ;;
esac
case "$PRODUCTION_BATCH" in
  1|4) ;;
  *) echo "GDN_SL_PRODUCTION_BATCH must be 1 or 4" >&2; exit 2 ;;
esac

CONTROL_ARM="${FIXED32_MODE}_gdn_w4_two_launch_b${PRODUCTION_BATCH}_${TAG}"
CANDIDATE_ARM="${FIXED32_MODE}_gdn_w4_single_launch_b${PRODUCTION_BATCH}_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for input in "$GDN_SL_CREDENTIAL" "$FORKED_FA2_SO"; do
  [[ "$input" == /* && -f "$input" && ! -L "$input" ]] \
    || { echo "timing input must be an absolute regular non-symlink file: $input" >&2; exit 2; }
done
[[ "$GDN_SL_CREDENTIAL_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$GDN_SL_CREDENTIAL" | awk '{print $1}')" == "$GDN_SL_CREDENTIAL_SHA256" ]] \
  || { echo "GDN single-launch credential identity mismatch" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical 16-task pool subset SHA-256 drift" >&2; exit 2; }
[[ -f "$DRAFT_VOCAB_BLOCKS_HOST" && ! -L "$DRAFT_VOCAB_BLOCKS_HOST" \
   && "$(sha256sum "$DRAFT_VOCAB_BLOCKS_HOST" | awk '{print $1}')" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
  || { echo "pinned root-64K draft-vocabulary block map drifted" >&2; exit 2; }
for required in "$PAIR_REDUCER" "$WINDOW_REDUCER" "$CREDENTIAL_VALIDATOR"; do
  [[ -f "$required" && ! -L "$required" ]] \
    || { echo "missing a required instrument: $required" >&2; exit 2; }
done
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

# EVERY PRECONDITION MUST BE ONE THE RUN CAN SATISFY. Campaign fossils were
# runners bound to an artifact nothing ever wrote -- this arm's own diagnostic
# bool is exactly that shape -- so the reducer and the credential are resolved
# BEFORE any GPU time is spent.
"$PYTHON_BIN" "$PAIR_REDUCER" --self-check \
  || { echo "width-4 pair reducer failed its own self-check" >&2; exit 2; }
# The credential must be bound to THIS commit. A credential from before this
# runner existed cannot be reused, because adapting the runner moves HEAD; the
# gate is therefore re-run at the final frozen HEAD before this pair.
"$PYTHON_BIN" "$CREDENTIAL_VALIDATOR" \
  --credential "$GDN_SL_CREDENTIAL" \
  --source-commit "$SOURCE_COMMIT" \
  --profile fixed32 \
  --mode "$FIXED32_MODE" \
  --batch "$PRODUCTION_BATCH" \
  || { echo "GDN single-launch credential is not valid at this HEAD" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS" "$PASS_DIR"

export BSIZE="$PRODUCTION_BATCH"
export CONC="$PRODUCTION_BATCH"
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
# The registry must not have been flipped underneath this pair. The screen NAMES
# both arms explicitly, so a flipped default cannot change what is served -- but
# it would mean the branch now ships an arm this screen has not yet justified,
# and that is a fact the run should record rather than discover later.
printf 'registry_single_launch_default=%s\n' \
  "${FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_DEFAULT:-unset}" \
  > "$RUNROOT_ABS/registry_state.at_launch.txt"

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

printf 'classification=%s\ntiming_eligible=1\nformal_floor_acceptance_eligible=0\nonly_arm_delta=%s\ncandidate_arm_selector=single_launch\ntopology=%s\nlogical_topology=%s\nactive_drafts=%s\nvalid_mask=%s\nphysical_drafts=31\nphysical_rows_root_inclusive=32\nbatch_size=%s\nconcurrency=%s\nslots=%s\ntask_pool=%s\ntask_refill=1\nagent_wall=none\nfixed_rows=128\ntask_ids=%s\nsubset=%s\nsubset_sha256=%s\ndraft_vocab_root=%s\ndraft_vocab_k=%s\ndraft_vocab_blocks=%s\ndraft_vocab_blocks_sha256=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\nlauncher_pid=%s\nrunroot=%s\npass_dir=%s\npass_index=%s\narm_order=%s\ncontrol_arm=%s\ncandidate_arm=%s\nsource_commit=%s\ncredential_sha256=%s\nrunner_sha256=%s\nreducer_sha256=%s\npatch_source_sha256=%s\nmetrics=1\nenforce_eager=0\ncudagraph_mode=FULL_AND_PIECEWISE\nkv_cache_memory_bytes=%s\nstarted=%s\n' \
  "$LAUNCH_CLASSIFICATION" "$ONLY_ARM_DELTA" \
  "$FIXED32_MODE" "$LOGICAL_TOPOLOGY" "$ACTIVE_DRAFTS" "$VALID_MASK" \
  "$PRODUCTION_BATCH" "$PRODUCTION_BATCH" "$SLOTS" "$TASK_COUNT" \
  "$TASK_IDS" "$SUBSET" "$SUBSET_SHA256" "$DRAFT_VOCAB_ROOT" "$DRAFT_VOCAB_K" \
  "$DRAFT_VOCAB_BLOCKS_CONTAINER" "$DRAFT_VOCAB_BLOCKS_SHA256" \
  "$FR13_MANDATORY_WEIGHT_BYTES" "$FR13_WEIGHT_FLOOR_MS" \
  "$$" "$RUNROOT_ABS" "$PASS_DIR" "$PASS_INDEX" "$ARM_ORDER" \
  "$CONTROL_ARM" "$CANDIDATE_ARM" "$SOURCE_COMMIT" "$GDN_SL_CREDENTIAL_SHA256" \
  "$RUNNER_SHA256" "$REDUCER_SHA256" "$PATCH_SOURCE_SHA256" \
  "$B4_KV_CACHE_MEMORY_BYTES" "$(date -u +%FT%TZ)" \
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
    || { echo "runtime/source manifest changed during timing" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" \
     && "$(sha256sum "$PAIR_REDUCER" | awk '{print $1}')" == "$REDUCER_SHA256" \
     && "$(sha256sum "$PATCH_SOURCE" | awk '{print $1}')" == "$PATCH_SOURCE_SHA256" ]] \
    || { echo "B4 GDN width-4 timing source changed during execution" >&2; return 14; }
  [[ "$(sha256sum "$GDN_SL_CREDENTIAL" | awk '{print $1}')" == "$GDN_SL_CREDENTIAL_SHA256" ]] \
    || { echo "GDN single-launch credential changed during timing" >&2; return 14; }
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

# The ONLY difference between the two invocations is the production arm trio.
# Everything else -- the FA2 binary, the subset, the pool depth, the refill flag,
# the sampling, the topology, the vocabulary, the graph mode, the metrics flag --
# is byte-for-byte the same.
run_arm() {
  local arm=$1
  local production=$2
  local batch=""
  local credential=""
  # The control arm carries the metrics-matched control flag; the candidate does
  # not, because the production arm already puts itself in the metrics=1 class.
  # This is the ONE field that differs between the arms besides the selector
  # itself, and it selects no kernel -- it only equalises the counter state so
  # the pair is comparable. Deriving it here rather than passing it means the two
  # can never be set inconsistently by a caller.
  local timing_control=1
  if [[ "$production" == "1" ]]; then
    batch=$PRODUCTION_BATCH
    credential=$GDN_SL_CREDENTIAL
    timing_control=0
  fi
  echo "===== $arm: pool16 B${PRODUCTION_BATCH} GDN single_launch_production=$production ====="
  # AGENT_WALL_S is EMPTY on purpose: the width-4 baseline was measured with no
  # wall, and a wall would truncate tasks and deform the admission ledger that
  # DEFINES the window.
  if env \
      RUNROOT="$PASS_DIR" \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR="$PRODUCTION_BATCH" \
      SWE_CONCURRENCY="$PRODUCTION_BATCH" AGENT_WALL_S= \
      FR13_B4_TASK_REFILL=1 \
      KV_CACHE_MEMORY_BYTES="$B4_KV_CACHE_MEMORY_BYTES" \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT" \
      FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K" \
      FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER" \
      FR13_NEEDS_ALLOW= \
      FR10_METRICS=1 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
      FR13_SCAN_ALIGN=0 FR13_NPAD_INVARIANT=0 \
      FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json" \
      FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION="$production" \
      FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_BATCH="$batch" \
      FR13_FIXED32_GDN_SINGLE_LAUNCH_PASS_JSON="$credential" \
      FR13_FIXED32_GDN_SINGLE_LAUNCH_TIMING_CONTROL="$timing_control" \
      FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=0 \
      FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH= \
      FR13_FIXED32_GDN_GQA_GROUP3_PASS_JSON= \
      FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
      FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
      FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH= \
      FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
      FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
      FR13_FA2_QROW32_B4_TIMING_ARM= \
      FR13_FA2_QROW32_LIVE_PAGED_AB=0 \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
      FR13_FA2_QROW32_B1_LIVE_AB_ARM= FR13_FA2_QROW32_B1_PRODUCTION_ARM= \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
      FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
      FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
      FR13_FIXED32_CUTLASS_WAVE=stock \
      FR13_FIXED32_CUTLASS_WAVE_SO= \
      FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$FORKED_FA2_SO" \
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
     && "$(grep -Fxc "FR13_B4_TASK_REFILL=1" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR10_METRICS=1" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION=$production" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_BATCH=$batch" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FIXED32_GDN_SINGLE_LAUNCH_TIMING_CONTROL=$timing_control" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=0" "$container_env")" -eq 1 ]] \
    || { echo "$arm did not run the declared single-variable B4 pool16 selector" >&2; return 4; }
  local credential_sidecar="$arm_dir/logs/fr13_fixed32_gdn_single_launch.production_credential.json"
  if [[ "$production" == "1" ]]; then
    [[ -f "$credential_sidecar" && ! -L "$credential_sidecar" ]] \
      || { echo "$arm lacks the GDN single-launch production credential" >&2; return 4; }
    [[ "$(sha256sum "$credential_sidecar" | awk '{print $1}')" == "$GDN_SL_CREDENTIAL_SHA256" ]] \
      || { echo "$arm served a credential that is not the presented one" >&2; return 4; }
  else
    # A control arm carrying a credential sidecar would mean the arm leaked
    # across the pair, which would invalidate the comparison outright.
    [[ ! -e "$credential_sidecar" && ! -L "$credential_sidecar" ]] \
      || { echo "$arm emitted a production credential on the control arm" >&2; return 4; }
  fi
  # THE WINDOW IS DEFINED BY THIS LEDGER. Without it there is no width-4 phase to
  # reduce and the verdict is vacuous, so it is required here rather than
  # discovered missing hours later at reduce time.
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
    --expected-tok-per-draft 31 --batch-size "$PRODUCTION_BATCH" \
    --work-census "$deploy_census" \
    --out "$arm_dir/deploy_speed_fullwall.json"
  printf 'arm=%s serve_rc=0 container_env_sha256=%s ended=%s\n' \
    "$arm" "$(sha256sum "$container_env" | awk '{print $1}')" \
    "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
}

# Both orders run the SAME two arms with the SAME single-variable delta; only
# which one boots first changes, and launcher_meta.txt records it so a reader
# never has to infer it from timestamps.
if [[ "$ARM_ORDER" == "TC" ]]; then
  run_arm "$CONTROL_ARM" 0
  [[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
    || { echo "Docker state was not clean between the paired arms" >&2; exit 2; }
  run_arm "$CANDIDATE_ARM" 1
else
  run_arm "$CANDIDATE_ARM" 1
  [[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
    || { echo "Docker state was not clean between the paired arms" >&2; exit 2; }
  run_arm "$CONTROL_ARM" 0
fi

finalize_manifests

# ------------------------------------------------------------------ verdict ---
# ALL timing math lives in the reducer, over the sealed window class. The runner
# serves and attests; it does not decide.
set +e
"$PYTHON_BIN" "$PAIR_REDUCER" \
  --runroot "$RUNROOT_ABS" \
  --control-arm "$PASS_DIR/$CONTROL_ARM" \
  --candidate-arm "$PASS_DIR/$CANDIDATE_ARM" \
  --mode "$FIXED32_MODE" \
  --source-commit "$SOURCE_COMMIT" \
  --out "$RUNROOT_ABS/width4_timing_pair.json"
REDUCE_RC=$?
set -e

# An INDEPENDENT second read of the same bytes by the sealed window reducer,
# unmodified. It cannot return a verdict for a 2-arm pair and is not asked to --
# it is run for its per-arm windowed records, which must agree with the pair
# reducer's.
"$PYTHON_BIN" "$WINDOW_REDUCER" \
  --gate-root "$RUNROOT_ABS" \
  --source-commit "$SOURCE_COMMIT" \
  --out "$RUNROOT_ABS/width4_window_independent_read.json" || true

printf 'width4_timing_pair=%s reduce_rc=%s completed=%s\n' \
  "$RUNROOT_ABS/width4_timing_pair.json" "$REDUCE_RC" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
exit "$REDUCE_RC"

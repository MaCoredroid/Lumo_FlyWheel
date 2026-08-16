#!/usr/bin/env bash
# Shared fixed32 Qrow16 + cooperative-target + SFWD conv/post-prep serve
# environment.
#
# Sourced by both scripts/fr13_run_b1_target_sfwd_exact4_timing.sh (the citable
# timing pair) and scripts/fr13_run_b1_sfwd_fusion_boot_diag.sh (the boot-only
# screen). The screen exists to catch EngineCore init/capture failures in
# minutes instead of paying a 90-minute pair, which is only sound if it boots
# byte-identical environment -- hence one definition, sourced twice.
#
# fr13_fixed32_sfwd_fusion_env <arm> <production> assigns into the CALLER's
# scope (bash locals are dynamically scoped, so a caller that declares these
# `local` keeps them local):
#
#   device_kernel target_selector target_so target_pass target_pass_sha
#   target_qualification_source_commit sfwd_fusion sfwd_pass sfwd_pass_sha
#   sfwd_manifest sfwd_manifest_sha sfwd_commit conv_wb_batched
#   FR13_FIXED32_SFWD_FUSION_ENV   (the env-prefix array)
#
# Callers must already have set: RUNROOT_ABS RUNROOT_REL QROW16_FA2_SO
# QROW16_SHA256 QROW16_PASS QROW16_PASS_SHA256 TARGET_SELECTOR
# CUTLASS_TARGET_SO TARGET_PASS TARGET_PASS_SHA256
# TARGET_QUALIFICATION_SOURCE_COMMIT SFWD_PASS_CONTAINER
# SFWD_CONV_POSTPREP_PASS_SHA256 SFWD_MANIFEST_CONTAINER
# SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256 SOURCE_COMMIT.

fr13_fixed32_sfwd_fusion_env() {
  local arm=$1
  local production=$2

  [[ "$production" == "0" || "$production" == "1" ]] \
    || { echo "fixed32 SFWD fusion env: production must be 0 or 1" >&2; return 2; }
  [[ -n "$arm" ]] \
    || { echo "fixed32 SFWD fusion env: arm must be non-empty" >&2; return 2; }
  local required
  for required in RUNROOT_ABS RUNROOT_REL QROW16_FA2_SO QROW16_SHA256 \
      QROW16_PASS QROW16_PASS_SHA256; do
    [[ -n "${!required:-}" ]] \
      || { echo "fixed32 SFWD fusion env lacks $required" >&2; return 2; }
  done

  device_kernel=/workspace/scripts/fr13_device_multidraft_kernel.py
  target_selector=stock
  target_so=""
  target_pass=""
  target_pass_sha=""
  target_qualification_source_commit=""
  sfwd_fusion=0
  sfwd_pass=""
  sfwd_pass_sha=""
  sfwd_manifest=""
  sfwd_manifest_sha=""
  sfwd_commit=""
  conv_wb_batched=1
  if [[ "$production" == "1" ]]; then
    for required in TARGET_SELECTOR CUTLASS_TARGET_SO TARGET_PASS \
        TARGET_PASS_SHA256 TARGET_QUALIFICATION_SOURCE_COMMIT \
        SFWD_PASS_CONTAINER SFWD_CONV_POSTPREP_PASS_SHA256 \
        SFWD_MANIFEST_CONTAINER SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256 \
        SOURCE_COMMIT; do
      [[ -n "${!required:-}" ]] \
        || { echo "fixed32 SFWD fusion env lacks $required" >&2; return 2; }
    done
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

  FR13_FIXED32_SFWD_FUSION_ENV=(
    RUNROOT="$RUNROOT_ABS"
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S=
    LUMO_SWE_AUTOCOMMIT=0
    FR13_FIXED32_B1_DIAGNOSTIC=0
    FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE
    FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1
    FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536
    FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
    FR13_DEVICE_MULTIDRAFT=1 FR13_DEVICE_MULTIDRAFT_KERNEL="$device_kernel"
    FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1
    FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}.json"
    FR13_DFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_dfwd.json"
    FR13_CFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_cfwd.json"
    FR13_B1_U8_CFWD_SFWD_STACK_TIMING=0
    FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0
    FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB=0
    FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION=0
    FR13_DRAFT_HEAD_FP8=0 FR13_DFWD_K64_TOP3=0
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON=
    FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0
    FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0
    FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_JSON=
    FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_SHA256=
    FR13_FA2_QROW16_LIVE_PAGED_AB=0
    FR13_FA2_QROW16_SO_SHA256="$QROW16_SHA256"
    FR13_FA2_QROW16_PRODUCTION=1
    FR13_FA2_QROW16_LIVE_PASS_JSON="$QROW16_PASS"
    FR13_FA2_QROW16_LIVE_PASS_SHA256="$QROW16_PASS_SHA256"
    FR13_FA2_QROW32_LIVE_PAGED_AB=0
    FR13_FA2_QROW32_B1_LIVE_AB_ARM= FR13_FA2_QROW32_B1_PRODUCTION_ARM=
    FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=0
    FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH=
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= FR13_FIXED32_GDN_PATH_BV_PRODUCTION=
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 FR13_FIXED32_BATCH_GDN_BV_CANDIDATE=
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0
    FR13_FIXED32_CUTLASS_WAVE="$target_selector"
    FR13_FIXED32_CUTLASS_WAVE_SO="$target_so"
    FR13_FIXED32_CUTLASS_WAVE_PRODUCTION="$production"
    FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE=k64_root
    FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_SOURCE_COMMIT="$target_qualification_source_commit"
    FR13_FIXED32_CUTLASS_WAVE_DIAGNOSTIC_TASK_PROFILE=astropy12907
    FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_JSON="$target_pass"
    FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_SHA256="$target_pass_sha"
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0
    FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0
    FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB=0
    FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION="$sfwd_fusion"
    FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=0
    FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_JSON="$sfwd_pass"
    FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_SHA256="$sfwd_pass_sha"
    FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_PATH="$sfwd_manifest"
    FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256="$sfwd_manifest_sha"
    FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_COMMIT="$sfwd_commit"
    FR13_CONV_WB_BATCHED="$conv_wb_batched"
    FR13_TREE_CONV_FUSED=1 FR13_FIXED32_CONV_SOURCE_BATCH=0
    FR13_FIXED32_ATTRIBUTION_ONLY=0
    FORKED_FA2_SO="$QROW16_FA2_SO"
  )
}

# Integer route pins the launcher validates before it starts a container, and
# that the caller's process environment (not the variant's XFLAGS) must carry.
# The launcher reads them as "${NAME:-}", so an unexported pin arrives as the
# empty string and int("") fails deep inside the launcher as the opaque
# "fixed32 integer route pin is malformed" -- which is how the boot screen
# failed at 050d5ae9b. Assert them here, by name, where the cause is obvious.
# (FR13_FIXED32_VALID_MASK / _ACTIVE_NODES / _PHYSICAL_DRAFTS are the variant's
# XFLAGS and are deliberately not asserted here.)
FR13_FIXED32_SFWD_FUSION_ROUTE_PINS=(
  FR13_FIXED32_TAW_WALK_CAP
)

# The exported boot preamble. This is NOT optional decoration: the floor
# sequence is the only place FR13_FIXED32_TAW_WALK_CAP and ~50 other fixed32
# route/committer/census pins are exported, and the launcher inherits them
# from the process environment rather than from the env-prefix array. A caller
# that skips this boots a materially different engine.
fr13_fixed32_sfwd_fusion_route_preamble() {
  local sequence=${1:-scripts/fr13_fixed32_floor_timers_seq.sh}

  [[ -f "$sequence" ]] \
    || { echo "fixed32 SFWD fusion preamble lacks sequence: $sequence" >&2; return 2; }
  # The sequence names its arms with $TAG even when run_variant is stubbed.
  [[ -n "${TAG:-}" ]] \
    || { echo "fixed32 SFWD fusion preamble requires TAG" >&2; return 2; }
  export BSIZE=1 CONC=1 WALL=0
  export FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536
  export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
  export FR13_FLOOR_ORDER=TH
  source scripts/fr13_canonical_env.sh
  # The sequence dispatches arms through run_variant; stub it so sourcing only
  # publishes the environment.
  run_variant() { :; }
  source "$sequence"
  unset -f run_variant

  local pin
  for pin in "${FR13_FIXED32_SFWD_FUSION_ROUTE_PINS[@]}"; do
    [[ -n "${!pin:-}" ]] \
      || { echo "fixed32 SFWD fusion preamble left $pin unset" >&2; return 2; }
  done
  [[ "${FR13_FIXED32_TAW_WALK_CAP:-}" =~ ^[0-9]+$ ]] \
    || { echo "fixed32 SFWD fusion preamble walk cap is not an integer" >&2; return 2; }
  [[ "${FR13_MANDATORY_WEIGHT_BYTES:-}" == "27977022848" ]] \
    || { echo "fixed32 SFWD fusion preamble weight contract drifted" >&2; return 2; }
  [[ "${LUMO_SWE_AUTOCOMMIT:-}" == "0" ]] \
    || { echo "fixed32 SFWD fusion preamble must not autocommit" >&2; return 2; }
}

# The seven fallback/preseed strings a healthy SFWD conv/post-prep production
# boot must never emit. Shared so the timing pair's engagement validator and
# the boot screen reject exactly the same set.
FR13_FIXED32_SFWD_FUSION_FORBIDDEN=(
  "linear fallback engaged"
  "FR13 SFWD conv/post-prep capture lacks preseeded output bindings"
  "FR13 SFWD conv/post-prep output preseed ran during capture"
  "FR13 SFWD conv/post-prep profile output preseed ran during capture"
  "FR13 SFWD conv/post-prep profile output preseed is incomplete"
  "FR13 SFWD conv/post-prep graph output preseed is incomplete"
  "[FR13_STEP_GRAPH] S1 DISABLED"
)

# The runtime census/commit/replay validator family is self-diagnosing: every
# site raises a distinctive fragment into the container log the instant an
# invariant fails, and not one of them can run before a MEASURED forward --
# reaching /health proves only EngineCore init and the final capture. A boot
# that has since served real requests and emitted none of these is the positive
# evidence that the pre-freeze census gate, the replay provenance and
# census-restore checks, the commit-time counter checks, the forward-graph
# registry, the capture manifest fused section, the attested kernel shape and
# the v13 work census all ran and passed.
#
# Exact substrings, deliberately not a loose "fixed32.*census" regex: that
# pattern also matches the benign runtime path
# /logs/fr13_fixed32_work_census.jsonl, which a healthy engine is free to print.
# tests/test_fr13_sfwd_fusion_boot_diag.py pins every fragment to a live raise
# site, so a renamed validator fails the suite instead of quietly turning the
# sweep into a no-op.
FR13_FIXED32_SFWD_FUSION_CENSUS_FORBIDDEN=(
  "census drift"
  "census observation is missing"
  "counters mismatch"
  "counters are incoherent"
  "forward work is incomplete"
  "geometry drifted"
  "capture set is incomplete before freeze"
  "unscoped eager work after capture freeze"
  "eager work mixed with graph replay"
  "provenance drift"
  "kernel shape drifted"
  "manifest semantic drift"
  "forward manifest registry drift"
  "replay identity/signature/batch drift"
  "replay evidence did not bind completed event"
  "event lacks one exact full-graph replay"
  "drafter graph replay drift"
  "requires sampled temp>0"
)

# The terminal-flush failure fragments. The campaign driver runs the fixed32
# terminal flush from its own EXIT trap, so a failure lands either in the
# driver runlog (harness side) or in the container log the driver captured
# before removing the container (engine side). The screen sweeps both.
FR13_FIXED32_SFWD_FUSION_FLUSH_FORBIDDEN=(
  "[FR13_FIXED32_FLUSH] failed generation"
  "[FR13_FIXED32_FLUSH] rejected unbound request"
  "fixed32 flush saw a pending TAW callback"
  "fixed32 flush saw an incomplete"
  "fixed32 flush quiescence is already active"
  "FAIL: fixed32 terminal flush"
  "fixed32 final flush missing PID or ready ack"
  "fixed32 teardown skipped container operations"
)

# The campaign's sampling shape, for the boot screen's smoke traffic.
#
# The pair's requests reach the engine through the offload proxy, which stamps
# these onto every /v1/chat/completions body from LUMO_PROXY_FORCE_* (see
# normalize_chat_completions_request_payload). The screen talks to the engine's
# fixed32 ingress DIRECTLY -- it freezes the driver before the proxy exists --
# so nothing stamps them for it, and whatever it sends IS the sampling shape.
#
# fixed32 refuses greedy decoding outright: with all_greedy the rejection
# sampler raises "FR13 fixed32 requires sampled temp>0, no draft_probs, and
# max_spec_len=31" before the proposal seal, which kills EngineCore and answers
# the request with a bare 500. The 2026-08-08 screen sent temperature 0.0 and
# died on its first forward for exactly that reason.
#
# An array, not scalars: bash cannot export an array, so these can never reach
# the launcher's `compgen -v | grep -E '^(FR[0-9]+_|LUMO_|VLLM_)'` -e sweep and
# perturb the very engine the screen exists to compare against. Each value is
# bound by test to its authoritative default in the proxy relaunch helper.
FR13_FIXED32_SFWD_FUSION_SMOKE_SAMPLING=(
  "temperature=0.6"
  "top_p=0.95"
  "top_k=20"
  "presence_penalty=1.0"
  "min_p=0"
)

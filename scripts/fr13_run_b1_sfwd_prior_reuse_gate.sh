#!/usr/bin/env bash
# One real SWE-Verified K64/root1 B1 byte gate for the register-local
# channel-serial two-lane candidate with one long-edge reload (C128/W2 at B1).
# The candidate is shadow-only and the incumbent tensors remain served.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the exact FA2 shared object}"

RUNROOT_ABS=$(realpath -m "$RUNROOT")
SOURCE_COMMIT=$(git rev-parse HEAD)
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
DRAFT_VOCAB_BLOCKS=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
ARM="hydra27_fixed32_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"
TASK_ID=astropy__astropy-12907
MANIFEST_LAUNCH="$RUNROOT_ABS/sfwd_prior_reuse_source_manifest.at_launch.json"
MANIFEST_END="$RUNROOT_ABS/sfwd_prior_reuse_source_manifest.at_end.json"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "current source identity is invalid" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical one-task B1 subset SHA-256 drift" >&2; exit 2; }
[[ -f "$DRAFT_VOCAB_BLOCKS" && ! -L "$DRAFT_VOCAB_BLOCKS" ]] \
  || { echo "K64 block map must be a regular source file" >&2; exit 2; }
[[ "$(sha256sum "$DRAFT_VOCAB_BLOCKS" | awk '{print $1}')" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
  || { echo "K64 block map SHA-256 drift" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
.venv/bin/python scripts/fr13_sfwd_prior_reuse_gate.py source-manifest \
  --repo "$REPO" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$MANIFEST_LAUNCH"
MANIFEST_SHA256=$(sha256sum "$MANIFEST_LAUNCH" | awk '{print $1}')
MANIFEST_CONTAINER="/workspace/${MANIFEST_LAUNCH#"$REPO/"}"

export RUNROOT=${RUNROOT_ABS#"$REPO/"}
export FR13_GATE_QROW16=0
export FR13_GATE_TAW_NATIVE=0
export FR13_GATE_DRAFT_HEAD_PAD=0
export FR13_GATE_GDN_BV=0
export FR13_GATE_BM8=0
export FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0
export FR13_FA2_QROW16_PRODUCTION=0
export FR13_DFWD_UNIFIED_BM8_LIVE_AB=0
export FR13_DFWD_UNIFIED_BM8_PRODUCTION=0
export FR13_FIXED32_GDN_PATH_BV_CANDIDATE=
export FR13_FIXED32_GDN_PATH_BV_PRODUCTION=
export FR13_FIXED32_BATCH_GDN_BYTE_AB=0
export FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0
export FR13_FIXED32_BATCH_GDN_PRODUCTION=0
export FR13_FIXED32_BATCH_GDN_BV_CANDIDATE=
export FR13_FIXED32_BATCH_GDN_BV_PRODUCTION=
export FR13_FIXED32_BATCH_GDN_BV8_TIMING=0
export FR13_FIXED32_CUTLASS_WAVE=stock
export FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0
export FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0
export FR13_FIXED32_SFWD_STATE_FUSION_TIMING_AB=0
export FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0
export FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB=1
export FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_MANIFEST_PATH="$MANIFEST_CONTAINER"
export FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_MANIFEST_SHA256="$MANIFEST_SHA256"
export FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_COMMIT="$SOURCE_COMMIT"
export FR13_CONV_WB_BATCHED=1
export FR13_TREE_CONV_FUSED=1
export ENFORCE_EAGER=1
export FR10_METRICS=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER"
export FR13_MANDATORY_WEIGHT_BYTES=32666638208
export FR13_WEIGHT_FLOOR_MS=119.658015414
unset FR13_NEEDS_ALLOW FR10_ALLOW_LINEAR_FALLBACK

MANIFEST_FINALIZED=0
finalize_source_manifest() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  .venv/bin/python scripts/fr13_sfwd_prior_reuse_gate.py source-manifest \
    --repo "$REPO" \
    --source-commit "$SOURCE_COMMIT" \
    --output "$MANIFEST_END"
  cmp -s "$MANIFEST_LAUNCH" "$MANIFEST_END" \
    || { echo "SFWD prior-reuse source manifest changed during the task" >&2; return 14; }
  MANIFEST_FINALIZED=1
}

runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    if finalize_source_manifest; then
      :
    else
      local manifest_rc=$?
      (( rc == 0 )) && rc=$manifest_rc
    fi
  fi
  exit "$rc"
}
trap runner_exit EXIT

bash scripts/fr13_run_b1_kernel_live_gate.sh

finalize_source_manifest

printf '%s\n' \
  'classification=one_real_swe_verified_k64_root_b1_sfwd_prior_reuse_byte_diagnostic' \
  'task_set=one' \
  'task_count=1' \
  'reference_returned=true' \
  'no_fallback=true' \
  'timing_eligible=false' \
  'floor_acceptance_eligible=false' \
  'production_enabled=false' \
  'physical_rows_per_request=32' \
  'conv_rows_per_program=32' \
  'conv_block_c=128' \
  'conv_num_warps=2' \
  'topology_host_validation=exact_parent_each_launch' \
  'source_descriptor_device_validation=false' \
  'source_descriptor_launcher_argument=false' \
  'source_descriptor_in_kernel=false' \
  'x_global_loads_per_channel=33' \
  'long_edge_reload=row21_from_row4' \
  'x_stride=16384,1' \
  'out_stride=10240,1' \
  'source_stage_stride=10240,1' \
  'conv_weights_stride=4,1' \
  'conv_state_layout=bank,channel,state' \
  'conv_state_stride=2097152,1,10240' \
  'spec_state_indices_width=32' \
  'spec_state_indices_contiguous=true' \
  "source_commit=$SOURCE_COMMIT" \
  "source_manifest_sha256=$MANIFEST_SHA256" \
  >> "$RUNROOT_ABS/launcher_meta.txt"

.venv/bin/python scripts/fr13_sfwd_prior_reuse_gate.py validate \
  --arm-dir "$ARMDIR" \
  --source-commit "$SOURCE_COMMIT" \
  --task-id "$TASK_ID" \
  --manifest-launch "$MANIFEST_LAUNCH" \
  --manifest-end "$MANIFEST_END" \
  --output "$ARMDIR/sfwd_prior_reuse_k64_root_b1_gate.json"
trap - EXIT

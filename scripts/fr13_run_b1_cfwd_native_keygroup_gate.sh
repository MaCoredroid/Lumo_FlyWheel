#!/usr/bin/env bash
# One resolved real SWE-Verified K64/root1 B1 all-depth byte gate for native CFWD.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the exact FA2 shared object}"
: "${FR13_FIXED32_CFWD_NATIVE_KEYGROUP_SO:?set the source-bound vLLM _C.abi3.so}"
: "${FR13_FIXED32_CFWD_NATIVE_KEYGROUP_BINDING_JSON:?set its private binding JSON}"

RUNROOT_ABS=$(realpath -m "$RUNROOT")
SOURCE_COMMIT=$(git rev-parse HEAD)
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
DRAFT_VOCAB_BLOCKS=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
TASK_ID=astropy__astropy-12907
ARM="hydra27_fixed32_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"
MANIFEST_LAUNCH="$RUNROOT_ABS/runtime_manifest.at_launch.json"
MANIFEST_END="$RUNROOT_ABS/runtime_manifest.at_end.json"

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
[[ -f "$DRAFT_VOCAB_BLOCKS" && ! -L "$DRAFT_VOCAB_BLOCKS" \
   && "$(sha256sum "$DRAFT_VOCAB_BLOCKS" | awk '{print $1}')" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
  || { echo "K64 block map SHA-256 drift" >&2; exit 2; }
for binary in "$FORKED_FA2_SO" "$FR13_FIXED32_CFWD_NATIVE_KEYGROUP_SO"; do
  [[ "$binary" == /* && "$binary" != *:* && -f "$binary" && ! -L "$binary" ]] \
    || { echo "candidate binary must be an absolute regular non-symlink" >&2; exit 2; }
done
[[ "$FR13_FIXED32_CFWD_NATIVE_KEYGROUP_BINDING_JSON" == /* \
   && "$FR13_FIXED32_CFWD_NATIVE_KEYGROUP_BINDING_JSON" != *:* \
   && -f "$FR13_FIXED32_CFWD_NATIVE_KEYGROUP_BINDING_JSON" \
   && ! -L "$FR13_FIXED32_CFWD_NATIVE_KEYGROUP_BINDING_JSON" \
   && "$(stat -c '%h:%a' "$FR13_FIXED32_CFWD_NATIVE_KEYGROUP_BINDING_JSON")" == "1:400" ]] \
  || { echo "binary binding must be an absolute mode-0400 single-link file" >&2; exit 2; }

.venv/bin/python scripts/fr13_cfwd_native_keygroup_binary.py verify \
  --repo "$REPO" \
  --candidate-so "$FR13_FIXED32_CFWD_NATIVE_KEYGROUP_SO" \
  --binding "$FR13_FIXED32_CFWD_NATIVE_KEYGROUP_BINDING_JSON" \
  >/dev/null

source scripts/fr13_canonical_env.sh
export RUNROOT=${RUNROOT_ABS#"$REPO/"}
export BSIZE=1
export CONC=1
export WALL=0
export FR13_FLOOR_ORDER=TH
export FR13_FIXED32_B1_DIAGNOSTIC=1
export FR13_FIXED32_COMMITTER_LAYER_BATCH=1
export FR13_FIXED32_COMMITTER_LAYER_BATCH_QUALIFICATION=1
export FR13_FIXED32_CFWD_NATIVE_KEYGROUP_PRECOMPUTE_CUDA=diagnostic
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
export FR13_MANDATORY_WEIGHT_BYTES=32666638208
export FR13_WEIGHT_FLOOR_MS=119.658015414
export FR13_FIXED32_CUTLASS_WAVE=stock
export FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0
export FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0
export FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0
export FR13_FA2_QROW16_LIVE_PAGED_AB=0
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
export FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0
export FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0
export FR13_FIXED32_CONV_SOURCE_BATCH=0
export FR13_DRAFT_HEAD_PAD_ROWS=0
export FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0
export FR13_DRAFT_HEAD_M32_LIVE_AB=0
export FR13_DRAFT_HEAD_M32_PRODUCTION=0
export FR13_DRAFT_HEAD_M32_TIMING_ARM=0
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
export FR13_GRAPH_TIMER=0
export FR13_REPLAY_GPU_TIMER=0
export FR13_COMMIT_FULL_GPU_TIMER=0
export ENFORCE_EAGER=0
export FR10_METRICS=0
export LUMO_SWE_AUTOCOMMIT=0
unset FR13_NEEDS_ALLOW FR10_ALLOW_LINEAR_FALLBACK

mkdir -p "$RUNROOT_ABS"
printf '%s\n' \
  'classification=one_real_swe_verified_k64_root_b1_cfwd_native_keygroup_byte_gate' \
  'task_count=1' \
  "task_id=$TASK_ID" \
  "source_commit=$SOURCE_COMMIT" \
  "subset_sha256=$SUBSET_SHA256" \
  "draft_vocab_blocks_sha256=$DRAFT_VOCAB_BLOCKS_SHA256" \
  'draft_vocab_k=65536' \
  'draft_vocab_root=1' \
  'reference_served=true' \
  'timing_eligible=false' \
  'floor_acceptance_eligible=false' \
  'production_authorized=false' \
  > "$RUNROOT_ABS/launcher_meta.txt"

.venv/bin/python scripts/fr13_runtime_manifest.py \
  --repo "$REPO" --profile fixed32 \
  --sequence scripts/fr13_run_b1_cfwd_native_keygroup_gate.sh \
  --output "$MANIFEST_LAUNCH"

MANIFEST_FINALIZED=0
finalize_manifest() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  .venv/bin/python scripts/fr13_runtime_manifest.py \
    --repo "$REPO" --profile fixed32 \
    --sequence scripts/fr13_run_b1_cfwd_native_keygroup_gate.sh \
    --output "$MANIFEST_END"
  cmp -s "$MANIFEST_LAUNCH" "$MANIFEST_END" \
    || { echo "runtime manifest changed during the real task" >&2; return 14; }
  MANIFEST_FINALIZED=1
}

runner_exit() {
  local rc=$?
  trap - EXIT
  if finalize_manifest; then
    :
  else
    local manifest_rc=$?
    (( rc == 0 )) && rc=$manifest_rc
  fi
  exit "$rc"
}
trap runner_exit EXIT

if OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
  FORKED_FA2_SO="$FORKED_FA2_SO" \
  FR13_FIXED32_CFWD_NATIVE_KEYGROUP_SO="$FR13_FIXED32_CFWD_NATIVE_KEYGROUP_SO" \
  FR13_FIXED32_CFWD_NATIVE_KEYGROUP_BINDING_JSON="$FR13_FIXED32_CFWD_NATIVE_KEYGROUP_BINDING_JSON" \
  bash scripts/fr13_bigdenom_swe_serve_variant.sh \
    "$ARM" hydra27_fixed32 "$SUBSET" \
    > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi
printf 'serve_rc=%s\n' "$serve_rc" >> "$RUNROOT_ABS/launcher_meta.txt"
(( serve_rc == 0 )) || exit "$serve_rc"

finalize_manifest
.venv/bin/python scripts/fr13_cfwd_native_keygroup_b1_gate.py \
  --repo "$REPO" \
  --arm-dir "$ARMDIR" \
  --source-commit "$SOURCE_COMMIT" \
  --candidate-so "$FR13_FIXED32_CFWD_NATIVE_KEYGROUP_SO" \
  --binding "$FR13_FIXED32_CFWD_NATIVE_KEYGROUP_BINDING_JSON" \
  --runtime-manifest-launch "$MANIFEST_LAUNCH" \
  --runtime-manifest-end "$MANIFEST_END" \
  --output "$ARMDIR/cfwd_native_keygroup_k64_root_b1_gate.json"
trap - EXIT

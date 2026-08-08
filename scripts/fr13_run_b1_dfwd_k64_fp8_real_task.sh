#!/usr/bin/env bash
# Real SWE-Verified B1 diagnostic for the static-I/O K64 block-FP8 draft head.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

case "${FR13_RUN_B1_DFWD_K64_FP8_REAL_TASK:-0}" in
  1) ;;
  0)
    echo "block-FP8 real task is disabled; set FR13_RUN_B1_DFWD_K64_FP8_REAL_TASK=1" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_B1_DFWD_K64_FP8_REAL_TASK must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned Qrow16 FA2 binary}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SOURCE_COMMIT=$(git rev-parse --verify HEAD)
SOURCE_SHA256=c40d429eb4a5e6521ba7134fe3ab647733c83436700eadd7f0ae8dd43823b6a4
SELECTOR=scripts/fr13_dfwd_k64_fp8_selector.py
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}
ARM="hydra27_fixed32_k64_root_${TAG}"

# Use the same exact candidate and exclusion contract for preflight and for the
# process that launches the real task.  Empty assignments are intentional: an
# ambient credential must not reactivate or authenticate a competing head.
FP8_EXACT_ENV=(
  FR13_DRAFT_HEAD_FP8=1
  FR13_DRAFT_HEAD_FP8_STATIC_IO=1
  "FR13_DRAFT_HEAD_FP8_ARM=$ARM"
  FR13_DRAFT_HEAD_FP8_ENGAGEMENT_JSON=/logs/fr13_draft_head_fp8.engagement.json
  "FR13_DRAFT_HEAD_FP8_SOURCE_COMMIT=$SOURCE_COMMIT"
  FR13_FIXED32_MODE=hydra27_fixed32
  MAX_NUM_SEQS=1
  SWE_CONCURRENCY=1
  ENFORCE_EAGER=0
  CUDAGRAPH_MODE=FULL_AND_PIECEWISE
  FR13_DRAFT_VOCAB_ROOT=1
  FR13_DRAFT_VOCAB_K=65536
  FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
  FR13_DRAFTER_SINGLE_LOGITS=1
  FR13_FIXED32_PHYSICAL_DRAFTS=31
  FR13_FIXED32_ACTIVE_NODES=27
  NUM_SPECULATIVE_TOKENS=31
  FR13_MANDATORY_WEIGHT_BYTES=30989326208
  FR13_WEIGHT_FLOOR_MS=113.514015414
)
FP8_DISABLED_ENV=(
  FR13_DRAFT_HEAD_K64_TC=0
  FR13_DRAFT_HEAD_B14_WARP4_PAIR8=0
  FR13_DRAFT_HEAD_PAD_ROWS=0
  FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0
  FR13_DRAFT_HEAD_M32_LIVE_AB=0
  FR13_DRAFT_HEAD_M32_PRODUCTION=0
  FR13_DRAFT_HEAD_M32_TIMING_ARM=0
  FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB=0
  FR13_DRAFT_HEAD_M1_R64_U8_QUALITY_GATE=0
  FR13_DRAFT_HEAD_M1_R64_U8_TAW_QUALITY_GATE=0
  FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION=0
  FR13_DRAFT_HEAD_M4_R64_U8_LIVE_AB=0
  FR13_DRAFT_HEAD_M4_R64_U8_QUALITY_GATE=0
  FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION=0
  FR13_DFWD_K64_TOP3=0
  FR13_B1_U8_CFWD_SFWD_STACK_TIMING=0
)
FP8_CLEAR_ENV=(
  FR13_DRAFT_HEAD_K64_TC_SOURCE_COMMIT=
  FR13_DRAFT_HEAD_B14_WARP4_PAIR8_SOURCE_COMMIT=
  FR13_DRAFT_HEAD_M32_INSTANCE_ID=
  FR13_DRAFT_HEAD_M32_LIVE_PASS_JSON=
  FR13_DRAFT_HEAD_M32_LIVE_PASS_SHA256=
  FR13_DRAFT_HEAD_M32_LIVE_FINAL_FLUSH_JSON=
  FR13_DRAFT_HEAD_M32_LIVE_BOUNDARY_SNAPSHOT_JSON=
  FR13_DRAFT_HEAD_M32_LIVE_CHAT_TRAFFIC_AUDIT_JSON=
  FR13_DRAFT_HEAD_M32_PRODUCTION_PASS_SIDECAR=
  FR13_DRAFT_HEAD_M32_PRODUCTION_PASS_SIDECAR_SHA256=
  FR13_DRAFT_HEAD_M32_INTERNAL_PRODUCTION_ATTESTED=
  FR13_DRAFT_HEAD_M1_R64_U8_SO=
  FR13_DRAFT_HEAD_M1_R64_U8_SO_SHA256=
  FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_SHA256=
  FR13_DRAFT_HEAD_M1_R64_U8_BUILD_ATTESTATION_SHA256=
  FR13_DRAFT_HEAD_M1_R64_U8_PATCH_SOURCE_SHA256=
  FR13_DRAFT_HEAD_M1_R64_U8_RUNNER_SHA256=
  FR13_DRAFT_HEAD_M1_R64_U8_SUBSET_SHA256=
  FR13_DRAFT_HEAD_M1_R64_U8_VOCAB_BLOCKS_SHA256=
  FR13_DRAFT_HEAD_M1_R64_U8_FA2_SHA256=
  FR13_DRAFT_HEAD_M1_R64_U8_TAW_SOURCE_SHA256=
  FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_COMMIT=
  FR13_DRAFT_HEAD_M1_R64_U8_INSTANCE_ID=
  FR13_DRAFT_HEAD_M1_R64_U8_LIVE_PASS_JSON=
  FR13_DRAFT_HEAD_M1_R64_U8_LIVE_PASS_SHA256=
  FR13_DRAFT_HEAD_M1_R64_U8_LIVE_FINAL_FLUSH_JSON=
  FR13_DRAFT_HEAD_M1_R64_U8_LIVE_BOUNDARY_SNAPSHOT_JSON=
  FR13_DRAFT_HEAD_M1_R64_U8_LIVE_CHAT_TRAFFIC_AUDIT_JSON=
  FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION_PASS_SIDECAR=
  FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION_PASS_SIDECAR_SHA256=
  FR13_DRAFT_HEAD_M1_R64_U8_INTERNAL_PRODUCTION_ATTESTED=
  FR13_DRAFT_HEAD_M4_R64_U8_SO=
  FR13_DRAFT_HEAD_M4_R64_U8_SO_SHA256=
  FR13_DRAFT_HEAD_M4_R64_U8_SOURCE_SHA256=
  FR13_DRAFT_HEAD_M4_R64_U8_BUILD_ATTESTATION_SHA256=
  FR13_DRAFT_HEAD_M4_R64_U8_PATCH_SOURCE_SHA256=
  FR13_DRAFT_HEAD_M4_R64_U8_RUNNER_SHA256=
  FR13_DRAFT_HEAD_M4_R64_U8_SUBSET_SHA256=
  FR13_DRAFT_HEAD_M4_R64_U8_VOCAB_BLOCKS_SHA256=
  FR13_DRAFT_HEAD_M4_R64_U8_FA2_SHA256=
  FR13_DRAFT_HEAD_M4_R64_U8_TAW_SOURCE_SHA256=
  FR13_DRAFT_HEAD_M4_R64_U8_SOURCE_COMMIT=
  FR13_DRAFT_HEAD_M4_R64_U8_TASK_IDS=
  FR13_DRAFT_HEAD_M4_R64_U8_LIVE_PASS_JSON=
  FR13_DRAFT_HEAD_M4_R64_U8_LIVE_PASS_SHA256=
  FR13_DRAFT_HEAD_M4_R64_U8_LIVE_FINAL_FLUSH_JSON=
  FR13_DRAFT_HEAD_M4_R64_U8_LIVE_BOUNDARY_SNAPSHOT_JSON=
  FR13_DRAFT_HEAD_M4_R64_U8_LIVE_CHAT_TRAFFIC_AUDIT_JSON=
  FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION_PASS_SIDECAR=
  FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION_PASS_SIDECAR_SHA256=
  FR13_DRAFT_HEAD_M4_R64_U8_INTERNAL_PRODUCTION_ATTESTED=
  FR13_DFWD_K64_TOP3_SO=
  FR13_DFWD_K64_TOP3_SHA256=
  FR13_GATE_DRAFT_HEAD_U8_SO=
  FR13_GATE_DFWD_TOP3_SO=
  FR13_FIXED32_CUTLASS_WAVE_SO=
  FR13_FIXED32_CUTLASS_WAVE_RESOURCE_CREDENTIAL=
  FR13_FIXED32_CUTLASS_WAVE_RESOURCE_CREDENTIAL_SHA256=
  FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_JSON=
  FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_SHA256=
  FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_SOURCE_COMMIT=
  FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE=
  FR13_FIXED32_CUTLASS_WAVE_DIAGNOSTIC_TASK_PROFILE=
)

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* \
   && ! -e "$RUNROOT_ABS" \
   && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be a new path below $REPO/output" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ \
   && -z "$(git status --porcelain=v1 --untracked-files=no)" \
   && "$(git rev-parse '@{upstream}')" == "$SOURCE_COMMIT" ]] \
  || { echo "block-FP8 real task requires a clean source commit pushed upstream" >&2; exit 2; }
[[ -f scripts/fr10_phase4_patch_vllm_tree_gdn.py \
   && ! -L scripts/fr10_phase4_patch_vllm_tree_gdn.py \
   && "$(sha256sum scripts/fr10_phase4_patch_vllm_tree_gdn.py | awk '{print $1}')" \
      == "$SOURCE_SHA256" \
   && -f "$SELECTOR" \
   && ! -L "$SELECTOR" ]] \
  || { echo "block-FP8 source or selector identity drifted" >&2; exit 2; }
[[ -f "$FORKED_FA2_SO" && ! -L "$FORKED_FA2_SO" \
   && "$FORKED_FA2_SO" == /* ]] \
  || { echo "FORKED_FA2_SO must be an absolute regular non-symlink" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the real task" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
env \
  "${FP8_EXACT_ENV[@]}" \
  "${FP8_DISABLED_ENV[@]}" \
  "${FP8_CLEAR_ENV[@]}" \
  "$PYTHON_BIN" "$SELECTOR" --repo "$PWD" --source-commit "$SOURCE_COMMIT" \
  > "$RUNROOT_ABS/selector.json"

printf 'classification=one_real_swe_verified_b1_hydra27_fixed32_k64_fp8_static_diagnostic\nacceptance_valid=0\ntiming_eligible=0\nfloor_acceptance_eligible=0\ncandidate_served=1\ntarget_authority_unchanged=1\nbatch_size=1\nconcurrency=1\nphysical_rows=32\nactive_nodes=27\ndraft_vocab_root=1\ndraft_vocab_k=65536\nmandatory_weight_bytes=30989326208\nmandatory_weight_floor_ms=113.514015414\none_sided_u95_cap_ms=130.541117726\nsource=%s\nsource_sha256=%s\narm=%s\n' \
  "$SOURCE_COMMIT" "$SOURCE_SHA256" "$ARM" \
  > "$RUNROOT_ABS/fp8_real_task_scope.txt"

env \
  "${FP8_EXACT_ENV[@]}" \
  "${FP8_DISABLED_ENV[@]}" \
  "${FP8_CLEAR_ENV[@]}" \
  RUNROOT="$RUNROOT_REL" TAG="$TAG" FORKED_FA2_SO="$FORKED_FA2_SO" \
  FR13_B1_DIAGNOSTIC_TASK_PROFILE=astropy12907 \
  FR13_B1_WORKLOAD_PROFILE=k64_root \
  FR13_GATE_QROW16=0 FR13_GATE_TAW_NATIVE=0 \
  FR13_GATE_DRAFT_HEAD_PAD=0 FR13_GATE_DRAFT_HEAD_M32=0 \
  FR13_GATE_DRAFT_HEAD_U8=0 FR13_GATE_DRAFT_HEAD_U8_QUALITY=0 \
  FR13_GATE_DRAFT_HEAD_U8_TAW_QUALITY=0 \
  FR13_GATE_DRAFT_HEAD_FP8=1 FR13_GATE_DRAFT_HEAD_FP8_STATIC_IO=1 \
  FR13_GATE_DFWD_TOP3=0 FR13_GATE_BM8=0 FR13_GATE_GDN_BV=0 \
  FR13_GATE_SFWD_CONV_POSTPREP=0 \
  FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
  bash scripts/fr13_run_b1_kernel_live_gate.sh

GATE="$RUNROOT_ABS/$ARM/draft_head_fp8_real_b1_gate.json"
ENGAGEMENT="$RUNROOT_ABS/$ARM/logs/fr13_draft_head_fp8.engagement.json"
ACCEPTANCE="$RUNROOT_ABS/$ARM/draft_head_fp8_acceptance.json"
for evidence in "$GATE" "$ENGAGEMENT" "$ACCEPTANCE"; do
  [[ -f "$evidence" && ! -L "$evidence" ]] \
    || { echo "block-FP8 real-task evidence is missing: $evidence" >&2; exit 4; }
done

grep -F '[FR13_DRAFT_HEAD_FP8] static weight ready' \
  "$RUNROOT_ABS/$ARM/docker_after_tasks.log" >/dev/null \
  || { echo "block-FP8 static weight marker is missing" >&2; exit 4; }
grep -F '[FR13_DRAFT_HEAD_FP8] engaged batch=1 proposal_logits=fp8_output_direct' \
  "$RUNROOT_ABS/$ARM/docker_after_tasks.log" >/dev/null \
  || { echo "block-FP8 candidate-served marker is missing" >&2; exit 4; }

printf 'gate=%s\nengagement=%s\nmeasurement=%s\ncompleted=%s\n' \
  "$GATE" "$ENGAGEMENT" "$ACCEPTANCE" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/fp8_real_task_scope.txt"

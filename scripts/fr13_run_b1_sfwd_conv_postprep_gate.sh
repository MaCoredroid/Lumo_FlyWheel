#!/usr/bin/env bash
# One authenticated SWE-Verified Hydra27 K64/root1 B1 byte gate. The fused
# candidate is shadow-only; incumbent tensors and pinned Qrow16 remain served.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned Qrow16 shared object}"
FR13_FIXED32_SFWD_EMBED_GATE_CTA=${FR13_FIXED32_SFWD_EMBED_GATE_CTA:-0}
case "$FR13_FIXED32_SFWD_EMBED_GATE_CTA" in
  0|1) ;;
  *) echo "FR13_FIXED32_SFWD_EMBED_GATE_CTA must be exactly 0 or 1" >&2; exit 2 ;;
esac
FR13_FIXED32_SFWD_NODEGROUP8_DIRECT=${FR13_FIXED32_SFWD_NODEGROUP8_DIRECT:-0}
case "$FR13_FIXED32_SFWD_NODEGROUP8_DIRECT" in
  0|1) ;;
  *) echo "FR13_FIXED32_SFWD_NODEGROUP8_DIRECT must be exactly 0 or 1" >&2; exit 2 ;;
esac
DIRECT_ARGS=()
if [[ "$FR13_FIXED32_SFWD_NODEGROUP8_DIRECT" == "1" ]]; then
  DIRECT_ARGS+=(--direct-nodegroup8)
fi

RUNROOT_ABS=$(realpath -m "$RUNROOT")
SOURCE_COMMIT=$(git rev-parse HEAD)
TASK_ID=astropy__astropy-12907
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
BLOCKS=scripts/fr13_dvk_subset_blocks.json
BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
QROW16_SHA256=1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86
QROW16_BYTES=299507792
ARM="hydra27_fixed32_k64_root_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"
MANIFEST_LAUNCH="$RUNROOT_ABS/sfwd_conv_postprep_source_manifest.at_launch.json"
MANIFEST_END="$RUNROOT_ABS/sfwd_conv_postprep_source_manifest.at_end.json"
READINESS="$RUNROOT_ABS/sfwd_conv_postprep_host_readiness.json"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ \
   && -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "source commit must be clean and exact" >&2; exit 2; }
[[ -x .venv/bin/python ]] \
  || { echo "repository virtual-environment Python is unavailable" >&2; exit 2; }
[[ -f "$FORKED_FA2_SO" && ! -L "$FORKED_FA2_SO" \
   && "$(stat -c '%s' "$FORKED_FA2_SO")" == "$QROW16_BYTES" \
   && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" == "$QROW16_SHA256" ]] \
  || { echo "FORKED_FA2_SO is not the pinned Qrow16 binary" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$BLOCKS" | awk '{print $1}')" == "$BLOCKS_SHA256" ]] \
  || { echo "one-task subset or K64 block-map identity drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
.venv/bin/python scripts/fr13_sfwd_conv_postprep_gate.py source-manifest \
  "${DIRECT_ARGS[@]}" \
  --repo "$REPO" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$MANIFEST_LAUNCH"
MANIFEST_SHA256=$(sha256sum "$MANIFEST_LAUNCH" | awk '{print $1}')
MANIFEST_CONTAINER="/workspace/${MANIFEST_LAUNCH#"$REPO/"}"
.venv/bin/python scripts/fr13_sfwd_conv_postprep_gate.py host-readiness \
  "${DIRECT_ARGS[@]}" \
  --repo "$REPO" \
  --source-commit "$SOURCE_COMMIT" \
  --source-manifest "$MANIFEST_LAUNCH" \
  --fa2-so "$FORKED_FA2_SO" \
  --output "$READINESS" \
  >/dev/null

export RUNROOT=${RUNROOT_ABS#"$REPO/"}
export FR13_B1_WORKLOAD_PROFILE=k64_root
export FR13_B1_DIAGNOSTIC_TASK_PROFILE=astropy12907
export FR13_GATE_QROW16=0
export FR13_GATE_TAW_NATIVE=0
export FR13_GATE_DRAFT_HEAD_PAD=0
export FR13_GATE_DRAFT_HEAD_M32=0
export FR13_GATE_DRAFT_HEAD_FP8=0
export FR13_GATE_DFWD_TOP3=0
export FR13_GATE_BM8=0
export FR13_GATE_GDN_BV=0
export FR13_GATE_SFWD_CONV_POSTPREP=1
export FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0
export FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0
export FR13_DFWD_UNIFIED_BM8_LIVE_AB=0
export FR13_DFWD_UNIFIED_BM8_PRODUCTION=0
export FR13_FIXED32_BATCH_GDN_BYTE_AB=0
export FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0
export FR13_FIXED32_BATCH_GDN_PRODUCTION=0
export FR13_FIXED32_BATCH_GDN_BV_CANDIDATE=
export FR13_FIXED32_BATCH_GDN_BV_PRODUCTION=
export FR13_FIXED32_CUTLASS_WAVE=stock
export FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0
export FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0
export FR13_FIXED32_SFWD_STATE_FUSION_TIMING_AB=0
export FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0
export FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB=0
export FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0
export FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=1
export FR13_FIXED32_SFWD_EMBED_GATE_CTA
export FR13_FIXED32_SFWD_NODEGROUP8_DIRECT
export FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_JSON=
export FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_SHA256=
export FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_PATH="$MANIFEST_CONTAINER"
export FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256="$MANIFEST_SHA256"
export FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_COMMIT="$SOURCE_COMMIT"
export FR13_CONV_WB_BATCHED=1
export FR13_TREE_CONV_FUSED=1
export FR13_FIXED32_CONV_SOURCE_BATCH=0
export ENFORCE_EAGER=1
export FR10_METRICS=0
unset FR10_ALLOW_LINEAR_FALLBACK FR13_NEEDS_ALLOW

MANIFEST_FINALIZED=0
finalize_manifest() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  .venv/bin/python scripts/fr13_sfwd_conv_postprep_gate.py source-manifest \
    "${DIRECT_ARGS[@]}" \
    --repo "$REPO" \
    --source-commit "$SOURCE_COMMIT" \
    --output "$MANIFEST_END"
  cmp -s "$MANIFEST_LAUNCH" "$MANIFEST_END" \
    || { echo "source manifest changed during the task" >&2; return 14; }
  MANIFEST_FINALIZED=1
}

runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    if finalize_manifest; then
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
finalize_manifest

if [[ "$FR13_FIXED32_SFWD_EMBED_GATE_CTA" == "1" ]]; then
  .venv/bin/python scripts/fr13_sfwd_conv_postprep_gate.py validate-embedded \
    "${DIRECT_ARGS[@]}" \
    --repo "$REPO" \
    --arm-dir "$ARMDIR" \
    --source-commit "$SOURCE_COMMIT" \
    --batch-size 1 \
    --manifest-launch "$MANIFEST_LAUNCH" \
    --manifest-end "$MANIFEST_END" \
    --output "$ARMDIR/sfwd_embedded_gate_k64_root_b1_gate.json"
else
  .venv/bin/python scripts/fr13_sfwd_conv_postprep_gate.py validate \
    "${DIRECT_ARGS[@]}" \
    --repo "$REPO" \
    --arm-dir "$ARMDIR" \
    --source-commit "$SOURCE_COMMIT" \
    --task-id "$TASK_ID" \
    --manifest-launch "$MANIFEST_LAUNCH" \
    --manifest-end "$MANIFEST_END" \
    --output "$ARMDIR/sfwd_conv_postprep_k64_root_b1_gate.json"
fi
trap - EXIT

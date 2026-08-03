#!/usr/bin/env bash
# One real SWE-Verified eager boot issuing target-GEMM and SFWD byte credentials.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

case "${FR13_RUN_B1_TARGET_SFWD_GATE:-0}" in
  1) ;;
  0)
    echo "combined target/SFWD gate is disabled; set FR13_RUN_B1_TARGET_SFWD_GATE=1" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_B1_TARGET_SFWD_GATE must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

: "${RUNROOT:?set RUNROOT to a fresh path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned Qrow16 binary}"
: "${CUTLASS_STREAMK_SO:?set CUTLASS_STREAMK_SO to the pinned target binary}"

SOURCE_COMMIT=$(git rev-parse HEAD)
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}
ARM="hydra27_fixed32_k64_root_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"
MANIFEST_LAUNCH="$RUNROOT_ABS/sfwd_conv_postprep_source_manifest.at_launch.json"
MANIFEST_END="$RUNROOT_ABS/sfwd_conv_postprep_source_manifest.at_end.json"
READINESS="$RUNROOT_ABS/sfwd_conv_postprep_host_readiness.json"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* \
   && ! -e "$RUNROOT_ABS" \
   && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be a fresh path below $REPO/output" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" \
   && "$(git rev-parse '@{upstream}')" == "$SOURCE_COMMIT" ]] \
  || { echo "combined gate requires a clean source commit pushed to upstream" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
.venv/bin/python scripts/fr13_sfwd_conv_postprep_gate.py source-manifest \
  --repo "$REPO" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$MANIFEST_LAUNCH"
MANIFEST_SHA256=$(sha256sum "$MANIFEST_LAUNCH" | awk '{print $1}')
.venv/bin/python scripts/fr13_sfwd_conv_postprep_gate.py host-readiness \
  --repo "$REPO" \
  --source-commit "$SOURCE_COMMIT" \
  --source-manifest "$MANIFEST_LAUNCH" \
  --fa2-so "$FORKED_FA2_SO" \
  --output "$READINESS" \
  >/dev/null

# The nested kernel runner prefixes container paths with /workspace, so keep its
# RUNROOT repository-relative while retaining absolute paths for this wrapper.
export RUNROOT="$RUNROOT_REL"
export FR13_STREAMK_GATE_CANDIDATE=identity_wide256_fullgrid_b1
export FR13_STREAMK_QUALIFICATION_PROFILE=k64_root
export FR13_B1_DIAGNOSTIC_TASK_PROFILE=astropy12907
export FR13_STREAMK_SFWD_COMBINED_GATE=1
export FR13_GATE_SFWD_CONV_POSTPREP=1
export FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0
export FR13_FIXED32_SFWD_STATE_FUSION_TIMING_AB=0
export FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0
export FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB=0
export FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0
export FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=1
export FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_JSON=
export FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_SHA256=
export FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_PATH="/workspace/$RUNROOT_REL/sfwd_conv_postprep_source_manifest.at_launch.json"
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
    --repo "$REPO" \
    --source-commit "$SOURCE_COMMIT" \
    --output "$MANIFEST_END"
  cmp -s "$MANIFEST_LAUNCH" "$MANIFEST_END" \
    || { echo "source manifest changed during target/SFWD gate" >&2; return 14; }
  MANIFEST_FINALIZED=1
}
runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    finalize_manifest || { local mrc=$?; (( rc == 0 )) && rc=$mrc; }
  fi
  exit "$rc"
}
trap runner_exit EXIT

bash scripts/fr13_run_b1_cutlass_streamk_live_gate.sh
finalize_manifest

.venv/bin/python scripts/fr13_sfwd_conv_postprep_gate.py validate \
  --repo "$REPO" \
  --arm-dir "$ARMDIR" \
  --source-commit "$SOURCE_COMMIT" \
  --task-id astropy__astropy-12907 \
  --manifest-launch "$MANIFEST_LAUNCH" \
  --manifest-end "$MANIFEST_END" \
  --target-live-pass "$ARMDIR/cutlass_identity_wide256_fullgrid_b1_k64_root_byte_gate.json" \
  --target-candidate-so "$CUTLASS_STREAMK_SO" \
  --output "$ARMDIR/sfwd_conv_postprep_k64_root_b1_gate.json"
trap - EXIT

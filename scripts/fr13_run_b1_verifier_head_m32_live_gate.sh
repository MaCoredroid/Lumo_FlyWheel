#!/usr/bin/env bash
# One real SWE-Verified K64/root1 B1 raw-BF16 verifier-head shadow gate.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path below output/}"
: "${TAG:?set TAG to a unique run tag}"

FORKED_FA2_SO=${FORKED_FA2_SO:-/home/mark/lumoFlyWheel-kernel-integrated/output/fr13_qrow16_production_assets/_vllm_fa2_C.qrow16_num_splits0.abi3.so}
VERIFIER_HEAD_M32_SO=${VERIFIER_HEAD_M32_SO:-/home/mark/fr13_bf16_verifier_head_m32_sm121a_20260805/fr13_bf16_verifier_head_m32_sm121a.abi3.so}
VERIFIER_HEAD_M32_SHA256=5b5e8c3051f29bc4f65ef93c96ed22ef38ef07a1754e9c36a167e5158f71f4b7
VERIFIER_HEAD_M32_BYTES=186048
KERNEL_SOURCE=csrc/fr13_bf16_verifier_head_m32_sm121a.cu
KERNEL_SOURCE_SHA256=7cbc9f5157d8e93ee35930b028d97d0c3b1a26a9d79aa87ec6061928f8161768

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
RUNROOT_ABS=$(realpath -m "$RUNROOT")
[[ "$RUNROOT_ABS" == "$REPO/output/"* \
   && ! -e "$RUNROOT_ABS" \
   && ! -L "$RUNROOT_ABS" ]] || {
  echo "RUNROOT must be a new path below $REPO/output" >&2
  exit 2
}
for binary in "$FORKED_FA2_SO" "$VERIFIER_HEAD_M32_SO"; do
  [[ "$binary" == /* && -f "$binary" && ! -L "$binary" ]] || {
    echo "required binary is not an absolute regular file: $binary" >&2
    exit 2
  }
done
unset binary
[[ "$(stat -c '%s' "$VERIFIER_HEAD_M32_SO")" == "$VERIFIER_HEAD_M32_BYTES" \
   && "$(sha256sum "$VERIFIER_HEAD_M32_SO" | awk '{print $1}')" \
      == "$VERIFIER_HEAD_M32_SHA256" \
   && "$(sha256sum "$KERNEL_SOURCE" | awk '{print $1}')" \
      == "$KERNEL_SOURCE_SHA256" ]] || {
  echo "verifier-head M32 candidate identity drifted" >&2
  exit 2
}
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] || {
  echo "tracked worktree must be clean" >&2
  exit 2
}
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] || {
  echo "all Docker containers must be absent before the live gate" >&2
  exit 2
}

export RUNROOT=${RUNROOT_ABS#"$REPO/"}
export FORKED_FA2_SO
export FR13_B1_WORKLOAD_PROFILE=k64_root
export FR13_B1_DIAGNOSTIC_TASK_PROFILE=astropy12907
export FR13_GATE_QROW16=0
export FR13_GATE_TAW_NATIVE=0
export FR13_GATE_DRAFT_HEAD_PAD=0
export FR13_GATE_DRAFT_HEAD_M32=0
export FR13_GATE_DRAFT_HEAD_FP8=0
export FR13_GATE_DRAFT_HEAD_FP8_STATIC_IO=0
export FR13_GATE_DFWD_TOP3=0
export FR13_GATE_VERIFIER_HEAD_M32=1
export FR13_GATE_VERIFIER_HEAD_M32_SO="$VERIFIER_HEAD_M32_SO"
export FR13_GATE_BM8=0
export FR13_GATE_GDN_BV=0
export FR13_GATE_SFWD_CONV_POSTPREP=0
export FR13_FIXED32_CUTLASS_WAVE=stock
export FR13_FIXED32_CUTLASS_WAVE_SO=
export FR13_FIXED32_B1_FP8_QUANT_REGCACHE=0
export FR13_FIXED32_B1_FP8_QUANT_REGCACHE_SO=
export FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0
export FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0
export FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0
export FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=0
export FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0
export FR13_DFWD_UNIFIED_BM8_LIVE_AB=0
export FR13_DFWD_UNIFIED_BM8_PRODUCTION=0
export FR13_FIXED32_ATTRIBUTION_ONLY=0
export ENFORCE_EAGER=1

exec bash scripts/fr13_run_b1_kernel_live_gate.sh

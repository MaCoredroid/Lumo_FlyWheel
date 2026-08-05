#!/usr/bin/env bash
# One real SWE-Verified Hydra27 B1 fixed-K64/root1 DFWD U8 shadow gate.
# It compares all 65,536 configured head logits, not the full model vocabulary.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path below output/}"
: "${TAG:?set TAG to a unique run tag}"

FORKED_FA2_SO=${FORKED_FA2_SO:-/home/mark/shared/lumoFlyWheel-fa2-suffix-only/output/fr13_fa2_suffix_fc855e59_build/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so}
DFWD_U8_SO=${DFWD_U8_SO:-/home/mark/fr13_dfwd_u8_linked_build_3bdd984c2/det-primary-bin/fr13_bf16_k64_m1_r64_u8.abi3.so}

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "TAG contains unsafe characters" >&2
  exit 2
}
RUNROOT_ABS=$(realpath -m "$RUNROOT")
[[ "$RUNROOT_ABS" == "$REPO/output/"* \
   && ! -e "$RUNROOT_ABS" \
   && ! -L "$RUNROOT_ABS" ]] || {
  echo "RUNROOT must be a new path below $REPO/output" >&2
  exit 2
}
for binary in "$FORKED_FA2_SO" "$DFWD_U8_SO"; do
  [[ "$binary" == /* && -f "$binary" && ! -L "$binary" ]] || {
    echo "required binary is not an absolute regular file: $binary" >&2
    exit 2
  }
done
unset binary
[[ "$(stat -c '%s' "$FORKED_FA2_SO")" == "299183936" \
   && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" \
      == "f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d" \
   && "$(stat -c '%s' "$DFWD_U8_SO")" == "117904" \
   && "$(sha256sum "$DFWD_U8_SO" | awk '{print $1}')" \
      == "8b27df4f3c6a5a0574261ee984159582a87615c3e6d83f2a267f4fa46a3e421e" \
   && "$(sha256sum csrc/fr13_bf16_gemvx_k64_m1_shuffle_r64_u8.cu | awk '{print $1}')" \
      == "af0044edd84ff58d353a816f6887894d05a62b221e0efa5af933c2c59676b01b" \
   && "$(sha256sum results/fr13_fixed32_dfwd_k64_m1_r64_u8_linked_build_20260805/build_attestation.json | awk '{print $1}')" \
      == "e7ec95d1fff3b665373ad7b3a14f7e3fad346cf77a5f2f992a90a689e5672c8f" ]] || {
  echo "DFWD U8 qualification input identity drifted" >&2
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
# The generic gate uses this outer selector for isolation and terminal evidence.
export FR13_GATE_DRAFT_HEAD_M32=1
export FR13_GATE_DRAFT_HEAD_U8=1
export FR13_GATE_DRAFT_HEAD_U8_SO="$DFWD_U8_SO"
export FR13_GATE_DRAFT_HEAD_FP8=0
export FR13_GATE_DRAFT_HEAD_FP8_STATIC_IO=0
export FR13_GATE_DFWD_TOP3=0
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
export ENFORCE_EAGER=0
export CUDAGRAPH_MODE=FULL_AND_PIECEWISE

exec bash scripts/fr13_run_b1_kernel_live_gate.sh

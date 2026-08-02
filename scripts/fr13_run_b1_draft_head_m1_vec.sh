#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)

export FR13_GATE_QROW16=0
export FR13_GATE_TAW_NATIVE=0
export FR13_GATE_DRAFT_HEAD_PAD=0
export FR13_GATE_DRAFT_HEAD_M32=0
export FR13_GATE_DRAFT_HEAD_M1_VEC=pair8bits
export FR13_GATE_BM8=0
export FR13_GATE_GDN_BV=0
export FR13_DRAFT_HEAD_M1_VEC_SO=${FR13_DRAFT_HEAD_M1_VEC_SO:-$REPO/results/fr13_fixed32_dfwd_k64_m1_warp32_r32_pair8bits_build_20260802/fr13_bf16_gemvx_k64_m1_warp32_r32_pair8bits.abi3.so}

exec bash "$SCRIPT_DIR/fr13_run_b1_kernel_live_gate.sh"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

: "${FR13_GATE_DFWD_TOP3_SO:?set FR13_GATE_DFWD_TOP3_SO to the pinned candidate shared object}"

export FR13_B1_WORKLOAD_PROFILE=k64_root
export FR13_GATE_QROW16=0
export FR13_GATE_TAW_NATIVE=0
export FR13_GATE_DRAFT_HEAD_PAD=0
export FR13_GATE_DRAFT_HEAD_M32=0
export FR13_GATE_DRAFT_HEAD_FP8=0
export FR13_GATE_DFWD_TOP3=1
export FR13_GATE_BM8=0
export FR13_GATE_GDN_BV=0
export FR13_FIXED32_CUTLASS_WAVE=stock
export FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE=k64_root

exec bash "$SCRIPT_DIR/fr13_run_b1_kernel_live_gate.sh"

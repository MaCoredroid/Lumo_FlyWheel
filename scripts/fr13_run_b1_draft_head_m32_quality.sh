#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export FR13_GATE_QROW16=0
export FR13_GATE_TAW_NATIVE=0
export FR13_GATE_DRAFT_HEAD_PAD=0
export FR13_GATE_DRAFT_HEAD_PAD_ROWS=32
export FR13_GATE_DRAFT_HEAD_M32=0
export FR13_GATE_DRAFT_HEAD_M1_VEC=0
export FR13_GATE_BM8=0
export FR13_GATE_GDN_BV=0

exec bash "$SCRIPT_DIR/fr13_run_b1_kernel_live_gate.sh"

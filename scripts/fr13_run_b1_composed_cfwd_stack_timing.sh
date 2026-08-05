#!/usr/bin/env bash
# Exact4 timing for the smoke-qualified B1 CFWD production stack.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

case "${FR13_RUN_B1_COMPOSED_CFWD_TIMING:-0}" in
  1) ;;
  0)
    echo "composed CFWD timing is disabled; set FR13_RUN_B1_COMPOSED_CFWD_TIMING=1" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_B1_COMPOSED_CFWD_TIMING must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

export FR13_RUN_QROW32_SPLIT2_TIMING=1
export FR13_B1_COMPOSED_STACK_TIMING=1
export FR13_B1_COMPOSED_CFWD_PRODUCTION=1
export FR13_B1_COMPOSED_CFWD_SMOKE=0

exec bash "$SCRIPT_DIR/fr13_run_b1_k64_qrow32_b1_sfwd_stack_timing.sh"

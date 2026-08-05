#!/usr/bin/env bash
# Final-HEAD candidate-only exact4 timing for the credentialed B1 kernel stack.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

case "${FR13_RUN_B1_COMPOSED_STACK_TIMING:-0}" in
  1) ;;
  0)
    echo "composed B1 timing is disabled; set FR13_RUN_B1_COMPOSED_STACK_TIMING=1" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_B1_COMPOSED_STACK_TIMING must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

export FR13_RUN_QROW32_NOSPLIT_TIMING=1
export FR13_B1_COMPOSED_STACK_TIMING=1

exec bash "$SCRIPT_DIR/fr13_run_b1_k64_qrow32_b1_sfwd_stack_timing.sh"

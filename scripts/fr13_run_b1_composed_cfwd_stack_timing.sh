#!/usr/bin/env bash
# Exact4/exact16 timing for the fresh U8, packed-CFWD, target/SFWD stack.
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

export FR13_RUN_B1_U8_CFWD_SFWD_TIMING=1

exec bash "$SCRIPT_DIR/fr13_run_b1_u8_cfwd_sfwd_stack_timing.sh"

#!/usr/bin/env bash
# One real SWE-Verified K64/root1 B1 ordered-root-loop GDN byte diagnostic.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENTRYPOINT=$(realpath "${BASH_SOURCE[0]}")
exec env QUALIFICATION_ENTRYPOINT="$ENTRYPOINT" \
  bash "$SCRIPT_DIR/fr13_run_gdn_single_launch_live_gate.sh" b1

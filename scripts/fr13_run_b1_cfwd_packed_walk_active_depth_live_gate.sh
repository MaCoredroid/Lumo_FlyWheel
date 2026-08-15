#!/usr/bin/env bash
# Real SWE-Verified B1 byte gate for the default-off active-depth packed walk.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

ACTIVE_DEPTH_SOURCE=scripts/fr13_cfwd_packed_walk_active_depth_kernel.py
ACTIVE_DEPTH_SOURCE_SHA256=e2f3354a2e7120c7f1a6c6ccf9381fda12cb985b129c2a15bd444fbc39b086ef
ACTIVE_DEPTH_OVERLAY=scripts/fr13_cfwd_packed_walk_active_depth_runtime_overlay.py
ACTIVE_DEPTH_OVERLAY_SHA256=8c8ef918c09102244587ba3fc46339b86cd8448bcb62db6ba04035713c07caee
CFWD_RUNTIME=scripts/fr13_device_multidraft_cfwd_packed_v3.py
CFWD_RUNTIME_SHA256=5e629adadf85a20c8aced5beb3753a4b7b0fa03f2523f7309180b38c86a7b766
BASE_RUNNER=scripts/fr13_run_b1_cfwd_logit_direct_live_gate.sh
BASE_RUNNER_SHA256=e1030ce75e4d012e3e59801e1b00076b0cb9dbfa4a2aed219309df2e7208ff3f

for binding in \
  "$ACTIVE_DEPTH_SOURCE:$ACTIVE_DEPTH_SOURCE_SHA256" \
  "$ACTIVE_DEPTH_OVERLAY:$ACTIVE_DEPTH_OVERLAY_SHA256" \
  "$CFWD_RUNTIME:$CFWD_RUNTIME_SHA256" \
  "$BASE_RUNNER:$BASE_RUNNER_SHA256"; do
  path=${binding%%:*}
  expected=${binding#*:}
  [[ -f "$path" && ! -L "$path" \
     && "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "packed-walk active-depth source binding drifted: $path" >&2
    exit 2
  }
done
unset binding path expected

exec env \
  FR13_CFWD_PACKED_WALK_NODE_TRUST_BYTE_AB=0 \
  FR13_CFWD_PACKED_WALK_NODE_TRUST_PRODUCTION=0 \
  FR13_CFWD_PACKED_WALK_ACTIVE_DEPTH_BYTE_AB=1 \
  bash "$BASE_RUNNER"

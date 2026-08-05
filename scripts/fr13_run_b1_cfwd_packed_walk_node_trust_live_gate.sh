#!/usr/bin/env bash
# Real SWE-Verified B1 byte gate for the default-off packed-walk node trust.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

NODE_TRUST_SOURCE=scripts/fr13_cfwd_packed_walk_node_trust_kernel.py
NODE_TRUST_SOURCE_SHA256=07cd03173ab1a6e6b9aa597d9c912475034f5b8100c2c57d819b2b7bbcf3bc37
NODE_TRUST_OVERLAY=scripts/fr13_cfwd_packed_walk_node_trust_runtime_overlay.py
NODE_TRUST_OVERLAY_SHA256=8f334ec0ff99e1297415f15b70f59378127e81965e9417d8091a7e801768d153
CFWD_RUNTIME=scripts/fr13_device_multidraft_cfwd_packed_v3.py
CFWD_RUNTIME_SHA256=67a7a6ad2ad9e3dcb5972d9b654a6325c65970c87a7e0304c7a98c9aa3c3a967
BASE_RUNNER=scripts/fr13_run_b1_cfwd_logit_direct_live_gate.sh
BASE_RUNNER_SHA256=e1030ce75e4d012e3e59801e1b00076b0cb9dbfa4a2aed219309df2e7208ff3f

for binding in \
  "$NODE_TRUST_SOURCE:$NODE_TRUST_SOURCE_SHA256" \
  "$NODE_TRUST_OVERLAY:$NODE_TRUST_OVERLAY_SHA256" \
  "$CFWD_RUNTIME:$CFWD_RUNTIME_SHA256" \
  "$BASE_RUNNER:$BASE_RUNNER_SHA256"; do
  path=${binding%%:*}
  expected=${binding#*:}
  [[ -f "$path" && ! -L "$path" \
     && "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "packed-walk node-trust source binding drifted: $path" >&2
    exit 2
  }
done
unset binding path expected

exec env \
  FR13_CFWD_PACKED_WALK_NODE_TRUST_BYTE_AB=1 \
  bash "$BASE_RUNNER"

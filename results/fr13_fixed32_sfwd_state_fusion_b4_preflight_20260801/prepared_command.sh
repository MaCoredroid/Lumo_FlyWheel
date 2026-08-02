#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

: "${FORKED_FA2_SO:?set the absolute pinned stock FA2 shared-object path}"
RUNROOT=${RUNROOT:-output/fr13_sfwd_state_fusion_exact4_b4_gate_20260801}
TAG=${TAG:-exact4_b4_sfwd_v1}

exec env \
  RUNROOT="$RUNROOT" \
  TAG="$TAG" \
  FORKED_FA2_SO="$FORKED_FA2_SO" \
  bash scripts/fr13_run_b4_sfwd_state_fusion_live_gate.sh

#!/usr/bin/env bash
set -euo pipefail

: "${PINNED_SOURCE_COMMIT:?set the exact trusted production-capable commit}"
: "${FORKED_FA2_SO:?set the absolute canonical FA2 binary path}"

REPO=$(git rev-parse --show-toplevel)
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$PINNED_SOURCE_COMMIT" ]] \
  || { echo "checkout does not match PINNED_SOURCE_COMMIT" >&2; exit 2; }

ARTIFACT=results/fr13_fixed32_bf16_gemvx_m1_b1_ready_20260801
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
UTC=$(date -u +%Y%m%dT%H%M%SZ)

RUNROOT="output/fr13_bf16_gemvx_m1_b1_live_${UTC}" \
TAG="bf16_gemvx_m1_b1_live_${UTC}" \
PYTHON_BIN="$PYTHON_BIN" \
FORKED_FA2_SO="$FORKED_FA2_SO" \
FR13_DRAFT_HEAD_M1_SO="$REPO/$ARTIFACT/fr13_bf16_gemvx_m1.abi3.so" \
FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION="$REPO/$ARTIFACT/build_attestation.json" \
FR13_DRAFT_HEAD_M1_EXPECTED_SOURCE_COMMIT="$PINNED_SOURCE_COMMIT" \
bash scripts/fr13_run_b1_draft_head_m1_live.sh

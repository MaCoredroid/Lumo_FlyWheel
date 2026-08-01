#!/usr/bin/env bash
set -euo pipefail

: "${PINNED_SOURCE_COMMIT:?set the exact trusted branch or merged commit}"
: "${FORKED_FA2_SO:?set an absolute path to the canonical FA2 binary}"

PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}
CANONICAL_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
CANONICAL_FA2_SIZE=299183936
[[ -x "$PYTHON_BIN" ]] \
  || { echo "host Python is not executable: $PYTHON_BIN" >&2; exit 2; }
[[ "$FORKED_FA2_SO" == /* \
   && -f "$FORKED_FA2_SO" \
   && ! -L "$FORKED_FA2_SO" \
   && "$(stat -c %s "$FORKED_FA2_SO")" == "$CANONICAL_FA2_SIZE" \
   && "$(sha256sum "$FORKED_FA2_SO" | cut -d' ' -f1)" \
      == "$CANONICAL_FA2_SHA256" ]] \
  || { echo "FORKED_FA2_SO does not have the canonical identity" >&2; exit 2; }

REPO=$(git rev-parse --show-toplevel)
cd "$REPO"
UTC=$(date -u +%Y%m%dT%H%M%SZ)
ARTIFACT=results/fr13_fixed32_bf16_gemvx_m1_b1_ready_20260801

RUNROOT="output/fr13_bf16_gemvx_m1_b1_live_${UTC}" \
TAG="bf16_gemvx_m1_b1_live_${UTC}" \
PYTHON_BIN="$PYTHON_BIN" \
FORKED_FA2_SO="$FORKED_FA2_SO" \
FR13_DRAFT_HEAD_M1_SO="$REPO/$ARTIFACT/fr13_bf16_gemvx_m1.abi3.so" \
FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION="$REPO/$ARTIFACT/build_attestation.json" \
FR13_DRAFT_HEAD_M1_EXPECTED_SOURCE_COMMIT="$PINNED_SOURCE_COMMIT" \
bash scripts/fr13_run_b1_draft_head_m1_live.sh

#!/usr/bin/env bash
set -euo pipefail

: "${PINNED_SOURCE_COMMIT:?set the exact trusted branch or merged commit}"

REPO=$(git rev-parse --show-toplevel)
cd "$REPO"
UTC=$(date -u +%Y%m%dT%H%M%SZ)
ARTIFACT=results/fr13_fixed32_bf16_gemvx_m1_b1_ready_20260801
CANONICAL_FA2="$REPO/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so"

RUNROOT="output/fr13_bf16_gemvx_m1_b1_live_${UTC}" \
TAG="bf16_gemvx_m1_b1_live_${UTC}" \
FORKED_FA2_SO="$CANONICAL_FA2" \
FR13_DRAFT_HEAD_M1_SO="$REPO/$ARTIFACT/fr13_bf16_gemvx_m1.abi3.so" \
FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION="$REPO/$ARTIFACT/build_attestation.json" \
FR13_DRAFT_HEAD_M1_EXPECTED_SOURCE_COMMIT="$PINNED_SOURCE_COMMIT" \
bash scripts/fr13_run_b1_draft_head_m1_live.sh


#!/usr/bin/env bash
set -euo pipefail

REPO=/home/mark/shared/lumoFlyWheel-fullhead-m32
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
cd "$REPO"

RUNROOT="output/fr13_b1_draft_head_msweep_live_${STAMP}" \
TAG="draft_head_msweep_fullvocab_b1_${STAMP}" \
FORKED_FA2_SO="$REPO/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so" \
  bash scripts/fr13_run_b1_draft_head_msweep_live.sh


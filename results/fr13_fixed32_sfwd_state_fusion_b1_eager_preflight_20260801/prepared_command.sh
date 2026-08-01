#!/usr/bin/env bash
set -euo pipefail

cd /home/mark/lumoFlyWheel-sfwd-state-fusion-timing-b1
UTC=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT="$PWD/output/fr13_b1_sfwd_state_fusion_live_gate_${UTC}" \
TAG="sfwd_state_fusion_fullvocab_b1_${UTC}" \
FORKED_FA2_SO="$PWD/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so" \
bash scripts/fr13_run_b1_sfwd_state_fusion_gate.sh

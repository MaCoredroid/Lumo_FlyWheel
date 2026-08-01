#!/usr/bin/env bash
set -euo pipefail

cd /home/mark/lumoFlyWheel-sfwd-timing-eager-preflight
: "${SFWD_PASS_JSON:?set SFWD_PASS_JSON to the absolute final live PASS from the completed B1 every-byte gate}"
SFWD_PASS_JSON=$(realpath "$SFWD_PASS_JSON")
[[ -f "$SFWD_PASS_JSON" && ! -L "$SFWD_PASS_JSON" ]] || {
  echo "SFWD_PASS_JSON must be a regular non-symlink file" >&2
  exit 2
}
SFWD_PASS_SHA256=$(sha256sum "$SFWD_PASS_JSON" | awk '{print $1}')
UTC=$(date -u +%Y%m%dT%H%M%SZ)

RUNROOT="$PWD/output/fr13_b1_sfwd_state_fusion_timing_${UTC}" \
TAG="sfwd_state_fusion_fullvocab_b1_timing_${UTC}" \
FORKED_FA2_SO="/home/mark/fr13_runs/auto_research_streamk_source_20260801/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so" \
SFWD_PASS_JSON="$SFWD_PASS_JSON" \
SFWD_PASS_SHA256="$SFWD_PASS_SHA256" \
bash scripts/fr13_run_b1_sfwd_state_fusion_timing.sh


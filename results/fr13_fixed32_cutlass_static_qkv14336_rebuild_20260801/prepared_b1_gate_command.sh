#!/usr/bin/env bash
set -euo pipefail

cd /home/mark/lumoFlyWheel-static-qkv14336
UTC=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT="$PWD/output/fr13_b1_cutlass_static_qkv14336_live_gate_${UTC}" \
TAG="cutlass_static_qkv14336_fullvocab_b1_${UTC}" \
FORKED_FA2_SO="$PWD/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so" \
CUTLASS_STREAMK_SO="/home/mark/fr13_static_qkv14336_build/build/_C_stable_libtorch.abi3.so" \
FR13_STREAMK_GATE_CANDIDATE=static_persistent_stocktile \
bash scripts/fr13_run_b1_cutlass_streamk_live_gate.sh

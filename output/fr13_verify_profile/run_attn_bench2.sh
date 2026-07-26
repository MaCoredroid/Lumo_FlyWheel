#!/usr/bin/env bash
# Verifier V2 microbench v3: FORK .so installed (matches serving kernel), fixed
# exception handling. Waits for the dvkg128L live arm to free the GPU.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
until grep -q "DVKG128L_DONE" output/fr13_msr/dvkg128L_console.log 2>/dev/null; do sleep 300; done
while docker ps --format '{{.Names}}' | grep -q fr13; do sleep 30; done
IMAGE="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
FORK_SO="output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so"
mkdir -p output/fr13_verify_profile/attn_bench_logs
docker run --rm --gpus all --entrypoint bash \
  -v "$PWD:/workspace" -v "$PWD/output/fr13_verify_profile/attn_bench_logs:/logs" \
  --name fr13-attn-bench "$IMAGE" -c "
    cp /workspace/$FORK_SO /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so &&
    python3 /workspace/scripts/fr13_attn_mgeom_bench.py" \
  > output/fr13_verify_profile/attn_mgeom_bench2.log 2>&1
echo "ATTN_BENCH2_DONE rc=$?"

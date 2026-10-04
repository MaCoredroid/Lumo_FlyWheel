#!/usr/bin/env bash
# q1v3 native (spec-off, sequential single-token decode) reference engine with the q1v3 hooks.
# Usage: serve_native_q1v3.sh <ARM A|B|V|P> <RUN_ROOT> <KV_CACHE_MEMORY_BYTES> <PACKED 0|1>
# Mirrors scripts/v2exp/serve_native.sh (same image, same lm_head + forked-FA2 install, same serving flags) with:
#   + --kv-cache-memory-bytes <pinned>        (KV capacity no longer derived from free memory)
#   + q1v3 patcher (after the production patches) and PYTHONPATH=/q1v3 for the hooks
#   + VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=<PACKED> (0 = deployment mirror; 1 = stock packed GDN decode, arm P)
# Starts the container detached and returns; the caller waits for /health and stops it.
set -euo pipefail
ARM=${1:?arm}; RUN=${2:?run root}; KVB=${3:?kv bytes}; PACKED=${4:?packed 0|1}
[[ "$KVB" =~ ^[1-9][0-9]*$ ]] || { echo "KV bytes must be a positive integer" >&2; exit 2; }
[[ "$PACKED" == 0 || "$PACKED" == 1 ]] || { echo "PACKED must be 0 or 1" >&2; exit 2; }
WT=/home/mark/shared/lumotree-v2exp-20260930
Q1V3=$WT/scripts/v2exp/q1v3
IMAGE=vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
FA2_SO=/home/mark/fr14_splitk_build_20260818/_vllm_fa2_qrow32_gqa_pair_splitk_b1_sm121a.abi3.so
GPU_UTIL=${GPU_UTIL:-0.70}
NAME=q1v3-native-$ARM
OUT=$RUN/$ARM
mkdir -p "$OUT/engine-logs"
docker ps -q --filter publish=9950 | grep -q . && { echo "port 9950 busy" >&2; exit 3; }
SERVE="vllm serve /models/qwen3.8-27b-nvfp4-radixark --tokenizer /models/qwen3.8-27b-nvfp4-radixark \
 --served-model-name qwen3.8-27b-nvfp4-radixark --host 0.0.0.0 --port 9950 --max-num-seqs 1 \
 --gpu-memory-utilization $GPU_UTIL --kv-cache-memory-bytes $KVB --max-model-len 131072 --seed 0 --attention-backend FLASH_ATTN \
 --gdn-prefill-backend triton --enable-prefix-caching --enable-chunked-prefill --mamba-block-size 1024 \
 --mamba-ssm-cache-dtype float32 --block-size 1024 --mamba-cache-mode align --max-num-batched-tokens 4096 \
 --long-prefill-token-threshold 1024 --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}' \
 --chat-template /workspace/docker/chat_templates/qwen3-openai-codex.jinja --enable-auto-tool-choice \
 --tool-call-parser qwen3_xml --reasoning-parser qwen3"
echo "$SERVE" > "$OUT/serve_cmd.txt"
docker run -d --name "$NAME" --gpus all --ipc=host --memory=105g --memory-swap=105g \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 9950:9950 \
  -v "$WT":/workspace:ro -v /models:/models:ro -v "$OUT/engine-logs":/logs \
  -v "$FA2_SO":/tmp/fr13_fork_fa2.so:ro -v "$WT/scripts/v2exp/assets":/v2assets:ro \
  -v "$Q1V3":/q1v3:ro -v "$RUN":/q1run \
  -e MKL_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/q1v3:/workspace/src \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_BATCH_INVARIANT=0 \
  -e VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE="$PACKED" -e VLLM_SERVER_DEV_MODE=1 \
  -e Q1V3_ARM_KIND=native -e Q1V3_ARM="$ARM" -e Q1V3_OUT="/q1run/$ARM" -e Q1V3_RUN=/q1run -e Q1V3_CASES=/q1v3/cases.v1.json \
  --entrypoint bash "$IMAGE" -lc "set -euo pipefail; \
   python3 /workspace/scripts/fr14_patch_nvfp4_lmhead.py && \
   python3 /v2assets/q1_reference_fa2_install.py --fork /tmp/fr13_fork_fa2.so \
     --patcher /workspace/scripts/fr13_patch_fa2_tree_bias.py --receipt /logs/fa2_install_receipt.json && \
   python3 /q1v3/patch_runner.py --mode native --apply --receipt /q1run/$ARM/patch_receipt.json && \
   exec $SERVE" > "$OUT/container_id.txt"
echo "started $NAME $(cat "$OUT/container_id.txt")"

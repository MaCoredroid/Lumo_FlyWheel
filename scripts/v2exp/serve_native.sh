#!/usr/bin/env bash
# Same-build native serving arms for the LumoTree v2 experiments.
# Usage: serve_native.sh <ar|mtpK> <run_dir>
# Stock vLLM 0.19.2rc1.dev134 image (the LumoTree deployment base) with only:
#   - the NVFP4 lm_head loader patch (required to load the RadixArk checkpoint)
#   - the same forked FA2 .so the LumoTree route installs (same attention kernel)
# Serving flags mirror the LumoTree ten-task deployment except tree-only options.
set -euo pipefail
ARM=${1:?arm}; OUT=${2:?run dir}
WT=/home/mark/shared/lumotree-v2exp-20260930
IMAGE=vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
FA2_SO=/home/mark/fr14_splitk_build_20260818/_vllm_fa2_qrow32_gqa_pair_splitk_b1_sm121a.abi3.so
GPU_UTIL=${GPU_UTIL:-0.70}
NAME=v2exp-$ARM
if [[ "$ARM" == sglang-s*k*d* ]]; then
  # SGLang EAGLE baseline, same checkpoint and chat template as the vLLM arms.
  # ARM=sglang-s<steps>k<topk>d<draft_tokens>; flags otherwise follow the Codex
  # owned SGLang launcher v3 (flashinfer, chunked prefill 8192, bf16 KV, 1 request).
  read -r S K DT <<< "$(sed -E 's/^sglang-s([0-9]+)k([0-9]+)d([0-9]+)$/\1 \2 \3/' <<< "$ARM")"
  SG_IMAGE=sha256:0076dffa60b76b7bf033c04d05e0cc69d46f2b8cd60aa2468827782afe9bc38f
  docker ps -q --filter publish=9950 | grep -q . && { echo "port 9950 busy" >&2; exit 3; }
  mkdir -p "$OUT/engine-logs"
  SG="python3 -m sglang.launch_server --model-path /models/qwen3.8-27b-nvfp4-radixark --tokenizer-path /models/qwen3.8-27b-nvfp4-radixark \
 --served-model-name qwen3.8-27b-nvfp4-radixark --host 0.0.0.0 --port 9950 --trust-remote-code --attention-backend flashinfer \
 --chunked-prefill-size 8192 --random-seed ${V2_SEED:-0} --mem-fraction-static ${SG_MEM:-0.70} --speculative-algorithm EAGLE --speculative-num-steps $S \
 --speculative-eagle-topk $K --speculative-num-draft-tokens $DT --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
 --enable-metrics --context-length 131072 --max-running-requests 1 --kv-cache-dtype bf16 \
 --chat-template /workspace/docker/chat_templates/qwen3-openai-codex.jinja"
  echo "$SG" > "$OUT/serve_cmd.txt"
  docker run -d --name "$NAME" --gpus all --network host --shm-size 16g --ipc=host \
    -v "$WT":/workspace:ro -v /home/mark/shared/models:/models:ro -e PYTHONDONTWRITEBYTECODE=1 \
    "$SG_IMAGE" bash -lc "exec $SG" > "$OUT/container_id.txt"
  echo "started $NAME $(cat "$OUT/container_id.txt")"
  exit 0
fi
SPEC=""
case "$ARM" in
  ar) ;;
  mtp[0-9]*) K=${ARM#mtp}; SPEC="--speculative-config '{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$K}'" ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
docker ps -q --filter publish=9950 | grep -q . && { echo "port 9950 busy" >&2; exit 3; }
mkdir -p "$OUT/engine-logs"
SERVE="vllm serve /models/qwen3.8-27b-nvfp4-radixark --tokenizer /models/qwen3.8-27b-nvfp4-radixark \
 --served-model-name qwen3.8-27b-nvfp4-radixark --host 0.0.0.0 --port 9950 --max-num-seqs 1 \
 --gpu-memory-utilization $GPU_UTIL --max-model-len 131072 --seed ${V2_SEED:-0} --attention-backend FLASH_ATTN \
 --gdn-prefill-backend triton --enable-prefix-caching --enable-chunked-prefill --mamba-block-size 1024 \
 --mamba-ssm-cache-dtype float32 --block-size 1024 --mamba-cache-mode align --max-num-batched-tokens 4096 \
 --long-prefill-token-threshold 1024 --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}' \
 --chat-template /workspace/docker/chat_templates/qwen3-openai-codex.jinja --enable-auto-tool-choice \
 --tool-call-parser qwen3_xml --reasoning-parser qwen3 $SPEC"
echo "$SERVE" > "$OUT/serve_cmd.txt"
docker run -d --name "$NAME" --gpus all --ipc=host --memory=105g --memory-swap=105g \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 9950:9950 \
  -v "$WT":/workspace:ro -v /models:/models:ro -v "$OUT/engine-logs":/logs \
  -v "$FA2_SO":/tmp/fr13_fork_fa2.so:ro -v "$WT/scripts/v2exp/assets":/v2assets:ro \
  -e MKL_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/workspace/src \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_BATCH_INVARIANT=0 \
  -e VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0 -e VLLM_SERVER_DEV_MODE=1 \
  --entrypoint bash "$IMAGE" -lc "set -euo pipefail; \
   python3 /workspace/scripts/fr14_patch_nvfp4_lmhead.py && \
   python3 /v2assets/q1_reference_fa2_install.py --fork /tmp/fr13_fork_fa2.so \
     --patcher /workspace/scripts/fr13_patch_fa2_tree_bias.py --receipt /logs/fa2_install_receipt.json && \
   exec $SERVE" > "$OUT/container_id.txt"
echo "started $NAME $(cat "$OUT/container_id.txt")"

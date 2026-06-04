#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/mark/shared/lumoFlyWheel}
IMAGE=${IMAGE:-"vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"}
CONTAINER=${CONTAINER:-fr10-speed-start}
PORT=${PORT:-9950}
GPU_UTIL=${GPU_UTIL:-0.88}
BATCH_INVARIANT=${BATCH_INVARIANT:-0}
FR10_METRICS=${FR10_METRICS:-0}
FR10_DECODE_MODE_DEFAULT=${FR10_DECODE_MODE_DEFAULT:-tree_mtp}
LOG_DIR=${LOG_DIR:-"$REPO/output/fr10_speed_starting_point/live_logs"}
TREE=${TREE:-"[(0,), (1,), (0, 0), (1, 0), (0, 0, 0), (1, 0, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0, 0), (1, 0, 0, 0, 0)]"}
SPEC_CONFIG=${SPEC_CONFIG:-"{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":10,\"speculative_token_tree\":\"$TREE\"}"}

mkdir -p "$LOG_DIR"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

docker run -d --name "$CONTAINER" --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 -p "$PORT:9950" \
  -v "$REPO:/workspace" -v /models:/models -v "$LOG_DIR:/logs" \
  -e VLLM_BATCH_INVARIANT="$BATCH_INVARIANT" \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e PYTHONPATH=/workspace/src \
  -e FR10_ENABLE_TREE_GDN=1 \
  -e FR10_METRICS="$FR10_METRICS" \
  -e FR10_DECODE_MODE_DEFAULT="$FR10_DECODE_MODE_DEFAULT" \
  -e SPEC_CONFIG="$SPEC_CONFIG" \
  --entrypoint bash \
  "$IMAGE" \
  -lc "set -euo pipefail
python3 /workspace/scripts/fr10_phase4_patch_vllm_tree_gdn.py
exec vllm serve /models/qwen3.6-27b-fp8 --served-model-name qwen3.6-27b \
  --host 0.0.0.0 --port 9950 --max-num-seqs 4 \
  --gpu-memory-utilization '$GPU_UTIL' --max-model-len 131072 \
  --attention-backend FLASH_ATTN --gdn-prefill-backend triton \
  --chat-template /workspace/docker/chat_templates/qwen3-openai-codex.jinja \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --speculative-config \"\$SPEC_CONFIG\""

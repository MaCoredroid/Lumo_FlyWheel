#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/mark/shared/lumoFlyWheel}
OUT_DIR=${OUT_DIR:-"$REPO/output/fr10_phase4_tree_capture_probe_20260603"}
IMAGE=${IMAGE:-"vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"}
CONTAINER=${CONTAINER:-fr10-cu130-tree-capture}
PORT=${PORT:-9950}
GPU_UTIL=${GPU_UTIL:-0.85}
TREE=${TREE:-"[(0,), (1,), (0, 0), (1, 0), (0, 0, 0), (1, 0, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0, 0), (1, 0, 0, 0, 0)]"}

mkdir -p "$OUT_DIR"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

docker run -d --name "$CONTAINER" --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 -p "$PORT:8000" \
  -v "$REPO:/workspace" -v /models:/models \
  -e VLLM_BATCH_INVARIANT=1 -e VLLM_SERVER_DEV_MODE=1 \
  -e FR10_CAPTURE_GDN_TENSOR_DIR=/workspace/output/fr10_phase4_tree_capture_probe_20260603/tensors \
  --entrypoint bash \
  "$IMAGE" \
  -lc "set -euo pipefail
python3 /workspace/scripts/fr10_phase4_patch_gdn_capture.py
exec vllm serve /models/qwen3.6-27b-fp8 --served-model-name qwen3.6-27b \
  --host 0.0.0.0 --port 8000 --max-num-seqs 4 \
  --gpu-memory-utilization '$GPU_UTIL' --max-model-len 131072 \
  --attention-backend FLASH_ATTN --gdn-prefill-backend triton --enforce-eager \
  --speculative-config '{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":10,\"speculative_token_tree\":\"$TREE\"}'" \
  > "$OUT_DIR/container_id.txt"

cat "$OUT_DIR/container_id.txt"

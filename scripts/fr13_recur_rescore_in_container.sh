#!/usr/bin/env bash
# Run the RECURRENT decode oracle rescore INSIDE the pinned vLLM container
# (vllm is GPU-only / container-only). One arm per invocation.
#   scripts/fr13_recur_rescore_in_container.sh <arm> <capture_json> <out_json>
set -euo pipefail
ARM="${1:?arm}"; SRC="${2:?capture json}"; OUT="${3:?out json}"
REPO="${REPO:-/home/mark/shared/lumoFlyWheel}"
IMAGE="${IMAGE:-vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776}"
CONTAINER="${CONTAINER:-fr13-recur-oracle}"
SEED="${SEED:-1313}"; TOPK="${TOPK:-20}"; THRESH="${THRESH:-1.0}"
GPU_UTIL="${GPU_UTIL:-0.88}"

# relpaths inside container (/workspace == REPO)
REL_SRC="/workspace/${SRC#$REPO/}"
REL_OUT="/workspace/${OUT#$REPO/}"
mkdir -p "$(dirname "$OUT")"

# host-mem recovery before a fresh GPU boot
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run --rm --name "$CONTAINER" --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$REPO:/workspace" -v /models:/models \
  -e PYTHONPATH=/workspace/src \
  -e VLLM_USE_V1=1 \
  --entrypoint bash "$IMAGE" -lc "
set -euo pipefail
cd /workspace
python3 /workspace/scripts/fr13_recurrent_decode_oracle.py rescore \
  --arm '$ARM' \
  --src '$REL_SRC' \
  --out '$REL_OUT' \
  --seed $SEED --top-k $TOPK --threshold $THRESH \
  --gpu-util $GPU_UTIL --attn-backend FLASH_ATTN
"
echo "[rescore done] arm=$ARM out=$OUT"

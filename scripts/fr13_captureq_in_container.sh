#!/usr/bin/env bash
# Run capture-q-deploy (the SPEC VERIFY top-K q on the deployment served stream)
# INSIDE the pinned vLLM container. One arm per invocation.
#   scripts/fr13_captureq_in_container.sh <arm> <src_json> <out_json> <spec_config_json>
set -euo pipefail
ARM="${1:?arm}"; SRC="${2:?src json}"; OUT="${3:?out json}"; SPEC="${4:?spec_config json}"
REPO="${REPO:-/home/mark/shared/lumoFlyWheel}"
IMAGE="${IMAGE:-vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776}"
CONTAINER="${CONTAINER:-fr13-captureq-$ARM}"
SEED="${SEED:-1313}"; TOPK="${TOPK:-20}"; GPU_UTIL="${GPU_UTIL:-0.88}"
REL_SRC="/workspace/${SRC#$REPO/}"
REL_OUT="/workspace/${OUT#$REPO/}"
mkdir -p "$(dirname "$OUT")"
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run --rm --name "$CONTAINER" --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$REPO:/workspace" -v /models:/models \
  -e PYTHONPATH=/workspace/src -e VLLM_USE_V1=1 \
  --entrypoint bash "$IMAGE" -lc "
set -euo pipefail
cd /workspace
python3 /workspace/scripts/fr13_recurrent_decode_oracle.py capture-q-deploy \
  --arm '$ARM' --src '$REL_SRC' --out '$REL_OUT' \
  --spec-config '$SPEC' --seed $SEED --top-k $TOPK \
  --gpu-util $GPU_UTIL --attn-backend FLASH_ATTN
"
echo "[capture-q done] arm=$ARM out=$OUT"

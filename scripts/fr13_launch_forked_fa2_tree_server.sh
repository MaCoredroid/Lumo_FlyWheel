#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/mark/shared/lumoFlyWheel}
IMAGE=${IMAGE:-"vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"}
CONTAINER=${CONTAINER:-fr13-forked-fa2-tree}
PORT=${PORT:-9950}
GPU_UTIL=${GPU_UTIL:-0.88}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-TREE_ATTN}
FR10_DECODE_MODE_DEFAULT=${FR10_DECODE_MODE_DEFAULT:-tree_mtp}
FR10_METRICS=${FR10_METRICS:-0}
FR13_FA2_TREE_BIAS=${FR13_FA2_TREE_BIAS:-1}
FR13_FA2_PREFILL_NATIVE=${FR13_FA2_PREFILL_NATIVE:-1}
FR13_TREE_ATTN_EXP2_SOFTMAX=${FR13_TREE_ATTN_EXP2_SOFTMAX:-1}
LOG_DIR=${LOG_DIR:-"${FR13_RUN_DIR:-$REPO/output/fr13_fa2_tree_e2e/live}/logs"}
FORKED_FA2_SO=${FORKED_FA2_SO:-"$REPO/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so"}
TREE=${TREE:-"[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), (0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0, 1)]"}
NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-$(TREE="$TREE" python3 - <<'PY'
import ast
import os
print(len(ast.literal_eval(os.environ["TREE"])))
PY
)}
SPEC_CONFIG=${SPEC_CONFIG:-"{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS,\"speculative_token_tree\":\"$TREE\"}"}

if [[ ! -f "$FORKED_FA2_SO" ]]; then
  echo "forked FA2 .so not found: $FORKED_FA2_SO" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
LOG_DIR=$(realpath "$LOG_DIR")
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

set -a
if [[ -f "$REPO/.lumo.local.env" ]]; then
  source "$REPO/.lumo.local.env"
fi
set +a

PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
from lumo_flywheel_serving.model_server import recover_host_memory

recover_host_memory()
PY

free -h
python3 - <<'PY'
from pathlib import Path

fields = {}
for line in Path("/proc/meminfo").read_text().splitlines():
    key, value = line.split(":", 1)
    fields[key] = int(value.strip().split()[0])

available_gib = fields.get("MemAvailable", 0) / 1024 / 1024
swap_used_kib = fields.get("SwapTotal", 0) - fields.get("SwapFree", 0)
if available_gib < 80 or swap_used_kib != 0:
    raise SystemExit(
        "FR13 launch aborted: host memory recovery did not produce "
        f"MemAvailable>=80GiB and swap_used==0; "
        f"MemAvailable={available_gib:.2f}GiB "
        f"swap_used={swap_used_kib / 1024 / 1024:.2f}GiB"
    )
PY

docker run -d --name "$CONTAINER" --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 -p "$PORT:9950" \
  -v "$REPO:/workspace" -v /models:/models -v "$LOG_DIR:/logs" \
  -v "$FORKED_FA2_SO:/tmp/fr13_fork_fa2.so:ro" \
  -e VLLM_BATCH_INVARIANT=0 \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e PYTHONPATH=/workspace/src \
  -e FR10_ENABLE_TREE_GDN=1 \
  -e FR10_METRICS="$FR10_METRICS" \
  -e FR10_DECODE_MODE_DEFAULT="$FR10_DECODE_MODE_DEFAULT" \
  -e FR11_TREE_CONV_NATIVE_BF16_TAPS=1 \
  -e FR12_TREE_CONV_NATIVE_BF16_TAPS=1 \
  -e FR12_TREE_CONV_NATIVE_PRIOR_READ="${FR12_TREE_CONV_NATIVE_PRIOR_READ:-0}" \
  -e FR12_TREE_CONV_NATIVE_SPINE="${FR12_TREE_CONV_NATIVE_SPINE:-0}" \
  -e FR12_TREE_SCAN_NATIVE_SPINE="${FR12_TREE_SCAN_NATIVE_SPINE:-0}" \
  -e FR12_NATIVE_SPINE_ORACLE="${FR12_NATIVE_SPINE_ORACLE:-0}" \
  -e FR12_TREE_CONV_STATE_FULL_CAPTURE="${FR12_TREE_CONV_STATE_FULL_CAPTURE:-0}" \
  -e FR13_FA2_TREE_BIAS="$FR13_FA2_TREE_BIAS" \
  -e FR13_FA2_PREFILL_NATIVE="$FR13_FA2_PREFILL_NATIVE" \
  -e FR13_TREE_ATTN_EXP2_SOFTMAX="$FR13_TREE_ATTN_EXP2_SOFTMAX" \
  -e FR10_TREE_GDN_COUNTER_DUMP=/logs/fr10_tree_gdn_counters.json \
  -e FR10_TREE_GDN_CAPTURE_PAYLOAD="${FR10_TREE_GDN_CAPTURE_PAYLOAD:-}" \
  -e FR10_TREE_GDN_CAPTURE_PAYLOAD_LAYER_PREFIX="${FR10_TREE_GDN_CAPTURE_PAYLOAD_LAYER_PREFIX:-}" \
  -e FR10_TREE_GDN_CAPTURE_PAYLOAD_NUM_TOKENS="${FR10_TREE_GDN_CAPTURE_PAYLOAD_NUM_TOKENS:-}" \
  -e FR10_TREE_GDN_COMMIT_HANDOFF_LOG="${FR10_TREE_GDN_COMMIT_HANDOFF_LOG:-}" \
  -e FR10_TREE_GDN_COMMIT_HANDOFF_LAYER_PREFIX="${FR10_TREE_GDN_COMMIT_HANDOFF_LAYER_PREFIX:-}" \
  -e FR10_TREE_GDN_COMMIT_HANDOFF_LIMIT="${FR10_TREE_GDN_COMMIT_HANDOFF_LIMIT:-32}" \
  -e FR10_TREE_GDN_SRC_NATIVE_PAYLOAD="${FR10_TREE_GDN_SRC_NATIVE_PAYLOAD:-}" \
  -e FR10_TREE_DEPTH_POSITION_LOG=/logs/fr10_tree_depth_positions.jsonl \
  -e FR10_ROOT_HIDDEN_CAPTURE="${FR10_ROOT_HIDDEN_CAPTURE:-}" \
  -e FR10_ROOT_HIDDEN_CAPTURE_NUM_TOKENS="${FR10_ROOT_HIDDEN_CAPTURE_NUM_TOKENS:-}" \
  -e FR10_ROOT_HIDDEN_CAPTURE_ROOT_ROW="${FR10_ROOT_HIDDEN_CAPTURE_ROOT_ROW:-0}" \
  -e FR10_ROOT_HIDDEN_CAPTURE_POSITION="${FR10_ROOT_HIDDEN_CAPTURE_POSITION:-}" \
  -e FR10_ROOT_LOGIT_CAPTURE_NUM_TOKENS="${FR10_ROOT_LOGIT_CAPTURE_NUM_TOKENS:-}" \
  -e FR10_ROOT_LOGIT_CAPTURE_ROOT_ROW="${FR10_ROOT_LOGIT_CAPTURE_ROOT_ROW:-}" \
  -e FR10_LAYER_HIDDEN_CAPTURE="${FR10_LAYER_HIDDEN_CAPTURE:-}" \
  -e FR10_LAYER_HIDDEN_CAPTURE_NUM_TOKENS="${FR10_LAYER_HIDDEN_CAPTURE_NUM_TOKENS:-}" \
  -e FR10_LAYER_HIDDEN_CAPTURE_ROWS="${FR10_LAYER_HIDDEN_CAPTURE_ROWS:-}" \
  -e FR10_LAYER_HIDDEN_CAPTURE_SKIP="${FR10_LAYER_HIDDEN_CAPTURE_SKIP:-0}" \
  -e FR10_LAYER_HIDDEN_CAPTURE_LIMIT="${FR10_LAYER_HIDDEN_CAPTURE_LIMIT:-1}" \
  -e FR12_FULL_ATTN_CAPTURE="${FR12_FULL_ATTN_CAPTURE:-}" \
  -e FR12_FULL_ATTN_CAPTURE_LAYER_PREFIX="${FR12_FULL_ATTN_CAPTURE_LAYER_PREFIX:-}" \
  -e FR12_FULL_ATTN_CAPTURE_NUM_TOKENS="${FR12_FULL_ATTN_CAPTURE_NUM_TOKENS:-}" \
  -e FR12_FULL_ATTN_CAPTURE_SKIP="${FR12_FULL_ATTN_CAPTURE_SKIP:-0}" \
  -e FR12_FULL_ATTN_CAPTURE_LIMIT="${FR12_FULL_ATTN_CAPTURE_LIMIT:-1}" \
  -e FR12_SUBKERNEL_CAPTURE="${FR12_SUBKERNEL_CAPTURE:-}" \
  -e FR12_SUBKERNEL_CAPTURE_DEBUG_LOG="${FR12_SUBKERNEL_CAPTURE_DEBUG_LOG:-}" \
  -e FR12_SUBKERNEL_CAPTURE_LAYER_PREFIX="${FR12_SUBKERNEL_CAPTURE_LAYER_PREFIX:-language_model.model.layers.0.linear_attn}" \
  -e FR12_SUBKERNEL_CAPTURE_NUM_TOKENS="${FR12_SUBKERNEL_CAPTURE_NUM_TOKENS:-}" \
  -e FR12_SUBKERNEL_CAPTURE_SKIP="${FR12_SUBKERNEL_CAPTURE_SKIP:-0}" \
  -e FR12_SUBKERNEL_CAPTURE_LIMIT="${FR12_SUBKERNEL_CAPTURE_LIMIT:-1}" \
  -e FR12_SUBKERNEL_CAPTURE_Z="${FR12_SUBKERNEL_CAPTURE_Z:-0}" \
  -e FR12_SUBKERNEL_CAPTURE_INPUT="${FR12_SUBKERNEL_CAPTURE_INPUT:-0}" \
  -e FR13_TREE_ATTN_OP_CAPTURE="${FR13_TREE_ATTN_OP_CAPTURE:-}" \
  -e FR13_TREE_ATTN_OP_CAPTURE_LAYER="${FR13_TREE_ATTN_OP_CAPTURE_LAYER:-language_model.model.layers.3.self_attn}" \
  -e FR13_TREE_ATTN_OP_CAPTURE_SKIP="${FR13_TREE_ATTN_OP_CAPTURE_SKIP:-0}" \
  -e FR13_TREE_ATTN_OP_CAPTURE_LIMIT="${FR13_TREE_ATTN_OP_CAPTURE_LIMIT:-1}" \
  -e FR13_FLASH_ATTN_OP_CAPTURE="${FR13_FLASH_ATTN_OP_CAPTURE:-}" \
  -e FR13_FLASH_ATTN_OP_CAPTURE_LAYER="${FR13_FLASH_ATTN_OP_CAPTURE_LAYER:-language_model.model.layers.3.self_attn}" \
  -e FR13_FLASH_ATTN_OP_CAPTURE_SKIP="${FR13_FLASH_ATTN_OP_CAPTURE_SKIP:-0}" \
  -e FR13_FLASH_ATTN_OP_CAPTURE_LIMIT="${FR13_FLASH_ATTN_OP_CAPTURE_LIMIT:-1}" \
  -e FR13_PREPROCESS_INPUT_CAPTURE="${FR13_PREPROCESS_INPUT_CAPTURE:-}" \
  -e FR13_PREPROCESS_INPUT_CAPTURE_NUM_TOKENS="${FR13_PREPROCESS_INPUT_CAPTURE_NUM_TOKENS:-}" \
  -e FR13_PREPROCESS_INPUT_CAPTURE_SKIP="${FR13_PREPROCESS_INPUT_CAPTURE_SKIP:-0}" \
  -e FR13_PREPROCESS_INPUT_CAPTURE_LIMIT="${FR13_PREPROCESS_INPUT_CAPTURE_LIMIT:-1}" \
  -e FR13_PREFILL_GDN_CAPTURE="${FR13_PREFILL_GDN_CAPTURE:-}" \
  -e FR13_PREFILL_GDN_CAPTURE_LAYER_PREFIX="${FR13_PREFILL_GDN_CAPTURE_LAYER_PREFIX:-}" \
  -e FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX="${FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX:-1}" \
  -e FR10_SPINE_LOGIT_CAPTURE="${FR10_SPINE_LOGIT_CAPTURE:-}" \
  -e FR10_SPINE_LOGIT_CAPTURE_SKIP="${FR10_SPINE_LOGIT_CAPTURE_SKIP:-0}" \
  -e FR10_SPINE_LOGIT_CAPTURE_LIMIT="${FR10_SPINE_LOGIT_CAPTURE_LIMIT:-1}" \
  -e FR13_FINAL_LOGIT_CAPTURE="${FR13_FINAL_LOGIT_CAPTURE:-}" \
  -e FR13_FINAL_LOGIT_CAPTURE_NUM_TOKENS="${FR13_FINAL_LOGIT_CAPTURE_NUM_TOKENS:-}" \
  -e FR13_FINAL_LOGIT_CAPTURE_ROWS="${FR13_FINAL_LOGIT_CAPTURE_ROWS:-}" \
  -e FR13_FINAL_LOGIT_CAPTURE_SKIP="${FR13_FINAL_LOGIT_CAPTURE_SKIP:-0}" \
  -e FR13_FINAL_LOGIT_CAPTURE_LIMIT="${FR13_FINAL_LOGIT_CAPTURE_LIMIT:-1}" \
  -e LUMO_MTP_DRAFT_TRACE_FILE=/logs/fr10_mtp_draft_trace.jsonl \
  -e LUMO_TREE_SAMPLER_DEBUG_LOG=/logs/tree_sampler_debug.jsonl \
  -e LUMO_TREE_PATH_LCP_LOG=/logs/tree_path_lcp.jsonl \
  -e SPEC_CONFIG="$SPEC_CONFIG" \
  --entrypoint bash \
  "$IMAGE" \
  -lc "set -euo pipefail
unset FR10_ALLOW_LINEAR_FALLBACK
cp /tmp/fr13_fork_fa2.so /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
sha256sum /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so | tee /logs/fr13_forked_fa2.sha256
python3 /workspace/scripts/fr10_phase4_patch_vllm_tree_gdn.py
python3 /workspace/scripts/fr13_patch_fa2_tree_bias.py --skip-source
python3 - <<'PY'
from pathlib import Path

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/tree_attn.py')
text = path.read_text()
needle = 'FR13_FA2_PREFILL_NATIVE'
if needle not in text:
    raise SystemExit(f'{needle} patch missing in {path}')
PY
exec vllm serve /models/qwen3.6-27b-fp8 --served-model-name qwen3.6-27b \
  --host 0.0.0.0 --port 9950 --max-num-seqs '$MAX_NUM_SEQS' \
  --gpu-memory-utilization '$GPU_UTIL' --max-model-len '$MAX_MODEL_LEN' \
  --attention-backend '$ATTENTION_BACKEND' --gdn-prefill-backend triton \
  --chat-template /workspace/docker/chat_templates/qwen3-openai-codex.jinja \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --speculative-config \"\$SPEC_CONFIG\" \
  $(if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then printf '%s' '--enforce-eager'; fi)"

#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# FR13 NATIVE-MTP-5 vLLM launcher (the fr9 decode path).
#
# STOCK vLLM Multi-Token-Prediction, num_speculative_tokens=5. NONE of the FR13
# machinery: NO forked-fa2 kernel, NO speculative tree, NO tree_attn backend,
# NO APC / prefix-cache patches, NO in-container GDN/tree patcher. This is the
# apples half of the chain5(forked)-vs-native-MTP A/B that localizes the char-8
# tool-call regression.
#
# WHY NO PATCHER (verified 2026-07-01 against the exact stock image
# vllm/vllm-openai@sha256:3dbe092... == local tag vllm/vllm-openai:cu130-nightly):
#   * `qwen3_5_mtp` is a STOCK SpeculativeMethod (config/speculative.py Literal
#     registry contains "qwen3_next_mtp","qwen3_5_mtp","mtp"). The served model
#     /models/qwen3.6-27b-fp8 is model_type=qwen3_5 with mtp_num_hidden_layers=1
#     + an mtp.fc head, which speculative.py coerces qwen3_5 -> qwen3_5_mtp ->
#     arch Qwen3_5MTP, then normalizes method=qwen3_5_mtp -> "mtp" internally
#     (a deprecation warning, NOT an error). So the model's native MTP head loads
#     and runs on the STOCK code path with NO edits.
#   * scripts/fr10_phase4_patch_vllm_tree_gdn.py registers/renames NO spec method;
#     every one of its patch steps is an additive tree/GDN/APC/capture/replay
#     instrument, all default-OFF by env. Native MTP needs none of them, so this
#     launcher does NOT run it (that is the whole point: fr9's unpatched path).
#   * Stock attention backend for the qwen3-next full-attn layers on this GB10/CUDA
#     platform is FLASH_ATTN (platforms/cuda.py default; --attention-backend
#     FLASH_ATTN is a valid stock value). GDN prefill uses --gdn-prefill-backend
#     triton (a stock CLI arg: engine/arg_utils.py Literal["flashinfer","triton"]).
#
# Same IMAGE / CONTAINER default / PORT / host-mem-recovery / OOM-guard / docker
# run skeleton as fr13_launch_forked_fa2_tree_server.sh so this arm plugs into the
# identical harness (fr13_bigdenom_swe_serve_variant.sh).
# ============================================================================

REPO=${REPO:-/home/mark/shared/lumoFlyWheel}
IMAGE=${IMAGE:-"vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"}
CONTAINER=${CONTAINER:-fr13-native-mtp}
PORT=${PORT:-9950}
GPU_UTIL=${GPU_UTIL:-0.6}
# DURABLE OOM GUARD (see forked launcher): ALWAYS cap the container cgroup so the
# host keeps headroom -> Claude/watchdog/tmux can NEVER be the kernel OOM victim.
DOCKER_MEM_CAP=${DOCKER_MEM_CAP:-105g}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
GPU_OOM_GUARD=${GPU_OOM_GUARD:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
BATCH_INVARIANT=${BATCH_INVARIANT:-0}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
# Stock backend for the qwen3-next full-attention layers on GB10/CUDA. Let vLLM
# use the platform default (FLASH_ATTN) unless the caller overrides. This is NOT
# the forked TREE_ATTN backend.
if [[ -z "${ATTENTION_BACKEND+x}" ]]; then
  ATTENTION_BACKEND=FLASH_ATTN
fi
# Native MTP-5: stock spec method + 5 speculative tokens, NO speculative_token_tree.
NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-5}
SPEC_METHOD=${SPEC_METHOD:-qwen3_5_mtp}
SPEC_CONFIG=${SPEC_CONFIG:-"{\"method\":\"$SPEC_METHOD\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS}"}

LOG_DIR=${LOG_DIR:-"${FR13_RUN_DIR:-$REPO/output/fr13_native_mtp/live}/logs"}

mkdir -p "$LOG_DIR"
LOG_DIR=$(realpath "$LOG_DIR")
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

set -a
if [[ -f "$REPO/.lumo.local.env" ]]; then
  source "$REPO/.lumo.local.env"
fi
set +a

# ---- host-memory recovery + hard pre-boot assert (same as forked launcher) ----
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
        "FR13 native-MTP launch aborted: host memory recovery did not produce "
        f"MemAvailable>=80GiB and swap_used==0; "
        f"MemAvailable={available_gib:.2f}GiB "
        f"swap_used={swap_used_kib / 1024 / 1024:.2f}GiB"
    )
PY

# ---- docker run skeleton (identical to the forked launcher, stripped) ----
# NO -v $FORKED_FA2_SO, NO TREE_ATTN, NO APC flags, NO FR13_* tree/APC env,
# NO in-container patcher. Only the minimal stock-serve env.
docker run -d --name "$CONTAINER" --gpus all --ipc=host \
  --memory="$DOCKER_MEM_CAP" --memory-swap="$DOCKER_MEM_CAP" \
  --ulimit memlock=-1 --ulimit stack=67108864 -p "$PORT:9950" \
  -v "$REPO:/workspace" -v /models:/models -v "$LOG_DIR:/logs" \
  -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  -e VLLM_BATCH_INVARIANT="$BATCH_INVARIANT" \
  -e LUMO_BATCH_INVARIANT_VLLM="${LUMO_BATCH_INVARIANT_VLLM:-$BATCH_INVARIANT}" \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e PYTHONPATH=/workspace/src \
  -e SPEC_CONFIG="$SPEC_CONFIG" \
  --entrypoint bash \
  "$IMAGE" \
  -lc "set -euo pipefail
# NATIVE path: NO forked-fa2 .so copy, NO fr10_phase4 patcher, NO fa2-tree-bias
# patch. Stock vLLM loads the model's native MTP head from the qwen3_5_mtp method.
exec vllm serve /models/qwen3.6-27b-fp8 --served-model-name qwen3.6-27b \
  --host 0.0.0.0 --port 9950 --max-num-seqs '$MAX_NUM_SEQS' \
  --gpu-memory-utilization '$GPU_UTIL' --max-model-len '$MAX_MODEL_LEN' --seed '${SEED:-0}' \
  --attention-backend '$ATTENTION_BACKEND' --gdn-prefill-backend triton \
  --chat-template /workspace/docker/chat_templates/qwen3-openai-codex.jinja \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --speculative-config \"\$SPEC_CONFIG\" \
  $(if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then printf '%s' '--enforce-eager'; fi)"

# DURABLE OOM BACKSTOP: spawn the detached GPU/unified-mem guard for THIS container
# (identical to the forked launcher). Converts an incipient GPU OOM into a
# relaunchable+loud container kill instead of a systemd --user session kill.
if [[ "${GPU_OOM_GUARD:-1}" == "1" ]]; then
  GPU_GUARD_NAME_GLOB="$CONTAINER" setsid bash "$(dirname "$0")/gpu_oom_guard.sh" \
    >/dev/null 2>&1 </dev/null &
  disown 2>/dev/null || true
  echo "[launch] gpu_oom_guard armed for container=$CONTAINER (floor=${GPU_GUARD_FLOOR_MIB:-9000}MiB)"
fi

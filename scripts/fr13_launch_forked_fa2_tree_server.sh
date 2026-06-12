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
BATCH_INVARIANT=${BATCH_INVARIANT:-0}
FR13_FA2_TREE_BIAS=${FR13_FA2_TREE_BIAS:-1}
FR13_FA2_PREFILL_NATIVE=${FR13_FA2_PREFILL_NATIVE:-1}
# FR13 Method-A: BI allowlist for TREE_ATTN (inert by default; only relevant
# with BATCH_INVARIANT=1, and requires the two FR13_FA2_* flags above).
FR13_BI_TREE_ATTN=${FR13_BI_TREE_ATTN:-0}
FR13_TREE_ATTN_EXP2_SOFTMAX=${FR13_TREE_ATTN_EXP2_SOFTMAX:-1}
# FR13_CONV_COMMITTED_PATH (default ON): the next event's prior conv window is
# read from the COMMITTED path's accepted-leaf NODE column (pre-remap), so
# BRANCH winners ([0,2], [0,1,4]) commit a window built from committed-path
# tokens only; spine winners are byte-identical to the legacy linear read.
# =0 restores the legacy post-remap linear-column read.
FR13_CONV_COMMITTED_PATH=${FR13_CONV_COMMITTED_PATH:-1}
# FR13_FORCE_SPINE_COMMIT (default OFF) — DIAGNOSTIC ONLY, like
# FR10_ALLOW_LINEAR_FALLBACK: the greedy committer scores all paths (alts
# still verified) but always commits the spine path's own prefix. For the
# S3/m1 decisive A/B (caterpillar forced-spine vs chain boot) ONLY. NEVER
# bind =1 into a committed serving config or a gate result.
FR13_FORCE_SPINE_COMMIT=${FR13_FORCE_SPINE_COMMIT:-0}
# FR13_DRAFTER_SINGLE_LOGITS (FIX-1, default ON): the caterpillar drafter
# takes draft tokens as argmax of the single already-computed logits tensor
# instead of _greedy_sample's second compute_logits (double full-vocab bf16
# lm-head read per drafter step, FR13_B1_SPEED_ATTRIBUTION_BIND.md). =0 is
# the exact legacy double-logits path (the A/B instrument).
FR13_DRAFTER_SINGLE_LOGITS=${FR13_DRAFTER_SINGLE_LOGITS:-1}
# FR13_FIX1_SELFCHECK (default OFF) — DIAGNOSTIC ONLY, like
# FR13_FORCE_SPINE_COMMIT: with the single-logits drafter serving, ALSO run
# legacy _greedy_sample per drafter step and raise on any token mismatch
# (in-process FIX-1 OFF==ON byte-identity proof; counters dumped to
# FR13_FIX1_SELFCHECK_DUMP). NEVER bind =1 into a serving config or a speed
# number.
FR13_FIX1_SELFCHECK=${FR13_FIX1_SELFCHECK:-0}
FR13_FIX1_SELFCHECK_DUMP=${FR13_FIX1_SELFCHECK_DUMP:-/logs/fr13_fix1_selfcheck.json}
LUMO_MTP_DRAFT_TRACE_FILE=${LUMO_MTP_DRAFT_TRACE_FILE:-}
LUMO_TREE_SAMPLER_DEBUG_LOG=${LUMO_TREE_SAMPLER_DEBUG_LOG:-}
LUMO_TREE_PATH_LCP_LOG=${LUMO_TREE_PATH_LCP_LOG:-}
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

_lumo_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

LUMO_NSYS_WRAP_VLLM=${LUMO_NSYS_WRAP_VLLM:-0}
LUMO_NSYS_BIN=${LUMO_NSYS_BIN:-/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys}
LUMO_NSYS_DELAY_S=${LUMO_NSYS_DELAY_S:-600}
LUMO_NSYS_DURATION_S=${LUMO_NSYS_DURATION_S:-150}
# Periodic CUPTI buffer flush (ms). Without it, per-kernel records (incl. graph
# node-level kernels) are dropped as "incomplete" at the delayed-duration session
# stop on GB10 (fr13_b1_profile_bind: 55k/78k events dropped, zero kernel rows).
LUMO_NSYS_FLUSH_MS=${LUMO_NSYS_FLUSH_MS:-100}
# Semicolon-separated lines appended to the in-container nsys user config
# ("$nsys -z"). Default works around the GB10 drop class where ALL per-kernel
# rows are "incomplete CUPTI events dropped ... GPU timestamp information have
# not been retrieved" even with periodic flushes (NVIDIA-documented
# CuptiUseRawGpuTimestamps=false workaround; fr13_b1_profile_node: 102,320
# dropped with --cuda-flush-interval 100 and zero kernel tables).
LUMO_NSYS_CONFIG_DIRECTIVES=${LUMO_NSYS_CONFIG_DIRECTIVES:-CuptiUseRawGpuTimestamps=false}
# nsys --trace value. On GB10 + CUDA 13 the default 'cuda' engages the HARDWARE
# trace engine for kernel records; in delayed-duration sessions ALL kernel rows
# are then dropped ("GPU timestamp information have not been retrieved").
# 'cuda,cuda-sw' forces the software CUPTI kernel-record path (memcpy/memset/
# runtime rows always survived; only hw-trace kernel rows dropped).
LUMO_NSYS_TRACE=${LUMO_NSYS_TRACE:-cuda,nvtx}
LUMO_NSYS_OUTPUT=${LUMO_NSYS_OUTPUT:-/logs/nsys_vllm_${CONTAINER}}
NSYS_DOCKER_ARGS=()
if _lumo_truthy "$LUMO_NSYS_WRAP_VLLM"; then
  for nsight_mount in /opt/nvidia /usr/local/cuda-13.0; do
    if [[ ! -e "$nsight_mount" ]]; then
      echo "LUMO_NSYS_WRAP_VLLM enabled but Nsight mount path is missing: $nsight_mount" >&2
      exit 2
    fi
    NSYS_DOCKER_ARGS+=(-v "$nsight_mount:$nsight_mount:ro")
  done
fi

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
  "${NSYS_DOCKER_ARGS[@]}" \
  -e VLLM_BATCH_INVARIANT="$BATCH_INVARIANT" \
  -e LUMO_BATCH_INVARIANT_VLLM="${LUMO_BATCH_INVARIANT_VLLM:-$BATCH_INVARIANT}" \
  -e LUMO_NSYS_WRAP_VLLM="$LUMO_NSYS_WRAP_VLLM" \
  -e LUMO_NSYS_BIN="$LUMO_NSYS_BIN" \
  -e LUMO_NSYS_DELAY_S="$LUMO_NSYS_DELAY_S" \
  -e LUMO_NSYS_DURATION_S="$LUMO_NSYS_DURATION_S" \
  -e LUMO_NSYS_FLUSH_MS="$LUMO_NSYS_FLUSH_MS" \
  -e LUMO_NSYS_CONFIG_DIRECTIVES="$LUMO_NSYS_CONFIG_DIRECTIVES" \
  -e LUMO_NSYS_TRACE="$LUMO_NSYS_TRACE" \
  -e LUMO_NSYS_OUTPUT="$LUMO_NSYS_OUTPUT" \
  -e FR13_BI_TREE_ATTN="$FR13_BI_TREE_ATTN" \
  -e FR13_TORCH_DET_WARN="${FR13_TORCH_DET_WARN:-0}" \
  -e FR13_TORCH_DET_WARN_LOG="${FR13_TORCH_DET_WARN_LOG:-/logs/fr13_torch_det_warn.log}" \
  -e FR13_TREE_PER_REQ_GEN="${FR13_TREE_PER_REQ_GEN:-1}" \
  -e FR13_TREE_REQKEY="${FR13_TREE_REQKEY:-1}" \
  -e FR13_TREE_REMAP_SEQ="${FR13_TREE_REMAP_SEQ:-1}" \
  -e FR13_TREE_BONUS_SELF="${FR13_TREE_BONUS_SELF:-1}" \
  -e FR13_CONV_COMMITTED_PATH="$FR13_CONV_COMMITTED_PATH" \
  -e FR13_FORCE_SPINE_COMMIT="$FR13_FORCE_SPINE_COMMIT" \
  -e FR13_DRAFTER_SINGLE_LOGITS="$FR13_DRAFTER_SINGLE_LOGITS" \
  -e FR13_FIX1_SELFCHECK="$FR13_FIX1_SELFCHECK" \
  -e FR13_FIX1_SELFCHECK_DUMP="$FR13_FIX1_SELFCHECK_DUMP" \
  -e FR13_REPLAY_ROUTE="${FR13_REPLAY_ROUTE:-1}" \
  -e FR13_REPLAY_BOUNDARY_LOG="${FR13_REPLAY_BOUNDARY_LOG:-0}" \
  -e FR13_REPLAY_BOUNDARY_LAYERS="${FR13_REPLAY_BOUNDARY_LAYERS:-layers.0.linear_attn}" \
  -e FR13_REPLAY_BOUNDARY_PATH="${FR13_REPLAY_BOUNDARY_PATH:-/logs/fr13_replay_boundary.jsonl}" \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" \
  -e TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-0}" \
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
  -e LUMO_MTP_DRAFT_TRACE_FILE="$LUMO_MTP_DRAFT_TRACE_FILE" \
  -e LUMO_TREE_SAMPLER_DEBUG_LOG="$LUMO_TREE_SAMPLER_DEBUG_LOG" \
  -e LUMO_TREE_PATH_LCP_LOG="$LUMO_TREE_PATH_LCP_LOG" \
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
import os
from pathlib import Path

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/tree_attn.py')
text = path.read_text()
needle = 'FR13_FA2_PREFILL_NATIVE'
if needle not in text:
    raise SystemExit(f'{needle} patch missing in {path}')
if os.environ.get('FR13_BI_TREE_ATTN', '0') == '1':
    bi_path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/batch_invariant.py')
    bi_text = bi_path.read_text()
    if 'FR13_BI_TREE_ATTN' not in bi_text:
        raise SystemExit(f'FR13_BI_TREE_ATTN allowlist patch missing in {bi_path}')
    decode_needle = (
        'num_splits=1 if envs.VLLM_BATCH_INVARIANT else 0,\n'
        '                    tree_bias=tree_bias,'
    )
    if decode_needle not in text:
        raise SystemExit(f'FR13 BI decode num_splits expression missing in {path}')
PY
NSYS_PREFIX=()
case \"\${LUMO_NSYS_WRAP_VLLM,,}\" in
  1|true|yes|on)
    if [[ -n \"\${LUMO_NSYS_CONFIG_DIRECTIVES:-}\" ]]; then
      NSYS_CFG_PATH=\$(\"\$LUMO_NSYS_BIN\" -z)
      mkdir -p \"\$(dirname \"\$NSYS_CFG_PATH\")\"
      printf '%s\n' \"\$LUMO_NSYS_CONFIG_DIRECTIVES\" | tr ';' '\n' >> \"\$NSYS_CFG_PATH\"
      echo \"nsys config directives appended to \$NSYS_CFG_PATH:\"
      cat \"\$NSYS_CFG_PATH\"
    fi
    NSYS_PREFIX=(
      \"\$LUMO_NSYS_BIN\"
      profile
      --delay \"\$LUMO_NSYS_DELAY_S\"
      --duration \"\$LUMO_NSYS_DURATION_S\"
      --trace=\"\$LUMO_NSYS_TRACE\"
      --cuda-graph-trace=node
      --cuda-flush-interval \"\$LUMO_NSYS_FLUSH_MS\"
      --sample=none
      --cpuctxsw=none
      --force-overwrite=true
      -o \"\$LUMO_NSYS_OUTPUT\"
    )
    ;;
esac
exec \"\${NSYS_PREFIX[@]}\" vllm serve /models/qwen3.6-27b-fp8 --served-model-name qwen3.6-27b \
  --host 0.0.0.0 --port 9950 --max-num-seqs '$MAX_NUM_SEQS' \
  --gpu-memory-utilization '$GPU_UTIL' --max-model-len '$MAX_MODEL_LEN' \
  --attention-backend '$ATTENTION_BACKEND' --gdn-prefill-backend triton \
  --chat-template /workspace/docker/chat_templates/qwen3-openai-codex.jinja \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --speculative-config \"\$SPEC_CONFIG\" \
  $(if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then printf '%s' '--enforce-eager'; fi)"

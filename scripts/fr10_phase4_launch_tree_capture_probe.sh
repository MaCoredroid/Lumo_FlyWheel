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
python3 - <<'PY'
from pathlib import Path

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py')
text = path.read_text()
if 'FR10_CAPTURE_GDN_TENSOR_DIR' not in text:
    text = text.replace('import torch\\n', 'import os\\nimport torch\\n', 1)
    helper = r'''

def _fr10_capture_spec_gdn(prefix, payload):
    capture_dir = os.environ.get("FR10_CAPTURE_GDN_TENSOR_DIR")
    if not capture_dir:
        return
    os.makedirs(capture_dir, exist_ok=True)
    safe_prefix = prefix.replace(".", "_").replace("/", "_")
    path = os.path.join(capture_dir, f"{safe_prefix}_spec_gdn.pt")
    if os.path.exists(path):
        return
    torch.save(payload, path)
    logger.warning("FR10 captured spec GDN tensor payload at %s", path)
'''
    text = text.replace('logger = init_logger(__name__)\\n', 'logger = init_logger(__name__)\\n' + helper, 1)
    needle = '''            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_spec_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
'''
    repl = '''            _fr10_initial_state_before_spec = ssm_state.detach().clone()
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_spec_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
            _fr10_capture_spec_gdn(self.prefix, {
                "prefix": self.prefix,
                "num_actual_tokens": int(num_actual_tokens),
                "num_spec_decodes": int(attn_metadata.num_spec_decodes),
                "mixed_qkv_spec": mixed_qkv_spec.detach().cpu(),
                "a": a.detach().cpu(),
                "b": b.detach().cpu(),
                "query_spec": query_spec.detach().cpu(),
                "key_spec": key_spec.detach().cpu(),
                "value_spec": value_spec.detach().cpu(),
                "A_log": self.A_log.detach().cpu(),
                "dt_bias": self.dt_bias.detach().cpu(),
                "spec_query_start_loc": spec_query_start_loc.detach().cpu(),
                "spec_state_indices_tensor": spec_state_indices_tensor.detach().cpu(),
                "num_accepted_tokens": num_accepted_tokens.detach().cpu(),
                "initial_state_before_spec": _fr10_initial_state_before_spec.detach().cpu(),
                "core_attn_out_spec_native": core_attn_out_spec.detach().cpu(),
                "last_recurrent_state_native": last_recurrent_state.detach().cpu(),
            })
'''
    if needle not in text:
        raise SystemExit('FR10 capture patch needle not found in gdn_linear_attn.py')
    text = text.replace(needle, repl, 1)
    path.write_text(text)
    print('[FR10] patched gdn_linear_attn.py for spec GDN tensor capture')
else:
    print('[FR10] gdn_linear_attn.py already patched')
PY
exec vllm serve /models/qwen3.6-27b-fp8 --served-model-name qwen3.6-27b \
  --host 0.0.0.0 --port 8000 --max-num-seqs 4 \
  --gpu-memory-utilization '$GPU_UTIL' --max-model-len 131072 \
  --attention-backend FLASH_ATTN --gdn-prefill-backend triton --enforce-eager \
  --speculative-config '{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":10,\"speculative_token_tree\":\"$TREE\"}'" \
  > "$OUT_DIR/container_id.txt"

cat "$OUT_DIR/container_id.txt"

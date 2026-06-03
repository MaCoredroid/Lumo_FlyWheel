#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


GDN_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/"
    "gdn_linear_attn.py"
)


HELPER = r'''

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


NEEDLE = '''            core_attn_out_spec, last_recurrent_state = (
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


REPLACEMENT = '''            _fr10_initial_state_before_spec = ssm_state.detach().clone()
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


def main() -> int:
    text = GDN_PATH.read_text()
    if "FR10_CAPTURE_GDN_TENSOR_DIR" in text:
        print("[FR10] gdn_linear_attn.py already patched")
        return 0
    text = text.replace("import torch\n", "import os\nimport torch\n", 1)
    text = text.replace("logger = init_logger(__name__)\n", "logger = init_logger(__name__)\n" + HELPER, 1)
    if NEEDLE not in text:
        raise RuntimeError("FR10 capture patch needle not found in gdn_linear_attn.py")
    text = text.replace(NEEDLE, REPLACEMENT, 1)
    GDN_PATH.write_text(text)
    print("[FR10] patched gdn_linear_attn.py for spec GDN tensor capture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

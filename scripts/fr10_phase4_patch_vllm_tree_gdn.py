#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path


GDN_ATTN_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/gdn_attn.py"
)
GDN_LINEAR_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/"
    "gdn_linear_attn.py"
)


def _patch_gdn_attn() -> bool:
    text = GDN_ATTN_PATH.read_text()
    if "fr10_tree_parent" in text:
        return False

    text = text.replace(
        "from dataclasses import dataclass\n",
        "from dataclasses import dataclass\nimport ast\nimport json\nimport os\nimport signal\nfrom pathlib import Path\n",
        1,
    )
    text = text.replace(
        "from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec\n",
        (
            "from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec\n"
            "\n"
            "_FR10_TREE_COUNTERS = []\n"
            "_FR10_SIGNAL_INSTALLED = False\n"
            "\n"
            "\n"
            "def _fr10_dump_tree_counters(signum=None, frame=None):\n"
            "    dump_path = os.environ.get(\n"
            "        \"FR10_TREE_GDN_COUNTER_DUMP\",\n"
            "        \"/workspace/output/fr10_phase4_tree_gdn_server/fr10_tree_gdn_counters.json\",\n"
            "    )\n"
            "    rows = []\n"
            "    for row in _FR10_TREE_COUNTERS:\n"
            "        counter = row.get(\"counter\")\n"
            "        try:\n"
            "            count = int(counter.detach().cpu().item()) if counter is not None else None\n"
            "        except Exception as exc:\n"
            "            count = f\"ERROR: {exc}\"\n"
            "        rows.append({\n"
            "            \"shape\": row.get(\"shape\"),\n"
            "            \"parent\": row.get(\"parent\"),\n"
            "            \"has_sibling\": row.get(\"has_sibling\"),\n"
            "            \"count\": count,\n"
            "        })\n"
            "    path = Path(dump_path)\n"
            "    path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    path.write_text(json.dumps(rows, indent=2, sort_keys=True))\n"
            "\n"
            "\n"
            "def _fr10_register_tree_counter(shape, parent, counter):\n"
            "    global _FR10_SIGNAL_INSTALLED\n"
            "    has_sibling = any(parent.count(p) > 1 for p in set(parent) if p >= 0)\n"
            "    _FR10_TREE_COUNTERS.append({\n"
            "        \"shape\": shape,\n"
            "        \"parent\": list(parent),\n"
            "        \"has_sibling\": has_sibling,\n"
            "        \"counter\": counter,\n"
            "    })\n"
            "    if not _FR10_SIGNAL_INSTALLED:\n"
            "        signal.signal(signal.SIGUSR1, _fr10_dump_tree_counters)\n"
            "        _FR10_SIGNAL_INSTALLED = True\n"
        ),
        1,
    )
    text = text.replace(
        "    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]\n",
        (
            "    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]\n"
            "\n"
            "    # FR10: static tree descriptors for GDN speculative verification.\n"
            "    fr10_tree_parent: torch.Tensor | None = None\n"
            "    fr10_tree_strict_mask: torch.Tensor | None = None\n"
            "    fr10_tree_visible_mask: torch.Tensor | None = None\n"
            "    fr10_tree_invocation_counter: torch.Tensor | None = None\n"
        ),
        1,
    )
    text = text.replace(
        "        self.num_accepted_tokens: torch.Tensor = torch.empty(\n"
        "            (self.decode_cudagraph_max_bs,),\n"
        "            dtype=torch.int32,\n"
        "            device=device,\n"
        "        )\n",
        (
            "        self.num_accepted_tokens: torch.Tensor = torch.empty(\n"
            "            (self.decode_cudagraph_max_bs,),\n"
            "            dtype=torch.int32,\n"
            "            device=device,\n"
            "        )\n"
            "\n"
            "        self.fr10_tree_parent = None\n"
            "        self.fr10_tree_strict_mask = None\n"
            "        self.fr10_tree_visible_mask = None\n"
            "        self.fr10_tree_invocation_counter = None\n"
            "        spec_token_tree = None\n"
            "        if self.speculative_config is not None:\n"
            "            spec_token_tree = self.speculative_config.speculative_token_tree\n"
            "        if spec_token_tree is not None:\n"
            "            tree_choices = ast.literal_eval(spec_token_tree)\n"
            "            index = {choice: i + 1 for i, choice in enumerate(tree_choices)}\n"
            "            parent = [-1]\n"
            "            for choice in tree_choices:\n"
            "                parent.append(0 if len(choice) == 1 else index[choice[:-1]])\n"
            "            n = len(parent)\n"
            "            n_pad = 1 << (n - 1).bit_length()\n"
            "            if n_pad > 16:\n"
            "                raise NotImplementedError(\n"
            "                    f\"FR10 GDN tree verifier only warms padded tree sizes <=16, got {n}\"\n"
            "                )\n"
            "            strict = torch.zeros((n_pad, n_pad), dtype=torch.int32, device=device)\n"
            "            visible = torch.zeros((n_pad, n_pad), dtype=torch.int32, device=device)\n"
            "            for node in range(n):\n"
            "                visible[node, node] = 1\n"
            "                cur = parent[node]\n"
            "                while cur >= 0:\n"
            "                    strict[node, cur] = 1\n"
            "                    visible[node, cur] = 1\n"
            "                    cur = parent[cur]\n"
            "            self.fr10_tree_parent = torch.tensor(parent, dtype=torch.int32, device=device)\n"
            "            self.fr10_tree_strict_mask = strict\n"
            "            self.fr10_tree_visible_mask = visible\n"
            "            self.fr10_tree_invocation_counter = torch.zeros((1,), dtype=torch.int64, device=device)\n"
            "            _fr10_register_tree_counter(\n"
            "                shape=f\"n{n}_pad{n_pad}\",\n"
            "                parent=parent,\n"
            "                counter=self.fr10_tree_invocation_counter,\n"
            "            )\n"
        ),
        1,
    )
    text = text.replace(
        "            num_accepted_tokens=num_accepted_tokens,\n"
        "            nums_dict=nums_dict,\n",
        (
            "            num_accepted_tokens=num_accepted_tokens,\n"
            "            fr10_tree_parent=self.fr10_tree_parent,\n"
            "            fr10_tree_strict_mask=self.fr10_tree_strict_mask,\n"
            "            fr10_tree_visible_mask=self.fr10_tree_visible_mask,\n"
            "            fr10_tree_invocation_counter=self.fr10_tree_invocation_counter,\n"
            "            nums_dict=nums_dict,\n"
        ),
        1,
    )
    GDN_ATTN_PATH.write_text(text)
    return True


def _patch_gdn_linear() -> bool:
    text = GDN_LINEAR_PATH.read_text()
    if "FR10_ENABLE_TREE_GDN" in text:
        return False

    text = text.replace("import torch\n", "import os\nimport torch\n", 1)
    text = text.replace(
        "from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata\n",
        (
            "from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata\n"
            "from lumo_flywheel_serving.fr10_gdn_tree_kernel import launch_tree_gdn_prepared\n"
        ),
        1,
    )

    needle = '''        if spec_sequence_masks is not None:
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
'''
    replacement = '''        if spec_sequence_masks is not None:
            use_fr10_tree = (
                os.environ.get("FR10_ENABLE_TREE_GDN") == "1"
                and getattr(attn_metadata, "fr10_tree_parent", None) is not None
                and attn_metadata.num_prefills == 0
                and attn_metadata.num_decodes == 0
            )
            if use_fr10_tree:
                assert spec_query_start_loc is not None
                assert spec_state_indices_tensor is not None
                assert attn_metadata.fr10_tree_parent is not None
                assert attn_metadata.fr10_tree_strict_mask is not None
                assert attn_metadata.fr10_tree_visible_mask is not None
                assert attn_metadata.fr10_tree_invocation_counter is not None
                logger.warning_once(
                    "FR10 tree GDN verifier branch active for layer %s", self.prefix
                )
                _, _, value_tree, g_tree, beta_tree = fused_post_conv_prep(
                    conv_output=mixed_qkv_spec,
                    a=a,
                    b=b,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    num_k_heads=self.num_k_heads // self.tp_size,
                    head_k_dim=self.head_k_dim,
                    head_v_dim=self.head_v_dim,
                    apply_l2norm=True,
                    output_g_exp=False,
                )
                tree_n = int(attn_metadata.fr10_tree_parent.numel())
                tree_n_pad = int(attn_metadata.fr10_tree_visible_mask.size(0))
                core_attn_out_spec = torch.empty(
                    (1, query_spec.size(1), value_tree.size(1), value_tree.size(2)),
                    dtype=query_spec.dtype,
                    device=query_spec.device,
                )
                tree_state = torch.empty(
                    (
                        tree_n_pad,
                        value_tree.size(1),
                        value_tree.size(2),
                        query_spec.size(3),
                    ),
                    dtype=torch.float32,
                    device=query_spec.device,
                )
                for fr10_b in range(attn_metadata.num_spec_decodes):
                    # Full CUDA graph capture cannot tolerate GPU->CPU syncs.
                    # In pure tree-spec decode vLLM lays each spec decode out as
                    # one fixed tree block, so offsets are static from tree_n.
                    start = fr10_b * tree_n
                    end = start + tree_n
                    tree_out, _ = launch_tree_gdn_prepared(
                        q=query_spec[0, start:end].contiguous(),
                        k=key_spec[0, start:end].contiguous(),
                        v=value_tree[start:end].contiguous(),
                        g=g_tree[start:end].contiguous(),
                        beta=beta_tree[start:end].contiguous(),
                        h0=ssm_state,
                        h0_indices=spec_state_indices_tensor,
                        h0_is_bank=True,
                        h0_index_row=fr10_b * spec_state_indices_tensor.size(-1),
                        n_actual=tree_n,
                        n_pad=tree_n_pad,
                        strict_mask=attn_metadata.fr10_tree_strict_mask,
                        visible_mask=attn_metadata.fr10_tree_visible_mask,
                        out=core_attn_out_spec[0, start:end],
                        state=tree_state,
                        output_scale=self.head_k_dim**-0.5,
                        use_qk_l2norm_in_kernel=True,
                        invocation_counter=attn_metadata.fr10_tree_invocation_counter,
                    )
                    core_attn_out_spec[0, start:end] = tree_out[:tree_n]
                _, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[
                        : attn_metadata.num_spec_decodes + 1
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
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
'''
    if needle not in text:
        raise RuntimeError("FR10 GDN linear spec branch needle not found")
    text = text.replace(needle, replacement, 1)
    GDN_LINEAR_PATH.write_text(text)
    return True


def main() -> int:
    patched = {
        str(GDN_ATTN_PATH): _patch_gdn_attn(),
        str(GDN_LINEAR_PATH): _patch_gdn_linear(),
    }
    print(patched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
SCHEDULER_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
)
REJECTION_SAMPLER_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py"
)
GPU_MODEL_RUNNER_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"
)
MAMBA_UTILS_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/mamba_utils.py"
)
REQUEST_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/request.py"
)
SCHED_OUTPUT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/output.py"
)
EAGLE_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py"
)
TREE_ATTN_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/tree_attn.py"
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
            "def _fr10_metrics_enabled():\n"
            "    return os.environ.get(\"FR10_METRICS\", \"0\") == \"1\"\n"
            "\n"
            "\n"
            "def _fr10_dump_tree_counters(signum=None, frame=None):\n"
            "    if not _fr10_metrics_enabled():\n"
            "        return\n"
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
            "        conv_diag = row.get(\"conv_diag\")\n"
            "        try:\n"
            "            conv_diag_values = (\n"
            "                [float(x) for x in conv_diag.detach().cpu().tolist()]\n"
            "                if conv_diag is not None\n"
            "                else None\n"
            "            )\n"
            "        except Exception as exc:\n"
            "            conv_diag_values = f\"ERROR: {exc}\"\n"
            "        rows.append({\n"
            "            \"shape\": row.get(\"shape\"),\n"
            "            \"parent\": row.get(\"parent\"),\n"
            "            \"path0_nodes\": row.get(\"path0_nodes\"),\n"
            "            \"has_sibling\": row.get(\"has_sibling\"),\n"
            "            \"count\": count,\n"
            "            \"conv_diag\": conv_diag_values,\n"
            "        })\n"
            "    path = Path(dump_path)\n"
            "    path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    path.write_text(json.dumps(rows, indent=2, sort_keys=True))\n"
            "\n"
            "\n"
            "def _fr10_register_tree_counter(\n"
            "    shape, parent, counter, conv_diag=None, path0_nodes=None\n"
            "):\n"
            "    if not _fr10_metrics_enabled():\n"
            "        return\n"
            "    global _FR10_SIGNAL_INSTALLED\n"
            "    has_sibling = any(parent.count(p) > 1 for p in set(parent) if p >= 0)\n"
            "    _FR10_TREE_COUNTERS.append({\n"
            "        \"shape\": shape,\n"
            "        \"parent\": list(parent),\n"
            "        \"path0_nodes\": list(path0_nodes) if path0_nodes is not None else None,\n"
            "        \"has_sibling\": has_sibling,\n"
            "        \"counter\": counter,\n"
            "        \"conv_diag\": conv_diag,\n"
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
            "    fr10_tree_conv_diag: torch.Tensor | None = None\n"
            "    fr10_tree_conv_source_indices: dict[int, torch.Tensor] | None = None\n"
            "    fr10_flat_conv_source_indices: dict[int, torch.Tensor] | None = None\n"
            "    fr10_path0_conv_source_indices: dict[int, torch.Tensor] | None = None\n"
            "    fr10_tree_path0_nodes: torch.Tensor | None = None\n"
            "    fr10_tree_path_node_tensors: list[torch.Tensor] | None = None\n"
            "    fr10_tree_has_sibling: bool = False\n"
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
            "        self.fr10_tree_conv_diag = None\n"
            "        self.fr10_tree_conv_source_indices = None\n"
            "        self.fr10_flat_conv_source_indices = None\n"
            "        self.fr10_path0_conv_source_indices = None\n"
            "        self.fr10_tree_path0_nodes = None\n"
            "        self.fr10_tree_path_node_tensors = None\n"
            "        self.fr10_tree_has_sibling = False\n"
            "        spec_token_tree = None\n"
            "        try:\n"
            "            spec_env = os.environ.get(\"SPEC_CONFIG\")\n"
            "            if spec_env:\n"
            "                spec_token_tree = json.loads(spec_env).get(\"speculative_token_tree\")\n"
            "        except Exception:\n"
            "            spec_token_tree = None\n"
            "        if spec_token_tree is None and self.speculative_config is not None:\n"
            "            spec_token_tree = self.speculative_config.speculative_token_tree\n"
            "        if spec_token_tree is not None:\n"
            "            tree_choices = sorted(ast.literal_eval(spec_token_tree), key=lambda _p: (len(_p), _p))\n"
            "            index = {choice: i + 1 for i, choice in enumerate(tree_choices)}\n"
            "            parent = [-1]\n"
            "            for choice in tree_choices:\n"
            "                parent.append(0 if len(choice) == 1 else index[choice[:-1]])\n"
            "            path0_nodes = [0] + [\n"
            "                index[choice]\n"
            "                for choice in tree_choices\n"
            "                if all(int(part) == 0 for part in choice)\n"
            "            ]\n"
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
            "            source_by_width = {}\n"
            "            flat_source_by_width = {}\n"
            "            path0_source_by_width = {}\n"
            "            path_node_tensors = []\n"
            "            for node in range(n):\n"
            "                path = []\n"
            "                cur = node\n"
            "                while cur >= 0:\n"
            "                    path.append(cur)\n"
            "                    cur = parent[cur]\n"
            "                path.reverse()\n"
            "                path_node_tensors.append(torch.tensor(path, dtype=torch.long, device=device))\n"
            "            for width in range(2, 7):\n"
            "                source_rows = []\n"
            "                for node in range(n):\n"
            "                    ancestry = []\n"
            "                    cur = parent[node]\n"
            "                    while cur >= 0:\n"
            "                        ancestry.append(cur)\n"
            "                        cur = parent[cur]\n"
            "                    ancestry.reverse()\n"
            "                    path = ancestry + [node]\n"
            "                    source = list(range(width - 1)) + [\n"
            "                        width - 1 + int(path_node) for path_node in path\n"
            "                    ]\n"
            "                    source_rows.append(source[-width:])\n"
            "                source_by_width[width] = torch.tensor(\n"
            "                    source_rows, dtype=torch.long, device=device\n"
            "                )\n"
            "                flat_rows = []\n"
            "                for node in range(n):\n"
            "                    source = list(range(width - 1)) + [\n"
            "                        width - 1 + int(path_node)\n"
            "                        for path_node in range(node + 1)\n"
            "                    ]\n"
            "                    flat_rows.append(source[-width:])\n"
            "                flat_source_by_width[width] = torch.tensor(\n"
            "                    flat_rows, dtype=torch.long, device=device\n"
            "                )\n"
            "                path0_rows = []\n"
            "                for node in range(len(path0_nodes)):\n"
            "                    source = list(range(width - 1)) + [\n"
            "                        width - 1 + int(path_node)\n"
            "                        for path_node in range(node + 1)\n"
            "                    ]\n"
            "                    path0_rows.append(source[-width:])\n"
            "                path0_source_by_width[width] = torch.tensor(\n"
            "                    path0_rows, dtype=torch.long, device=device\n"
            "                )\n"
            "            self.fr10_tree_conv_source_indices = source_by_width\n"
            "            self.fr10_flat_conv_source_indices = flat_source_by_width\n"
            "            self.fr10_path0_conv_source_indices = path0_source_by_width\n"
            "            self.fr10_tree_path0_nodes = torch.tensor(path0_nodes, dtype=torch.long, device=device)\n"
            "            self.fr10_tree_path_node_tensors = path_node_tensors\n"
            "            self.fr10_tree_has_sibling = any(parent.count(p) > 1 for p in set(parent) if p >= 0)\n"
            "            if _fr10_metrics_enabled():\n"
            "                self.fr10_tree_invocation_counter = torch.zeros((1,), dtype=torch.int32, device=device)\n"
            "                self.fr10_tree_conv_diag = torch.zeros((32,), dtype=torch.float32, device=device)\n"
            "                _fr10_register_tree_counter(\n"
            "                    shape=f\"n{n}_pad{n_pad}\",\n"
            "                    parent=parent,\n"
            "                    counter=self.fr10_tree_invocation_counter,\n"
            "                    conv_diag=self.fr10_tree_conv_diag,\n"
            "                    path0_nodes=path0_nodes,\n"
            "                )\n"
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
            "            fr10_tree_conv_diag=self.fr10_tree_conv_diag,\n"
            "            fr10_tree_conv_source_indices=self.fr10_tree_conv_source_indices,\n"
            "            fr10_flat_conv_source_indices=self.fr10_flat_conv_source_indices,\n"
            "            fr10_path0_conv_source_indices=self.fr10_path0_conv_source_indices,\n"
            "            fr10_tree_path0_nodes=self.fr10_tree_path0_nodes,\n"
            "            fr10_tree_path_node_tensors=self.fr10_tree_path_node_tensors,\n"
            "            fr10_tree_has_sibling=self.fr10_tree_has_sibling,\n"
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

    text = text.replace("import torch\n", "import ast\nimport json\nimport os\nimport torch\n", 1)
    text = text.replace(
        "from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata\n",
        (
            "from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata\n"
            "from lumo_flywheel_serving.fr10_gdn_tree_kernel import launch_tree_gdn_prepared\n"
            "\n"
            "_FR10_DECODE_MODE = os.environ.get(\"FR10_DECODE_MODE_DEFAULT\", \"tree_mtp\")\n"
        ),
        1,
    )

    conv_needle = '''        if spec_sequence_masks is not None:
            # spec_state_indices_tensor is always set when spec_sequence_masks is set
            assert spec_state_indices_tensor is not None
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                    : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )
'''
    conv_replacement = '''        if spec_sequence_masks is not None:
            # spec_state_indices_tensor is always set when spec_sequence_masks is set
            assert spec_state_indices_tensor is not None
            try:
                from vllm.v1.sample import rejection_sampler as _fr10_rs_mode
                _fr10_active_decode_mode = getattr(
                    _fr10_rs_mode, "_FR10_DECODE_MODE", _FR10_DECODE_MODE
                )
            except Exception:
                _fr10_active_decode_mode = _FR10_DECODE_MODE
            use_fr10_tree_conv = (
                os.environ.get("FR10_ENABLE_TREE_GDN") == "1"
                and _fr10_active_decode_mode == "tree_mtp"
                and getattr(attn_metadata, "fr10_tree_parent", None) is not None
                and attn_metadata.num_spec_decodes > 0
            )
            _fr10_conv_diag = getattr(attn_metadata, "fr10_tree_conv_diag", None)
            if os.environ.get("FR10_METRICS", "0") == "1" and _fr10_conv_diag is not None:
                _fr10_conv_diag[14].add_(float(attn_metadata.num_spec_decodes))
                if use_fr10_tree_conv:
                    _fr10_conv_diag[15].add_(float(attn_metadata.num_spec_decodes))
                else:
                    _fr10_conv_diag[16].add_(float(attn_metadata.num_spec_decodes))
                _fr10_conv_diag[20].add_(float(attn_metadata.num_prefills))
                _fr10_conv_diag[21].add_(float(attn_metadata.num_decodes))
            if use_fr10_tree_conv:
                _fr10_prior_conv_state_bank = torch.index_select(
                    conv_state,
                    0,
                    spec_state_indices_tensor[
                        : attn_metadata.num_spec_decodes, 0
                    ].to(torch.long),
                )
                try:
                    _fr10_tree_src = None
                    _fr10_spec_env = os.environ.get("SPEC_CONFIG")
                    if _fr10_spec_env:
                        _fr10_tree_src = json.loads(_fr10_spec_env).get(
                            "speculative_token_tree"
                        )
                    if _fr10_tree_src is None and self.speculative_config is not None:
                        _fr10_tree_src = self.speculative_config.speculative_token_tree
                    _fr10_choices = sorted(
                        ast.literal_eval(_fr10_tree_src), key=lambda _p: (len(_p), _p)
                    )
                    _fr10_index = {_p: _i + 1 for _i, _p in enumerate(_fr10_choices)}
                    _fr10_parent = [-1]
                    for _fr10_choice in _fr10_choices:
                        _fr10_parent.append(
                            0
                            if len(_fr10_choice) == 1
                            else _fr10_index[_fr10_choice[:-1]]
                        )
                    _fr10_path0_node_tensor = getattr(
                        attn_metadata, "fr10_tree_path0_nodes", None
                    )
                    assert _fr10_path0_node_tensor is not None
                    _fr10_path0_nodes_py = [0] + [
                        _fr10_index[_fr10_choice]
                        for _fr10_choice in _fr10_choices
                        if all(int(_fr10_part) == 0 for _fr10_part in _fr10_choice)
                    ]
                    _fr10_tree_n = len(_fr10_parent)
                    _fr10_branch_nodes_py = [
                        _fr10_i
                        for _fr10_i in range(_fr10_tree_n)
                        if _fr10_i not in set(_fr10_path0_nodes_py)
                    ]
                    _fr10_width = int(conv_weights.shape[1])
                    _fr10_tree_source_indices = getattr(
                        attn_metadata, "fr10_tree_conv_source_indices", None
                    )[_fr10_width]
                    _fr10_flat_source_indices = getattr(
                        attn_metadata, "fr10_flat_conv_source_indices", None
                    )[_fr10_width]
                    _fr10_path0_source_indices = getattr(
                        attn_metadata, "fr10_path0_conv_source_indices", None
                    )[_fr10_width]
                    _fr10_path_node_tensors = getattr(
                        attn_metadata, "fr10_tree_path_node_tensors", None
                    )
                    assert _fr10_path_node_tensors is not None
                    _fr10_source_flat = _fr10_tree_source_indices.reshape(-1)
                    _fr10_flat_source_flat = _fr10_flat_source_indices.reshape(-1)
                    _fr10_path0_source_flat = _fr10_path0_source_indices.reshape(-1)
                    _fr10_prior_col_base = torch.arange(
                        _fr10_width - 1,
                        dtype=torch.long,
                        device=mixed_qkv_spec.device,
                    )
                    _fr10_weight_f = conv_weights.to(torch.float32)
                    _fr10_tree_conv_out = torch.empty_like(mixed_qkv_spec)
                    _fr10_conv_diag = getattr(
                        attn_metadata, "fr10_tree_conv_diag", None
                    )
                    _fr10_log_conv_diag = (
                        os.environ.get("FR10_METRICS", "0") == "1"
                        and _fr10_conv_diag is not None
                    )
                    assert _fr10_prior_conv_state_bank is not None
                    for _fr10_b in range(attn_metadata.num_spec_decodes):
                        _fr10_start = _fr10_b * _fr10_tree_n
                        _fr10_end = _fr10_start + _fr10_tree_n
                        _fr10_x = mixed_qkv_spec[_fr10_start:_fr10_end]
                        _fr10_accept_offset = (
                            num_accepted_tokens[_fr10_b].to(torch.long) - 1
                            if num_accepted_tokens is not None
                            else torch.zeros((), dtype=torch.long, device=_fr10_x.device)
                        )
                        _fr10_prior_cols = _fr10_prior_col_base + _fr10_accept_offset
                        _fr10_prior_window = _fr10_prior_conv_state_bank[
                            _fr10_b
                        ].index_select(1, _fr10_prior_cols)
                        _fr10_source = torch.cat(
                            (_fr10_prior_window.transpose(0, 1), _fr10_x),
                            dim=0,
                        )
                        _fr10_window = _fr10_source.index_select(
                            0, _fr10_source_flat
                        ).view(_fr10_tree_n, _fr10_width, _fr10_x.size(1))
                        if self.conv1d.bias is None:
                            _fr10_acc = torch.zeros_like(_fr10_x, dtype=torch.float32)
                        else:
                            _fr10_acc = self.conv1d.bias.to(torch.float32).unsqueeze(
                                0
                            ).expand_as(_fr10_x.float()).clone()
                        for _fr10_col in range(_fr10_width):
                            _fr10_acc = _fr10_acc + (
                                _fr10_window[:, _fr10_col, :].to(torch.float32)
                                * _fr10_weight_f[:, _fr10_col].unsqueeze(0)
                            )
                        if self.activation in (True, "silu", "swish"):
                            _fr10_acc = torch.nn.functional.silu(_fr10_acc)
                        _fr10_out = _fr10_acc.to(dtype=mixed_qkv_spec.dtype)
                        _fr10_tree_conv_out[_fr10_start:_fr10_end] = _fr10_out
                        _fr10_path0_x = _fr10_x.index_select(
                            0, _fr10_path0_node_tensor
                        )
                        _fr10_node_state_rows = []
                        for _fr10_node_i in range(_fr10_tree_n):
                            _fr10_node_path = _fr10_path_node_tensors[_fr10_node_i]
                            _fr10_node_x = _fr10_x.index_select(0, _fr10_node_path)
                            _fr10_node_store_idx = (
                                _fr10_node_path.numel()
                                + torch.arange(
                                    conv_state.size(2),
                                    dtype=torch.long,
                                    device=mixed_qkv_spec.device,
                                )
                            )
                            _fr10_node_state_source = torch.cat(
                                (
                                    _fr10_prior_conv_state_bank[_fr10_b].transpose(0, 1),
                                    _fr10_node_x,
                                ),
                                dim=0,
                            )
                            _fr10_node_state_rows.append(
                                _fr10_node_state_source.index_select(
                                    0, _fr10_node_store_idx
                                ).transpose(0, 1)
                            )
                        _fr10_new_state = torch.stack(
                            _fr10_node_state_rows, dim=0
                        ).to(dtype=conv_state.dtype)
                        if os.environ.get("FR10_TREE_GDN_COMMIT_HANDOFF_LOG"):
                            try:
                                globals().setdefault(
                                    "_FR10_COMMIT_HANDOFF_CURR_CONV_BY_B", {}
                                )[int(_fr10_b)] = {
                                    "prior": _fr10_prior_conv_state_bank[
                                        _fr10_b
                                    ].detach().clone(),
                                    "rows": _fr10_new_state.detach().clone(),
                                }
                            except Exception:
                                pass
                        conv_state.index_copy_(
                            0,
                            spec_state_indices_tensor[
                                _fr10_b, :_fr10_tree_n
                            ].to(torch.long),
                            _fr10_new_state,
                        )
                        if _fr10_log_conv_diag:
                            _fr10_path0_source = torch.cat(
                                (_fr10_prior_window.transpose(0, 1), _fr10_path0_x),
                                dim=0,
                            )
                            _fr10_path0_window = _fr10_path0_source.index_select(
                                0, _fr10_path0_source_flat
                            ).view(
                                _fr10_path0_node_tensor.numel(),
                                _fr10_width,
                                _fr10_path0_x.size(1),
                            )
                            if self.conv1d.bias is None:
                                _fr10_path0_acc = torch.zeros_like(
                                    _fr10_path0_x, dtype=torch.float32
                                )
                            else:
                                _fr10_path0_acc = self.conv1d.bias.to(
                                    torch.float32
                                ).unsqueeze(0).expand_as(_fr10_path0_x.float()).clone()
                            for _fr10_col in range(_fr10_width):
                                _fr10_path0_acc = _fr10_path0_acc + (
                                    _fr10_path0_window[:, _fr10_col, :].to(
                                        torch.float32
                                    )
                                    * _fr10_weight_f[:, _fr10_col].unsqueeze(0)
                                )
                            if self.activation in (True, "silu", "swish"):
                                _fr10_path0_acc = torch.nn.functional.silu(
                                    _fr10_path0_acc
                                )
                            _fr10_path0_ref = _fr10_path0_acc.to(
                                dtype=mixed_qkv_spec.dtype
                            )
                            _fr10_tree_path0 = _fr10_out.index_select(
                                0, _fr10_path0_node_tensor
                            )
                            _fr10_flat_window = _fr10_source.index_select(
                                0, _fr10_flat_source_flat
                            ).view(_fr10_tree_n, _fr10_width, _fr10_x.size(1))
                            if self.conv1d.bias is None:
                                _fr10_flat_acc = torch.zeros_like(
                                    _fr10_x, dtype=torch.float32
                                )
                            else:
                                _fr10_flat_acc = self.conv1d.bias.to(
                                    torch.float32
                                ).unsqueeze(0).expand_as(_fr10_x.float()).clone()
                            for _fr10_col in range(_fr10_width):
                                _fr10_flat_acc = _fr10_flat_acc + (
                                    _fr10_flat_window[:, _fr10_col, :].to(
                                        torch.float32
                                    )
                                    * _fr10_weight_f[:, _fr10_col].unsqueeze(0)
                                )
                            if self.activation in (True, "silu", "swish"):
                                _fr10_flat_acc = torch.nn.functional.silu(
                                    _fr10_flat_acc
                                )
                            _fr10_native_flat_path0 = _fr10_flat_acc.to(
                                dtype=mixed_qkv_spec.dtype
                            ).index_select(0, _fr10_path0_node_tensor)
                            _fr10_serial_out = torch.empty_like(_fr10_out)
                            _fr10_serial_state_rows = []
                            _fr10_replay_pert_x = _fr10_x.clone()
                            for _fr10_branch_node in _fr10_branch_nodes_py:
                                _fr10_replay_pert_x[_fr10_branch_node].add_(10000.0)
                            _fr10_pert_source = torch.cat(
                                (_fr10_prior_window.transpose(0, 1), _fr10_replay_pert_x),
                                dim=0,
                            )
                            _fr10_pert_window = _fr10_pert_source.index_select(
                                0, _fr10_source_flat
                            ).view(_fr10_tree_n, _fr10_width, _fr10_x.size(1))
                            _fr10_flat_pert_window = _fr10_pert_source.index_select(
                                0, _fr10_flat_source_flat
                            ).view(_fr10_tree_n, _fr10_width, _fr10_x.size(1))
                            if self.conv1d.bias is None:
                                _fr10_pert_acc = torch.zeros_like(
                                    _fr10_x, dtype=torch.float32
                                )
                                _fr10_flat_pert_acc = torch.zeros_like(
                                    _fr10_x, dtype=torch.float32
                                )
                            else:
                                _fr10_pert_acc = self.conv1d.bias.to(
                                    torch.float32
                                ).unsqueeze(0).expand_as(_fr10_x.float()).clone()
                                _fr10_flat_pert_acc = self.conv1d.bias.to(
                                    torch.float32
                                ).unsqueeze(0).expand_as(_fr10_x.float()).clone()
                            for _fr10_col in range(_fr10_width):
                                _fr10_pert_acc = _fr10_pert_acc + (
                                    _fr10_pert_window[:, _fr10_col, :].to(torch.float32)
                                    * _fr10_weight_f[:, _fr10_col].unsqueeze(0)
                                )
                                _fr10_flat_pert_acc = _fr10_flat_pert_acc + (
                                    _fr10_flat_pert_window[:, _fr10_col, :].to(
                                        torch.float32
                                    )
                                    * _fr10_weight_f[:, _fr10_col].unsqueeze(0)
                                )
                            if self.activation in (True, "silu", "swish"):
                                _fr10_pert_acc = torch.nn.functional.silu(_fr10_pert_acc)
                                _fr10_flat_pert_acc = torch.nn.functional.silu(
                                    _fr10_flat_pert_acc
                                )
                            _fr10_pert_path0 = _fr10_pert_acc.to(
                                dtype=mixed_qkv_spec.dtype
                            ).index_select(0, _fr10_path0_node_tensor)
                            _fr10_flat_pert_path0 = _fr10_flat_pert_acc.to(
                                dtype=mixed_qkv_spec.dtype
                            ).index_select(0, _fr10_path0_node_tensor)
                            for _fr10_node_i in range(_fr10_tree_n):
                                _fr10_node_path = _fr10_path_node_tensors[_fr10_node_i]
                                _fr10_node_x = _fr10_x.index_select(0, _fr10_node_path)
                                _fr10_serial_source = torch.cat(
                                    (_fr10_prior_window.transpose(0, 1), _fr10_node_x),
                                    dim=0,
                                )
                                _fr10_serial_window_idx = (
                                    _fr10_node_path.numel()
                                    - 1
                                    + torch.arange(
                                        _fr10_width,
                                        dtype=torch.long,
                                        device=mixed_qkv_spec.device,
                                    )
                                )
                                _fr10_serial_window = _fr10_serial_source.index_select(
                                    0, _fr10_serial_window_idx
                                )
                                if self.conv1d.bias is None:
                                    _fr10_serial_acc = torch.zeros_like(
                                        _fr10_x[_fr10_node_i], dtype=torch.float32
                                    )
                                else:
                                    _fr10_serial_acc = self.conv1d.bias.to(
                                        torch.float32
                                    ).clone()
                                for _fr10_col in range(_fr10_width):
                                    _fr10_serial_acc = _fr10_serial_acc + (
                                        _fr10_serial_window[_fr10_col].to(torch.float32)
                                        * _fr10_weight_f[:, _fr10_col]
                                    )
                                if self.activation in (True, "silu", "swish"):
                                    _fr10_serial_acc = torch.nn.functional.silu(
                                        _fr10_serial_acc
                                    )
                                _fr10_serial_out[_fr10_node_i] = _fr10_serial_acc.to(
                                    dtype=mixed_qkv_spec.dtype
                                )
                                _fr10_serial_state_source = torch.cat(
                                    (
                                        _fr10_prior_conv_state_bank[_fr10_b].transpose(0, 1),
                                        _fr10_node_x,
                                    ),
                                    dim=0,
                                )
                                _fr10_serial_state_idx = (
                                    _fr10_node_path.numel()
                                    + torch.arange(
                                        conv_state.size(2),
                                        dtype=torch.long,
                                        device=mixed_qkv_spec.device,
                                    )
                                )
                                _fr10_serial_state_rows.append(
                                    _fr10_serial_state_source.index_select(
                                        0, _fr10_serial_state_idx
                                    ).transpose(0, 1)
                                )
                            _fr10_serial_state = torch.stack(
                                _fr10_serial_state_rows, dim=0
                            ).to(dtype=conv_state.dtype)
                            _fr10_tree_delta = (
                                _fr10_tree_path0.float() - _fr10_path0_ref.float()
                            ).abs()
                            _fr10_native_delta = (
                                _fr10_native_flat_path0.float()
                                - _fr10_path0_ref.float()
                            ).abs()
                            _fr10_tree_max = _fr10_tree_delta.max()
                            _fr10_native_max = _fr10_native_delta.max()
                            _fr10_serial_out_max = (
                                _fr10_out.float() - _fr10_serial_out.float()
                            ).abs().max()
                            _fr10_serial_state_max = (
                                _fr10_new_state.float() - _fr10_serial_state.float()
                            ).abs().max()
                            _fr10_sibling_path0_max = (
                                _fr10_pert_path0.float() - _fr10_tree_path0.float()
                            ).abs().max()
                            _fr10_flat_sibling_path0_max = (
                                _fr10_flat_pert_path0.float()
                                - _fr10_native_flat_path0.float()
                            ).abs().max()
                            _fr10_conv_diag[0].copy_(
                                torch.maximum(_fr10_conv_diag[0], _fr10_tree_max)
                            )
                            _fr10_conv_diag[1].copy_(
                                torch.maximum(_fr10_conv_diag[1], _fr10_native_max)
                            )
                            _fr10_conv_diag[2].add_(
                                (_fr10_tree_max != 0).to(dtype=torch.float32)
                            )
                            _fr10_conv_diag[3].add_(
                                (_fr10_native_max != 0).to(dtype=torch.float32)
                            )
                            _fr10_conv_diag[4].add_(1.0)
                            _fr10_conv_diag[5].fill_(float(_fr10_tree_n))
                            _fr10_conv_diag[6].copy_(
                                torch.maximum(_fr10_conv_diag[6], _fr10_serial_out_max)
                            )
                            _fr10_conv_diag[7].copy_(
                                torch.maximum(_fr10_conv_diag[7], _fr10_serial_state_max)
                            )
                            _fr10_conv_diag[8].copy_(
                                torch.maximum(_fr10_conv_diag[8], _fr10_sibling_path0_max)
                            )
                            _fr10_conv_diag[9].copy_(
                                torch.maximum(
                                    _fr10_conv_diag[9], _fr10_flat_sibling_path0_max
                                )
                            )
                            _fr10_conv_diag[10].add_(
                                (_fr10_sibling_path0_max != 0).to(dtype=torch.float32)
                            )
                            _fr10_conv_diag[11].add_(
                                (_fr10_flat_sibling_path0_max != 0).to(dtype=torch.float32)
                            )
                    mixed_qkv_spec = _fr10_tree_conv_out
                except Exception as _fr10_tree_conv_exc:
                    if os.environ.get("FR10_METRICS", "0") == "1":
                        logger.warning_once(
                            "FR10 tree causal-conv fallback to native flat order: %s",
                            _fr10_tree_conv_exc,
                        )
                    mixed_qkv_spec = causal_conv1d_update(
                        mixed_qkv_spec,
                        conv_state,
                        conv_weights,
                        self.conv1d.bias,
                        self.activation,
                        conv_state_indices=spec_state_indices_tensor[:, 0][
                            : attn_metadata.num_spec_decodes
                        ],
                        num_accepted_tokens=num_accepted_tokens,
                        query_start_loc=spec_query_start_loc,
                        max_query_len=spec_state_indices_tensor.size(-1),
                        validate_data=False,
                    )
            else:
                mixed_qkv_spec = causal_conv1d_update(
                    mixed_qkv_spec,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                        : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                    ],
                    num_accepted_tokens=num_accepted_tokens,
                    query_start_loc=spec_query_start_loc,
                    max_query_len=spec_state_indices_tensor.size(-1),
                    validate_data=False,
                )
'''
    if conv_needle not in text:
        raise RuntimeError("FR10 GDN causal conv spec branch needle not found")
    text = text.replace(conv_needle, conv_replacement, 1)

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
            try:
                from vllm.v1.sample import rejection_sampler as _fr10_rs_mode
                _fr10_active_decode_mode = getattr(
                    _fr10_rs_mode, "_FR10_DECODE_MODE", _FR10_DECODE_MODE
                )
            except Exception:
                _fr10_active_decode_mode = _FR10_DECODE_MODE
            use_fr10_tree = (
                os.environ.get("FR10_ENABLE_TREE_GDN") == "1"
                and _fr10_active_decode_mode == "tree_mtp"
                and getattr(attn_metadata, "fr10_tree_parent", None) is not None
                and attn_metadata.num_spec_decodes > 0
            )
            _fr10_scan_branch_diag = getattr(
                attn_metadata, "fr10_tree_conv_diag", None
            )
            if (
                os.environ.get("FR10_METRICS", "0") == "1"
                and _fr10_scan_branch_diag is not None
            ):
                _fr10_scan_branch_diag[17].add_(float(attn_metadata.num_spec_decodes))
                if use_fr10_tree:
                    _fr10_scan_branch_diag[18].add_(float(attn_metadata.num_spec_decodes))
                else:
                    _fr10_scan_branch_diag[19].add_(float(attn_metadata.num_spec_decodes))
                _fr10_scan_branch_diag[22].add_(float(attn_metadata.num_spec_decodes))
            if use_fr10_tree:
                assert spec_query_start_loc is not None
                assert spec_state_indices_tensor is not None
                assert attn_metadata.fr10_tree_parent is not None
                assert attn_metadata.fr10_tree_strict_mask is not None
                assert attn_metadata.fr10_tree_visible_mask is not None
                if os.environ.get("FR10_METRICS", "0") == "1":
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
                tree_state_all = torch.empty(
                    (
                        attn_metadata.num_spec_decodes,
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
                    tree_state = tree_state_all[fr10_b]
                    _fr10_capture_scan_payload = (
                        os.environ.get("FR10_TREE_GDN_CAPTURE_PAYLOAD")
                        and not globals().get("_FR10_TREE_GDN_CAPTURE_DONE", False)
                        and fr10_b == 0
                    )
                    if _fr10_capture_scan_payload:
                        _fr10_capture_state_index = int(
                            spec_state_indices_tensor[fr10_b, 0].detach().cpu().item()
                        )
                        _fr10_capture_h0 = (
                            ssm_state[_fr10_capture_state_index].detach().cpu().clone()
                        )
                    _fr10_commit_handoff_active = (
                        os.environ.get("FR10_TREE_GDN_COMMIT_HANDOFF_LOG")
                    )
                    if _fr10_commit_handoff_active:
                        try:
                            _fr10_commit_state_index = int(
                                spec_state_indices_tensor[fr10_b, 0]
                                .detach()
                                .cpu()
                                .item()
                            )
                            _fr10_commit_h0 = (
                                ssm_state[_fr10_commit_state_index].detach().clone()
                            )
                        except Exception:
                            _fr10_commit_handoff_active = False
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
                        invocation_counter=(
                            attn_metadata.fr10_tree_invocation_counter
                            if os.environ.get("FR10_METRICS", "0") == "1"
                            else None
                        ),
                    )
                    core_attn_out_spec[0, start:end] = tree_out[:tree_n]
                    if _fr10_capture_scan_payload:
                        try:
                            _fr10_payload_path = os.environ.get(
                                "FR10_TREE_GDN_CAPTURE_PAYLOAD"
                            )
                            torch.save(
                                {
                                    "schema": "fr10.tree_gdn_scan_capture.v1",
                                    "layer_prefix": str(self.prefix),
                                    "batch_index": int(fr10_b),
                                    "tree_parent": [
                                        int(_x)
                                        for _x in attn_metadata.fr10_tree_parent.detach()
                                        .cpu()
                                        .tolist()
                                    ],
                                    "n_actual": int(tree_n),
                                    "n_pad": int(tree_n_pad),
                                    "state_index": int(_fr10_capture_state_index),
                                    "output_scale": float(self.head_k_dim**-0.5),
                                    "query_spec": query_spec[0, start:end]
                                    .detach()
                                    .cpu()
                                    .clone(),
                                    "key_spec": key_spec[0, start:end]
                                    .detach()
                                    .cpu()
                                    .clone(),
                                    "value_spec": value_spec[0, start:end]
                                    .detach()
                                    .cpu()
                                    .clone(),
                                    "a": a[start:end].detach().cpu().clone(),
                                    "b": b[start:end].detach().cpu().clone(),
                                    "A_log": self.A_log.detach().cpu().clone(),
                                    "dt_bias": self.dt_bias.detach().cpu().clone(),
                                    "h0": _fr10_capture_h0,
                                    "value_tree": value_tree[start:end]
                                    .detach()
                                    .cpu()
                                    .clone(),
                                    "g_tree": g_tree[start:end].detach().cpu().clone(),
                                    "beta_tree": beta_tree[start:end]
                                    .detach()
                                    .cpu()
                                    .clone(),
                                    "serving_out": tree_out[:tree_n]
                                    .detach()
                                    .cpu()
                                    .clone(),
                                    "serving_state": tree_state[:tree_n]
                                    .detach()
                                    .cpu()
                                    .clone(),
                                },
                                _fr10_payload_path,
                            )
                            globals()["_FR10_TREE_GDN_CAPTURE_DONE"] = True
                        except Exception as _fr10_capture_exc:
                            logger.warning_once(
                                "FR10 tree GDN scan capture failed: %s",
                                _fr10_capture_exc,
                            )
                    ssm_state.index_copy_(
                        0,
                        spec_state_indices_tensor[fr10_b, :tree_n].to(torch.long),
                        tree_state[:tree_n].to(dtype=ssm_state.dtype),
                    )
                    if _fr10_commit_handoff_active:
                        try:
                            import json as _fr10_handoff_json
                            import time as _fr10_handoff_time

                            _fr10_prefix = str(self.prefix)
                            _fr10_filter = os.environ.get(
                                "FR10_TREE_GDN_COMMIT_HANDOFF_LAYER_PREFIX", ""
                            )
                            _fr10_count = int(
                                globals().get("_FR10_COMMIT_HANDOFF_LOG_COUNT", 0)
                            )
                            _fr10_limit = int(
                                os.environ.get(
                                    "FR10_TREE_GDN_COMMIT_HANDOFF_LIMIT", "32"
                                )
                            )
                            _fr10_layer_ok = (
                                not _fr10_filter or _fr10_prefix.endswith(_fr10_filter)
                            )
                            if _fr10_layer_ok and _fr10_count < _fr10_limit:
                                _fr10_key = (_fr10_prefix, int(fr10_b))
                                _fr10_prev_all = globals().setdefault(
                                    "_FR10_COMMIT_HANDOFF_PREV", {}
                                )
                                _fr10_prev = _fr10_prev_all.get(_fr10_key)
                                _fr10_rows = globals().get(
                                    "_LUMO_FA_LAST_ACCEPTED_TREE_ROWS", []
                                )
                                _fr10_lens = globals().get(
                                    "_LUMO_FA_LAST_ACCEPTED_TREE_LENS", []
                                )
                                _fr10_prev_accepted_row = (
                                    int(_fr10_rows[fr10_b])
                                    if fr10_b < len(_fr10_rows)
                                    else None
                                )
                                _fr10_prev_accepted_len = (
                                    int(_fr10_lens[fr10_b])
                                    if fr10_b < len(_fr10_lens)
                                    else None
                                )
                                _fr10_curr_conv = globals().get(
                                    "_FR10_COMMIT_HANDOFF_CURR_CONV_BY_B", {}
                                ).get(int(fr10_b), {})
                                _fr10_curr_conv_prior = _fr10_curr_conv.get("prior")
                                _fr10_curr_conv_rows = _fr10_curr_conv.get("rows")
                                if (
                                    _fr10_prev is not None
                                    and _fr10_prev_accepted_row is not None
                                ):
                                    _fr10_tree_n_prev = int(_fr10_prev["tree_n"])
                                    _fr10_row = max(
                                        0,
                                        min(
                                            int(_fr10_prev_accepted_row),
                                            _fr10_tree_n_prev - 1,
                                        ),
                                    )
                                    _fr10_has_accept = (
                                        _fr10_prev_accepted_len is None
                                        or int(_fr10_prev_accepted_len) > 0
                                    )
                                    _fr10_expected_ssm = (
                                        _fr10_prev["tree_state"][_fr10_row]
                                        if _fr10_has_accept
                                        else _fr10_prev["h0"]
                                    )
                                    _fr10_expected_conv = (
                                        _fr10_prev["conv_rows"][_fr10_row]
                                        if _fr10_has_accept
                                        else _fr10_prev["conv_prior"]
                                    )
                                    _fr10_prev_spec_indices = _fr10_prev.get(
                                        "spec_state_indices"
                                    )
                                    _fr10_accepted_bank_row = None
                                    if _fr10_prev_spec_indices is not None:
                                        _fr10_accepted_bank_row = int(
                                            _fr10_prev_spec_indices[int(_fr10_row)]
                                            .detach()
                                            .cpu()
                                            .item()
                                        )
                                    _fr10_address_coincide = bool(
                                        (not _fr10_has_accept)
                                        or (
                                            int(_fr10_prev_accepted_row)
                                            == int(_fr10_row)
                                            and _fr10_accepted_bank_row is not None
                                            and int(_fr10_accepted_bank_row)
                                            == int(_fr10_commit_state_index)
                                        )
                                    )
                                    _fr10_ssm_max = float(
                                        (
                                            _fr10_commit_h0.float()
                                            - _fr10_expected_ssm.float()
                                        )
                                        .abs()
                                        .max()
                                        .detach()
                                        .cpu()
                                        .item()
                                    )
                                    _fr10_conv_max = None
                                    if _fr10_curr_conv_prior is not None:
                                        _fr10_conv_max = float(
                                            (
                                                _fr10_curr_conv_prior.float()
                                                - _fr10_expected_conv.float()
                                            )
                                            .abs()
                                            .max()
                                            .detach()
                                            .cpu()
                                            .item()
                                        )
                                    _fr10_neg_row = (
                                        _fr10_row + 1
                                        if _fr10_row + 1 < _fr10_tree_n_prev
                                        else (_fr10_row - 1 if _fr10_row > 0 else None)
                                    )
                                    _fr10_neg_ssm_max = None
                                    _fr10_neg_conv_max = None
                                    if _fr10_neg_row is not None:
                                        _fr10_neg_ssm_max = float(
                                            (
                                                _fr10_commit_h0.float()
                                                - _fr10_prev["tree_state"][
                                                    int(_fr10_neg_row)
                                                ].float()
                                            )
                                            .abs()
                                            .max()
                                            .detach()
                                            .cpu()
                                            .item()
                                        )
                                        if _fr10_curr_conv_prior is not None:
                                            _fr10_neg_conv_max = float(
                                                (
                                                    _fr10_curr_conv_prior.float()
                                                    - _fr10_prev["conv_rows"][
                                                        int(_fr10_neg_row)
                                                    ].float()
                                                )
                                                .abs()
                                                .max()
                                                .detach()
                                                .cpu()
                                                .item()
                                            )
                                    _fr10_payload = {
                                        "schema": "fr10.commit_native_handoff.v1",
                                        "event": "commit_native_handoff",
                                        "ts": round(_fr10_handoff_time.time(), 4),
                                        "layer_prefix": _fr10_prefix,
                                        "batch_index": int(fr10_b),
                                        "state_index": int(_fr10_commit_state_index),
                                        "prev_accepted_row": int(
                                            _fr10_prev_accepted_row
                                        ),
                                        "prev_accepted_len": (
                                            None
                                            if _fr10_prev_accepted_len is None
                                            else int(_fr10_prev_accepted_len)
                                        ),
                                        "prev_accepted_node_id": (
                                            int(_fr10_prev_accepted_row)
                                            if _fr10_has_accept
                                            else None
                                        ),
                                        "tree_n": int(_fr10_tree_n_prev),
                                        "accepted_node_row": int(_fr10_row),
                                        "accepted_spec_state_bank_row": (
                                            None
                                            if _fr10_accepted_bank_row is None
                                            else int(_fr10_accepted_bank_row)
                                        ),
                                        "next_read_bank_row": int(_fr10_commit_state_index),
                                        "address_coincide": _fr10_address_coincide,
                                        "ssm_vs_native_max_abs": _fr10_ssm_max,
                                        "conv_vs_native_max_abs": _fr10_conv_max,
                                        "negative_row": (
                                            None
                                            if _fr10_neg_row is None
                                            else int(_fr10_neg_row)
                                        ),
                                        "negative_ssm_max_abs": _fr10_neg_ssm_max,
                                        "negative_conv_max_abs": _fr10_neg_conv_max,
                                        "ssm_pass": bool(_fr10_ssm_max == 0.0),
                                        "conv_pass": bool(
                                            _fr10_conv_max is not None
                                            and _fr10_conv_max == 0.0
                                        ),
                                        "negative_powered": bool(
                                            (
                                                _fr10_neg_ssm_max is not None
                                                and _fr10_neg_ssm_max > 0.0
                                            )
                                            or (
                                                _fr10_neg_conv_max is not None
                                                and _fr10_neg_conv_max > 0.0
                                            )
                                        ),
                                    }
                                    _fr10_path = os.environ.get(
                                        "FR10_TREE_GDN_COMMIT_HANDOFF_LOG",
                                        "/logs/fr10_commit_native_handoff.jsonl",
                                    )
                                    with open(_fr10_path, "a", buffering=1) as _fr10_fh:
                                        _fr10_fh.write(
                                            _fr10_handoff_json.dumps(_fr10_payload)
                                            + chr(10)
                                        )
                                    globals()["_FR10_COMMIT_HANDOFF_LOG_COUNT"] = (
                                        _fr10_count + 1
                                    )
                                if (
                                    _fr10_curr_conv_prior is not None
                                    and _fr10_curr_conv_rows is not None
                                ):
                                    _fr10_prev_all[_fr10_key] = {
                                        "tree_n": int(tree_n),
                                        "h0": _fr10_commit_h0.detach().clone(),
                                        "tree_state": tree_state[
                                            :tree_n
                                        ].detach().clone(),
                                        "spec_state_indices": spec_state_indices_tensor[
                                            fr10_b, :tree_n
                                        ].detach().clone(),
                                        "conv_prior": _fr10_curr_conv_prior.detach().clone(),
                                        "conv_rows": _fr10_curr_conv_rows[
                                            :tree_n
                                        ].detach().clone(),
                                    }
                        except Exception as _fr10_handoff_exc:
                            logger.warning_once(
                                "FR10 commit native handoff capture failed: %s",
                                _fr10_handoff_exc,
                            )
                    _fr10_scan_diag = getattr(
                        attn_metadata, "fr10_tree_conv_diag", None
                    )
                    if (
                        os.environ.get("FR10_METRICS", "0") == "1"
                        and _fr10_scan_diag is not None
                    ):
                        _fr10_staged_state = ssm_state.index_select(
                            0, spec_state_indices_tensor[fr10_b, :tree_n].to(torch.long)
                        )
                        _fr10_staged_delta = (
                            _fr10_staged_state.float()
                            - tree_state[:tree_n].to(dtype=ssm_state.dtype).float()
                        ).abs().max()
                        _fr10_scan_diag[12].copy_(
                            torch.maximum(_fr10_scan_diag[12], _fr10_staged_delta)
                        )
                        _fr10_scan_diag[13].add_(
                            (_fr10_staged_delta != 0).to(dtype=torch.float32)
                        )
                last_recurrent_state = tree_state_all
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


def _patch_request_decode_mode() -> bool:
    text = REQUEST_PATH.read_text()
    if "fr10_decode_mode" in text:
        return False

    text = text.replace(
        "from vllm.sampling_params import SamplingParams\n",
        (
            "from vllm.sampling_params import SamplingParams\n"
            "from lumo_flywheel_serving.fr10_decode_modes import decode_mode_from_sampling_params\n"
        ),
        1,
    )
    text = text.replace(
        "        self.sampling_params = sampling_params\n",
        (
            "        self.sampling_params = sampling_params\n"
            "        self.fr10_decode_mode = decode_mode_from_sampling_params(sampling_params)\n"
        ),
        1,
    )
    REQUEST_PATH.write_text(text)
    return True


def _patch_sched_output_decode_mode() -> bool:
    text = SCHED_OUTPUT_PATH.read_text()
    if "fr10_decode_mode" in text:
        return False
    text = text.replace(
        "    num_invalid_spec_tokens: dict[str, int] | None = None\n",
        (
            "    num_invalid_spec_tokens: dict[str, int] | None = None\n"
            "    # FR10: homogeneous per-request decode mode for this scheduler step.\n"
            "    fr10_decode_mode: str | None = None\n"
        ),
        1,
    )
    SCHED_OUTPUT_PATH.write_text(text)
    return True


def _patch_scheduler_decode_modes() -> bool:
    text = SCHEDULER_PATH.read_text()
    sentinel = "# FR10_DECODE_MODE_SAFETY"
    if sentinel in text:
        return False

    text = text.replace(
        "from vllm.v1.spec_decode.metrics import SpecDecodingStats\n",
        (
            "from vllm.v1.spec_decode.metrics import SpecDecodingStats\n"
            "from lumo_flywheel_serving.fr10_decode_modes import (\n"
            "    NAIVE_MTP,\n"
            "    NON_MTP,\n"
            "    decode_mode_from_request,\n"
            "    select_path0_spec_tokens,\n"
            ")\n"
        ),
        1,
    )
    text = text.replace(
        "        self.kv_cache_manager.new_step_starts()\n\n",
        (
            f"        {sentinel}: reject mixed decode modes within one forward pass.\n"
            "        fr10_step_decode_mode = None\n"
            "        self.kv_cache_manager.new_step_starts()\n\n"
        ),
        1,
    )
    text = text.replace(
        "            request = self.running[req_index]\n\n",
        (
            "            request = self.running[req_index]\n"
            "            fr10_req_mode = decode_mode_from_request(request)\n"
            "            if fr10_step_decode_mode is None:\n"
            "                fr10_step_decode_mode = fr10_req_mode\n"
            "            elif fr10_req_mode != fr10_step_decode_mode:\n"
            "                req_index += 1\n"
            "                continue\n\n"
        ),
        1,
    )
    text = text.replace(
        "                request = request_queue.peek_request()\n"
        "                request_id = request.request_id\n\n",
        (
            "                request = request_queue.peek_request()\n"
            "                request_id = request.request_id\n"
            "                fr10_req_mode = decode_mode_from_request(request)\n"
            "                if fr10_step_decode_mode is None:\n"
            "                    fr10_step_decode_mode = fr10_req_mode\n"
            "                elif fr10_req_mode != fr10_step_decode_mode:\n"
            "                    request_queue.pop_request()\n"
            "                    step_skipped_waiting.prepend_request(request)\n"
            "                    continue\n\n"
        ),
        1,
    )
    text = text.replace(
        "            new_block_ids_to_zero=new_block_ids_to_zero,\n"
        "        )\n",
        (
            "            new_block_ids_to_zero=new_block_ids_to_zero,\n"
            "            fr10_decode_mode=fr10_step_decode_mode,\n"
            "        )\n"
        ),
        1,
    )
    text = text.replace(
        "            # Add newly generated spec token ids to the request.\n"
        "            if self.structured_output_manager.should_advance(request):\n",
        (
            "            # FR10: mode-specific draft handling on a multi-mode server.\n"
            "            fr10_req_mode = decode_mode_from_request(request)\n"
            "            if fr10_req_mode == NON_MTP:\n"
            "                request.spec_token_ids = []\n"
            "                continue\n"
            "            if fr10_req_mode == NAIVE_MTP:\n"
            "                try:\n"
            "                    import ast as _fr10_ast\n"
            "                    _fr10_spec = getattr(self.vllm_config, \"speculative_config\", None)\n"
            "                    _fr10_tree = getattr(_fr10_spec, \"speculative_token_tree\", None) if _fr10_spec is not None else None\n"
            "                    if _fr10_tree:\n"
            "                        spec_token_ids = select_path0_spec_tokens(\n"
            "                            spec_token_ids, _fr10_ast.literal_eval(_fr10_tree)\n"
            "                        )\n"
            "                except Exception:\n"
            "                    request.spec_token_ids = []\n"
            "                    continue\n"
            "\n"
            "            # Add newly generated spec token ids to the request.\n"
            "            if self.structured_output_manager.should_advance(request):\n"
        ),
        1,
    )
    SCHEDULER_PATH.write_text(text)
    return True


def _patch_scheduler_spec_trace() -> bool:
    text = SCHEDULER_PATH.read_text()
    sentinel = "# LUMO_PER_AGENT_SPEC_TRACE"
    if sentinel in text:
        return False

    nl = "\n"
    anchor = (
        "    ) -> SpecDecodingStats | None:"
        + nl
        + "        if not self.log_stats or not num_draft_tokens:"
    )
    if anchor not in text:
        raise RuntimeError("make_spec_decoding_stats anchor not found")
    inject = nl.join(
        [
            "    ) -> SpecDecodingStats | None:",
            f"        {sentinel}",
            "        import json as _lj, time as _lt, os as _lo",
            '        if _lo.environ.get("FR10_METRICS", "0") == "1":',
            "            try:",
            "                global _LUMO_SPEC_FH",
            "                try:",
            "                    _LUMO_SPEC_FH",
            "                except NameError:",
            '                    _LUMO_SPEC_FH = open(_lo.environ.get("LUMO_PER_REQ_SPEC_TRACE", "/logs/per_req_spec_trace.jsonl"), "a", buffering=1)',
            "                _linv = (num_invalid_spec_tokens.get(request_id, 0) if num_invalid_spec_tokens else 0)",
            '                _LUMO_SPEC_FH.write(_lj.dumps({"ts": round(_lt.time(), 4), "rid": request_id, "draft": num_draft_tokens, "proposal_width": num_draft_tokens, "verify_width": num_draft_tokens, "acc": num_accepted_tokens, "inv": _linv}) + chr(10))',
            "            except Exception:",
            "                pass",
            "        if not self.log_stats or not num_draft_tokens:",
        ]
    )
    text = text.replace(anchor, inject, 1)
    SCHEDULER_PATH.write_text(text)
    return True


def _patch_rejection_sampler_tree_lcp() -> bool:
    text = REJECTION_SAMPLER_PATH.read_text()
    sentinel = "# LUMO_TREE_PATH_LCP_MAX"
    if sentinel in text:
        return False

    helper_anchor = "\n\ndef rejection_sample("
    helper = r'''

# LUMO_TREE_PATH_LCP_MAX: greedy N-spine verifier.
#
# Gate B is greedy/deterministic, so max-LCP over root-to-leaf paths is the
# correct deterministic tree accept rule. The same rows are diagnostics only for
# sampled Gate C; sampled production must use the distribution-preserving tree
# rejection sampler, not this max selector.
def _lumo_tree_path_lcp_max_greedy_sample(
    output_token_ids: torch.Tensor,
    accepted_tree_rows: torch.Tensor,
    num_draft_tokens,
    draft_token_ids: torch.Tensor,
    tree_parent_indices: torch.Tensor,
    parent_token_ids: torch.Tensor,
    self_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    max_spec_len: int,
) -> torch.Tensor:
    parents_cpu = [int(x) for x in tree_parent_indices.detach().cpu().tolist()]
    drafts_cpu = [int(x) for x in draft_token_ids.detach().cpu().tolist()]
    parent_targets_cpu = [int(x) for x in parent_token_ids.detach().cpu().tolist()]
    self_targets_cpu = [int(x) for x in self_token_ids.detach().cpu().tolist()]
    bonus_targets_raw = bonus_token_ids.detach().cpu().tolist()
    if bonus_targets_raw and isinstance(bonus_targets_raw[0], list):
        bonus_targets_cpu = [int(x[0]) for x in bonus_targets_raw]
    else:
        bonus_targets_cpu = [int(x) for x in bonus_targets_raw]
    if hasattr(num_draft_tokens, 'detach'):
        counts = [int(x) for x in num_draft_tokens.detach().cpu().tolist()]
    else:
        counts = [int(x) for x in num_draft_tokens]

    out_rows = []
    accepted_rows = []
    accepted_lens = []
    path_log_rows = []
    winner_log_rows = []
    start = 0
    for req_i, node_count in enumerate(counts):
        node_count = int(node_count)
        parents = parents_cpu[start:start + node_count]
        drafts = drafts_cpu[start:start + node_count]
        parent_targets = parent_targets_cpu[start:start + node_count]
        self_targets = self_targets_cpu[start:start + node_count]

        children = {-1: []}
        for node, parent in enumerate(parents):
            parent = int(parent)
            children.setdefault(parent, []).append(node)
            children.setdefault(node, [])
        leaves = [node for node in range(node_count) if not children.get(node)]
        if not leaves:
            leaves = list(range(node_count))

        best_path = []
        best_lcp = -1
        best_leaf = -1
        best_path_idx = 0
        path_scores = []
        for path_idx, leaf in enumerate(leaves):
            path = []
            node = int(leaf)
            guard = 0
            while 0 <= node < node_count and guard <= node_count:
                path.append(node)
                node = int(parents[node])
                guard += 1
            path.reverse()
            lcp = 0
            for node in path:
                if int(drafts[node]) != int(parent_targets[node]):
                    break
                lcp += 1
            path_scores.append({
                'leaf': int(leaf),
                'path': [int(x) for x in path],
                'lcp': int(lcp),
            })
            # Stable tie-break: earliest flattened leaf wins. With vLLM's sorted
            # tree choices this preserves the native top-1/path0 chain.
            if lcp > best_lcp:
                best_lcp = int(lcp)
                best_path = path
                best_leaf = int(leaf)
                best_path_idx = int(path_idx)

        best_lcp = max(0, int(best_lcp))
        path0_lcp = int(path_scores[0]['lcp']) if path_scores else 0
        row = []
        for node in best_path[:best_lcp]:
            row.append(int(drafts[node]))
        if best_path:
            if best_lcp < len(best_path):
                bonus_source = 'reject_parent_target'
                row.append(int(parent_targets[best_path[best_lcp]]))
            elif best_lcp > 0:
                if best_path_idx == 0 and req_i < len(bonus_targets_cpu):
                    bonus_source = 'path0_native_bonus'
                    row.append(int(bonus_targets_cpu[req_i]))
                else:
                    bonus_source = 'tree_self_target'
                    row.append(int(self_targets[best_path[best_lcp - 1]]))
            else:
                bonus_source = 'root_parent_target'
                row.append(int(parent_targets[best_path[0]]))
        else:
            bonus_source = 'no_path'
        row = row[:int(max_spec_len) + 1]
        out_rows.append(row)
        accepted_row = int(best_path[best_lcp - 1]) if best_lcp > 0 else 0
        accepted_rows.append(accepted_row)
        accepted_lens.append(int(best_lcp))
        path_log_rows.append({
            'req_index': int(req_i),
            'node_count': int(node_count),
            'accepted_len': int(best_lcp),
            'accepted_final_row': int(accepted_row),
            'accepted_node_ids': [int(x) for x in best_path[:best_lcp]],
            'path0_lcp': int(path0_lcp),
            'superset_violation': bool(int(best_lcp) < int(path0_lcp)),
            'superset_delta': int(best_lcp) - int(path0_lcp),
            'winner_leaf': int(best_leaf),
            'winner_path': [int(x) for x in best_path],
            'path_scores': path_scores,
            'emitted_tokens': [int(x) for x in row],
            'bonus_source': bonus_source,
            'native_bonus_token': (
                int(bonus_targets_cpu[req_i])
                if req_i < len(bonus_targets_cpu)
                else None
            ),
            'draft_token_ids': [int(x) for x in drafts],
            'parent_target_ids': [int(x) for x in parent_targets],
            'self_target_ids': [int(x) for x in self_targets],
        })
        counts_by_path = {
            str(i): int(score.get('lcp', 0))
            for i, score in enumerate(path_scores)
        }
        winner_log_rows.append({
            'primary': f'fr10_tree_req_{req_i}',
            'policy': 'greedy_tree_lcp_max',
            'selector_enabled': True,
            'lossless_public_stream': True,
            'temperature': 0.0,
            'winner_req_id': f'fr10_tree_req_{req_i}::path{best_path_idx}',
            'winner_spine': int(best_path_idx),
            'winner_acc': int(best_lcp),
            'spine0_acc': int(path0_lcp),
            'candidate_winner_req_id': f'fr10_tree_req_{req_i}::path{best_path_idx}',
            'candidate_winner_spine': int(best_path_idx),
            'candidate_winner_acc': int(best_lcp),
            'hidden_winner_suppressed_reason': None,
            'counts': counts_by_path,
            'members': [f'fr10_tree_req_{req_i}::path{i}' for i in range(len(path_scores))],
            'copy': {'missing': 0, 'copied': 0},
        })
        start += node_count

    output_token_ids.fill_(-1)
    for req_i, row in enumerate(out_rows):
        for pos, token_id in enumerate(row):
            output_token_ids[req_i, pos] = int(token_id)
    accepted_tree_rows.copy_(
        torch.tensor(accepted_rows, dtype=accepted_tree_rows.dtype,
                     device=accepted_tree_rows.device)
    )
    globals()['_LUMO_TREE_LAST_ACCEPTED_ROWS_KERNEL'] = [int(x) for x in accepted_rows]
    globals()['_LUMO_TREE_LAST_ACCEPTED_LENS_KERNEL'] = [int(x) for x in accepted_lens]
    try:
        from vllm.model_executor.layers.mamba import gdn_linear_attn as _lumo_tree_commit_gdn
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_ROWS = [
            int(x) for x in accepted_rows
        ]
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_LENS = [
            int(x) for x in accepted_lens
        ]
    except Exception:
        pass
    try:
        import json as _lcpj, os as _lcpo, time as _lcpt
        if _lcpo.environ.get('FR10_METRICS', '0') == '1':
            global _LUMO_TREE_PATH_LCP_FH
            try:
                _LUMO_TREE_PATH_LCP_FH
            except NameError:
                _LUMO_TREE_PATH_LCP_FH = open(
                    _lcpo.environ.get('LUMO_TREE_PATH_LCP_LOG',
                                      '/logs/tree_path_lcp_max.jsonl'),
                    'a',
                    buffering=1,
                )
            _now = round(_lcpt.time(), 4)
            for row in path_log_rows:
                row = dict(row)
                row['ts'] = _now
                row['event'] = 'tree_path_lcp_max'
                _LUMO_TREE_PATH_LCP_FH.write(_lcpj.dumps(row) + chr(10))
    except Exception:
        pass
    try:
        import json as _iwj, os as _iwo, time as _iwt
        if _iwo.environ.get('FR10_METRICS', '0') == '1':
            global _LUMO_IR_WINNER_TRACE_FH
            try:
                _LUMO_IR_WINNER_TRACE_FH
            except NameError:
                _LUMO_IR_WINNER_TRACE_FH = open(
                    _iwo.environ.get('LUMO_IR_WINNER_TRACE_FILE',
                                     '/logs/independent_winner_trace.jsonl'),
                    'a',
                    buffering=1,
                )
            _now = round(_iwt.time(), 4)
            for row in winner_log_rows:
                row = dict(row)
                row['ts'] = _now
                row['event'] = 'independent_winner_commit'
                _LUMO_IR_WINNER_TRACE_FH.write(_iwj.dumps(row) + chr(10))
    except Exception:
        pass
    return output_token_ids


def _lumo_tree_canonical_multidraft_sample(
    output_token_ids: torch.Tensor,
    accepted_tree_rows: torch.Tensor,
    num_draft_tokens,
    draft_token_ids: torch.Tensor,
    tree_parent_indices: torch.Tensor,
    target_logits: torch.Tensor,
    tree_self_logits: torch.Tensor,
    draft_probs: torch.Tensor | None,
    bonus_token_ids: torch.Tensor,
    max_spec_len: int,
) -> torch.Tensor:
    """Reference sampled tree committer using the verified FR10 rule."""
    import numpy as _fr10_np
    from lumo_flywheel_serving.fr10_tree_rejection_sampler import (
        sample_deterministic_multidraft_rejection_step as _fr10_sample_det_step,
        sample_multidraft_rejection_step as _fr10_sample_step,
    )

    parents_cpu = [int(x) for x in tree_parent_indices.detach().cpu().tolist()]
    drafts_cpu = [int(x) for x in draft_token_ids.detach().cpu().tolist()]
    if hasattr(num_draft_tokens, 'detach'):
        counts = [int(x) for x in num_draft_tokens.detach().cpu().tolist()]
    else:
        counts = [int(x) for x in num_draft_tokens]
    target_probs_cpu = target_logits.softmax(dim=-1, dtype=torch.float32).detach().cpu().numpy()
    self_probs_cpu = tree_self_logits.softmax(dim=-1, dtype=torch.float32).detach().cpu().numpy()
    draft_probs_cpu = (
        None if draft_probs is None else draft_probs.detach().cpu().numpy()
    )
    seed = int(torch.randint(0, 2**31 - 1, (1,), device=output_token_ids.device).cpu().item())
    rng = _fr10_np.random.default_rng(seed)

    out_rows = []
    accepted_rows = []
    accepted_lens = []
    sample_log_rows = []
    try:
        import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
        if _fr10_lo.environ.get('FR10_METRICS', '0') == '1':
            global _LUMO_TREE_SAMPLE_DEBUG_FH
            try:
                _LUMO_TREE_SAMPLE_DEBUG_FH
            except NameError:
                _LUMO_TREE_SAMPLE_DEBUG_FH = open(
                    _fr10_lo.environ.get('LUMO_TREE_SAMPLER_DEBUG_LOG',
                                         '/logs/tree_sampler_debug.jsonl'),
                    'a',
                    buffering=1,
                )
            _LUMO_TREE_SAMPLE_DEBUG_FH.write(_fr10_lj.dumps({
                'event': 'sample_helper_enter',
                'ts': round(_fr10_lt.time(), 4),
                'max_spec_len': int(max_spec_len),
                'has_draft_probs': draft_probs is not None,
            }) + chr(10))
    except Exception:
        pass
    start = 0
    for req_i, node_count in enumerate(counts):
        node_count = int(node_count)
        parents = parents_cpu[start:start + node_count]
        drafts = drafts_cpu[start:start + node_count]
        current_parent = -1
        accepted_row = 0
        accepted_path = []
        row = []
        for _step in range(int(max_spec_len) + 1):
            children = [
                node for node, parent in enumerate(parents)
                if int(parent) == int(current_parent)
            ]
            if not children:
                if current_parent >= 0:
                    row.append(
                        int(rng.choice(
                            self_probs_cpu.shape[1],
                            p=self_probs_cpu[start + current_parent],
                        ))
                    )
                elif req_i < int(bonus_token_ids.numel()):
                    row.append(int(bonus_token_ids.reshape(-1)[req_i].detach().cpu().item()))
                break

            if draft_probs_cpu is None:
                step = _fr10_sample_det_step(
                    target_probs_cpu[start + children[0]],
                    [drafts[child] for child in children],
                    rng=rng,
                )
            else:
                step = _fr10_sample_step(
                    target_probs_cpu[start + children[0]],
                    [draft_probs_cpu[start + child] for child in children],
                    rng=rng,
                )
            row.append(int(step.token_id))
            if not step.accepted:
                break
            accepted_child = int(children[int(step.source_index)])
            if int(step.token_id) != int(drafts[accepted_child]):
                break
            current_parent = accepted_child
            accepted_row = int(current_parent)
            accepted_path.append(int(current_parent))
        out_rows.append(row[:int(max_spec_len) + 1])
        accepted_rows.append(int(accepted_row))
        accepted_lens.append(int(len(accepted_path)))
        final_root = int(accepted_path[0]) if accepted_path else None
        sample_log_rows.append({
            'event': 'tree_sample_accept',
            'req_index': int(req_i),
            'node_count': int(node_count),
            'accepted_len': int(len(accepted_path)),
            'accepted_final_row': int(accepted_row),
            'accepted_node_ids': [int(x) for x in accepted_path],
            'accepted_root': final_root,
            'emitted_tokens': [int(x) for x in row[:int(max_spec_len) + 1]],
            'draft_token_ids': [int(x) for x in drafts],
        })
        start += node_count

    output_token_ids.fill_(-1)
    for req_i, row in enumerate(out_rows):
        for pos, token_id in enumerate(row):
            output_token_ids[req_i, pos] = int(token_id)
    accepted_tree_rows.copy_(
        torch.tensor(accepted_rows, dtype=accepted_tree_rows.dtype,
                     device=accepted_tree_rows.device)
    )
    globals()['_LUMO_TREE_LAST_ACCEPTED_ROWS_KERNEL'] = [int(x) for x in accepted_rows]
    globals()['_LUMO_TREE_LAST_ACCEPTED_LENS_KERNEL'] = [int(x) for x in accepted_lens]
    try:
        from vllm.model_executor.layers.mamba import gdn_linear_attn as _lumo_tree_commit_gdn
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_ROWS = [
            int(x) for x in accepted_rows
        ]
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_LENS = [
            int(x) for x in accepted_lens
        ]
    except Exception:
        pass
    try:
        import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
        if _fr10_lo.environ.get('FR10_METRICS', '0') == '1':
            global _LUMO_TREE_SAMPLE_ACCEPT_FH
            try:
                _LUMO_TREE_SAMPLE_ACCEPT_FH
            except NameError:
                _LUMO_TREE_SAMPLE_ACCEPT_FH = open(
                    _fr10_lo.environ.get('LUMO_TREE_PATH_LCP_LOG',
                                         '/logs/tree_path_lcp_max.jsonl'),
                    'a',
                    buffering=1,
                )
            _now = round(_fr10_lt.time(), 4)
            for _row in sample_log_rows:
                _row = dict(_row)
                _row['ts'] = _now
                _LUMO_TREE_SAMPLE_ACCEPT_FH.write(_fr10_lj.dumps(_row) + chr(10))
    except Exception:
        pass
    return output_token_ids
'''
    if helper_anchor not in text:
        raise RuntimeError("tree path-LCP helper anchor not found")
    text = text.replace(helper_anchor, helper + helper_anchor, 1)

    old = """        if sampling_metadata.all_greedy:
            lumo_tree_sample_kernel[(batch_size,)](
                output_token_ids,
                accepted_tree_rows,
                cu_num_draft_tokens,
                draft_token_ids,
                tree_parent_indices,
                tree_token_ids[0],
                tree_token_ids[1],
                bonus_token_ids,
                max_spec_len,
            )
        else:
"""
    new = """        if sampling_metadata.all_greedy:
            output_token_ids = _lumo_tree_path_lcp_max_greedy_sample(
                output_token_ids,
                accepted_tree_rows,
                num_draft_tokens,
                draft_token_ids,
                tree_parent_indices,
                tree_token_ids[0],
                tree_token_ids[1],
                max_spec_len,
            )
        else:
"""
    if old in text:
        text = text.replace(old, new, 1)
    else:
        stock_call = """        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
        )
"""
        stock_call_new = """        lumo_tree_parent_indices = getattr(metadata, "tree_parent_indices", None)
        lumo_tree_token_ids = None
        lumo_tree_self_logits = None
        if lumo_tree_parent_indices is not None:
            tree_self_logits = logits[metadata.tree_self_logits_indices]
            tree_self_logits = tree_self_logits.to(torch.float32)
            if not self.is_processed_logprobs_mode:
                tree_self_logits = tree_self_logits.clone()
            tree_self_logits = self.apply_logits_processors(
                tree_self_logits, sampling_metadata, metadata
            )
            lumo_tree_self_logits = apply_sampling_constraints(
                tree_self_logits,
                metadata.cu_num_draft_tokens,
                sampling_metadata,
            )
            if sampling_metadata.all_greedy:
                lumo_tree_token_ids = torch.stack(
                    [
                        target_logits.argmax(dim=-1).to(torch.int32),
                        lumo_tree_self_logits.argmax(dim=-1).to(torch.int32),
                    ],
                    dim=0,
                ).contiguous()

        try:
            import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
            if _fr10_lo.environ.get("FR10_METRICS", "0") == "1":
                global _LUMO_TREE_SAMPLER_DEBUG_FH
                try:
                    _LUMO_TREE_SAMPLER_DEBUG_FH
                except NameError:
                    _LUMO_TREE_SAMPLER_DEBUG_FH = open(
                        _fr10_lo.environ.get(
                            "LUMO_TREE_SAMPLER_DEBUG_LOG",
                            "/logs/tree_sampler_debug.jsonl",
                        ),
                        "a",
                        buffering=1,
                    )
                _LUMO_TREE_SAMPLER_DEBUG_FH.write(
                    _fr10_lj.dumps({
                        "event": "sampler_metadata",
                        "ts": round(_fr10_lt.time(), 4),
                        "has_tree_parent_indices": lumo_tree_parent_indices is not None,
                        "has_tree_self_logits": lumo_tree_self_logits is not None,
                        "all_greedy": bool(sampling_metadata.all_greedy),
                    }) + chr(10)
                )
        except Exception:
            pass

        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            tree_parent_indices=lumo_tree_parent_indices,
            tree_token_ids=lumo_tree_token_ids,
            tree_self_logits=lumo_tree_self_logits,
        )
"""
        if stock_call not in text:
            raise RuntimeError("stock rejection_sample call anchor not found")
        text = text.replace(stock_call, stock_call_new, 1)

        stock_sig = """    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
"""
        stock_sig_new = """    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    tree_parent_indices: torch.Tensor | None = None,
    tree_token_ids: torch.Tensor | None = None,
    tree_self_logits: torch.Tensor | None = None,
) -> torch.Tensor:
"""
        if stock_sig not in text:
            raise RuntimeError("stock rejection_sample signature anchor not found")
        text = text.replace(stock_sig, stock_sig_new, 1)

        stock_branch = """    if sampling_metadata.all_greedy:
        is_greedy = None
"""
        stock_branch_new = """    if (
        tree_parent_indices is not None
        and tree_token_ids is not None
        and sampling_metadata.all_greedy
    ):
        accepted_tree_rows = torch.empty(
            (batch_size,), dtype=torch.int32, device=device
        )
        return _lumo_tree_path_lcp_max_greedy_sample(
            output_token_ids,
            accepted_tree_rows,
            num_draft_tokens,
            draft_token_ids,
            tree_parent_indices,
            tree_token_ids[0],
            tree_token_ids[1],
            bonus_token_ids,
            max_spec_len,
        )

    if tree_parent_indices is not None and not sampling_metadata.all_greedy:
        try:
            import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
            if _fr10_lo.environ.get("FR10_METRICS", "0") == "1":
                global _LUMO_TREE_SAMPLER_BRANCH_FH
                try:
                    _LUMO_TREE_SAMPLER_BRANCH_FH
                except NameError:
                    _LUMO_TREE_SAMPLER_BRANCH_FH = open(
                        _fr10_lo.environ.get(
                            "LUMO_TREE_SAMPLER_DEBUG_LOG",
                            "/logs/tree_sampler_debug.jsonl",
                        ),
                        "a",
                        buffering=1,
                    )
                _LUMO_TREE_SAMPLER_BRANCH_FH.write(
                    _fr10_lj.dumps({
                        "event": "sampler_branch_enter",
                        "ts": round(_fr10_lt.time(), 4),
                        "batch_size": int(batch_size),
                        "max_spec_len": int(max_spec_len),
                        "has_tree_self_logits": tree_self_logits is not None,
                    }) + chr(10)
                )
        except Exception:
            pass
        if tree_self_logits is None:
            raise RuntimeError("FR10 sampled tree committer missing self logits")
        accepted_tree_rows = torch.empty(
            (batch_size,), dtype=torch.int32, device=device
        )
        return _lumo_tree_canonical_multidraft_sample(
            output_token_ids,
            accepted_tree_rows,
            num_draft_tokens,
            draft_token_ids,
            tree_parent_indices,
            target_logits,
            tree_self_logits,
            draft_probs,
            bonus_token_ids,
            max_spec_len,
        )

    if sampling_metadata.all_greedy:
        is_greedy = None
"""
        if stock_branch not in text:
            raise RuntimeError("stock tree path-LCP greedy branch anchor not found")
        text = text.replace(stock_branch, stock_branch_new, 1)

    text = sentinel + "\n" + text
    REJECTION_SAMPLER_PATH.write_text(text)
    return True


def _patch_gpu_model_runner_tree_metadata() -> bool:
    text = GPU_MODEL_RUNNER_PATH.read_text()
    sentinel = "# LUMO_TREE_METADATA"
    if sentinel in text:
        return False

    meta_anchor = """        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True
        )
"""
    meta_inject = """        # LUMO_TREE_METADATA: tree parent map + parent-logit remap.
        lumo_tree_parent_indices = None
        lumo_tree_self_logits_indices = None
        lumo_draft_token_indices = None
        _lumo_tree_meta_debug = {
            "event": "gpu_tree_metadata",
            "mode": None,
            "has_tree_src": False,
            "has_tree_parent_indices": False,
            "reason": "not_built",
        }
        try:
            try:
                from vllm.v1.sample import rejection_sampler as _fr10_rejection_sampler
                _fr10_mode = getattr(_fr10_rejection_sampler, "_FR10_DECODE_MODE", None)
            except Exception:
                _fr10_mode = None
            _fr10_mode = _fr10_mode or __import__("os").environ.get(
                "FR10_DECODE_MODE_DEFAULT", "tree_mtp"
            )
            _ltree_src = None
            try:
                _spec_env = __import__("os").environ.get("SPEC_CONFIG")
                if _spec_env:
                    _ltree_src = __import__("json").loads(_spec_env).get(
                        "speculative_token_tree"
                    )
            except Exception:
                _ltree_src = None
            _lspec = getattr(self.vllm_config, "speculative_config", None)
            if not _ltree_src:
                _ltree_src = getattr(_lspec, "speculative_token_tree", None) if _lspec is not None else None
            _lumo_tree_meta_debug["mode"] = _fr10_mode
            _lumo_tree_meta_debug["has_tree_src"] = bool(_ltree_src)
            if _fr10_mode == "tree_mtp" and _ltree_src:
                _choices = sorted(__import__("ast").literal_eval(_ltree_src), key=lambda _p: (len(_p), _p))
                _max_depth = max(len(_t) for _t in _choices)
                _lumo_tree_meta_debug["tree_len"] = int(len(_choices))
                _lumo_tree_meta_debug["max_depth"] = int(_max_depth)
                if len(_choices) > _max_depth:
                    _path_to_idx = {tuple(_p): _i for _i, _p in enumerate(_choices)}
                    _parents_template = np.array([
                        _path_to_idx.get(tuple(_p[:-1]), -1) for _p in _choices
                    ], dtype=np.int32)
                    _tree_len = int(len(_choices))
                    _parents = []
                    _target = []
                    _self = []
                    _draft = []
                    _sampled_start = 0
                    _ok = True
                    for _n in num_draft_tokens.tolist():
                        _n = int(_n)
                        if _n == 0:
                            _sampled_start += 1
                            continue
                        if _n != _tree_len:
                            _ok = False
                            _lumo_tree_meta_debug["reason"] = f"draft_count_mismatch:{_n}!={_tree_len}"
                            break
                        for _node_idx, _parent in enumerate(_parents_template.tolist()):
                            _parent_local = 0 if _parent < 0 else int(_parent) + 1
                            _parents.append(int(_parent))
                            _target.append(_sampled_start + _parent_local)
                            _self.append(_sampled_start + _node_idx + 1)
                            _draft.append(_sampled_start + _node_idx + 1)
                        _sampled_start += _n + 1
                    if _ok and len(_target) == int(cu_num_draft_tokens[-1]):
                        target_logits_indices = np.array(_target, dtype=np.int32)
                        lumo_tree_parent_indices = torch.from_numpy(
                            np.array(_parents, dtype=np.int32)
                        ).to(self.device, non_blocking=True)
                        lumo_tree_self_logits_indices = torch.from_numpy(
                            np.array(_self, dtype=np.int32)
                        ).to(self.device, non_blocking=True)
                        lumo_draft_token_indices = torch.from_numpy(
                            np.array(_draft, dtype=np.int32)
                        ).to(self.device, non_blocking=True)
                        _lumo_tree_meta_debug["reason"] = "ok"
                    elif _ok:
                        _lumo_tree_meta_debug["reason"] = (
                            f"target_total_mismatch:{len(_target)}!={int(cu_num_draft_tokens[-1])}"
                        )
                else:
                    _lumo_tree_meta_debug["reason"] = "linear_or_empty_tree"
            elif _fr10_mode != "tree_mtp":
                _lumo_tree_meta_debug["reason"] = f"mode:{_fr10_mode}"
            else:
                _lumo_tree_meta_debug["reason"] = "missing_tree_src"
        except Exception as _lumo_tree_meta_exc:
            lumo_tree_parent_indices = None
            lumo_tree_self_logits_indices = None
            lumo_draft_token_indices = None
            _lumo_tree_meta_debug["reason"] = (
                "exception:"
                + type(_lumo_tree_meta_exc).__name__
                + ":"
                + str(_lumo_tree_meta_exc)[:200]
            )
        try:
            import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
            if _fr10_lo.environ.get("FR10_METRICS", "0") == "1":
                global _LUMO_TREE_META_DEBUG_FH
                try:
                    _LUMO_TREE_META_DEBUG_FH
                except NameError:
                    _LUMO_TREE_META_DEBUG_FH = open(
                        _fr10_lo.environ.get(
                            "LUMO_TREE_SAMPLER_DEBUG_LOG",
                            "/logs/tree_sampler_debug.jsonl",
                        ),
                        "a",
                        buffering=1,
                    )
                _LUMO_TREE_META_DEBUG_FH.write(
                    _fr10_lj.dumps(dict(
                        _lumo_tree_meta_debug,
                        ts=round(_fr10_lt.time(), 4),
                        has_tree_parent_indices=lumo_tree_parent_indices is not None,
                        has_tree_self_logits_indices=lumo_tree_self_logits_indices is not None,
                        has_draft_token_indices=lumo_draft_token_indices is not None,
                        num_draft_tokens=[int(_x) for _x in num_draft_tokens.tolist()],
                        cu_total=int(cu_num_draft_tokens[-1]) if len(cu_num_draft_tokens) else 0,
                    )) + chr(10)
                )
        except Exception:
            pass

        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True
        )
"""
    if meta_anchor not in text:
        raise RuntimeError("metadata CPU-copy anchor not found")
    text = text.replace(meta_anchor, meta_inject, 1)

    draft_anchor = """        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        return SpecDecodeMetadata(
"""
    draft_inject = """        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        if lumo_draft_token_indices is not None:
            draft_token_ids = draft_token_ids[lumo_draft_token_indices]
        else:
            draft_token_ids = draft_token_ids[target_logits_indices + 1]

        _lumo_meta = SpecDecodeMetadata(
"""
    if draft_anchor not in text:
        raise RuntimeError("draft token gather anchor not found")
    text = text.replace(draft_anchor, draft_inject, 1)

    return_anchor = """            logits_indices=logits_indices,
        )

    def _prepare_kv_sharing_fast_prefill(
"""
    return_inject = """            logits_indices=logits_indices,
        )
        if lumo_tree_parent_indices is not None:
            _lumo_meta.tree_parent_indices = lumo_tree_parent_indices
            _lumo_meta.tree_self_logits_indices = lumo_tree_self_logits_indices
        return _lumo_meta

    def _prepare_kv_sharing_fast_prefill(
"""
    if return_anchor not in text:
        raise RuntimeError("metadata return anchor not found")
    text = text.replace(return_anchor, return_inject, 1)

    GPU_MODEL_RUNNER_PATH.write_text(text)
    return True


def _patch_gpu_model_runner_decode_mode_globals() -> bool:
    text = GPU_MODEL_RUNNER_PATH.read_text()
    sentinel = "# FR10_DECODE_MODE_GLOBALS"
    if sentinel in text:
        return False

    anchor = "        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0\n"
    inject = (
        f"        {sentinel}: homogeneous batch mode for GDN/sampler branches.\n"
        "        fr10_decode_mode = getattr(scheduler_output, \"fr10_decode_mode\", None) or \"tree_mtp\"\n"
        "        try:\n"
        "            from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr10_gdn_linear\n"
        "            _fr10_gdn_linear._FR10_DECODE_MODE = fr10_decode_mode\n"
        "        except Exception:\n"
        "            pass\n"
        "        try:\n"
        "            from vllm.v1.sample import rejection_sampler as _fr10_rejection_sampler\n"
        "            _fr10_rejection_sampler._FR10_DECODE_MODE = fr10_decode_mode\n"
        "        except Exception:\n"
        "            pass\n"
        "\n"
        "        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0\n"
    )
    if anchor not in text:
        raise RuntimeError("gpu_model_runner use_spec_decode anchor not found")
    text = text.replace(anchor, inject, 1)
    GPU_MODEL_RUNNER_PATH.write_text(text)
    return True


def _patch_eagle_tree_consumption_verify() -> bool:
    text = EAGLE_PATH.read_text()
    sentinel = "# FR10_TREE_DRAFT_CONSUMPTION_VERIFY"
    if sentinel in text:
        return False

    text = text.replace("import ast\n", "import ast\nimport json\nimport os\n", 1)

    tree_parse_old = """        # Parse the speculative token tree.
        spec_token_tree = self.speculative_config.speculative_token_tree
        assert spec_token_tree is not None
        self.tree_choices: list[tuple[int, ...]] = ast.literal_eval(spec_token_tree)
"""
    tree_parse_new = """        # Parse the speculative token tree.
        spec_token_tree = None
        try:
            spec_env = os.environ.get("SPEC_CONFIG")
            if spec_env:
                spec_token_tree = json.loads(spec_env).get("speculative_token_tree")
        except Exception:
            spec_token_tree = None
        if spec_token_tree is None:
            spec_token_tree = self.speculative_config.speculative_token_tree
        assert spec_token_tree is not None
        self.tree_choices: list[tuple[int, ...]] = sorted(
            ast.literal_eval(spec_token_tree), key=lambda _p: (len(_p), _p)
        )
"""
    if tree_parse_old not in text:
        raise RuntimeError("EAGLE tree parse anchor not found")
    text = text.replace(tree_parse_old, tree_parse_new, 1)

    old = """        if any(isinstance(md, TreeAttentionMetadata) for md in per_group_attn_metadata):
            # Draft using tree attention - requires full logits for top-k
            logits = self.model.compute_logits(sample_hidden_states)
            draft_token_ids_list = self.propose_tree(
                batch_size=batch_size,
                logits=logits,
                positions=positions,
                hidden_states=hidden_states,
                common_attn_metadata=common_attn_metadata,
                slot_mappings=slot_mappings,
            )
            # [batch_size, num_tree_tokens]
            return torch.cat(draft_token_ids_list, dim=1)
"""
    new = """        _fr10_active_decode_mode = os.environ.get("FR10_DECODE_MODE_DEFAULT", "tree_mtp")
        try:
            from vllm.v1.sample import rejection_sampler as _fr10_rs_mode
            _fr10_active_decode_mode = getattr(
                _fr10_rs_mode, "_FR10_DECODE_MODE", _fr10_active_decode_mode
            )
        except Exception:
            pass
        _fr10_caterpillar_choices = [
            (0,), (0, 0), (0, 1), (0, 0, 0), (0, 0, 1),
            (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 1),
        ]
        _fr10_is_caterpillar = (
            _fr10_active_decode_mode == "tree_mtp"
            and int(self.num_speculative_tokens) == 9
            and [tuple(_x) for _x in getattr(self, "tree_choices", [])]
            == _fr10_caterpillar_choices
        )
        if _fr10_is_caterpillar:
            # FR10_CATERPILLAR_NATIVE_SPINE_TOP2: read-only drafter fix.
            # Run the native causal MTP spine unchanged for depth 5. At each
            # post-root spine step, read the runner-up token from the same
            # logits and pack it into the caterpillar leaf slot. Leaves are
            # never fed back into any forward or recurrent state.
            _fr10_logits = self.model.compute_logits(sample_hidden_states)
            _fr10_top2 = torch.topk(_fr10_logits, 2, dim=-1).indices
            draft_token_ids = self._greedy_sample(sample_hidden_states)
            _fr10_spine_tokens = [draft_token_ids]
            _fr10_leaf_tokens = []

            if self.allowed_attn_types is not None:
                for group_md in per_group_attn_metadata:
                    if not isinstance(group_md, self.allowed_attn_types):
                        raise ValueError(
                            f"Unsupported attention metadata type for speculative "
                            "decoding with FR10 caterpillar native-spine drafting: "
                            f"{type(group_md)}. Supported types are: "
                            f"{self.allowed_attn_types}"
                        )

            cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
                self._determine_batch_execution_and_padding(batch_size)
            )

            common_attn_metadata.num_actual_tokens = batch_size
            common_attn_metadata.max_query_len = 1
            common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
            common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
                self.token_arange_np[: batch_size + 1]
            ).clone()

            if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
                common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
                common_attn_metadata._seq_lens_cpu = None
                common_attn_metadata._num_computed_tokens_cpu = None

            block_size = self.block_size
            assert block_size > 0, "block_size has not been initialized."
            for token_index in range(4):
                input_ids = _fr10_spine_tokens[-1].int()
                positions_1d = positions[0] if self.uses_mrope else positions
                if self.uses_mrope:
                    out_pos = self.mrope_positions[0, :batch_size]
                elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
                    out_pos = self.xdrope_positions[0, :batch_size]
                else:
                    out_pos = self.positions[:batch_size]
                eagle_step_update_slot_mapping_and_metadata(
                    positions_1d=positions_1d,
                    block_table_tensor=common_attn_metadata.block_table_tensor,
                    seq_lens=common_attn_metadata.seq_lens,
                    block_size=block_size,
                    max_model_len=self.max_model_len,
                    out_clamped_positions=out_pos,
                    out_slot_mapping=self._slot_mapping_buffer[:input_batch_size],
                    input_batch_size=input_batch_size,
                )
                common_attn_metadata.slot_mapping = self._slot_mapping_buffer[:batch_size]
                if self.uses_mrope:
                    self.mrope_positions[1:, :batch_size] = self.mrope_positions[
                        0, :batch_size
                    ]
                    positions = self.mrope_positions[:, :batch_size]
                elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
                    self.xdrope_positions[1:, :batch_size] = self.xdrope_positions[
                        0, :batch_size
                    ]
                    positions = self.xdrope_positions[0, :batch_size]
                else:
                    positions = self.positions[:batch_size]

                common_attn_metadata.max_seq_len = min(
                    common_attn_metadata.max_seq_len + 1, self.max_model_len
                )
                if common_attn_metadata._seq_lens_cpu is not None:
                    common_attn_metadata._seq_lens_cpu += 1
                if common_attn_metadata._num_computed_tokens_cpu is not None:
                    common_attn_metadata._num_computed_tokens_cpu += 1

                _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
                    common_attn_metadata, draft_index=token_index + 1
                )

                self.input_ids[:batch_size] = input_ids
                self.hidden_states[:batch_size] = hidden_states
                if self.supports_mm_inputs:
                    self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:input_batch_size]
                else:
                    input_ids = self.input_ids[:input_batch_size]
                    inputs_embeds = None

                model_kwargs = {
                    "input_ids": input_ids,
                    "positions": self._get_positions(input_batch_size),
                    "inputs_embeds": inputs_embeds,
                }
                if self.pass_hidden_states_to_model:
                    model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]

                with set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=input_batch_size,
                    num_tokens_across_dp=batch_size_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    slot_mapping=self._get_slot_mapping(input_batch_size),
                ):
                    ret_hidden_states = self.model(**model_kwargs)
                    if not self.model_returns_tuple():
                        last_hidden_states = ret_hidden_states
                        hidden_states = ret_hidden_states
                    else:
                        last_hidden_states, hidden_states = ret_hidden_states

                hidden_states = hidden_states[:batch_size]
                _fr10_step_logits = self.model.compute_logits(
                    last_hidden_states[:batch_size]
                )
                _fr10_step_top2 = torch.topk(_fr10_step_logits, 2, dim=-1).indices
                draft_token_ids = self._greedy_sample(last_hidden_states[:batch_size])
                _fr10_spine_tokens.append(draft_token_ids)
                _fr10_leaf_tokens.append(_fr10_step_top2[:, 1])

            _fr10_packed = torch.stack(
                [
                    _fr10_spine_tokens[0],
                    _fr10_spine_tokens[1],
                    _fr10_leaf_tokens[0],
                    _fr10_spine_tokens[2],
                    _fr10_leaf_tokens[1],
                    _fr10_spine_tokens[3],
                    _fr10_leaf_tokens[2],
                    _fr10_spine_tokens[4],
                    _fr10_leaf_tokens[3],
                ],
                dim=1,
            )
            try:
                import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
                if _fr10_lo.environ.get("FR10_METRICS", "0") == "1":
                    global _LUMO_CATERPILLAR_DRAFTER_FH
                    try:
                        _LUMO_CATERPILLAR_DRAFTER_FH
                    except NameError:
                        _LUMO_CATERPILLAR_DRAFTER_FH = open(
                            _fr10_lo.environ.get(
                                "LUMO_CATERPILLAR_DRAFTER_LOG",
                                "/logs/fr10_caterpillar_drafter.jsonl",
                            ),
                            "a",
                            buffering=1,
                        )
                    _LUMO_CATERPILLAR_DRAFTER_FH.write(
                        _fr10_lj.dumps({
                            "event": "fr10_caterpillar_native_spine_top2",
                            "ts": round(_fr10_lt.time(), 4),
                            "spine_slots": [0, 1, 3, 5, 7],
                            "leaf_slots": [2, 4, 6, 8],
                            "draft": _fr10_packed.detach().cpu().tolist(),
                            "spine": torch.stack(_fr10_spine_tokens, dim=1).detach().cpu().tolist(),
                            "leaves": torch.stack(_fr10_leaf_tokens, dim=1).detach().cpu().tolist(),
                        }) + chr(10)
                    )
            except Exception:
                pass
            return _fr10_packed

        _fr10_tree_draft_branch_seen = any(
            isinstance(md, TreeAttentionMetadata) for md in per_group_attn_metadata
        )
        try:
            import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
            if _fr10_lo.environ.get("FR10_METRICS", "0") == "1":
                global _LUMO_TREE_DRAFT_BRANCH_FH
                try:
                    _LUMO_TREE_DRAFT_BRANCH_FH
                except NameError:
                    _LUMO_TREE_DRAFT_BRANCH_FH = open(
                        _fr10_lo.environ.get(
                            "LUMO_TREE_DRAFT_BRANCH_LOG",
                            "/logs/fr10_tree_draft_branch.jsonl",
                        ),
                        "a",
                        buffering=1,
                    )
                _LUMO_TREE_DRAFT_BRANCH_FH.write(
                    _fr10_lj.dumps({
                        "event": "tree_draft_branch",
                        "ts": round(_fr10_lt.time(), 4),
                        "tree_branch_seen": bool(_fr10_tree_draft_branch_seen),
                        "metadata_types": [
                            type(md).__name__ for md in per_group_attn_metadata
                        ],
                        "num_speculative_tokens": int(self.num_speculative_tokens),
                        "speculative_token_tree": _fr10_lo.environ.get("SPEC_CONFIG"),
                    }) + chr(10)
                )
        except Exception:
            pass

        if _fr10_tree_draft_branch_seen:
            # FR10_TREE_DRAFT_CONSUMPTION_VERIFY: no separate spine overlay.
            # vLLM's native tree drafter consumes the MTP head directly:
            # child-rank 0 is the fed-back spine, child-rank 1 is recorded as
            # the side leaf and must not advance the spine recurrent state.
            logits = self.model.compute_logits(sample_hidden_states)
            draft_token_ids_list = self.propose_tree(
                batch_size=batch_size,
                logits=logits,
                positions=positions,
                hidden_states=hidden_states,
                common_attn_metadata=common_attn_metadata,
                slot_mappings=slot_mappings,
            )
            # [batch_size, num_tree_tokens]
            return torch.cat(draft_token_ids_list, dim=1)
"""
    if old not in text:
        raise RuntimeError("EAGLE tree propose branch anchor not found")
    text = text.replace(old, new, 1)

    root_log_anchor = """        draft_token_ids_list = [draft_token_ids]
        draft_hidden_states = hidden_states.view(batch_size, 1, -1)
"""
    root_log_new = """        draft_token_ids_list = [draft_token_ids]
        # FR10_TREE_DRAFT_CONSUMPTION_VERIFY: runtime node placement and q.
        _fr10_depth_choices = {}
        for _fr10_choice in self.tree_choices:
            _fr10_depth_choices.setdefault(len(_fr10_choice), []).append(tuple(_fr10_choice))
        _fr10_runtime_slot_by_choice = {
            tuple(_fr10_choice): int(_fr10_i)
            for _fr10_i, _fr10_choice in enumerate(self.tree_choices)
        }

        def _fr10_record_tree_consumption(
            _fr10_depth,
            _fr10_level_logits,
            _fr10_level_tokens,
            _fr10_level_num_parents,
            _fr10_level_num_children,
        ):
            import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
            if _fr10_lo.environ.get("FR10_METRICS", "0") != "1":
                return
            _fr10_now = round(_fr10_lt.time(), 4)
            def _fr10_log_consumption_error(_fr10_exc):
                global _LUMO_TREE_DRAFT_CONSUMPTION_ERR_FH
                try:
                    _LUMO_TREE_DRAFT_CONSUMPTION_ERR_FH
                except NameError:
                    _LUMO_TREE_DRAFT_CONSUMPTION_ERR_FH = open(
                        _fr10_lo.environ.get(
                            "LUMO_TREE_DRAFT_CONSUMPTION_ERROR_LOG",
                            "/logs/fr10_tree_draft_consumption_errors.jsonl",
                        ),
                        "a",
                        buffering=1,
                    )
                _LUMO_TREE_DRAFT_CONSUMPTION_ERR_FH.write(
                    _fr10_lj.dumps({
                        "event": "tree_draft_consumption_error",
                        "ts": _fr10_now,
                        "depth": int(_fr10_depth),
                        "level_tokens_shape": [
                            int(_x) for _x in getattr(_fr10_level_tokens, "shape", [])
                        ],
                        "level_logits_shape": [
                            int(_x) for _x in getattr(_fr10_level_logits, "shape", [])
                        ],
                        "level_num_parents": int(_fr10_level_num_parents),
                        "level_num_children": int(_fr10_level_num_children),
                        "error_type": type(_fr10_exc).__name__,
                        "error": str(_fr10_exc),
                    }) + chr(10)
                )
            try:
                global _LUMO_TREE_DRAFT_CONSUMPTION_FH
                try:
                    _LUMO_TREE_DRAFT_CONSUMPTION_FH
                except NameError:
                    _LUMO_TREE_DRAFT_CONSUMPTION_FH = open(
                        _fr10_lo.environ.get(
                            "LUMO_TREE_DRAFT_CONSUMPTION_LOG",
                            "/logs/fr10_tree_draft_consumption.jsonl",
                        ),
                        "a",
                        buffering=1,
                    )
                _fr10_choices = _fr10_depth_choices.get(int(_fr10_depth), [])
                _fr10_parent_choices = (
                    [tuple()]
                    if int(_fr10_depth) == 1
                    else _fr10_depth_choices.get(int(_fr10_depth) - 1, [])
                )
                _fr10_parent_slot = {
                    tuple(_fr10_choice): int(_fr10_i)
                    for _fr10_i, _fr10_choice in enumerate(_fr10_parent_choices)
                }
                _fr10_top = torch.topk(
                    _fr10_level_logits, int(_fr10_level_num_children), dim=-1
                )
                for _fr10_choice in _fr10_choices:
                    _fr10_parent = tuple(_fr10_choice[:-1])
                    _fr10_rank = int(_fr10_choice[-1])
                    _fr10_p_slot = int(_fr10_parent_slot.get(_fr10_parent, -1))
                    _fr10_flat_col = (
                        _fr10_p_slot * int(_fr10_level_num_children) + _fr10_rank
                    )
                    if (
                        _fr10_p_slot < 0
                        or _fr10_rank >= int(_fr10_level_num_children)
                        or _fr10_flat_col >= int(_fr10_level_tokens.size(1))
                    ):
                        _LUMO_TREE_DRAFT_CONSUMPTION_FH.write(
                            _fr10_lj.dumps({
                                "event": "tree_draft_consumption",
                                "ts": _fr10_now,
                                "depth": int(_fr10_depth),
                                "path": [int(_x) for _x in _fr10_choice],
                                "parent_path": [int(_x) for _x in _fr10_parent],
                                "child_rank": int(_fr10_rank),
                                "runtime_slot": int(_fr10_runtime_slot_by_choice.get(_fr10_choice, -1)),
                                "placement_ok": False,
                                "reason": "choice_not_represented_by_level_topk",
                            }) + chr(10)
                        )
                        continue
                    for _fr10_b in range(int(batch_size)):
                        _fr10_row = (
                            int(_fr10_b) * int(_fr10_level_num_parents) + _fr10_p_slot
                        )
                        _fr10_placed = _fr10_level_tokens[_fr10_b, _fr10_flat_col]
                        _fr10_expected = _fr10_top.indices[_fr10_row, _fr10_rank]
                        _fr10_probs = torch.softmax(
                            _fr10_level_logits[_fr10_row].float(), dim=-1
                        )
                        _fr10_q = _fr10_probs[_fr10_placed.to(torch.long)]
                        _LUMO_TREE_DRAFT_CONSUMPTION_FH.write(
                            _fr10_lj.dumps({
                                "event": "tree_draft_consumption",
                                "ts": _fr10_now,
                                "req_index": int(_fr10_b),
                                "depth": int(_fr10_depth),
                                "path": [int(_x) for _x in _fr10_choice],
                                "parent_path": [int(_x) for _x in _fr10_parent],
                                "runtime_slot": int(_fr10_runtime_slot_by_choice.get(_fr10_choice, -1)),
                                "parent_runtime_slot": int(_fr10_runtime_slot_by_choice.get(_fr10_parent, -1)),
                                "parent_level_slot": int(_fr10_p_slot),
                                "child_rank": int(_fr10_rank),
                                "rank_kind": "top1" if int(_fr10_rank) == 0 else ("top2" if int(_fr10_rank) == 1 else "topN"),
                                "flat_level_col": int(_fr10_flat_col),
                                "placed_token": int(_fr10_placed.detach().cpu().item()),
                                "expected_topk_token": int(_fr10_expected.detach().cpu().item()),
                                "q_prob": float(_fr10_q.detach().cpu().item()),
                                "placement_ok": bool(
                                    int(_fr10_placed.detach().cpu().item())
                                    == int(_fr10_expected.detach().cpu().item())
                                ),
                            }) + chr(10)
                        )
            except Exception as _fr10_exc:
                _fr10_log_consumption_error(_fr10_exc)

        _fr10_record_tree_consumption(1, logits, draft_token_ids, 1, num_children)
        draft_hidden_states = hidden_states.view(batch_size, 1, -1)
"""
    tree_fn_pos = text.find("    def propose_tree(")
    if tree_fn_pos < 0:
        raise RuntimeError("EAGLE propose_tree function anchor not found")
    root_log_pos = text.find(root_log_anchor, tree_fn_pos)
    if root_log_pos < 0:
        raise RuntimeError("EAGLE propose_tree root consumption anchor not found")
    text = (
        text[:root_log_pos]
        + root_log_new
        + text[root_log_pos + len(root_log_anchor) :]
    )

    level_log_anchor = """            draft_token_ids_list.append(draft_token_ids)
"""
    level_log_new = """            draft_token_ids_list.append(draft_token_ids)
            _fr10_record_tree_consumption(
                level + 2, logits, draft_token_ids, level_num_drafts, num_children
            )
"""
    level_log_pos = text.find(level_log_anchor, root_log_pos + len(root_log_new))
    if level_log_pos < 0:
        raise RuntimeError("EAGLE propose_tree level consumption anchor not found")
    text = (
        text[:level_log_pos]
        + level_log_new
        + text[level_log_pos + len(level_log_anchor) :]
    )
    EAGLE_PATH.write_text(text)
    return True


def _patch_eagle_mtp_draft_trace() -> bool:
    text = EAGLE_PATH.read_text()
    sentinel = "# FR10_MTP_DRAFT_TRACE"
    if sentinel in text:
        return False
    patch = r'''

# FR10_MTP_DRAFT_TRACE: final drafter tensor trace for native-vs-tree parity.
import json as _fr10_mtp_trace_json
import os as _fr10_mtp_trace_os
import time as _fr10_mtp_trace_time

_fr10_mtp_trace_orig_propose = EagleProposer.propose
_fr10_mtp_trace_idx = 0

def _fr10_mtp_trace_propose(self, target_token_ids, target_positions,
                            target_hidden_states, next_token_ids,
                            token_indices_to_sample, common_attn_metadata,
                            sampling_metadata, mm_embed_inputs=None,
                            num_rejected_tokens_gpu=None,
                            slot_mappings=None):
    global _fr10_mtp_trace_idx
    out = _fr10_mtp_trace_orig_propose(
        self, target_token_ids, target_positions, target_hidden_states,
        next_token_ids, token_indices_to_sample, common_attn_metadata,
        sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu, slot_mappings)
    trace_path = _fr10_mtp_trace_os.environ.get("LUMO_MTP_DRAFT_TRACE_FILE")
    if trace_path or _fr10_mtp_trace_os.environ.get("FR10_METRICS", "0") == "1":
        try:
            if not trace_path:
                trace_path = "/logs/fr10_mtp_draft_trace.jsonl"
            global _FR10_MTP_DRAFT_TRACE_FH
            try:
                _FR10_MTP_DRAFT_TRACE_FH
            except NameError:
                _FR10_MTP_DRAFT_TRACE_FH = open(trace_path, "a", buffering=1)
            try:
                from vllm.v1.sample import rejection_sampler as _fr10_rs_mode
                mode = getattr(
                    _fr10_rs_mode,
                    "_FR10_DECODE_MODE",
                    _fr10_mtp_trace_os.environ.get("FR10_DECODE_MODE_DEFAULT", "tree_mtp"),
                )
            except Exception:
                mode = _fr10_mtp_trace_os.environ.get("FR10_DECODE_MODE_DEFAULT", "tree_mtp")
            _FR10_MTP_DRAFT_TRACE_FH.write(_fr10_mtp_trace_json.dumps({
                "event": "mtp_draft",
                "idx": int(_fr10_mtp_trace_idx),
                "ts": round(_fr10_mtp_trace_time.time(), 4),
                "mode": str(mode),
                "speculative_token_tree": _fr10_mtp_trace_os.environ.get("SPEC_CONFIG"),
                "shape": [int(x) for x in out.shape],
                "draft": out.detach().cpu().tolist(),
            }) + chr(10))
            _fr10_mtp_trace_idx += 1
        except Exception:
            pass
    return out

EagleProposer.propose = _fr10_mtp_trace_propose
'''
    EAGLE_PATH.write_text(text + patch)
    return True


def _patch_tree_attn_spec_config_override() -> bool:
    text = TREE_ATTN_PATH.read_text()
    sentinel = "# FR10_SPEC_CONFIG_TREE_OVERRIDE"
    if sentinel in text:
        return False
    text = text.replace("import ast\n", "import ast\nimport json\nimport os\n", 1)
    old = """        spec_token_tree: str | None = None
        if spec := spec_config:
            spec_token_tree = spec.speculative_token_tree
        tree_choices: list[tuple[int, ...]] = (
            ast.literal_eval(spec_token_tree) if spec_token_tree is not None else [(0,)]
        )
"""
    new = f"""        {sentinel}: keep attention tree identical to the FR10 launch descriptor.
        spec_token_tree: str | None = None
        try:
            spec_env = os.environ.get("SPEC_CONFIG")
            if spec_env:
                spec_token_tree = json.loads(spec_env).get("speculative_token_tree")
        except Exception:
            spec_token_tree = None
        if spec_token_tree is None and (spec := spec_config):
            spec_token_tree = spec.speculative_token_tree
        tree_choices: list[tuple[int, ...]] = (
            sorted(ast.literal_eval(spec_token_tree), key=lambda _p: (len(_p), _p)) if spec_token_tree is not None else [(0,)]
        )
"""
    if old not in text:
        raise RuntimeError("tree attention spec tree anchor not found")
    text = text.replace(old, new, 1)
    TREE_ATTN_PATH.write_text(text)
    return True


def _patch_mamba_postprocess_tree_rows() -> bool:
    text = MAMBA_UTILS_PATH.read_text()
    sentinel = "# LUMO_TREE_STATE_COMMIT_ROWS"
    if sentinel in text:
        return False

    copy_old = '''            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                copy_spec = state_copy_func(
                    state, block_ids, src_block_idx, accept_token_bias + 1
                )

                src_ptrs_np[offset] = copy_spec.start_addr
                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1
'''
    copy_new = '''            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                # LUMO_TREE_STATE_COMMIT_ROWS: tree-mode recurrent states are
                # materialized per accepted node row. The stock copy helpers
                # interpret accept_token_bias as a flat-MTP suffix offset and
                # add one, which would skip node 0 and shift every accepted row.
                copy_spec = None
                try:
                    from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr10_gdn
                    _fr10_rows = getattr(_fr10_gdn, "_LUMO_FA_LAST_ACCEPTED_TREE_ROWS", None)
                    _fr10_req_has_accept = bool(
                        globals().get("_LUMO_TREE_COMMIT_CURRENT_HAS_ACCEPT", False)
                    )
                    _fr10_tree_row_copy = _fr10_rows is not None and _fr10_req_has_accept
                    if _fr10_tree_row_copy:
                        src_state = state[block_ids[src_block_idx + accept_token_bias]]
                        copy_spec = MambaCopySpec(
                            start_addr=src_state.data_ptr(),
                            num_elements=src_state.numel(),
                        )
                except Exception:
                    copy_spec = None
                if copy_spec is None:
                    copy_spec = state_copy_func(
                        state, block_ids, src_block_idx, accept_token_bias + 1
                    )

                try:
                    if (
                        globals().get("_LUMO_TREE_COMMIT_CAPTURE_ACTIVE", False)
                        and not globals().get("_LUMO_TREE_COMMIT_CAPTURE_DONE", False)
                    ):
                        _fr10_kind = (
                            "conv"
                            if getattr(state_copy_func, "__name__", "") == "get_conv_copy_spec"
                            else "ssm"
                        )
                        _fr10_src_row = state[block_ids[src_block_idx + accept_token_bias]]
                        _fr10_specs = globals().setdefault(
                            "_LUMO_TREE_COMMIT_CAPTURE_SPECS", []
                        )
                        _fr10_specs.append({
                            "kind": _fr10_kind,
                            "state": state,
                            "src_snapshot": _fr10_src_row.detach().clone(),
                            "dest_block_id": int(dest_block_id),
                            "num_elements": int(copy_spec.num_elements),
                            "element_size": int(state.element_size()),
                            "meta": dict(globals().get("_LUMO_TREE_COMMIT_CAPTURE_META", {})),
                        })
                except Exception:
                    pass

                src_ptrs_np[offset] = copy_spec.start_addr
                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1
'''
    if copy_old not in text:
        raise RuntimeError("mamba tree state row copy helper anchor not found")
    text = text.replace(copy_old, copy_new, 1)

    preprocess_old = '''        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                prev_state_idx,
                curr_state_idx,
                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state,
                forward_context,
            )
            input_batch.num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)
'''
    preprocess_new = '''        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            _fr10_native_accept_token_bias = int(input_batch.num_accepted_tokens_cpu[i]) - 1
            _fr10_tree_row = 0
            _fr10_accept_token_bias = _fr10_native_accept_token_bias
            _fr10_has_accept = False
            # LUMO_TREE_STATE_COMMIT_ROWS: preprocessing moves the previous
            # step's running state into the current running block. For tree
            # verification, the source row must be the accepted runtime tree
            # node, not the flat accepted-count row.
            try:
                from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr10_gdn
                _fr10_rows = getattr(_fr10_gdn, "_LUMO_FA_LAST_ACCEPTED_TREE_ROWS", None)
                _fr10_lens = getattr(_fr10_gdn, "_LUMO_FA_LAST_ACCEPTED_TREE_LENS", None)
                if _fr10_rows is not None and i < len(_fr10_rows):
                    _fr10_tree_row = int(_fr10_rows[i])
                    _fr10_has_accept = (
                        _fr10_lens is not None
                        and i < len(_fr10_lens)
                        and int(_fr10_lens[i]) > 0
                    )
                    if _fr10_has_accept:
                        _fr10_accept_token_bias = int(_fr10_tree_row)
            except Exception:
                pass
            try:
                import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
                if _fr10_lo.environ.get("FR10_METRICS", "0") == "1":
                    global _LUMO_TREE_COMMIT_PARITY_FH
                    try:
                        _LUMO_TREE_COMMIT_PARITY_FH
                    except NameError:
                        _LUMO_TREE_COMMIT_PARITY_FH = open(
                            _fr10_lo.environ.get(
                                "LUMO_TREE_COMMIT_PARITY_LOG",
                                "/logs/fr10_commit_copy_parity.jsonl",
                            ),
                            "a",
                            buffering=1,
                        )
                    _LUMO_TREE_COMMIT_PARITY_FH.write(
                        _fr10_lj.dumps({
                            "event": "mamba_preprocess_commit_copy_bias",
                            "ts": round(_fr10_lt.time(), 4),
                            "req_index": int(i),
                            "req_id": str(req_id),
                            "native_accept_token_bias": int(_fr10_native_accept_token_bias),
                            "tree_accepted_row": int(_fr10_tree_row),
                            "effective_accept_token_bias": int(_fr10_accept_token_bias),
                            "collect_num_accepted_tokens": int(_fr10_accept_token_bias) + 1,
                            "src_block_idx": int(prev_state_idx),
                            "dest_block_idx": int(curr_state_idx),
                            "accepted_len": int(_fr10_lens[i]) if _fr10_lens is not None and i < len(_fr10_lens) else None,
                            "row_redirect_active": bool(_fr10_has_accept),
                        }) + chr(10)
                    )
            except Exception:
                pass
            try:
                import os as _fr10_lo2
                _fr10_capture_commit = (
                    _fr10_lo2.environ.get("FR10_TREE_GDN_CAPTURE_PAYLOAD")
                    and bool(_fr10_has_accept)
                    and not globals().get("_LUMO_TREE_COMMIT_CAPTURE_DONE", False)
                )
                globals()["_LUMO_TREE_COMMIT_CAPTURE_ACTIVE"] = bool(_fr10_capture_commit)
                globals()["_LUMO_TREE_COMMIT_CAPTURE_META"] = {
                    "phase": "preprocess",
                    "req_index": int(i),
                    "req_id": str(req_id),
                    "native_accept_token_bias": int(_fr10_native_accept_token_bias),
                    "tree_accepted_row": int(_fr10_tree_row),
                    "effective_accept_token_bias": int(_fr10_accept_token_bias),
                    "src_block_idx": int(prev_state_idx),
                    "dest_block_idx": int(curr_state_idx),
                }
            except Exception:
                globals()["_LUMO_TREE_COMMIT_CAPTURE_ACTIVE"] = False
            globals()["_LUMO_TREE_COMMIT_CURRENT_HAS_ACCEPT"] = bool(_fr10_has_accept)
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                prev_state_idx,
                curr_state_idx,
                _fr10_accept_token_bias,
                req_state,
                forward_context,
            )
            globals()["_LUMO_TREE_COMMIT_CURRENT_HAS_ACCEPT"] = False
            globals()["_LUMO_TREE_COMMIT_CAPTURE_ACTIVE"] = False
            input_batch.num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)
    try:
        _fr10_specs = globals().pop("_LUMO_TREE_COMMIT_CAPTURE_SPECS", [])
        globals()["_LUMO_TREE_COMMIT_CAPTURE_ACTIVE"] = False
        if _fr10_specs and not globals().get("_LUMO_TREE_COMMIT_CAPTURE_DONE", False):
            import json as _fr10_lj2, os as _fr10_lo3, time as _fr10_lt2
            torch.cuda.synchronize()
            _fr10_by_kind = {}
            _fr10_rows = []
            for _fr10_spec in _fr10_specs:
                _fr10_kind = str(_fr10_spec.get("kind", "unknown"))
                _fr10_state = _fr10_spec["state"]
                _fr10_dest = _fr10_state[int(_fr10_spec["dest_block_id"])]
                _fr10_src = _fr10_spec["src_snapshot"]
                _fr10_exact = bool(torch.equal(_fr10_dest, _fr10_src))
                _fr10_max = float((_fr10_dest.float() - _fr10_src.float()).abs().max().item())
                _fr10_by_kind.setdefault(_fr10_kind, {
                    "rows": 0,
                    "bit_exact_all": True,
                    "max_abs": 0.0,
                })
                _fr10_by_kind[_fr10_kind]["rows"] += 1
                _fr10_by_kind[_fr10_kind]["bit_exact_all"] = (
                    bool(_fr10_by_kind[_fr10_kind]["bit_exact_all"]) and _fr10_exact
                )
                _fr10_by_kind[_fr10_kind]["max_abs"] = max(
                    float(_fr10_by_kind[_fr10_kind]["max_abs"]), _fr10_max
                )
                if len(_fr10_rows) < 8:
                    _fr10_row = {
                        "kind": _fr10_kind,
                        "bit_exact": _fr10_exact,
                        "max_abs": _fr10_max,
                        "num_elements": int(_fr10_spec.get("num_elements", 0)),
                        "element_size": int(_fr10_spec.get("element_size", 0)),
                    }
                    _fr10_row.update(dict(_fr10_spec.get("meta", {})))
                    _fr10_rows.append(_fr10_row)
            _fr10_payload = {
                "schema": "fr10.commit_state_capture.v1",
                "event": "mamba_commit_state_copy_capture",
                "ts": round(_fr10_lt2.time(), 4),
                "by_kind": _fr10_by_kind,
                "sample_rows": _fr10_rows,
            }
            _fr10_path = _fr10_lo3.environ.get(
                "LUMO_TREE_COMMIT_STATE_CAPTURE_LOG",
                "/logs/fr10_commit_state_capture.jsonl",
            )
            with open(_fr10_path, "a", buffering=1) as _fr10_fh:
                _fr10_fh.write(_fr10_lj2.dumps(_fr10_payload) + chr(10))
            globals()["_LUMO_TREE_COMMIT_CAPTURE_DONE"] = True
    except Exception:
        pass
'''
    if preprocess_old not in text:
        raise RuntimeError("mamba preprocess accepted-row copy anchor not found")
    text = text.replace(preprocess_old, preprocess_new, 1)

    old = '''        if aligned_new_computed_tokens >= num_tokens_running_state:
            accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
            src_block_idx = mamba_state_idx[req_id]
            dest_block_idx = aligned_new_computed_tokens // mamba_spec.block_size - 1
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                src_block_idx,
                dest_block_idx,
                accept_token_bias,
                req_state,
                forward_context,
            )
            if src_block_idx == dest_block_idx:
                num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)
'''
    new = '''        if aligned_new_computed_tokens >= num_tokens_running_state:
            accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
            _fr10_native_accept_token_bias = accept_token_bias
            _fr10_tree_row = 0
            _fr10_has_accept = False
            # LUMO_TREE_STATE_COMMIT_ROWS: for FR10 tree verification, the
            # accepted final recurrent state lives at the accepted tree node
            # row, not at the flat linear accepted-count row. The scheduler's
            # accepted-token accounting remains unchanged; only the source row
            # used for the state copy is redirected.
            try:
                from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr10_gdn
                _fr10_rows = getattr(_fr10_gdn, "_LUMO_FA_LAST_ACCEPTED_TREE_ROWS", None)
                _fr10_lens = getattr(_fr10_gdn, "_LUMO_FA_LAST_ACCEPTED_TREE_LENS", None)
                if _fr10_rows is not None and i < len(_fr10_rows):
                    _fr10_row = int(_fr10_rows[i])
                    _fr10_tree_row = int(_fr10_row)
                    _fr10_has_accept = (
                        _fr10_lens is not None
                        and i < len(_fr10_lens)
                        and int(_fr10_lens[i]) > 0
                    )
                    if _fr10_has_accept:
                        accept_token_bias = _fr10_row
            except Exception:
                pass
            src_block_idx = mamba_state_idx[req_id]
            dest_block_idx = aligned_new_computed_tokens // mamba_spec.block_size - 1
            try:
                import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
                if _fr10_lo.environ.get("FR10_METRICS", "0") == "1":
                    global _LUMO_TREE_COMMIT_PARITY_FH
                    try:
                        _LUMO_TREE_COMMIT_PARITY_FH
                    except NameError:
                        _LUMO_TREE_COMMIT_PARITY_FH = open(
                            _fr10_lo.environ.get(
                                "LUMO_TREE_COMMIT_PARITY_LOG",
                                "/logs/fr10_commit_copy_parity.jsonl",
                            ),
                            "a",
                            buffering=1,
                        )
                    _LUMO_TREE_COMMIT_PARITY_FH.write(
                        _fr10_lj.dumps({
                            "event": "mamba_commit_copy_bias",
                            "ts": round(_fr10_lt.time(), 4),
                            "req_index": int(i),
                            "req_id": str(req_id),
                            "num_accepted_tokens": int(num_accepted_tokens),
                            "num_draft_tokens": int(num_draft_tokens),
                            "native_accept_token_bias": int(_fr10_native_accept_token_bias),
                            "tree_accepted_row": int(_fr10_tree_row),
                            "effective_accept_token_bias": int(accept_token_bias),
                            "collect_num_accepted_tokens": int(accept_token_bias) + 1,
                            "src_block_idx": int(src_block_idx),
                            "dest_block_idx": int(dest_block_idx),
                            "accepted_len": int(_fr10_lens[i]) if _fr10_lens is not None and i < len(_fr10_lens) else None,
                            "row_redirect_active": bool(_fr10_has_accept),
                        }) + chr(10)
                    )
            except Exception:
                pass
            try:
                import os as _fr10_lo2
                _fr10_capture_commit = (
                    _fr10_lo2.environ.get("FR10_TREE_GDN_CAPTURE_PAYLOAD")
                    and bool(_fr10_has_accept)
                    and not globals().get("_LUMO_TREE_COMMIT_CAPTURE_DONE", False)
                )
                globals()["_LUMO_TREE_COMMIT_CAPTURE_ACTIVE"] = bool(_fr10_capture_commit)
                globals()["_LUMO_TREE_COMMIT_CAPTURE_META"] = {
                    "req_index": int(i),
                    "req_id": str(req_id),
                    "native_accept_token_bias": int(_fr10_native_accept_token_bias),
                    "tree_accepted_row": int(_fr10_tree_row),
                    "effective_accept_token_bias": int(accept_token_bias),
                    "src_block_idx": int(src_block_idx),
                    "dest_block_idx": int(dest_block_idx),
                }
            except Exception:
                globals()["_LUMO_TREE_COMMIT_CAPTURE_ACTIVE"] = False
            globals()["_LUMO_TREE_COMMIT_CURRENT_HAS_ACCEPT"] = bool(_fr10_has_accept)
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                src_block_idx,
                dest_block_idx,
                accept_token_bias,
                req_state,
                forward_context,
            )
            globals()["_LUMO_TREE_COMMIT_CURRENT_HAS_ACCEPT"] = False
            globals()["_LUMO_TREE_COMMIT_CAPTURE_ACTIVE"] = False
            if src_block_idx == dest_block_idx:
                num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)
    try:
        _fr10_specs = globals().pop("_LUMO_TREE_COMMIT_CAPTURE_SPECS", [])
        globals()["_LUMO_TREE_COMMIT_CAPTURE_ACTIVE"] = False
        if _fr10_specs and not globals().get("_LUMO_TREE_COMMIT_CAPTURE_DONE", False):
            import json as _fr10_lj2, os as _fr10_lo3, time as _fr10_lt2
            torch.cuda.synchronize()
            _fr10_by_kind = {}
            _fr10_rows = []
            for _fr10_spec in _fr10_specs:
                _fr10_kind = str(_fr10_spec.get("kind", "unknown"))
                _fr10_state = _fr10_spec["state"]
                _fr10_dest = _fr10_state[int(_fr10_spec["dest_block_id"])]
                _fr10_src = _fr10_spec["src_snapshot"]
                _fr10_exact = bool(torch.equal(_fr10_dest, _fr10_src))
                _fr10_max = float((_fr10_dest.float() - _fr10_src.float()).abs().max().item())
                _fr10_by_kind.setdefault(_fr10_kind, {
                    "rows": 0,
                    "bit_exact_all": True,
                    "max_abs": 0.0,
                })
                _fr10_by_kind[_fr10_kind]["rows"] += 1
                _fr10_by_kind[_fr10_kind]["bit_exact_all"] = (
                    bool(_fr10_by_kind[_fr10_kind]["bit_exact_all"]) and _fr10_exact
                )
                _fr10_by_kind[_fr10_kind]["max_abs"] = max(
                    float(_fr10_by_kind[_fr10_kind]["max_abs"]), _fr10_max
                )
                if len(_fr10_rows) < 8:
                    _fr10_row = {
                        "kind": _fr10_kind,
                        "bit_exact": _fr10_exact,
                        "max_abs": _fr10_max,
                        "num_elements": int(_fr10_spec.get("num_elements", 0)),
                        "element_size": int(_fr10_spec.get("element_size", 0)),
                    }
                    _fr10_row.update(dict(_fr10_spec.get("meta", {})))
                    _fr10_rows.append(_fr10_row)
            _fr10_payload = {
                "schema": "fr10.commit_state_capture.v1",
                "event": "mamba_commit_state_copy_capture",
                "ts": round(_fr10_lt2.time(), 4),
                "by_kind": _fr10_by_kind,
                "sample_rows": _fr10_rows,
            }
            _fr10_path = _fr10_lo3.environ.get(
                "LUMO_TREE_COMMIT_STATE_CAPTURE_LOG",
                "/logs/fr10_commit_state_capture.jsonl",
            )
            with open(_fr10_path, "a", buffering=1) as _fr10_fh:
                _fr10_fh.write(_fr10_lj2.dumps(_fr10_payload) + chr(10))
            globals()["_LUMO_TREE_COMMIT_CAPTURE_DONE"] = True
    except Exception:
        pass
'''
    if old not in text:
        raise RuntimeError("mamba postprocess accepted-row copy anchor not found")
    text = text.replace(old, new, 1)
    MAMBA_UTILS_PATH.write_text(text)
    return True


def main() -> int:
    patch_steps = [
        (REQUEST_PATH, _patch_request_decode_mode()),
        (SCHED_OUTPUT_PATH, _patch_sched_output_decode_mode()),
        (SCHEDULER_PATH, _patch_scheduler_decode_modes()),
        (GDN_ATTN_PATH, _patch_gdn_attn()),
        (GDN_LINEAR_PATH, _patch_gdn_linear()),
        (SCHEDULER_PATH, _patch_scheduler_spec_trace()),
        (EAGLE_PATH, _patch_eagle_tree_consumption_verify()),
        (EAGLE_PATH, _patch_eagle_mtp_draft_trace()),
        (TREE_ATTN_PATH, _patch_tree_attn_spec_config_override()),
        (GPU_MODEL_RUNNER_PATH, _patch_gpu_model_runner_tree_metadata()),
        (GPU_MODEL_RUNNER_PATH, _patch_gpu_model_runner_decode_mode_globals()),
        (MAMBA_UTILS_PATH, _patch_mamba_postprocess_tree_rows()),
        (REJECTION_SAMPLER_PATH, _patch_rejection_sampler_tree_lcp()),
    ]
    patched: dict[str, bool] = {}
    for path, did_patch in patch_steps:
        patched[str(path)] = patched.get(str(path), False) or did_patch
    import py_compile

    for path, did_patch in patched.items():
        if did_patch:
            py_compile.compile(path, doraise=True)
    print(patched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

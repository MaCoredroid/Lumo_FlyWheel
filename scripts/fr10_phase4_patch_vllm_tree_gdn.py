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
MAMBA_UTILS_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/mamba_utils.py"
)
QWEN3_NEXT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"
)
QWEN3_5_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py"
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
            "    fr10_tree_accepted_paths: torch.Tensor | None = None\n"
            "    fr10_tree_accepted_lens: torch.Tensor | None = None\n"
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
            "        self.fr10_tree_accepted_path_bs = max(\n"
            "            int(self.decode_cudagraph_max_bs),\n"
            "            int(self.vllm_config.scheduler_config.max_num_seqs),\n"
            "        )\n"
            "        self.fr10_tree_accepted_paths = torch.zeros(\n"
            "            (self.fr10_tree_accepted_path_bs, self.num_spec + 1),\n"
            "            dtype=torch.int32,\n"
            "            device=device,\n"
            "        )\n"
            "        self.fr10_tree_accepted_lens = torch.zeros(\n"
            "            (self.fr10_tree_accepted_path_bs,),\n"
            "            dtype=torch.int32,\n"
            "            device=device,\n"
            "        )\n"
            "        try:\n"
            "            from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr10_gdn_linear\n"
            "            _fr10_gdn_linear._LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR = self.fr10_tree_accepted_paths\n"
            "            _fr10_gdn_linear._LUMO_FA_ACCEPTED_TREE_LENS_TENSOR = self.fr10_tree_accepted_lens\n"
            "        except Exception as _fr10_path_buf_exc:\n"
            "            if os.environ.get(\"FR10_ALLOW_LINEAR_FALLBACK\", \"0\") != \"1\":\n"
            "                raise RuntimeError(\"FR10 tree accepted-path buffer export failed: \" + type(_fr10_path_buf_exc).__name__ + \":\" + str(_fr10_path_buf_exc)) from _fr10_path_buf_exc\n"
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
            "from lumo_flywheel_serving.fr10_gdn_tree_kernel import launch_tree_gdn_prepared, launch_tree_state_linear_remap\n"
            "\n"
            "_FR10_DECODE_MODE = os.environ.get(\"FR10_DECODE_MODE_DEFAULT\", \"tree_mtp\")\n"
            "_FR12_SUBKERNEL_CAPTURE_ACTIVE = {}\n"
            "\n"
            "\n"
            "def _fr12_subkernel_capture_get(self, num_tokens=None, create=False):\n"
            "    path = os.environ.get(\"FR12_SUBKERNEL_CAPTURE\")\n"
            "    if not path:\n"
            "        return None\n"
            "    try:\n"
            "        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():\n"
            "            return None\n"
            "    except Exception:\n"
            "        return None\n"
            "    prefix = str(getattr(self, \"prefix\", \"\"))\n"
            "    want_prefix = os.environ.get(\n"
            "        \"FR12_SUBKERNEL_CAPTURE_LAYER_PREFIX\",\n"
            "        \"language_model.model.layers.0.linear_attn\",\n"
            "    )\n"
            "    if want_prefix and prefix != want_prefix:\n"
            "        return None\n"
            "    active = _FR12_SUBKERNEL_CAPTURE_ACTIVE.get(prefix)\n"
            "    if active is not None:\n"
            "        return active\n"
            "    if not create:\n"
            "        return None\n"
            "    if num_tokens is not None:\n"
            "        desired = os.environ.get(\"FR12_SUBKERNEL_CAPTURE_NUM_TOKENS\")\n"
            "        if desired:\n"
            "            desired_counts = {\n"
            "                int(_x.strip()) for _x in desired.split(\",\") if _x.strip()\n"
            "            }\n"
            "            if int(num_tokens) not in desired_counts:\n"
            "                return None\n"
            "    seen = int(globals().get(\"_FR12_SUBKERNEL_CAPTURE_SEEN\", 0))\n"
            "    skip = int(os.environ.get(\"FR12_SUBKERNEL_CAPTURE_SKIP\", \"0\"))\n"
            "    limit = int(os.environ.get(\"FR12_SUBKERNEL_CAPTURE_LIMIT\", \"1\"))\n"
            "    saved = int(globals().get(\"_FR12_SUBKERNEL_CAPTURE_SAVED\", 0))\n"
            "    globals()[\"_FR12_SUBKERNEL_CAPTURE_SEEN\"] = seen + 1\n"
            "    if seen < skip or saved >= limit:\n"
            "        return None\n"
            "    root, ext = os.path.splitext(path)\n"
            "    call_path = root + \".call\" + str(saved) + (ext or \".pt\")\n"
            "    payload = {\n"
            "        \"schema\": \"fr12.gdn_l0_subkernel_capture.v1\",\n"
            "        \"path\": path,\n"
            "        \"call_path\": call_path,\n"
            "        \"capture_call_index\": int(seen),\n"
            "        \"capture_saved_index\": int(saved),\n"
            "        \"layer_prefix\": prefix,\n"
            "        \"num_tokens\": None if num_tokens is None else int(num_tokens),\n"
            "        \"stages\": {},\n"
            "        \"meta\": {},\n"
            "    }\n"
            "    _FR12_SUBKERNEL_CAPTURE_ACTIVE[prefix] = payload\n"
            "    return payload\n"
            "\n"
            "\n"
            "def _fr12_subkernel_capture_flush(payload, final=False):\n"
            "    if payload is None:\n"
            "        return\n"
            "    try:\n"
            "        out = str(payload[\"call_path\"])\n"
            "        parent = os.path.dirname(out)\n"
            "        if parent:\n"
            "            os.makedirs(parent, exist_ok=True)\n"
            "        torch.save(payload, out)\n"
            "        if int(payload.get(\"capture_saved_index\", 0)) == 0:\n"
            "            torch.save(payload, str(payload[\"path\"]))\n"
            "        if final:\n"
            "            globals()[\"_FR12_SUBKERNEL_CAPTURE_SAVED\"] = (\n"
            "                int(globals().get(\"_FR12_SUBKERNEL_CAPTURE_SAVED\", 0)) + 1\n"
            "            )\n"
            "            _FR12_SUBKERNEL_CAPTURE_ACTIVE.pop(\n"
            "                str(payload.get(\"layer_prefix\", \"\")), None\n"
            "            )\n"
            "    except Exception as exc:\n"
            "        logger.warning(\"FR12 subkernel capture flush failed: %s\", exc)\n"
            "\n"
            "\n"
            "def _fr12_subkernel_capture_tensor(self, name, tensor, create=False, extra=None):\n"
            "    if tensor is None:\n"
            "        return\n"
            "    try:\n"
            "        if extra and extra.get(\"num_actual_tokens\") is not None:\n"
            "            num_tokens = int(extra.get(\"num_actual_tokens\"))\n"
            "        else:\n"
            "            num_tokens = int(tensor.shape[0]) if tensor.ndim > 0 else None\n"
            "        payload = _fr12_subkernel_capture_get(\n"
            "            self, num_tokens=num_tokens, create=create\n"
            "        )\n"
            "        if payload is None:\n"
            "            return\n"
            "        item = {\n"
            "            \"shape\": [int(_x) for _x in tensor.shape],\n"
            "            \"dtype\": str(tensor.dtype),\n"
            "            \"tensor\": tensor.detach().to(torch.float32).cpu(),\n"
            "        }\n"
            "        if extra:\n"
            "            item[\"extra\"] = extra\n"
            "        payload[\"stages\"][name] = item\n"
            "        _fr12_subkernel_capture_flush(payload)\n"
            "    except Exception as exc:\n"
            "        logger.warning(\"FR12 subkernel capture stage %s failed: %s\", name, exc)\n"
            "\n"
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
            _fr10_tree_conv_expected = (
                _fr10_active_decode_mode == "tree_mtp"
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
            _fr12_pre_conv_spec = mixed_qkv_spec
            try:
                _fr12_pre_extra = {
                    "num_spec_decodes": int(attn_metadata.num_spec_decodes),
                    "num_actual_tokens": int(num_actual_tokens),
                    "tree_conv_active": bool(use_fr10_tree_conv),
                    "tree_conv_expected": bool(_fr10_tree_conv_expected),
                }
                if getattr(attn_metadata, "fr10_tree_parent", None) is not None:
                    _fr12_pre_extra["tree_parent"] = [
                        int(_x)
                        for _x in attn_metadata.fr10_tree_parent.detach().cpu().tolist()
                    ]
                if spec_token_indx is not None:
                    _fr12_pre_extra["spec_token_indx"] = [
                        int(_x) for _x in spec_token_indx.detach().cpu().tolist()
                    ]
                if spec_query_start_loc is not None:
                    _fr12_pre_extra["spec_query_start_loc"] = [
                        int(_x) for _x in spec_query_start_loc.detach().cpu().tolist()
                    ]
                _fr12_subkernel_capture_tensor(
                    self,
                    "pre_conv",
                    mixed_qkv_spec,
                    create=True,
                    extra=_fr12_pre_extra,
                )
            except Exception as _fr12_pre_cap_exc:
                logger.warning("FR12 pre-conv capture failed: %s", _fr12_pre_cap_exc)
            if use_fr10_tree_conv:
                _fr12_native_prior_read = (
                    os.environ.get("FR12_TREE_CONV_NATIVE_PRIOR_READ", "0") == "1"
                )
                _fr12_native_prior_conv_bank_rows = None
                _fr12_native_prior_conv_state_bank = None
                if _fr12_native_prior_read:
                    _fr12_native_prior_conv_bank_rows = spec_state_indices_tensor[
                        : attn_metadata.num_spec_decodes, 0
                    ].to(torch.long).view(-1, 1)
                    _fr12_native_prior_conv_state_bank = torch.index_select(
                        conv_state,
                        0,
                        _fr12_native_prior_conv_bank_rows.reshape(-1),
                    )
                try:
                    _fr10_accepted_paths_tensor = globals().get(
                        "_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR"
                    )
                    _fr10_accepted_lens_tensor = globals().get(
                        "_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR"
                    )
                    if _fr10_accepted_paths_tensor is None:
                        raise RuntimeError("missing_accepted_path_device_tensor")
                    if _fr10_accepted_lens_tensor is None:
                        raise RuntimeError("missing_accepted_lens_device_tensor")
                    if os.environ.get("FR10_METRICS", "0") == "1":
                        try:
                            import time as _fr10_len_time

                            _fr10_len_log = os.environ.get(
                                "FR10_TREE_GDN_COMMIT_HANDOFF_LOG"
                            )
                            _fr10_len_count = int(
                                globals().get(
                                    "_FR10_TREE_LENGTH_ALIGNMENT_LOG_COUNT", 0
                                )
                            )
                            _fr10_len_limit = int(
                                os.environ.get(
                                    "FR10_TREE_GDN_COMMIT_HANDOFF_LIMIT", "32"
                                )
                            )
                            if _fr10_len_log and _fr10_len_count < _fr10_len_limit:
                                _fr10_len_rows = []
                                for _fr10_len_b in range(
                                    int(attn_metadata.num_spec_decodes)
                                ):
                                    _fr10_meta_len = None
                                    if num_accepted_tokens is not None:
                                        _fr10_meta_len = int(
                                            num_accepted_tokens[_fr10_len_b]
                                            .detach()
                                            .cpu()
                                            .item()
                                        )
                                    _fr10_path_len = int(
                                        _fr10_accepted_lens_tensor[_fr10_len_b]
                                        .detach()
                                        .cpu()
                                        .item()
                                    )
                                    _fr10_len_rows.append(
                                        {
                                            "batch_index": int(_fr10_len_b),
                                            "metadata_num_accepted_tokens": _fr10_meta_len,
                                            "accepted_tree_len": int(_fr10_path_len),
                                            "metadata_read_col": (
                                                None
                                                if _fr10_meta_len is None
                                                else max(0, int(_fr10_meta_len) - 1)
                                            ),
                                            "accepted_len_read_col": max(
                                                0, int(_fr10_path_len) - 1
                                            ),
                                        }
                                    )
                                with open(_fr10_len_log, "a", buffering=1) as _fr10_fh:
                                    _fr10_fh.write(
                                        json.dumps(
                                            {
                                                "schema": "fr10.length_alignment.v1",
                                                "event": "tree_length_alignment",
                                                "ts": round(_fr10_len_time.time(), 4),
                                                "layer_prefix": str(self.prefix),
                                                "rows": _fr10_len_rows,
                                            }
                                        )
                                        + chr(10)
                                    )
                                globals()[
                                    "_FR10_TREE_LENGTH_ALIGNMENT_LOG_COUNT"
                                ] = _fr10_len_count + 1
                        except Exception as _fr10_len_exc:
                            if os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1":
                                raise RuntimeError(
                                    "FR10 tree length-alignment logging failed: "
                                    + type(_fr10_len_exc).__name__
                                    + ":"
                                    + str(_fr10_len_exc)
                                ) from _fr10_len_exc
                    launch_tree_state_linear_remap(
                        ssm_state=ssm_state,
                        conv_state=conv_state,
                        spec_state_indices=spec_state_indices_tensor,
                        accepted_paths=_fr10_accepted_paths_tensor,
                        num_accepted_tokens=_fr10_accepted_lens_tensor,
                        num_spec_decodes=int(attn_metadata.num_spec_decodes),
                        max_path_len=int(spec_state_indices_tensor.size(-1)),
                    )
                except Exception as _fr10_seed_conv_exc:
                    if (
                        _fr10_tree_conv_expected
                        and os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1"
                    ):
                        raise RuntimeError(
                            "FR10 tree state linear remap failed: "
                            + type(_fr10_seed_conv_exc).__name__
                            + ":"
                            + str(_fr10_seed_conv_exc)
                        ) from _fr10_seed_conv_exc
                if _fr12_native_prior_read:
                    _fr10_conv_read_cols = torch.zeros(
                        (int(attn_metadata.num_spec_decodes), 1),
                        dtype=torch.long,
                        device=spec_state_indices_tensor.device,
                    )
                    _fr10_prior_conv_bank_rows = _fr12_native_prior_conv_bank_rows
                    _fr10_prior_conv_state_bank = _fr12_native_prior_conv_state_bank
                else:
                    if _fr10_accepted_lens_tensor is None:
                        _fr10_conv_read_cols = torch.zeros(
                            (int(attn_metadata.num_spec_decodes), 1),
                            dtype=torch.long,
                            device=spec_state_indices_tensor.device,
                        )
                    else:
                        _fr10_conv_read_cols = torch.clamp(
                            _fr10_accepted_lens_tensor[
                                : attn_metadata.num_spec_decodes
                            ].to(torch.long)
                            - 1,
                            min=0,
                            max=int(spec_state_indices_tensor.size(-1)) - 1,
                        ).view(-1, 1)
                    _fr10_prior_conv_bank_rows = spec_state_indices_tensor[
                        : attn_metadata.num_spec_decodes
                    ].gather(1, _fr10_conv_read_cols)
                    _fr10_prior_conv_state_bank = torch.index_select(
                        conv_state,
                        0,
                        _fr10_prior_conv_bank_rows.reshape(-1).to(torch.long),
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
                    _fr12_native_prior_col_base = torch.arange(
                        max(0, int(conv_state.size(2)) - (_fr10_width - 1)),
                        int(conv_state.size(2)),
                        dtype=torch.long,
                        device=mixed_qkv_spec.device,
                    )
                    _fr11_native_bf16_taps = (
                        os.environ.get("FR11_TREE_CONV_NATIVE_BF16_TAPS", "0") == "1"
                    )
                    _fr10_weight_f = conv_weights.to(torch.float32)
                    def _fr11_conv_tap_product(_fr11_x, _fr11_w):
                        if _fr11_native_bf16_taps:
                            _fr11_dtype = mixed_qkv_spec.dtype
                            _fr11_w_cast = _fr11_w.to(_fr11_dtype)
                            if _fr11_x.ndim == 2:
                                _fr11_w_cast = _fr11_w_cast.unsqueeze(0)
                            return (
                                _fr11_x.to(_fr11_dtype) * _fr11_w_cast
                            ).to(_fr11_dtype).to(torch.float32)
                        _fr11_w_f = _fr11_w.to(torch.float32)
                        if _fr11_x.ndim == 2:
                            _fr11_w_f = _fr11_w_f.unsqueeze(0)
                        return _fr11_x.to(torch.float32) * _fr11_w_f
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
                        if _fr12_native_prior_read:
                            _fr10_prior_cols = _fr12_native_prior_col_base
                        else:
                            _fr10_prior_cols = _fr10_prior_col_base
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
                        _fr10_path0_x = _fr10_x.index_select(
                            0, _fr10_path0_node_tensor
                        )
                        if _fr10_b == 0:
                            try:
                                _fr12_payload = _fr12_subkernel_capture_get(
                                    self, create=False
                                )
                                if (
                                    _fr12_payload is not None
                                    and not _fr12_payload["meta"].get(
                                        "tree_conv_detail_done", False
                                    )
                                ):
                                    _fr12_path0_window = _fr10_window.index_select(
                                        0, _fr10_path0_node_tensor
                                    )
                                    _fr12_path0_source_indices = (
                                        _fr10_tree_source_indices.index_select(
                                            0, _fr10_path0_node_tensor
                                        )
                                    )
                                    _fr12_path0_native_chain_indices = (
                                        _fr10_path0_source_indices
                                    )
                                    _fr12_taps_fp32 = []
                                    _fr12_taps_bf16 = []
                                    _fr12_tap_dtype = mixed_qkv_spec.dtype
                                    for _fr12_col in range(_fr10_width):
                                        _fr12_w_fp32 = conv_weights[
                                            :, _fr12_col
                                        ].to(torch.float32).unsqueeze(0)
                                        _fr12_w_bf16 = conv_weights[
                                            :, _fr12_col
                                        ].to(_fr12_tap_dtype).unsqueeze(0)
                                        _fr12_x_col = _fr12_path0_window[
                                            :, _fr12_col, :
                                        ]
                                        _fr12_taps_fp32.append(
                                            _fr12_x_col.to(torch.float32)
                                            * _fr12_w_fp32
                                        )
                                        _fr12_taps_bf16.append(
                                            (
                                                _fr12_x_col.to(_fr12_tap_dtype)
                                                * _fr12_w_bf16
                                            )
                                            .to(_fr12_tap_dtype)
                                            .to(torch.float32)
                                        )
                                    _fr12_payload["meta"]["tree_conv_detail"] = {
                                        "schema": "fr12.tree_conv_detail.v1",
                                        "prior_bank_rows": _fr10_prior_conv_bank_rows.detach()
                                        .cpu()
                                        .clone(),
                                        "read_cols": _fr10_conv_read_cols.detach()
                                        .cpu()
                                        .clone(),
                                        "prior_read_mode": (
                                            "native_tail_pre_remap"
                                            if _fr12_native_prior_read
                                            else "legacy_remapped_head"
                                        ),
                                        "prior_cols": _fr10_prior_cols.detach()
                                        .cpu()
                                        .clone(),
                                        "path0_nodes": _fr10_path0_node_tensor.detach()
                                        .cpu()
                                        .clone(),
                                        "tree_source_indices_path0": _fr12_path0_source_indices.detach()
                                        .cpu()
                                        .clone(),
                                        "native_chain_source_indices_path0": _fr12_path0_native_chain_indices.detach()
                                        .cpu()
                                        .clone(),
                                        "prior_window": _fr10_prior_window.detach()
                                        .to(torch.float32)
                                        .cpu()
                                        .clone(),
                                        "pre_conv_path0": _fr10_path0_x.detach()
                                        .to(torch.float32)
                                        .cpu()
                                        .clone(),
                                        "window_path0": _fr12_path0_window.detach()
                                        .to(torch.float32)
                                        .cpu()
                                        .clone(),
                                        "tap_products_fp32_path0": torch.stack(
                                            _fr12_taps_fp32, dim=1
                                        )
                                        .detach()
                                        .cpu()
                                        .clone(),
                                        "tap_products_bf16_path0": torch.stack(
                                            _fr12_taps_bf16, dim=1
                                        )
                                        .detach()
                                        .cpu()
                                        .clone(),
                                    }
                                    _fr12_payload["meta"][
                                        "tree_conv_detail_done"
                                    ] = True
                                    _fr12_subkernel_capture_flush(_fr12_payload)
                            except Exception as _fr12_tree_detail_exc:
                                logger.warning(
                                    "FR12 tree conv detail capture failed: %s",
                                    _fr12_tree_detail_exc,
                                )
                        if self.conv1d.bias is None:
                            _fr10_acc = torch.zeros_like(_fr10_x, dtype=torch.float32)
                        else:
                            _fr10_acc = self.conv1d.bias.to(torch.float32).unsqueeze(
                                0
                            ).expand_as(_fr10_x.float()).clone()
                        for _fr10_col in range(_fr10_width):
                            _fr10_acc = _fr10_acc + _fr11_conv_tap_product(
                                _fr10_window[:, _fr10_col, :],
                                conv_weights[:, _fr10_col],
                            )
                        if self.activation in (True, "silu", "swish"):
                            _fr10_acc = torch.nn.functional.silu(_fr10_acc)
                        _fr10_out = _fr10_acc.to(dtype=mixed_qkv_spec.dtype)
                        _fr10_tree_conv_out[_fr10_start:_fr10_end] = _fr10_out
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
                                _fr10_path0_acc = (
                                    _fr10_path0_acc
                                    + _fr11_conv_tap_product(
                                        _fr10_path0_window[:, _fr10_col, :],
                                        conv_weights[:, _fr10_col],
                                    )
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
                                _fr10_flat_acc = (
                                    _fr10_flat_acc
                                    + _fr11_conv_tap_product(
                                        _fr10_flat_window[:, _fr10_col, :],
                                        conv_weights[:, _fr10_col],
                                    )
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
                                _fr10_pert_acc = (
                                    _fr10_pert_acc
                                    + _fr11_conv_tap_product(
                                        _fr10_pert_window[:, _fr10_col, :],
                                        conv_weights[:, _fr10_col],
                                    )
                                )
                                _fr10_flat_pert_acc = (
                                    _fr10_flat_pert_acc
                                    + _fr11_conv_tap_product(
                                        _fr10_flat_pert_window[:, _fr10_col, :],
                                        conv_weights[:, _fr10_col],
                                    )
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
                                    _fr10_serial_acc = (
                                        _fr10_serial_acc
                                        + _fr11_conv_tap_product(
                                            _fr10_serial_window[_fr10_col],
                                            conv_weights[:, _fr10_col],
                                        )
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
                    if (
                        _fr10_tree_conv_expected
                        and os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1"
                    ):
                        raise RuntimeError(
                            "FR10 tree causal-conv disengaged: "
                            + type(_fr10_tree_conv_exc).__name__
                            + ":"
                            + str(_fr10_tree_conv_exc)
                        ) from _fr10_tree_conv_exc
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
                if (
                    _fr10_tree_conv_expected
                    and os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1"
                ):
                    raise RuntimeError(
                        "FR10 tree causal-conv disengaged: eligible_tree_spec_row_flat_fallback"
                    )
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
            try:
                _fr12_conv_extra = {
                    "num_spec_decodes": int(attn_metadata.num_spec_decodes),
                    "num_actual_tokens": int(num_actual_tokens),
                    "tree_conv_active": bool(use_fr10_tree_conv),
                    "tree_conv_expected": bool(_fr10_tree_conv_expected),
                }
                if getattr(attn_metadata, "fr10_tree_parent", None) is not None:
                    _fr12_conv_extra["tree_parent"] = [
                        int(_x)
                        for _x in attn_metadata.fr10_tree_parent.detach().cpu().tolist()
                    ]
                if spec_token_indx is not None:
                    _fr12_conv_extra["spec_token_indx"] = [
                        int(_x) for _x in spec_token_indx.detach().cpu().tolist()
                    ]
                _fr12_subkernel_capture_tensor(
                    self,
                    "conv1d_out",
                    mixed_qkv_spec,
                    create=True,
                    extra=_fr12_conv_extra,
                )
                _fr12_payload = _fr12_subkernel_capture_get(self, create=False)
                if (
                    _fr12_payload is not None
                    and not bool(use_fr10_tree_conv)
                    and not _fr12_payload["meta"].get(
                        "native_conv_detail_done", False
                    )
                    and spec_state_indices_tensor is not None
                    and spec_query_start_loc is not None
                    and attn_metadata.num_spec_decodes > 0
                ):
                    _fr12_width = int(conv_weights.shape[1])
                    _fr12_native_start = int(
                        spec_query_start_loc[0].detach().cpu().item()
                    )
                    _fr12_native_end = int(
                        spec_query_start_loc[1].detach().cpu().item()
                    )
                    _fr12_native_len = max(
                        0, _fr12_native_end - _fr12_native_start
                    )
                    _fr12_native_rows = torch.arange(
                        _fr12_native_len,
                        dtype=torch.long,
                        device=mixed_qkv_spec.device,
                    )
                    _fr12_prior_row = spec_state_indices_tensor[0, 0].to(torch.long)
                    _fr12_prior_window = conv_state.index_select(
                        0, _fr12_prior_row.view(1)
                    )[0].index_select(
                        1,
                        torch.arange(
                            _fr12_width - 1,
                            dtype=torch.long,
                            device=mixed_qkv_spec.device,
                        ),
                    )
                    _fr12_native_x = _fr12_pre_conv_spec[
                        _fr12_native_start:_fr12_native_end
                    ]
                    _fr12_native_source = torch.cat(
                        (_fr12_prior_window.transpose(0, 1), _fr12_native_x),
                        dim=0,
                    )
                    _fr12_src_rows = []
                    for _fr12_i in range(_fr12_native_len):
                        _fr12_src_rows.append(
                            list(
                                range(
                                    int(_fr12_i),
                                    int(_fr12_i) + int(_fr12_width),
                                )
                            )
                        )
                    _fr12_native_source_indices = torch.tensor(
                        _fr12_src_rows,
                        dtype=torch.long,
                        device=mixed_qkv_spec.device,
                    )
                    _fr12_native_window = _fr12_native_source.index_select(
                        0, _fr12_native_source_indices.reshape(-1)
                    ).view(_fr12_native_len, _fr12_width, _fr12_native_x.size(1))
                    _fr12_taps_fp32 = []
                    _fr12_taps_bf16 = []
                    _fr12_tap_dtype = mixed_qkv_spec.dtype
                    for _fr12_col in range(_fr12_width):
                        _fr12_w_fp32 = conv_weights[:, _fr12_col].to(
                            torch.float32
                        ).unsqueeze(0)
                        _fr12_w_bf16 = conv_weights[:, _fr12_col].to(
                            _fr12_tap_dtype
                        ).unsqueeze(0)
                        _fr12_x_col = _fr12_native_window[:, _fr12_col, :]
                        _fr12_taps_fp32.append(
                            _fr12_x_col.to(torch.float32) * _fr12_w_fp32
                        )
                        _fr12_taps_bf16.append(
                            (
                                _fr12_x_col.to(_fr12_tap_dtype)
                                * _fr12_w_bf16
                            )
                            .to(_fr12_tap_dtype)
                            .to(torch.float32)
                        )
                    _fr12_payload["meta"]["native_conv_detail"] = {
                        "schema": "fr12.native_conv_detail.v1",
                        "prior_bank_row": int(_fr12_prior_row.detach().cpu().item()),
                        "query_start_loc": spec_query_start_loc.detach()
                        .cpu()
                        .clone(),
                        "source_indices": _fr12_native_source_indices.detach()
                        .cpu()
                        .clone(),
                        "prior_window": _fr12_prior_window.detach()
                        .to(torch.float32)
                        .cpu()
                        .clone(),
                        "pre_conv_rows": _fr12_native_x.detach()
                        .to(torch.float32)
                        .cpu()
                        .clone(),
                        "window": _fr12_native_window.detach()
                        .to(torch.float32)
                        .cpu()
                        .clone(),
                        "tap_products_fp32": torch.stack(
                            _fr12_taps_fp32, dim=1
                        )
                        .detach()
                        .cpu()
                        .clone(),
                        "tap_products_bf16": torch.stack(
                            _fr12_taps_bf16, dim=1
                        )
                        .detach()
                        .cpu()
                        .clone(),
                    }
                    _fr12_payload["meta"]["native_conv_detail_done"] = True
                    _fr12_subkernel_capture_flush(_fr12_payload)
            except Exception as _fr12_conv_cap_exc:
                logger.warning("FR12 conv capture failed: %s", _fr12_conv_cap_exc)
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
            _fr10_tree_scan_expected = (
                _fr10_active_decode_mode == "tree_mtp"
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
                _fr10_accepted_lens_tensor = globals().get(
                    "_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR"
                )
                if _fr10_accepted_lens_tensor is None:
                    raise RuntimeError("missing_accepted_lens_device_tensor")
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
                    _fr10_commit_handoff_active = os.environ.get(
                        "FR10_TREE_GDN_COMMIT_HANDOFF_LOG"
                    )
                    _fr10_read_col = 0
                    if _fr10_capture_scan_payload or _fr10_commit_handoff_active:
                        try:
                            _fr10_read_col = max(
                                0,
                                min(
                                    int(
                                        _fr10_accepted_lens_tensor[
                                            fr10_b
                                        ].detach().cpu().item()
                                    )
                                    - 1,
                                    int(spec_state_indices_tensor.size(-1)) - 1,
                                )
                            )
                        except Exception:
                            _fr10_read_col = 0
                    _fr10_capture_state_index = None
                    _fr10_capture_h0 = None
                    if _fr10_capture_scan_payload:
                        _fr10_capture_state_index = int(
                            spec_state_indices_tensor[fr10_b, _fr10_read_col]
                            .detach()
                            .cpu()
                            .item()
                        )
                        _fr10_capture_h0 = (
                            ssm_state[_fr10_capture_state_index].detach().cpu().clone()
                        )
                    if _fr10_commit_handoff_active:
                        try:
                            _fr10_commit_state_index = int(
                                spec_state_indices_tensor[fr10_b, _fr10_read_col]
                                .detach()
                                .cpu()
                                .item()
                            )
                            _fr10_commit_h0 = (
                                ssm_state[_fr10_commit_state_index].detach().clone()
                            )
                        except Exception:
                            _fr10_commit_handoff_active = False
                    try:
                        _fr10_prev_read = None
                        if _fr10_capture_scan_payload or _fr10_commit_handoff_active:
                            _fr10_prev_read = globals().setdefault(
                                "_FR10_TREE_READ_PREV", {}
                            ).get((str(self.prefix), int(fr10_b)))
                        _fr10_rows = globals().get(
                            "_LUMO_FA_LAST_ACCEPTED_TREE_ROWS", []
                        )
                        _fr10_lens = globals().get(
                            "_LUMO_FA_LAST_ACCEPTED_TREE_LENS", []
                        )
                        _fr10_node_paths = globals().get(
                            "_LUMO_FA_LAST_ACCEPTED_TREE_NODE_PATHS", []
                        )
                        _fr10_has_accept = (
                            _fr10_lens is not None
                            and fr10_b < len(_fr10_lens)
                            and int(_fr10_lens[fr10_b]) > 0
                        )
                        if (
                            _fr10_prev_read is not None
                            and _fr10_has_accept
                            and fr10_b < len(_fr10_rows)
                        ):
                            _fr10_seed_row = max(
                                0,
                                min(
                                    int(_fr10_rows[fr10_b]),
                                    int(_fr10_prev_read["tree_n"]) - 1,
                                ),
                            )
                            _fr10_seed_path = (
                                [int(_x) for _x in _fr10_node_paths[fr10_b]]
                                if fr10_b < len(_fr10_node_paths)
                                else [int(_fr10_seed_row)]
                            )
                            _fr10_seed_path_len = min(
                                len(_fr10_seed_path),
                                int(_fr10_lens[fr10_b]),
                                int(spec_state_indices_tensor.size(-1)),
                                int(_fr10_prev_read["tree_n"]),
                            )
                            if _fr10_seed_path_len <= 0:
                                raise RuntimeError("accepted path is empty")
                            _fr10_seed_state_index = int(
                                spec_state_indices_tensor[
                                    fr10_b, _fr10_read_col
                                ]
                                .detach()
                                .cpu()
                                .item()
                            )
                            if _fr10_commit_handoff_active:
                                _fr10_commit_h0 = (
                                    ssm_state[_fr10_commit_state_index].detach().clone()
                                )
                            if _fr10_capture_scan_payload:
                                _fr10_capture_h0 = (
                                    ssm_state[_fr10_capture_state_index]
                                    .detach()
                                    .cpu()
                                    .clone()
                                )
                            try:
                                import json as _fr10_seed_json
                                import time as _fr10_seed_time

                                _fr10_seed_log = os.environ.get(
                                    "FR10_TREE_GDN_COMMIT_HANDOFF_LOG"
                                )
                                _fr10_seed_count = int(
                                    globals().get(
                                        "_FR10_TREE_READ_HANDOFF_LOG_COUNT", 0
                                    )
                                )
                                _fr10_seed_limit = int(
                                    os.environ.get(
                                        "FR10_TREE_GDN_COMMIT_HANDOFF_LIMIT", "32"
                                    )
                                )
                                if _fr10_seed_log and _fr10_seed_count < _fr10_seed_limit:
                                    _fr10_seed_accepted_bank_row = None
                                    _fr10_prev_spec_indices = _fr10_prev_read.get(
                                        "spec_state_indices"
                                    )
                                    if _fr10_prev_spec_indices is not None:
                                        _fr10_seed_accepted_bank_row = int(
                                            _fr10_prev_spec_indices[int(_fr10_seed_row)]
                                            .detach()
                                            .cpu()
                                            .item()
                                        )
                                    _fr10_seed_next_ssm = ssm_state[
                                        _fr10_seed_state_index
                                    ].detach().clone()
                                    _fr10_seed_expected_ssm = _fr10_prev_read[
                                        "tree_state"
                                    ][_fr10_seed_row]
                                    _fr10_seed_curr_conv = globals().get(
                                        "_FR10_COMMIT_HANDOFF_CURR_CONV_BY_B", {}
                                    ).get(int(fr10_b), {})
                                    _fr10_seed_next_conv = _fr10_seed_curr_conv.get(
                                        "prior"
                                    )
                                    _fr10_seed_expected_conv = _fr10_prev_read[
                                        "conv_rows"
                                    ][_fr10_seed_row]
                                    _fr10_seed_ssm_max = float(
                                        (
                                            _fr10_seed_next_ssm.float()
                                            - _fr10_seed_expected_ssm.float()
                                        )
                                        .abs()
                                        .max()
                                        .detach()
                                        .cpu()
                                        .item()
                                    )
                                    _fr10_seed_conv_max = None
                                    if _fr10_seed_next_conv is not None:
                                        _fr10_seed_conv_max = float(
                                            (
                                                _fr10_seed_next_conv.float()
                                                - _fr10_seed_expected_conv.float()
                                            )
                                            .abs()
                                            .max()
                                            .detach()
                                            .cpu()
                                            .item()
                                        )
                                    with open(_fr10_seed_log, "a", buffering=1) as _fr10_fh:
                                        _fr10_fh.write(
                                            _fr10_seed_json.dumps(
                                                {
                                                    "schema": "fr10.commit_native_handoff.v1",
                                                    "event": "commit_native_handoff",
                                                    "ts": round(_fr10_seed_time.time(), 4),
                                                    "layer_prefix": str(self.prefix),
                                                    "batch_index": int(fr10_b),
                                                    "prev_accepted_len": int(_fr10_lens[fr10_b]),
                                                    "accepted_node_row": int(_fr10_seed_row),
                                                    "accepted_node_path": [
                                                        int(_x)
                                                        for _x in _fr10_seed_path[
                                                            :_fr10_seed_path_len
                                                        ]
                                                    ],
                                                    "accepted_linear_read_col": int(
                                                        _fr10_seed_path_len - 1
                                                    ),
                                                    "accepted_spec_state_bank_row": (
                                                        None
                                                        if _fr10_seed_accepted_bank_row is None
                                                        else int(_fr10_seed_accepted_bank_row)
                                                    ),
                                                    "accepted_bank_row": (
                                                        None
                                                        if _fr10_seed_accepted_bank_row is None
                                                        else int(_fr10_seed_accepted_bank_row)
                                                    ),
                                                    "next_read_bank_row": int(
                                                        _fr10_seed_state_index
                                                    ),
                                                    "address_coincide": bool(
                                                        _fr10_seed_accepted_bank_row is not None
                                                        and int(_fr10_seed_accepted_bank_row)
                                                        == int(_fr10_seed_state_index)
                                                    ),
                                                    "linear_column_coincide": bool(
                                                        _fr10_seed_path_len
                                                        == int(_fr10_lens[fr10_b])
                                                    ),
                                                    "cache_state_coincide": bool(
                                                        _fr10_seed_ssm_max == 0.0
                                                        and _fr10_seed_conv_max == 0.0
                                                    ),
                                                    "ssm_bank_vs_cached_tree_state_max_abs": _fr10_seed_ssm_max,
                                                    "conv_cache_vs_cache_max_abs": _fr10_seed_conv_max,
                                                }
                                            )
                                            + chr(10)
                                        )
                                    _fr10_src_native_path = os.environ.get(
                                        "FR10_TREE_GDN_SRC_NATIVE_PAYLOAD"
                                    )
                                    if (
                                        _fr10_src_native_path
                                        and not globals().get(
                                            "_FR10_TREE_GDN_SRC_NATIVE_PAYLOAD_DONE",
                                            False,
                                        )
                                        and _fr10_prev_read.get("query_spec") is not None
                                        and _fr10_prev_read.get("h0_cpu") is not None
                                    ):
                                        _fr10_token_ids = globals().get(
                                            "_LUMO_FA_LAST_ACCEPTED_TREE_TOKEN_IDS", []
                                        )
                                        torch.save(
                                            {
                                                "schema": "fr10.src_native_handoff_payload.v1",
                                                "layer_prefix": str(self.prefix),
                                                "batch_index": int(fr10_b),
                                                "accepted_len": int(_fr10_lens[fr10_b]),
                                                "accepted_node_id": int(_fr10_seed_row),
                                                "accepted_node_path": [
                                                    int(_x)
                                                    for _x in _fr10_seed_path[
                                                        :_fr10_seed_path_len
                                                    ]
                                                ],
                                                "accepted_token_ids": (
                                                    [
                                                        int(_x)
                                                        for _x in _fr10_token_ids[fr10_b]
                                                    ]
                                                    if fr10_b < len(_fr10_token_ids)
                                                    else []
                                                ),
                                                "tree_parent": list(
                                                    _fr10_prev_read["tree_parent"]
                                                ),
                                                "output_scale": float(
                                                    _fr10_prev_read["output_scale"]
                                                ),
                                                "query_spec": _fr10_prev_read["query_spec"],
                                                "key_spec": _fr10_prev_read["key_spec"],
                                                "value_spec": _fr10_prev_read["value_spec"],
                                                "value_tree": _fr10_prev_read["value_tree"],
                                                "a": _fr10_prev_read["a"],
                                                "b": _fr10_prev_read["b"],
                                                "g_tree": _fr10_prev_read["g_tree"],
                                                "beta_tree": _fr10_prev_read["beta_tree"],
                                                "A_log": _fr10_prev_read["A_log"],
                                                "dt_bias": _fr10_prev_read["dt_bias"],
                                                "prev_h0": _fr10_prev_read["h0_cpu"],
                                                "serving_tree_state": _fr10_prev_read[
                                                    "tree_state_cpu"
                                                ],
                                                "next_read_ssm_state": _fr10_seed_next_ssm.detach()
                                                .cpu()
                                                .clone(),
                                                "prev_conv_prior": _fr10_prev_read[
                                                    "conv_prior_cpu"
                                                ],
                                                "serving_conv_rows": _fr10_prev_read[
                                                    "conv_rows_cpu"
                                                ],
                                                "next_read_conv_state": (
                                                    None
                                                    if _fr10_seed_next_conv is None
                                                    else _fr10_seed_next_conv.detach()
                                                    .cpu()
                                                    .clone()
                                                ),
                                                "accepted_spec_state_bank_row": (
                                                    None
                                                    if _fr10_seed_accepted_bank_row is None
                                                    else int(_fr10_seed_accepted_bank_row)
                                                ),
                                                "accepted_bank_row": (
                                                    None
                                                    if _fr10_seed_accepted_bank_row is None
                                                    else int(_fr10_seed_accepted_bank_row)
                                                ),
                                                "next_read_bank_row": int(
                                                    _fr10_seed_state_index
                                                ),
                                            },
                                            _fr10_src_native_path,
                                        )
                                        globals()[
                                            "_FR10_TREE_GDN_SRC_NATIVE_PAYLOAD_DONE"
                                        ] = True
                                    globals()[
                                        "_FR10_TREE_READ_HANDOFF_LOG_COUNT"
                                    ] = _fr10_seed_count + 1
                            except Exception as _fr10_seed_log_exc:
                                if os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1":
                                    raise RuntimeError(
                                        "FR10 tree SSM handoff logging failed: "
                                        + type(_fr10_seed_log_exc).__name__
                                        + ":"
                                        + str(_fr10_seed_log_exc)
                            ) from _fr10_seed_log_exc
                    except Exception as _fr10_seed_exc:
                        if os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1":
                            raise RuntimeError(
                                "FR10 tree SSM handoff oracle failed: "
                                + type(_fr10_seed_exc).__name__
                                + ":"
                                + str(_fr10_seed_exc)
                            ) from _fr10_seed_exc
                    _fr10_event_h0 = None
                    if _fr10_capture_scan_payload or _fr10_commit_handoff_active:
                        try:
                            _fr10_event_h0 = (
                                ssm_state[
                                    int(
                                        spec_state_indices_tensor[
                                            fr10_b, _fr10_read_col
                                        ].detach().cpu().item()
                                    )
                                ]
                                .detach()
                                .clone()
                            )
                        except Exception:
                            _fr10_event_h0 = None
                    tree_out, _ = launch_tree_gdn_prepared(
                        q=query_spec[0, start:end].contiguous(),
                        k=key_spec[0, start:end].contiguous(),
                        v=value_tree[start:end].contiguous(),
                        g=g_tree[start:end].contiguous(),
                        beta=beta_tree[start:end].contiguous(),
                        h0=ssm_state,
                        h0_indices=spec_state_indices_tensor,
                        h0_num_accepted_tokens=_fr10_accepted_lens_tensor,
                        h0_is_bank=True,
                        h0_index_row=fr10_b * spec_state_indices_tensor.size(-1),
                        h0_batch_index=fr10_b,
                        h0_use_accepted_column=True,
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
                    _fr10_root_h0_log = os.environ.get("FR10_TREE_GDN_ROOT_H0_LOG")
                    _fr10_root_h0_log_prefix = os.environ.get(
                        "FR10_TREE_GDN_ROOT_H0_LOG_LAYER_PREFIX",
                        "language_model.model.layers.0.linear_attn",
                    )
                    _fr10_root_h0_log_limit = int(
                        os.environ.get("FR10_TREE_GDN_ROOT_H0_LOG_LIMIT", "20")
                    )
                    _fr10_root_h0_log_count = int(
                        globals().get("_FR10_TREE_GDN_ROOT_H0_LOG_COUNT", 0)
                    )
                    _fr10_root_h0_in_cuda_capture = bool(
                        torch.cuda.is_available()
                        and torch.cuda.is_current_stream_capturing()
                    )
                    if (
                        _fr10_root_h0_log
                        and fr10_b == 0
                        and str(self.prefix) == _fr10_root_h0_log_prefix
                        and _fr10_root_h0_log_count < _fr10_root_h0_log_limit
                        and not _fr10_root_h0_in_cuda_capture
                    ):
                        try:
                            _fr10_live_col = max(
                                0,
                                min(
                                    int(
                                        _fr10_accepted_lens_tensor[fr10_b]
                                        .detach()
                                        .cpu()
                                        .item()
                                    )
                                    - 1,
                                    int(spec_state_indices_tensor.size(-1)) - 1,
                                ),
                            )
                            _fr10_col0_row = int(
                                spec_state_indices_tensor[fr10_b, 0]
                                .detach()
                                .cpu()
                                .item()
                            )
                            _fr10_live_row = int(
                                spec_state_indices_tensor[fr10_b, _fr10_live_col]
                                .detach()
                                .cpu()
                                .item()
                            )
                            _fr10_root_row = {
                                "event_index": int(_fr10_root_h0_log_count),
                                "schema": "fr10.tree_root_h0_probe_row.v2",
                                "layer_prefix": str(self.prefix),
                                "batch_index": int(fr10_b),
                                "tree_n": int(tree_n),
                                "query_spec_rows": int(query_spec.size(1)),
                                "start": int(start),
                                "end": int(end),
                                "accepted_len": int(_fr10_live_col + 1),
                                "col0": 0,
                                "live_col": int(_fr10_live_col),
                                "col0_bank_row": int(_fr10_col0_row),
                                "live_bank_row": int(_fr10_live_row),
                                "h0_col0": ssm_state[_fr10_col0_row]
                                .detach()
                                .cpu()
                                .clone(),
                                "h0_live": ssm_state[_fr10_live_row]
                                .detach()
                                .cpu()
                                .clone(),
                                "q_root": query_spec[0, start].detach().cpu().clone(),
                                "k_root": key_spec[0, start].detach().cpu().clone(),
                                "value_spec_root": value_spec[start]
                                .detach()
                                .cpu()
                                .clone(),
                                "value_tree_root": value_tree[start]
                                .detach()
                                .cpu()
                                .clone(),
                                "a_root": a[start].detach().cpu().clone(),
                                "b_root": b[start].detach().cpu().clone(),
                                "g_tree_root": g_tree[start].detach().cpu().clone(),
                                "beta_tree_root": beta_tree[start]
                                .detach()
                                .cpu()
                                .clone(),
                                "A_log": self.A_log.detach().cpu().clone(),
                                "dt_bias": self.dt_bias.detach().cpu().clone(),
                                "serving_tree_out_root": tree_out[0]
                                .detach()
                                .cpu()
                                .clone(),
                                "serving_tree_state_root": tree_state[0]
                                .detach()
                                .cpu()
                                .clone(),
                                "output_scale": float(self.head_k_dim**-0.5),
                            }
                            _fr10_root_rows = list(
                                globals().get("_FR10_TREE_GDN_ROOT_H0_LOG_ROWS", [])
                            )
                            _fr10_root_rows.append(_fr10_root_row)
                            globals()["_FR10_TREE_GDN_ROOT_H0_LOG_ROWS"] = (
                                _fr10_root_rows
                            )
                            torch.save(
                                {
                                    "schema": "fr10.tree_root_h0_probe.v2",
                                    "layer_prefix": _fr10_root_h0_log_prefix,
                                    "limit": int(_fr10_root_h0_log_limit),
                                    "rows": _fr10_root_rows,
                                },
                                _fr10_root_h0_log,
                            )
                            globals()["_FR10_TREE_GDN_ROOT_H0_LOG_COUNT"] = (
                                _fr10_root_h0_log_count + 1
                            )
                        except Exception as _fr10_root_h0_exc:
                            if os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1":
                                raise RuntimeError(
                                    "FR10 tree root h0 probe failed: "
                                    + type(_fr10_root_h0_exc).__name__
                                    + ":"
                                    + str(_fr10_root_h0_exc)
                                ) from _fr10_root_h0_exc
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
                    # Persist every verified tree-node state. The sampled
                    # committer publishes the accepted path after this forward;
                    # the next decode remaps those node rows into stock linear
                    # columns with launch_tree_state_linear_remap before the
                    # recurrent consumers read them.
                    ssm_state.index_copy_(
                        0,
                        spec_state_indices_tensor[fr10_b, :tree_n].to(torch.long),
                        tree_state[:tree_n].to(dtype=ssm_state.dtype),
                    )
                    if (
                        os.environ.get("FR10_TREE_GDN_COMMIT_HANDOFF_LOG")
                        or os.environ.get("FR10_TREE_GDN_SRC_NATIVE_PAYLOAD")
                    ):
                        try:
                            _fr10_curr_conv = globals().get(
                                "_FR10_COMMIT_HANDOFF_CURR_CONV_BY_B", {}
                            ).get(int(fr10_b), {})
                            _fr10_curr_conv_rows = _fr10_curr_conv.get("rows")
                            if _fr10_curr_conv_rows is not None:
                                globals().setdefault(
                                    "_FR10_TREE_READ_PREV", {}
                                )[(str(self.prefix), int(fr10_b))] = {
                                    "tree_n": int(tree_n),
                                    "tree_state": tree_state[:tree_n].detach().clone(),
                                    "conv_rows": _fr10_curr_conv_rows[
                                        :tree_n
                                    ].detach().clone(),
                                    "spec_state_indices": spec_state_indices_tensor[
                                        fr10_b, :tree_n
                                    ].detach().clone(),
                                    "tree_parent": [
                                        int(_x)
                                        for _x in attn_metadata.fr10_tree_parent.detach()
                                        .cpu()
                                        .tolist()
                                    ],
                                    "output_scale": float(self.head_k_dim**-0.5),
                                    "query_spec": query_spec[
                                        0, start:end
                                    ].detach().cpu().clone(),
                                    "key_spec": key_spec[
                                        0, start:end
                                    ].detach().cpu().clone(),
                                    "value_spec": value_spec[
                                        0, start:end
                                    ].detach().cpu().clone(),
                                    "value_tree": value_tree[
                                        start:end
                                    ].detach().cpu().clone(),
                                    "a": a[start:end].detach().cpu().clone(),
                                    "b": b[start:end].detach().cpu().clone(),
                                    "g_tree": g_tree[start:end].detach().cpu().clone(),
                                    "beta_tree": beta_tree[
                                        start:end
                                    ].detach().cpu().clone(),
                                    "A_log": self.A_log.detach().cpu().clone(),
                                    "dt_bias": self.dt_bias.detach().cpu().clone(),
                                    "h0_cpu": (
                                        None
                                        if _fr10_event_h0 is None
                                        else _fr10_event_h0.detach().cpu().clone()
                                    ),
                                    "tree_state_cpu": tree_state[
                                        :tree_n
                                    ].detach().cpu().clone(),
                                    "conv_prior_cpu": _fr10_curr_conv.get(
                                        "prior"
                                    ).detach().cpu().clone(),
                                    "conv_rows_cpu": _fr10_curr_conv_rows[
                                        :tree_n
                                    ].detach().cpu().clone(),
                                }
                        except Exception:
                            pass
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
                if (
                    _fr10_tree_scan_expected
                    and os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1"
                ):
                    raise RuntimeError(
                        "FR10 tree scan disengaged: eligible_tree_spec_row_flat_fallback"
                    )
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
            try:
                _fr12_scan_extra = {
                    "num_spec_decodes": int(attn_metadata.num_spec_decodes),
                    "num_actual_tokens": int(num_actual_tokens),
                    "tree_scan_active": bool(use_fr10_tree),
                    "tree_scan_expected": bool(_fr10_tree_scan_expected),
                }
                if getattr(attn_metadata, "fr10_tree_parent", None) is not None:
                    _fr12_scan_extra["tree_parent"] = [
                        int(_x)
                        for _x in attn_metadata.fr10_tree_parent.detach().cpu().tolist()
                    ]
                if spec_token_indx is not None:
                    _fr12_scan_extra["spec_token_indx"] = [
                        int(_x) for _x in spec_token_indx.detach().cpu().tolist()
                    ]
                _fr12_subkernel_capture_tensor(
                    self,
                    "gdn_scan_out",
                    core_attn_out_spec.squeeze(0),
                    create=False,
                    extra=_fr12_scan_extra,
                )
            except Exception as _fr12_scan_cap_exc:
                logger.warning("FR12 scan capture failed: %s", _fr12_scan_cap_exc)
'''
    if needle not in text:
        raise RuntimeError("FR10 GDN linear spec branch needle not found")
    text = text.replace(needle, replacement, 1)

    output_projection_needle = '''        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)
'''
    output_projection_replacement = '''        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        _fr12_subkernel_capture_tensor(
            self,
            "gate_out",
            core_attn_out[:num_tokens],
            create=False,
            extra={"num_tokens": int(num_tokens)},
        )
        output[:num_tokens], _ = self.out_proj(core_attn_out)
        _fr12_subkernel_capture_tensor(
            self,
            "o_proj_out",
            output[:num_tokens],
            create=False,
            extra={"num_tokens": int(num_tokens)},
        )
        _fr12_payload = _fr12_subkernel_capture_get(self, create=False)
        if _fr12_payload is not None:
            _fr12_subkernel_capture_flush(_fr12_payload, final=True)
'''
    if output_projection_needle not in text:
        raise RuntimeError("FR12 output projection needle not found")
    text = text.replace(output_projection_needle, output_projection_replacement, 1)
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
    accepted_node_paths = []
    accepted_token_rows = []
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
        accepted_node_paths.append([int(x) for x in best_path[:best_lcp]])
        accepted_token_rows.append([int(drafts[x]) for x in best_path[:best_lcp]])
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
    globals()['_LUMO_TREE_LAST_ACCEPTED_NODE_PATHS_KERNEL'] = [
        [int(x) for x in row] for row in accepted_node_paths
    ]
    try:
        from vllm.model_executor.layers.mamba import gdn_linear_attn as _lumo_tree_commit_gdn
        accepted_gdn_node_paths = []
        accepted_gdn_rows = []
        for _accepted_path, _accepted_len, _accepted_row in zip(
            accepted_node_paths, accepted_lens, accepted_rows
        ):
            _gdn_path = [
                int(_node_id) + 1
                for _node_id in _accepted_path[: int(_accepted_len)]
            ]
            accepted_gdn_node_paths.append(_gdn_path)
            accepted_gdn_rows.append(
                int(_gdn_path[-1]) if _gdn_path else 0
            )
        _accepted_path_buf = getattr(
            _lumo_tree_commit_gdn, "_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR", None
        )
        _accepted_lens_buf = getattr(
            _lumo_tree_commit_gdn, "_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR", None
        )
        if _accepted_path_buf is None or _accepted_lens_buf is None:
            raise RuntimeError("missing_accepted_path_device_tensor")
        if int(_accepted_path_buf.size(0)) < len(accepted_gdn_node_paths):
            raise RuntimeError("accepted_path_device_tensor_batch_too_small")
        _accepted_path_cols = int(_accepted_path_buf.size(1))
        _accepted_path_rows = []
        for _accepted_path in accepted_gdn_node_paths:
            _row = [0 for _ in range(_accepted_path_cols)]
            for _pos, _node_id in enumerate(_accepted_path[:_accepted_path_cols]):
                _row[_pos] = int(_node_id)
            _accepted_path_rows.append(_row)
        if _accepted_path_rows:
            _accepted_path_buf[: len(_accepted_path_rows), :_accepted_path_cols].copy_(
                torch.tensor(
                    _accepted_path_rows,
                    dtype=_accepted_path_buf.dtype,
                    device=_accepted_path_buf.device,
                )
            )
            _accepted_lens_buf[: len(accepted_lens)].copy_(
                torch.tensor(
                    accepted_lens,
                    dtype=_accepted_lens_buf.dtype,
                    device=_accepted_lens_buf.device,
                )
            )
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_ROWS = [
            int(x) for x in accepted_gdn_rows
        ]
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_LENS = [
            int(x) for x in accepted_lens
        ]
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_NODE_PATHS = [
            [int(x) for x in row] for row in accepted_gdn_node_paths
        ]
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_TOKEN_IDS = [
            [int(x) for x in row] for row in accepted_token_rows
        ]
    except Exception as _fr10_tree_lcp_log_exc:
        if __import__('os').environ.get('FR10_ALLOW_LINEAR_FALLBACK', '0') != '1':
            raise RuntimeError(
                'FR10 tree path-LCP log failed: '
                + type(_fr10_tree_lcp_log_exc).__name__
                + ':'
                + str(_fr10_tree_lcp_log_exc)
            ) from _fr10_tree_lcp_log_exc
    try:
        import json as _lcpj, os as _lcpo, time as _lcpt
        if (
            _lcpo.environ.get('FR10_METRICS', '0') == '1'
            or _lcpo.environ.get('LUMO_TREE_PATH_LCP_LOG')
        ):
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
    except Exception as _fr10_winner_log_exc:
        if __import__('os').environ.get('FR10_ALLOW_LINEAR_FALLBACK', '0') != '1':
            raise RuntimeError(
                'FR10 independent-winner log failed: '
                + type(_fr10_winner_log_exc).__name__
                + ':'
                + str(_fr10_winner_log_exc)
            ) from _fr10_winner_log_exc
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
    except Exception as _fr10_commit_globals_exc:
        if __import__('os').environ.get('FR10_ALLOW_LINEAR_FALLBACK', '0') != '1':
            raise RuntimeError(
                'FR10 tree committer failed to publish accepted rows: '
                + type(_fr10_commit_globals_exc).__name__
                + ':'
                + str(_fr10_commit_globals_exc)
            ) from _fr10_commit_globals_exc
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
    accepted_node_paths = []
    accepted_token_rows = []
    try:
        import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
        if (
            _fr10_lo.environ.get('FR10_METRICS', '0') == '1'
            or _fr10_lo.environ.get('LUMO_TREE_SAMPLER_DEBUG_LOG')
        ):
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
        step_trace_rows = []
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

            _target_row = int(start + children[0])
            _child_drafts = [int(drafts[child]) for child in children]
            _target_probs = target_probs_cpu[_target_row]
            if draft_probs_cpu is None:
                step = _fr10_sample_det_step(
                    _target_probs,
                    _child_drafts,
                    rng=rng,
                )
            else:
                step = _fr10_sample_step(
                    _target_probs,
                    [draft_probs_cpu[start + child] for child in children],
                    rng=rng,
                )
            _selected_child = int(children[int(step.source_index)])
            step_trace_rows.append({
                'step': int(_step),
                'parent_node_id': int(current_parent),
                'child_node_ids': [int(x) for x in children],
                'target_prob_row': int(_target_row),
                'target_argmax': int(_fr10_np.argmax(_target_probs)),
                'draft_token_ids': [int(x) for x in _child_drafts],
                'target_prob_at_draft_token_ids': [
                    float(_target_probs[int(_tok)]) for _tok in _child_drafts
                ],
                'selected_source_index': int(step.source_index),
                'selected_child_node_id': int(_selected_child),
                'selected_token_id': int(step.token_id),
                'accepted': bool(step.accepted),
            })
            row.append(int(step.token_id))
            if not step.accepted:
                break
            accepted_child = _selected_child
            if int(step.token_id) != int(drafts[accepted_child]):
                break
            current_parent = accepted_child
            accepted_row = int(current_parent)
            accepted_path.append(int(current_parent))
        out_rows.append(row[:int(max_spec_len) + 1])
        accepted_rows.append(int(accepted_row))
        accepted_lens.append(int(len(accepted_path)))
        accepted_node_paths.append([int(x) for x in accepted_path])
        accepted_token_rows.append([int(drafts[x]) for x in accepted_path])
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
            'committer_step_trace': step_trace_rows,
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
    globals()['_LUMO_TREE_LAST_ACCEPTED_NODE_PATHS_KERNEL'] = [
        [int(x) for x in row] for row in accepted_node_paths
    ]
    try:
        from vllm.model_executor.layers.mamba import gdn_linear_attn as _lumo_tree_commit_gdn
        accepted_gdn_node_paths = []
        accepted_gdn_rows = []
        for _accepted_path, _accepted_len, _accepted_row in zip(
            accepted_node_paths, accepted_lens, accepted_rows
        ):
            _gdn_path = [
                int(_node_id) + 1
                for _node_id in _accepted_path[: int(_accepted_len)]
            ]
            accepted_gdn_node_paths.append(_gdn_path)
            accepted_gdn_rows.append(
                int(_gdn_path[-1]) if _gdn_path else 0
            )
        _accepted_path_buf = getattr(
            _lumo_tree_commit_gdn, "_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR", None
        )
        _accepted_lens_buf = getattr(
            _lumo_tree_commit_gdn, "_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR", None
        )
        if _accepted_path_buf is None or _accepted_lens_buf is None:
            raise RuntimeError("missing_accepted_path_device_tensor")
        if int(_accepted_path_buf.size(0)) < len(accepted_gdn_node_paths):
            raise RuntimeError("accepted_path_device_tensor_batch_too_small")
        _accepted_path_cols = int(_accepted_path_buf.size(1))
        _accepted_path_rows = []
        for _accepted_path in accepted_gdn_node_paths:
            _row = [0 for _ in range(_accepted_path_cols)]
            for _pos, _node_id in enumerate(_accepted_path[:_accepted_path_cols]):
                _row[_pos] = int(_node_id)
            _accepted_path_rows.append(_row)
        if _accepted_path_rows:
            _accepted_path_buf[: len(_accepted_path_rows), :_accepted_path_cols].copy_(
                torch.tensor(
                    _accepted_path_rows,
                    dtype=_accepted_path_buf.dtype,
                    device=_accepted_path_buf.device,
                )
            )
            _accepted_lens_buf[: len(accepted_lens)].copy_(
                torch.tensor(
                    accepted_lens,
                    dtype=_accepted_lens_buf.dtype,
                    device=_accepted_lens_buf.device,
                )
            )
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_ROWS = [
            int(x) for x in accepted_gdn_rows
        ]
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_LENS = [
            int(x) for x in accepted_lens
        ]
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_NODE_PATHS = [
            [int(x) for x in row] for row in accepted_gdn_node_paths
        ]
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_TOKEN_IDS = [
            [int(x) for x in row] for row in accepted_token_rows
        ]
    except Exception as _fr10_commit_globals_exc:
        if __import__('os').environ.get('FR10_ALLOW_LINEAR_FALLBACK', '0') != '1':
            raise RuntimeError(
                'FR10 tree committer failed to publish accepted rows: '
                + type(_fr10_commit_globals_exc).__name__
                + ':'
                + str(_fr10_commit_globals_exc)
            ) from _fr10_commit_globals_exc
    try:
        import json as _fr10_lj, os as _fr10_lo, time as _fr10_lt
        if (
            _fr10_lo.environ.get('FR10_METRICS', '0') == '1'
            or _fr10_lo.environ.get('LUMO_TREE_PATH_LCP_LOG')
        ):
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
        target_constraint = """        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )
"""
        target_processor_anchor = """        target_logits = self.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )
        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `apply_sampling_constraints` function.
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
            if (
                _fr10_lo.environ.get("FR10_METRICS", "0") == "1"
                or _fr10_lo.environ.get("LUMO_TREE_SAMPLER_DEBUG_LOG")
            ):
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
                if not sampling_metadata.all_greedy:
                    _fr10_draft_ids_cpu = [
                        int(_x) for _x in metadata.draft_token_ids.detach().cpu().tolist()
                    ]
                    _fr10_target_indices_cpu = [
                        int(_x) for _x in metadata.target_logits_indices.detach().cpu().tolist()
                    ]
                    _fr10_target_probs_cpu = (
                        target_logits.softmax(dim=-1, dtype=torch.float32)
                        .detach()
                        .cpu()
                    )
                    if lumo_tree_parent_indices is not None:
                        _fr10_parents_cpu = [
                            int(_x) for _x in lumo_tree_parent_indices.detach().cpu().tolist()
                        ]
                        _fr10_self_indices_cpu = [
                            int(_x)
                            for _x in metadata.tree_self_logits_indices.detach().cpu().tolist()
                        ]
                        _fr10_rows = []
                        _fr10_start = 0
                        for _fr10_req_i, _fr10_node_count in enumerate(metadata.num_draft_tokens):
                            _fr10_node_count = int(_fr10_node_count)
                            for _fr10_node in range(_fr10_node_count):
                                _fr10_flat = _fr10_start + _fr10_node
                                _fr10_tok = int(_fr10_draft_ids_cpu[_fr10_flat])
                                _fr10_prob_row = _fr10_target_probs_cpu[_fr10_flat]
                                _fr10_rows.append({
                                    "req_index": int(_fr10_req_i),
                                    "node_id": int(_fr10_node),
                                    "parent_node_id": int(_fr10_parents_cpu[_fr10_flat]),
                                    "target_logits_index": int(_fr10_target_indices_cpu[_fr10_flat]),
                                    "self_logits_index": int(_fr10_self_indices_cpu[_fr10_flat]),
                                    "draft_token_id": int(_fr10_tok),
                                    "target_argmax": int(torch.argmax(_fr10_prob_row).item()),
                                    "target_prob_draft": float(_fr10_prob_row[_fr10_tok].item()),
                                })
                            _fr10_start += _fr10_node_count
                        _LUMO_TREE_SAMPLER_DEBUG_FH.write(
                            _fr10_lj.dumps({
                                "event": "tree_logit_gather",
                                "ts": round(_fr10_lt.time(), 4),
                                "rows": _fr10_rows,
                            }) + chr(10)
                        )
                    elif len(_fr10_draft_ids_cpu):
                        _fr10_rows = []
                        _fr10_start = 0
                        for _fr10_req_i, _fr10_node_count in enumerate(metadata.num_draft_tokens):
                            _fr10_node_count = int(_fr10_node_count)
                            for _fr10_pos in range(_fr10_node_count):
                                _fr10_flat = _fr10_start + _fr10_pos
                                _fr10_tok = int(_fr10_draft_ids_cpu[_fr10_flat])
                                _fr10_prob_row = _fr10_target_probs_cpu[_fr10_flat]
                                _fr10_rows.append({
                                    "req_index": int(_fr10_req_i),
                                    "position": int(_fr10_pos),
                                    "target_logits_index": int(_fr10_target_indices_cpu[_fr10_flat]),
                                    "draft_logits_index": int(_fr10_target_indices_cpu[_fr10_flat] + 1),
                                    "draft_token_id": int(_fr10_tok),
                                    "target_argmax": int(torch.argmax(_fr10_prob_row).item()),
                                    "target_prob_draft": float(_fr10_prob_row[_fr10_tok].item()),
                                })
                            _fr10_start += _fr10_node_count
                        _LUMO_TREE_SAMPLER_DEBUG_FH.write(
                            _fr10_lj.dumps({
                                "event": "linear_logit_gather",
                                "ts": round(_fr10_lt.time(), 4),
                                "rows": _fr10_rows,
                            }) + chr(10)
                        )
                    _fr10_capture_path = _fr10_lo.environ.get("FR10_SPINE_LOGIT_CAPTURE")
                    _fr10_capture_has_spec = any(
                        int(_x) > 0 for _x in metadata.num_draft_tokens
                    )
                    if (
                        _fr10_capture_path
                        and _fr10_capture_has_spec
                    ):
                        _fr10_capture_seen = int(
                            globals().get("_FR10_SPINE_LOGIT_CAPTURE_SEEN", 0)
                        )
                        _fr10_capture_skip = int(
                            _fr10_lo.environ.get("FR10_SPINE_LOGIT_CAPTURE_SKIP", "0")
                        )
                        _fr10_capture_limit = int(
                            _fr10_lo.environ.get("FR10_SPINE_LOGIT_CAPTURE_LIMIT", "1")
                        )
                        _fr10_capture_saved = int(
                            globals().get("_FR10_SPINE_LOGIT_CAPTURE_SAVED", 0)
                        )
                        globals()["_FR10_SPINE_LOGIT_CAPTURE_SEEN"] = (
                            _fr10_capture_seen + 1
                        )
                        if (
                            _fr10_capture_seen >= _fr10_capture_skip
                            and _fr10_capture_saved < _fr10_capture_limit
                        ):
                            try:
                                from pathlib import Path as _fr10_Path
                                _fr10_cap = {
                                    "schema": "fr10.spine_logit_capture.v1",
                                    "capture_call_index": int(_fr10_capture_seen),
                                    "capture_saved_index": int(_fr10_capture_saved),
                                    "mode": str(
                                        getattr(
                                            __import__("vllm.v1.sample.rejection_sampler", fromlist=["_FR10_DECODE_MODE"]),
                                            "_FR10_DECODE_MODE",
                                            _fr10_lo.environ.get("FR10_DECODE_MODE_DEFAULT", "tree_mtp"),
                                        )
                                    ),
                                    "has_tree_parent_indices": lumo_tree_parent_indices is not None,
                                    "num_draft_tokens": [
                                        int(_x) for _x in metadata.num_draft_tokens
                                    ],
                                    "draft_token_ids": metadata.draft_token_ids.detach().cpu(),
                                    "target_logits_indices": metadata.target_logits_indices.detach().cpu(),
                                    "target_logits": target_logits.detach().to(torch.float32).cpu(),
                                }
                                if lumo_tree_parent_indices is not None:
                                    _fr10_cap["tree_parent_indices"] = (
                                        lumo_tree_parent_indices.detach().cpu()
                                    )
                                    _fr10_cap["tree_self_logits_indices"] = (
                                        metadata.tree_self_logits_indices.detach().cpu()
                                    )
                                    _fr10_cap["tree_self_logits"] = (
                                        lumo_tree_self_logits.detach().to(torch.float32).cpu()
                                        if lumo_tree_self_logits is not None else None
                                    )
                                _fr10_out = _fr10_Path(_fr10_capture_path)
                                _fr10_out.parent.mkdir(parents=True, exist_ok=True)
                                _fr10_call_out = _fr10_out.with_name(
                                    _fr10_out.stem
                                    + ".call"
                                    + str(int(_fr10_capture_saved))
                                    + _fr10_out.suffix
                                )
                                torch.save(_fr10_cap, _fr10_call_out)
                                if _fr10_capture_saved == 0:
                                    torch.save(_fr10_cap, _fr10_out)
                                globals()["_FR10_SPINE_LOGIT_CAPTURE_SAVED"] = (
                                    _fr10_capture_saved + 1
                                )
                                if _fr10_capture_saved + 1 >= _fr10_capture_limit:
                                    globals()["_FR10_SPINE_LOGIT_CAPTURED"] = True
                            except Exception as _fr10_capture_exc:
                                raise RuntimeError(
                                    "FR10 spine logit capture failed: "
                                    + type(_fr10_capture_exc).__name__
                                    + ":"
                                    + str(_fr10_capture_exc)
                                ) from _fr10_capture_exc
        except RuntimeError as _fr10_debug_exc:
            if str(_fr10_debug_exc).startswith("FR10 spine logit capture failed:"):
                raise
        except Exception:
            pass

        if (
            globals().get(
                "_FR10_DECODE_MODE",
                __import__("os").environ.get("FR10_DECODE_MODE_DEFAULT", "tree_mtp"),
            )
            == "tree_mtp"
            and hasattr(metadata, "tree_parent_indices")
            and lumo_tree_parent_indices is None
            and __import__("os").environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1"
        ):
            raise RuntimeError(
                "FR10 sampled tree committer disengaged: missing_tree_parent_indices"
            )

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
        stock_call_pos = text.find(stock_call)
        if stock_call_pos < 0:
            raise RuntimeError("stock rejection_sample call anchor not found")
        if target_constraint not in text[:stock_call_pos]:
            if target_processor_anchor not in text[:stock_call_pos]:
                raise RuntimeError("target logits sampling-constraints anchor not found")
            text = text.replace(
                target_processor_anchor,
                target_processor_anchor + target_constraint + "\n",
                1,
            )
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
            if (
                _fr10_lo.environ.get("FR10_METRICS", "0") == "1"
                or _fr10_lo.environ.get("LUMO_TREE_SAMPLER_DEBUG_LOG")
            ):
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
            if (
                _fr10_lo.environ.get("FR10_METRICS", "0") == "1"
                or _fr10_lo.environ.get("LUMO_TREE_SAMPLER_DEBUG_LOG")
            ):
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
        if (
            _lumo_tree_meta_debug.get("mode") == "tree_mtp"
            and _lumo_tree_meta_debug.get("has_tree_src")
            and _lumo_tree_meta_debug.get("reason") != "ok"
            and __import__("os").environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1"
        ):
            raise RuntimeError(
                "FR10 tree metadata disengaged: "
                + str(_lumo_tree_meta_debug.get("reason"))
            )

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


def _patch_gpu_model_runner_tree_depth_positions() -> bool:
    text = GPU_MODEL_RUNNER_PATH.read_text()
    sentinel = "# FR10_TREE_DEPTH_POSITIONS"
    if sentinel in text:
        return False

    anchor = """        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            self.query_start_loc.gpu[: num_reqs + 1],
            self.positions[:total_num_scheduled_tokens],
        )

        # Copy the tensors to the GPU.
"""
    inject = """        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            self.query_start_loc.gpu[: num_reqs + 1],
            self.positions[:total_num_scheduled_tokens],
        )

        # FR10_TREE_DEPTH_POSITIONS: keep token/KV slot addressing flat and
        # unique, then rewrite only the model-visible RoPE positions for tree
        # verify rows. Tree siblings share their causal depth; flattened row
        # offsets are not valid RoPE positions for non-spine caterpillar leaves.
        try:
            _fr10_mode = getattr(scheduler_output, "fr10_decode_mode", None) or __import__(
                "os"
            ).environ.get("FR10_DECODE_MODE_DEFAULT", "tree_mtp")
            _fr10_tree_src = None
            try:
                _fr10_spec_env = __import__("os").environ.get("SPEC_CONFIG")
                if _fr10_spec_env:
                    _fr10_tree_src = __import__("json").loads(_fr10_spec_env).get(
                        "speculative_token_tree"
                    )
            except Exception:
                _fr10_tree_src = None
            _fr10_lspec = getattr(self.vllm_config, "speculative_config", None)
            if not _fr10_tree_src:
                _fr10_tree_src = (
                    getattr(_fr10_lspec, "speculative_token_tree", None)
                    if _fr10_lspec is not None
                    else None
                )
            if (
                _fr10_mode == "tree_mtp"
                and _fr10_tree_src
                and len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ):
                _fr10_choices = sorted(
                    __import__("ast").literal_eval(_fr10_tree_src),
                    key=lambda _p: (len(_p), _p),
                )
                _fr10_depth_offsets = np.array(
                    [0] + [len(_fr10_choice) for _fr10_choice in _fr10_choices],
                    dtype=np.int64,
                )
                _fr10_spine_choices = [
                    _fr10_choice
                    for _fr10_choice in _fr10_choices
                    if all(int(_fr10_part) == 0 for _fr10_part in _fr10_choice)
                ]
                _fr10_leaf_choices = [
                    _fr10_choice
                    for _fr10_choice in _fr10_choices
                    if not all(int(_fr10_part) == 0 for _fr10_part in _fr10_choice)
                ]
                _fr10_spine_first_depth_offsets = np.array(
                    [0]
                    + [len(_fr10_choice) for _fr10_choice in _fr10_spine_choices]
                    + [len(_fr10_choice) for _fr10_choice in _fr10_leaf_choices],
                    dtype=np.int64,
                )
                _fr10_tree_n = int(len(_fr10_depth_offsets))
                _fr10_depth_pos = np.empty(
                    int(cu_num_tokens[-1]), dtype=np.int64
                )
                _fr10_spec_req_ids = set(
                    getattr(scheduler_output, "scheduled_spec_decode_tokens", {}).keys()
                )
                _fr10_out = 0
                _fr10_ok = True
                _fr10_bad = []
                for _fr10_req_idx, _fr10_sched in enumerate(
                    num_scheduled_tokens.tolist()
                ):
                    _fr10_sched = int(_fr10_sched)
                    _fr10_req_id = (
                        self.input_batch.req_ids[_fr10_req_idx]
                        if _fr10_req_idx < len(self.input_batch.req_ids)
                        else None
                    )
                    _fr10_is_spec_row = _fr10_req_id in _fr10_spec_req_ids
                    _fr10_base = max(
                        0,
                        int(self.input_batch.num_computed_tokens_cpu[_fr10_req_idx])
                        - 1,
                    )
                    if _fr10_is_spec_row and _fr10_sched == _fr10_tree_n:
                        _fr10_depth_pos[
                            _fr10_out : _fr10_out + _fr10_sched
                        ] = _fr10_base + _fr10_depth_offsets
                    else:
                        _fr10_depth_pos[
                            _fr10_out : _fr10_out + _fr10_sched
                        ] = positions_np[_fr10_out : _fr10_out + _fr10_sched]
                        if _fr10_is_spec_row:
                            _fr10_bad.append(
                                {
                                    "req_index": int(_fr10_req_idx),
                                    "req_id": str(_fr10_req_id),
                                    "scheduled": int(_fr10_sched),
                                    "expected": int(_fr10_tree_n),
                                }
                            )
                    _fr10_out += _fr10_sched
                if _fr10_out != int(cu_num_tokens[-1]):
                    _fr10_ok = False
                    _fr10_bad.append(
                        {
                            "reason": "total_mismatch",
                            "out": int(_fr10_out),
                            "total": int(cu_num_tokens[-1]),
                        }
                    )
                if _fr10_bad and __import__("os").environ.get(
                    "FR10_ALLOW_LINEAR_FALLBACK", "0"
                ) != "1":
                    raise RuntimeError(
                        "FR10 tree depth-position remap found non-tree spec rows: "
                        + repr(_fr10_bad[:8])
                    )
                if _fr10_ok:
                    self.positions[:total_num_scheduled_tokens].copy_(
                        torch.from_numpy(_fr10_depth_pos).to(
                            device=self.device, non_blocking=True
                        )
                    )
                    if __import__("os").environ.get("FR10_METRICS", "0") == "1":
                        global _FR10_TREE_DEPTH_POS_FH
                        try:
                            _FR10_TREE_DEPTH_POS_FH
                        except NameError:
                            _FR10_TREE_DEPTH_POS_FH = open(
                                __import__("os").environ.get(
                                    "FR10_TREE_DEPTH_POSITION_LOG",
                                    "/logs/fr10_tree_depth_positions.jsonl",
                                ),
                                "a",
                                buffering=1,
                            )
                        _FR10_TREE_DEPTH_POS_FH.write(
                            __import__("json").dumps(
                                {
                                    "event": "tree_depth_positions",
                                    "tree_n": int(_fr10_tree_n),
                                    "depth_offsets_row_order": [
                                        int(_x) for _x in _fr10_depth_offsets.tolist()
                                    ],
                                    "depth_offsets_spine_first": [
                                        int(_x)
                                        for _x in _fr10_spine_first_depth_offsets.tolist()
                                    ],
                                    "base_contract": "num_computed_tokens_cpu-1",
                                    "flat_first_tree": [
                                        int(_x)
                                        for _x in self.query_pos.np[
                                            : min(_fr10_tree_n, int(cu_num_tokens[-1]))
                                        ].tolist()
                                    ],
                                    "depth_first_tree": [
                                        int(_x)
                                        for _x in (
                                            _fr10_depth_pos[
                                                : min(
                                                    _fr10_tree_n,
                                                    int(cu_num_tokens[-1]),
                                                )
                                            ]
                                            - _fr10_depth_pos[0]
                                        ).tolist()
                                    ],
                                    "num_scheduled_tokens": [
                                        int(_x) for _x in num_scheduled_tokens.tolist()
                                    ],
                                },
                                sort_keys=True,
                            )
                            + chr(10)
                        )
        except Exception as _fr10_pos_exc:
            if __import__("os").environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1":
                raise RuntimeError(
                    "FR10 tree depth-position remap failed: "
                    + type(_fr10_pos_exc).__name__
                    + ":"
                    + str(_fr10_pos_exc)
                ) from _fr10_pos_exc

        # Copy the tensors to the GPU.
"""
    if anchor not in text:
        raise RuntimeError("gpu_model_runner slot-mapping position anchor not found")
    text = text.replace(anchor, inject, 1)
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
        "        except Exception as _fr10_mode_exc:\n"
        "            if __import__(\"os\").environ.get(\"FR10_ALLOW_LINEAR_FALLBACK\", \"0\") != \"1\":\n"
        "                raise RuntimeError(\"FR10 tree decode-mode global failed for gdn_linear_attn: \" + type(_fr10_mode_exc).__name__ + \":\" + str(_fr10_mode_exc)) from _fr10_mode_exc\n"
        "        try:\n"
        "            from vllm.v1.sample import rejection_sampler as _fr10_rejection_sampler\n"
        "            _fr10_rejection_sampler._FR10_DECODE_MODE = fr10_decode_mode\n"
        "        except Exception as _fr10_mode_exc:\n"
        "            if __import__(\"os\").environ.get(\"FR10_ALLOW_LINEAR_FALLBACK\", \"0\") != \"1\":\n"
        "                raise RuntimeError(\"FR10 tree decode-mode global failed for rejection_sampler: \" + type(_fr10_mode_exc).__name__ + \":\" + str(_fr10_mode_exc)) from _fr10_mode_exc\n"
        "\n"
        "        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0\n"
    )
    if anchor not in text:
        raise RuntimeError("gpu_model_runner use_spec_decode anchor not found")
    text = text.replace(anchor, inject, 1)
    GPU_MODEL_RUNNER_PATH.write_text(text)
    return True


def _patch_mamba_utils_tree_accept_bias() -> bool:
    text = MAMBA_UTILS_PATH.read_text()
    sentinel = "# FR10_TREE_MAMBA_ACCEPTED_COPY_BIAS"
    if sentinel in text:
        return False

    text = text.replace(
        "import dataclasses\nimport itertools\n",
        "import dataclasses\nimport itertools\nimport json\nimport os\nimport time\n",
        1,
    )
    helper_anchor = """
def collect_mamba_copy_meta(
"""
    helper = r'''
# FR10_TREE_MAMBA_ACCEPTED_COPY_BIAS: stock hybrid Mamba copies use a
# linear accepted-token offset. For tree_mtp, translate that offset through
# the accepted tree path before the copy helper dereferences state rows.
_FR10_TREE_ACCEPTED_PATH_BY_REQ_ID: dict[str, list[int]] = {}


def _fr10_tree_mamba_mode_active() -> bool:
    try:
        from vllm.v1.sample import rejection_sampler as _fr10_rs
        mode = getattr(
            _fr10_rs,
            "_FR10_DECODE_MODE",
            os.environ.get("FR10_DECODE_MODE_DEFAULT", "tree_mtp"),
        )
    except Exception:
        mode = os.environ.get("FR10_DECODE_MODE_DEFAULT", "tree_mtp")
    return (
        mode == "tree_mtp"
        and os.environ.get("FR10_ENABLE_TREE_GDN", "1") == "1"
    )


def _fr10_tree_current_accepted_path(batch_index: int) -> list[int] | None:
    try:
        from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr10_gdn
        lens = getattr(_fr10_gdn, "_LUMO_FA_LAST_ACCEPTED_TREE_LENS", [])
        paths = getattr(_fr10_gdn, "_LUMO_FA_LAST_ACCEPTED_TREE_NODE_PATHS", [])
        rows = getattr(_fr10_gdn, "_LUMO_FA_LAST_ACCEPTED_TREE_ROWS", [])
        if batch_index >= len(lens):
            return None
        accepted_len = int(lens[batch_index])
        if accepted_len <= 0:
            return []
        if batch_index < len(paths) and paths[batch_index]:
            return [int(x) for x in paths[batch_index][:accepted_len]]
        if batch_index < len(rows):
            return [int(rows[batch_index])]
    except Exception as exc:
        if os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1":
            raise RuntimeError(
                "FR10 tree Mamba accepted-path lookup failed: "
                + type(exc).__name__
                + ":"
                + str(exc)
            ) from exc
    return None


def _fr10_tree_record_request_accept(req_id: str, batch_index: int) -> None:
    if not _fr10_tree_mamba_mode_active():
        return
    path = _fr10_tree_current_accepted_path(batch_index)
    if path is None:
        return
    if path:
        _FR10_TREE_ACCEPTED_PATH_BY_REQ_ID[str(req_id)] = path
    else:
        _FR10_TREE_ACCEPTED_PATH_BY_REQ_ID.pop(str(req_id), None)


def _fr10_tree_accept_token_bias(
    req_id: str,
    batch_index: int,
    linear_bias: int,
    *,
    phase: str,
) -> int:
    if not _fr10_tree_mamba_mode_active():
        return int(linear_bias)
    path = None
    if phase == "postprocess":
        path = _fr10_tree_current_accepted_path(batch_index)
    if path is None:
        path = _FR10_TREE_ACCEPTED_PATH_BY_REQ_ID.get(str(req_id))
    if not path:
        if int(linear_bias) <= 0:
            return int(linear_bias)
        if os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") == "1":
            return int(linear_bias)
        raise RuntimeError(
            "FR10 tree Mamba copy missing accepted path: "
            f"phase={phase} req_id={req_id} batch_index={batch_index} "
            f"linear_bias={linear_bias}"
        )
    path_index = max(0, min(int(linear_bias), len(path) - 1))
    tree_bias = int(path[path_index])
    if os.environ.get("FR10_METRICS", "0") == "1":
        try:
            global _FR10_TREE_MAMBA_COPY_FH
            try:
                _FR10_TREE_MAMBA_COPY_FH
            except NameError:
                _FR10_TREE_MAMBA_COPY_FH = open(
                    os.environ.get(
                        "FR10_TREE_MAMBA_COPY_LOG",
                        "/logs/fr10_tree_mamba_copy.jsonl",
                    ),
                    "a",
                    buffering=1,
                )
            _FR10_TREE_MAMBA_COPY_FH.write(
                json.dumps(
                    {
                        "event": "tree_mamba_copy_bias",
                        "ts": round(time.time(), 4),
                        "phase": phase,
                        "req_id": str(req_id),
                        "batch_index": int(batch_index),
                        "linear_bias": int(linear_bias),
                        "tree_bias": int(tree_bias),
                        "accepted_path": [int(x) for x in path],
                    }
                )
                + chr(10)
            )
        except Exception:
            pass
    return tree_bias


def collect_mamba_copy_meta(
'''
    if helper_anchor not in text:
        raise RuntimeError("mamba collect_mamba_copy_meta anchor not found")
    text = text.replace(helper_anchor, helper, 1)

    preprocess_old = """                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state,
"""
    preprocess_new = """                _fr10_tree_accept_token_bias(
                    req_id,
                    i,
                    input_batch.num_accepted_tokens_cpu[i] - 1,
                    phase="preprocess",
                ),
                req_state,
"""
    if preprocess_old not in text:
        raise RuntimeError("mamba preprocess accept-token-bias anchor not found")
    text = text.replace(preprocess_old, preprocess_new, 1)

    postprocess_anchor = """        req_state = requests[req_id]
        num_computed_tokens = req_state.num_computed_tokens
"""
    postprocess_inject = """        req_state = requests[req_id]
        _fr10_tree_record_request_accept(req_id, i)
        num_computed_tokens = req_state.num_computed_tokens
"""
    if postprocess_anchor not in text:
        raise RuntimeError("mamba postprocess req_state anchor not found")
    text = text.replace(postprocess_anchor, postprocess_inject, 1)

    postprocess_old = """            accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
            src_block_idx = mamba_state_idx[req_id]
"""
    postprocess_new = """            accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
            accept_token_bias = _fr10_tree_accept_token_bias(
                req_id,
                i,
                accept_token_bias,
                phase="postprocess",
            )
            src_block_idx = mamba_state_idx[req_id]
"""
    if postprocess_old not in text:
        raise RuntimeError("mamba postprocess accept-token-bias anchor not found")
    text = text.replace(postprocess_old, postprocess_new, 1)

    MAMBA_UTILS_PATH.write_text(text)
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
        _fr10_tree_choices_current = [
            tuple(_x) for _x in getattr(self, "tree_choices", [])
        ]
        _fr10_is_caterpillar = (
            _fr10_active_decode_mode == "tree_mtp"
            and int(self.num_speculative_tokens) == 9
            and _fr10_tree_choices_current == _fr10_caterpillar_choices
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
        _fr10_tree_expected = (
            _fr10_active_decode_mode == "tree_mtp"
            and (
                "speculative_token_tree" in os.environ.get("SPEC_CONFIG", "")
                or len(_fr10_tree_choices_current) == 9
            )
        )
        if (
            _fr10_tree_expected
            and not _fr10_is_caterpillar
            and os.environ.get("FR10_ALLOW_LINEAR_FALLBACK", "0") != "1"
        ):
            raise RuntimeError(
                "FR10 caterpillar drafter disengaged: "
                + "num_speculative_tokens="
                + str(int(self.num_speculative_tokens))
                + " tree_choices="
                + repr(_fr10_tree_choices_current)
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
    did_patch = False
    text = text.replace("import ast\n", "import ast\nimport json\nimport os\n", 1)
    if sentinel not in text:
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
        did_patch = True

    mask_sentinel = "# FR10_ROOT_ATTENTION_BIAS_CAPTURE"
    if mask_sentinel not in text:
        old_return = """    return tree_attn_mask
"""
        new_return = f"""    {mask_sentinel}: dump the runtime root/bonus attention bias row.
    try:
        _fr10_mask_path = os.environ.get("FR10_ROOT_HIDDEN_CAPTURE")
        if _fr10_mask_path and not os.path.exists(_fr10_mask_path + ".tree_attn_bias.pt"):
            torch.save(
                {{
                    "root_row": tree_attn_mask[0].detach().cpu(),
                    "full_bias": tree_attn_mask.detach().cpu(),
                    "sorted_tree_choices": [tuple(_p) for _p in sorted_tree_choices],
                    "depth_counts": [int(_x) for _x in depth_counts],
                }},
                _fr10_mask_path + ".tree_attn_bias.pt",
            )
    except Exception:
        pass
    return tree_attn_mask
"""
        if old_return not in text:
            raise RuntimeError("tree attention return anchor not found")
        text = text.replace(old_return, new_return, 1)
        did_patch = True
    TREE_ATTN_PATH.write_text(text)
    return did_patch


def _patch_qwen_root_hidden_capture() -> bool:
    """Diagnostic-only root hidden/logit capture for FR10 verify-forward bisection."""

    did_patch = False
    text = QWEN3_NEXT_PATH.read_text()
    sentinel = "# FR10_ROOT_HIDDEN_CAPTURE"
    if sentinel not in text:
        text = text.replace(
            "from itertools import islice\n",
            "from itertools import islice\nimport os\nfrom pathlib import Path\n",
            1,
        )
        helper_anchor = "logger = init_logger(__name__)\n"
        helper = '''logger = init_logger(__name__)


# FR10_ROOT_HIDDEN_CAPTURE
_FR10_ROOT_HIDDEN_CAPTURED = False


def _fr10_root_hidden_capture_start(self, positions, hidden_states):
    global _FR10_ROOT_HIDDEN_CAPTURED
    path = os.environ.get("FR10_ROOT_HIDDEN_CAPTURE")
    if not path or _FR10_ROOT_HIDDEN_CAPTURED:
        return None
    try:
        desired = os.environ.get("FR10_ROOT_HIDDEN_CAPTURE_NUM_TOKENS")
        if desired:
            desired_counts = {
                int(_x.strip()) for _x in desired.split(",") if _x.strip()
            }
            if int(hidden_states.shape[0]) not in desired_counts:
                return None
        pos_cpu = positions.detach().cpu()
        root_position_env = os.environ.get("FR10_ROOT_HIDDEN_CAPTURE_POSITION")
        if root_position_env:
            token_positions = pos_cpu[0] if pos_cpu.ndim == 2 else pos_cpu
            matches = (token_positions.reshape(-1) == int(root_position_env)).nonzero()
            if matches.numel() == 0:
                return None
            root_row = int(matches.reshape(-1)[0].item())
        else:
            root_row = int(os.environ.get("FR10_ROOT_HIDDEN_CAPTURE_ROOT_ROW", "0"))
        if root_row < 0 or root_row >= int(hidden_states.shape[0]):
            return None
        layer_types = list(getattr(getattr(self, "config", None), "layer_types", []))
        token_positions = pos_cpu[0] if pos_cpu.ndim == 2 else pos_cpu
        return {
            "source": "Qwen3NextModel.forward",
            "path": path,
            "num_tokens": int(hidden_states.shape[0]),
            "hidden_size": int(hidden_states.shape[-1]),
            "root_row": root_row,
            "input_root_hidden": hidden_states[root_row]
            .detach()
            .to(torch.float32)
            .cpu(),
            "positions_shape": list(positions.shape),
            "positions": pos_cpu,
            "root_position": token_positions.reshape(-1)[root_row].item(),
            "positions_first16": pos_cpu.reshape(-1)[:16].tolist(),
            "layer_types": layer_types,
            "layers": [],
        }
    except Exception as exc:
        logger.warning("FR10 root hidden capture start failed: %s", exc)
        return None


def _fr10_root_hidden_capture_layer(payload, layer_idx, hidden_states, residual):
    if payload is None:
        return
    try:
        root_row = int(payload["root_row"])
        root = hidden_states[root_row].detach().to(torch.float32).cpu()
        residual_root = (
            residual[root_row].detach().to(torch.float32).cpu()
            if residual is not None
            else None
        )
        layer_types = payload.get("layer_types") or []
        layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else None
        payload["layers"].append({
            "layer_idx": int(layer_idx),
            "layer_type": layer_type,
            "root_hidden": root,
            "root_residual": residual_root,
        })
    except Exception as exc:
        logger.warning("FR10 root hidden capture layer failed: %s", exc)


def _fr10_root_hidden_capture_finish(payload, hidden_states):
    global _FR10_ROOT_HIDDEN_CAPTURED
    if payload is None:
        return
    try:
        root_row = int(payload["root_row"])
        payload["final_norm_root_hidden"] = (
            hidden_states[root_row].detach().to(torch.float32).cpu()
        )
        out = Path(str(payload["path"]))
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out)
        _FR10_ROOT_HIDDEN_CAPTURED = True
    except Exception as exc:
        logger.warning("FR10 root hidden capture finish failed: %s", exc)


'''
        if helper_anchor not in text:
            raise RuntimeError("qwen3_next logger anchor not found")
        text = text.replace(helper_anchor, helper, 1)
        old_loop = """        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
"""
        new_loop = """        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        _fr10_root_capture = _fr10_root_hidden_capture_start(
            self, positions, hidden_states
        )
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )
            _fr10_root_hidden_capture_layer(
                _fr10_root_capture, layer_idx, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
"""
        if old_loop not in text:
            raise RuntimeError("qwen3_next layer loop anchor not found")
        text = text.replace(old_loop, new_loop, 1)
        old_norm = """        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states
"""
        new_norm = """        hidden_states, _ = self.norm(hidden_states, residual)
        _fr10_root_hidden_capture_finish(_fr10_root_capture, hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states
"""
        if old_norm not in text:
            raise RuntimeError("qwen3_next norm anchor not found")
        text = text.replace(old_norm, new_norm, 1)
        QWEN3_NEXT_PATH.write_text(text)
        did_patch = True

    layer_sentinel = "# FR10_LAYER_HIDDEN_CAPTURE"
    if layer_sentinel not in text:
        helper_anchor = "def _fr10_root_hidden_capture_start(self, positions, hidden_states):\n"
        helper = '''# FR10_LAYER_HIDDEN_CAPTURE
def _fr10_layer_hidden_capture_start(self, positions, hidden_states):
    path = os.environ.get("FR10_LAYER_HIDDEN_CAPTURE")
    if not path:
        return None
    try:
        desired = os.environ.get("FR10_LAYER_HIDDEN_CAPTURE_NUM_TOKENS")
        if desired:
            desired_counts = {
                int(_x.strip()) for _x in desired.split(",") if _x.strip()
            }
            if int(hidden_states.shape[0]) not in desired_counts:
                return None
        seen = int(globals().get("_FR10_LAYER_HIDDEN_CAPTURE_SEEN", 0))
        skip = int(os.environ.get("FR10_LAYER_HIDDEN_CAPTURE_SKIP", "0"))
        limit = int(os.environ.get("FR10_LAYER_HIDDEN_CAPTURE_LIMIT", "1"))
        saved = int(globals().get("_FR10_LAYER_HIDDEN_CAPTURE_SAVED", 0))
        globals()["_FR10_LAYER_HIDDEN_CAPTURE_SEEN"] = seen + 1
        if seen < skip or saved >= limit:
            return None
        rows_env = os.environ.get("FR10_LAYER_HIDDEN_CAPTURE_ROWS", "")
        if rows_env:
            rows = [
                int(_x.strip()) for _x in rows_env.split(",") if _x.strip()
            ]
        else:
            rows = list(range(int(hidden_states.shape[0])))
        rows = [row for row in rows if 0 <= row < int(hidden_states.shape[0])]
        if not rows:
            return None
        row_index = torch.tensor(rows, dtype=torch.long, device=hidden_states.device)
        pos_cpu = positions.detach().cpu()
        layer_types = list(getattr(getattr(self, "config", None), "layer_types", []))
        out = Path(str(path))
        call_out = out.with_name(out.stem + ".call" + str(saved) + out.suffix)
        return {
            "source": "Qwen3NextModel.forward",
            "path": str(path),
            "call_path": str(call_out),
            "capture_call_index": int(seen),
            "capture_saved_index": int(saved),
            "num_tokens": int(hidden_states.shape[0]),
            "hidden_size": int(hidden_states.shape[-1]),
            "rows": rows,
            "positions_shape": list(positions.shape),
            "positions": pos_cpu,
            "positions_first16": pos_cpu.reshape(-1)[:16].tolist(),
            "layer_types": layer_types,
            "input_hidden": hidden_states.index_select(0, row_index)
            .detach()
            .to(torch.float32)
            .cpu(),
            "layers": [],
        }
    except Exception as exc:
        logger.warning("FR10 layer hidden capture start failed: %s", exc)
        return None


def _fr10_layer_hidden_capture_layer(payload, layer_idx, hidden_states, residual):
    if payload is None:
        return
    try:
        rows = torch.tensor(
            [int(_x) for _x in payload["rows"]],
            dtype=torch.long,
            device=hidden_states.device,
        )
        layer_types = payload.get("layer_types") or []
        layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else None
        item = {
            "layer_idx": int(layer_idx),
            "layer_type": layer_type,
            "hidden": hidden_states.index_select(0, rows)
            .detach()
            .to(torch.float32)
            .cpu(),
        }
        if residual is not None:
            item["residual"] = residual.index_select(0, rows).detach().to(torch.float32).cpu()
        payload["layers"].append(item)
    except Exception as exc:
        logger.warning("FR10 layer hidden capture layer failed: %s", exc)


def _fr10_layer_hidden_capture_finish(payload, hidden_states):
    if payload is None:
        return
    try:
        rows = torch.tensor(
            [int(_x) for _x in payload["rows"]],
            dtype=torch.long,
            device=hidden_states.device,
        )
        payload["final_norm_hidden"] = hidden_states.index_select(0, rows).detach().to(torch.float32).cpu()
        out = Path(str(payload["call_path"]))
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out)
        if int(payload.get("capture_saved_index", 0)) == 0:
            torch.save(payload, Path(str(payload["path"])))
        globals()["_FR10_LAYER_HIDDEN_CAPTURE_SAVED"] = (
            int(globals().get("_FR10_LAYER_HIDDEN_CAPTURE_SAVED", 0)) + 1
        )
    except Exception as exc:
        logger.warning("FR10 layer hidden capture finish failed: %s", exc)


'''
        if helper_anchor not in text:
            raise RuntimeError("qwen3_next root hidden helper anchor not found")
        text = text.replace(helper_anchor, helper + helper_anchor, 1)
        root_start = """        _fr10_root_capture = _fr10_root_hidden_capture_start(
            self, positions, hidden_states
        )
"""
        layer_start = root_start + """        _fr10_layer_capture = _fr10_layer_hidden_capture_start(
            self, positions, hidden_states
        )
"""
        if root_start not in text:
            raise RuntimeError("qwen3_next FR10 root capture start anchor not found")
        text = text.replace(root_start, layer_start, 1)
        root_layer = """            _fr10_root_hidden_capture_layer(
                _fr10_root_capture, layer_idx, hidden_states, residual
            )
"""
        layer_layer = root_layer + """            _fr10_layer_hidden_capture_layer(
                _fr10_layer_capture, layer_idx, hidden_states, residual
            )
"""
        if root_layer not in text:
            raise RuntimeError("qwen3_next FR10 root capture layer anchor not found")
        text = text.replace(root_layer, layer_layer, 1)
        root_finish = """        _fr10_root_hidden_capture_finish(_fr10_root_capture, hidden_states)
"""
        layer_finish = root_finish + """        _fr10_layer_hidden_capture_finish(_fr10_layer_capture, hidden_states)
"""
        if root_finish not in text:
            raise RuntimeError("qwen3_next FR10 root capture finish anchor not found")
        text = text.replace(root_finish, layer_finish, 1)
        QWEN3_NEXT_PATH.write_text(text)
        did_patch = True

    text = QWEN3_5_PATH.read_text()
    logit_sentinel = "# FR10_ROOT_LOGIT_CAPTURE"
    if logit_sentinel not in text:
        text = text.replace(
            "import typing\n",
            "import typing\nimport os\nfrom pathlib import Path\n",
            1,
        )
        old_compute = """    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)
"""
        new_compute = """    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        # FR10_ROOT_LOGIT_CAPTURE: dump the same root row scored by the LM head.
        try:
            _fr10_path = os.environ.get("FR10_ROOT_HIDDEN_CAPTURE")
            _fr10_done = bool(globals().get("_FR10_ROOT_LOGIT_CAPTURED", False))
            if _fr10_path and not _fr10_done and logits is not None:
                _fr10_desired = os.environ.get(
                    "FR10_ROOT_LOGIT_CAPTURE_NUM_TOKENS",
                    os.environ.get("FR10_ROOT_HIDDEN_CAPTURE_NUM_TOKENS"),
                )
                if _fr10_desired:
                    _fr10_desired_counts = {
                        int(_x.strip())
                        for _x in _fr10_desired.split(",")
                        if _x.strip()
                    }
                else:
                    _fr10_desired_counts = set()
                if (
                    not _fr10_desired_counts
                    or int(hidden_states.shape[0]) in _fr10_desired_counts
                ):
                    _fr10_root_row = int(os.environ.get(
                        "FR10_ROOT_LOGIT_CAPTURE_ROOT_ROW",
                        os.environ.get("FR10_ROOT_HIDDEN_CAPTURE_ROOT_ROW", "0"),
                    ))
                    if 0 <= _fr10_root_row < int(logits.shape[0]):
                        _fr10_out = Path(_fr10_path + ".logits.pt")
                        _fr10_out.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({
                            "source": "Qwen3_5ForCausalLMBase.compute_logits",
                            "num_tokens": int(hidden_states.shape[0]),
                            "root_row": int(_fr10_root_row),
                            "root_logits": logits[_fr10_root_row].detach().to(torch.float32).cpu(),
                        }, _fr10_out)
                        globals()["_FR10_ROOT_LOGIT_CAPTURED"] = True
        except Exception as _fr10_exc:
            logger.warning("FR10 root logit capture failed: %s", _fr10_exc)
        return logits
"""
        if old_compute not in text:
            raise RuntimeError("qwen3_5 compute_logits anchor not found")
        text = text.replace(old_compute, new_compute, 1)
        QWEN3_5_PATH.write_text(text)
        did_patch = True

    return did_patch



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
        (QWEN3_NEXT_PATH, _patch_qwen_root_hidden_capture()),
        (GPU_MODEL_RUNNER_PATH, _patch_gpu_model_runner_tree_metadata()),
        (GPU_MODEL_RUNNER_PATH, _patch_gpu_model_runner_tree_depth_positions()),
        (GPU_MODEL_RUNNER_PATH, _patch_gpu_model_runner_decode_mode_globals()),
        (MAMBA_UTILS_PATH, _patch_mamba_utils_tree_accept_bias()),
        (REJECTION_SAMPLER_PATH, _patch_rejection_sampler_tree_lcp()),
    ]
    patched: dict[str, bool] = {}
    for path, did_patch in patch_steps:
        patched[str(path)] = patched.get(str(path), False) or did_patch
    if patched.get(str(QWEN3_NEXT_PATH), False):
        patched[str(QWEN3_5_PATH)] = True
    import py_compile

    for path, did_patch in patched.items():
        if did_patch:
            py_compile.compile(path, doraise=True)
    print(patched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

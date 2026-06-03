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
            "    if not _fr10_metrics_enabled():\n"
            "        return\n"
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
            "            if _fr10_metrics_enabled():\n"
            "                self.fr10_tree_invocation_counter = torch.zeros((1,), dtype=torch.int32, device=device)\n"
            "                _fr10_register_tree_counter(\n"
            "                    shape=f\"n{n}_pad{n_pad}\",\n"
            "                    parent=parent,\n"
            "                    counter=self.fr10_tree_invocation_counter,\n"
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
                        invocation_counter=(
                            attn_metadata.fr10_tree_invocation_counter
                            if os.environ.get("FR10_METRICS", "0") == "1"
                            else None
                        ),
                    )
                    core_attn_out_spec[0, start:end] = tree_out[:tree_n]
                    ssm_state.index_copy_(
                        0,
                        spec_state_indices_tensor[fr10_b, :tree_n].to(torch.long),
                        tree_state[:tree_n].to(ssm_state.dtype),
                    )
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
        accepted_row = int(best_path[best_lcp - 1]) + 1 if best_lcp > 0 else 0
        accepted_rows.append(accepted_row)
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
    try:
        from vllm.model_executor.layers.mamba import gdn_linear_attn as _lumo_tree_commit_gdn
        _lumo_tree_commit_gdn._LUMO_FA_LAST_ACCEPTED_TREE_ROWS = [
            int(x) for x in accepted_rows
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
        if lumo_tree_parent_indices is not None and sampling_metadata.all_greedy:
            tree_self_logits = logits[metadata.tree_self_logits_indices]
            tree_self_logits = tree_self_logits.to(torch.float32)
            if not self.is_processed_logprobs_mode:
                tree_self_logits = tree_self_logits.clone()
            tree_self_logits = self.apply_logits_processors(
                tree_self_logits, sampling_metadata, metadata
            )
            tree_self_logits = apply_sampling_constraints(
                tree_self_logits,
                metadata.cu_num_draft_tokens,
                sampling_metadata,
            )
            lumo_tree_token_ids = torch.stack(
                [
                    target_logits.argmax(dim=-1).to(torch.int32),
                    tree_self_logits.argmax(dim=-1).to(torch.int32),
                ],
                dim=0,
            ).contiguous()

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
        try:
            _lspec = getattr(self.vllm_config, "speculative_config", None)
            _ltree_src = getattr(_lspec, "speculative_token_tree", None) if _lspec is not None else None
            if _ltree_src:
                _choices = __import__("ast").literal_eval(_ltree_src)
                _max_depth = max(len(_t) for _t in _choices)
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
        except Exception:
            lumo_tree_parent_indices = None
            lumo_tree_self_logits_indices = None
            lumo_draft_token_indices = None

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


def _patch_mamba_postprocess_tree_rows() -> bool:
    text = MAMBA_UTILS_PATH.read_text()
    sentinel = "# LUMO_TREE_STATE_COMMIT_ROWS"
    if sentinel in text:
        return False

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
            # LUMO_TREE_STATE_COMMIT_ROWS: for FR10 tree verification, the
            # accepted final recurrent state lives at the accepted tree node
            # row, not at the flat linear accepted-count row. The scheduler's
            # accepted-token accounting remains unchanged; only the source row
            # used for the state copy is redirected.
            try:
                from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr10_gdn
                _fr10_rows = getattr(_fr10_gdn, "_LUMO_FA_LAST_ACCEPTED_TREE_ROWS", None)
                if _fr10_rows is not None and i < len(_fr10_rows):
                    _fr10_row = int(_fr10_rows[i])
                    if _fr10_row > 0:
                        accept_token_bias = _fr10_row
            except Exception:
                pass
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
    if old not in text:
        raise RuntimeError("mamba postprocess accepted-row copy anchor not found")
    text = text.replace(old, new, 1)
    MAMBA_UTILS_PATH.write_text(text)
    return True


def main() -> int:
    patched = {
        str(GDN_ATTN_PATH): _patch_gdn_attn(),
        str(GDN_LINEAR_PATH): _patch_gdn_linear(),
        str(SCHEDULER_PATH): _patch_scheduler_spec_trace(),
        str(GPU_MODEL_RUNNER_PATH): _patch_gpu_model_runner_tree_metadata(),
        str(MAMBA_UTILS_PATH): _patch_mamba_postprocess_tree_rows(),
        str(REJECTION_SAMPLER_PATH): _patch_rejection_sampler_tree_lcp(),
    }
    import py_compile

    for path, did_patch in patched.items():
        if did_patch:
            py_compile.compile(path, doraise=True)
    print(patched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

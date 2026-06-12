from __future__ import annotations

from pathlib import Path


def test_phase4_patcher_routes_sampled_tree_to_verified_committer() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()

    assert "sample_multidraft_rejection_step as _fr10_sample_step" in text
    assert "sample_deterministic_multidraft_rejection_step as _fr10_sample_det_step" in text
    assert "if tree_parent_indices is not None and not sampling_metadata.all_greedy" in text
    assert "return _lumo_tree_canonical_multidraft_sample(" in text
    assert "if draft_probs_cpu is None:" in text
    assert "'committer_step_trace': step_trace_rows" in text
    assert "'target_prob_at_draft_token_ids':" in text


def test_phase4_patcher_logs_tree_and_linear_logit_gathers() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()

    assert '"event": "tree_logit_gather"' in text
    assert '"event": "linear_logit_gather"' in text
    assert '"target_logits_index":' in text
    assert '"self_logits_index":' in text
    assert '"draft_logits_index":' in text
    assert '"target_prob_draft":' in text


def test_phase4_tree_committer_receives_constrained_target_logits_once() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()

    target_constraint = """target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )
"""
    tree_branch = 'lumo_tree_parent_indices = getattr(metadata, "tree_parent_indices", None)'
    committer_call = "return _lumo_tree_canonical_multidraft_sample("

    assert target_constraint in text
    assert text.index(target_constraint) < text.index(tree_branch)
    assert text.index(tree_branch) < text.index(committer_call)
    assert text.count(target_constraint) == 1


def test_phase4_tree_committer_publishes_accepted_node_without_dead_mamba_redirect() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()

    assert "accepted_row = int(best_path[best_lcp - 1]) if best_lcp > 0 else 0" in text
    assert "accepted_row = int(current_parent)" in text
    assert "_LUMO_FA_LAST_ACCEPTED_TREE_LENS" in text
    assert "_patch_mamba_postprocess_tree_rows" not in text
    assert "LUMO_TREE_STATE_COMMIT_ROWS" not in text
    assert "and accept_token_bias > 0" not in text


def test_phase4_mamba_stock_copies_use_tree_accepted_bias() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()

    assert "MAMBA_UTILS_PATH" in text
    assert "_patch_mamba_utils_tree_accept_bias()" in text
    assert "FR10_TREE_MAMBA_ACCEPTED_COPY_BIAS" in text
    assert "_fr10_tree_accept_token_bias(" in text
    assert "phase=\"preprocess\"" in text
    assert "phase=\"postprocess\"" in text
    assert "_FR10_TREE_ACCEPTED_PATH_BY_REQ_ID" in text
    assert "tree_mamba_copy_bias" in text
    assert "FR13_TREE_MAMBA_CONV_NODE_COPY" in text
    assert "src_state = state[block_ids[src_block_pos]]" in text
    assert "_LUMO_FA_SPEC_ROW_REQ_IDS" in text
    assert "missing request-keyed replay row ids" in text
    assert "missing committer rows for staged" in text
    assert "_fr13_replay_gdn_node_paths" in text


def test_phase4_patcher_exports_src_native_handoff_payload() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()
    gate_text = Path("scripts/fr10_src_native_handoff_gate.py").read_text()

    assert "_LUMO_FA_LAST_ACCEPTED_TREE_NODE_PATHS" in text
    assert "_LUMO_FA_LAST_ACCEPTED_TREE_TOKEN_IDS" in text
    assert "FR10_TREE_GDN_SRC_NATIVE_PAYLOAD" in text
    assert "next_read_ssm_state" in text
    assert "next_read_conv_state" in text
    assert '"value_tree": _fr10_prev_read["value_tree"]' in text
    assert '"g_tree": _fr10_prev_read["g_tree"]' in text
    assert '"beta_tree": _fr10_prev_read["beta_tree"]' in text
    assert "accepted_gdn_node_paths" in text
    assert "int(_node_id) + 1" in text
    assert '_node_major_spec_tensor(payload, "value_spec")' in gate_text
    assert '_node_major_spec_tensor(payload, "a")' in gate_text
    assert '_node_major_spec_tensor(payload, "b")' in gate_text


def test_phase4_tree_disengagement_raises_by_default() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()

    assert "FR10_ALLOW_LINEAR_FALLBACK" in text
    assert "FR10 tree metadata disengaged:" in text
    assert "FR10 tree causal-conv disengaged:" in text
    assert "FR10 tree scan disengaged:" in text
    assert "eligible_tree_spec_row_flat_fallback" in text
    assert "FR10 sampled tree committer disengaged: missing_tree_parent_indices" in text
    assert 'hasattr(metadata, "tree_parent_indices")' in text
    assert "FR10 caterpillar drafter disengaged:" in text
    assert "_fr10_tree_expected = (" in text
    assert '"speculative_token_tree" in os.environ.get("SPEC_CONFIG", "")' in text
    assert "or len(_fr10_tree_choices_current) == 9" in text
    assert "FR10 tree decode-mode global failed for gdn_linear_attn" in text
    assert "FR10 tree committer failed to publish accepted rows:" in text


def test_phase4_handoff_logs_accepted_bank_row_alias() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()

    assert '"accepted_spec_state_bank_row":' in text
    assert '"accepted_bank_row":' in text
    assert '"value_spec": _fr10_prev_read["value_spec"]' in text
    assert '"value_spec": value_spec[0, start:end]' in text


def test_phase4_seeds_next_tree_read_linear_path_from_accepted_nodes() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()
    kernel_text = Path("src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py").read_text()

    assert "_FR10_TREE_READ_PREV" in text
    assert "_LUMO_FA_LAST_ACCEPTED_TREE_NODE_PATHS" in text
    assert "_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR" in text
    assert "_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR" in text
    assert "launch_tree_state_linear_remap(" in text
    assert "missing_accepted_path_device_tensor" in text
    assert "missing_accepted_lens_device_tensor" in text
    assert '"event": "tree_length_alignment"' in text
    assert '"metadata_num_accepted_tokens": _fr10_meta_len' in text
    assert '"accepted_tree_len": int(_fr10_path_len)' in text
    assert "_fr10_seed_path_tensor = torch.tensor" not in text
    assert "_fr10_path0_state_rows = _fr10_path0_nodes[1:]" not in text
    assert '"value_tree": _fr10_prev_read["value_tree"]' in text
    assert '"g_tree": _fr10_prev_read["g_tree"]' in text
    assert '"beta_tree": _fr10_prev_read["beta_tree"]' in text
    assert "_LUMO_FA_LAST_ACCEPTED_TREE_NODE_PATHS = [\n            [int(x) for x in row] for row in accepted_gdn_node_paths\n        ]" in text
    assert "h0_num_accepted_tokens=_fr10_accepted_lens_tensor" in text
    assert "num_accepted_tokens=_fr10_accepted_lens_tensor" in text
    assert "_fr10_conv_read_cols = torch.clamp(" in text
    assert ".to(torch.long),\n                            min=0," in text
    assert ".to(torch.long)\n                            - 1," not in text
    assert "_fr10_prior_conv_bank_rows = spec_state_indices_tensor" in text
    assert "_fr10_prior_cols = _fr10_prior_col_base" in text
    assert "_fr10_prior_col_base = torch.arange(" in text
    assert "compact_head" in text
    assert "_fr10_use_rolled_tail_prior" not in text
    assert 'int(_fr13_layer_idx) >= 4' not in text
    assert "rolled_tail_remapped" not in text
    assert "legacy_remapped_head" not in text
    # Co-updated for FR13_TREE_CONV_FUSED (FIX-3): the legacy per-node
    # write-back loop moved one indent level into the byte-verbatim OFF arm
    # (else branch); the ops are unchanged.
    assert "_fr10_node_state_source = torch.cat(\n                                    (_fr10_prior_window.transpose(0, 1), _fr10_node_x)," in text
    assert "FR11_TREE_CONV_NATIVE_BF16_TAPS" in text
    assert "def _fr11_conv_tap_product" in text
    assert "h0_use_accepted_column=True" in text
    assert "accepted_linear_read_col" in text
    assert "linear_column_coincide" in text
    assert "_fr10_read_col" in text
    assert "self.fr10_tree_accepted_path_bs = max(" in text
    assert "self.vllm_config.scheduler_config.max_num_seqs" in text
    assert "_linear_remap_rows_kernel" in kernel_text
    assert "src_col = tl.load(" in kernel_text
    assert "spec_state_indices + pid_b * SPEC_COLS + src_col" in kernel_text
    assert "spec_state_indices_tensor[\n                                    _fr10_seed_b, 0" not in text
    assert "spec_state_indices_tensor[\n                                    fr10_b, 0" not in text


def test_phase4_tree_kernel_gathers_h0_from_accepted_linear_column() -> None:
    text = Path("src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py").read_text()

    assert "h0_num_accepted_tokens" in text
    assert "H0_USE_ACCEPTED_COLUMN" in text
    assert "H0_BATCH_INDEX" in text
    assert "tl.load(h0_num_accepted_tokens + H0_BATCH_INDEX)" in text
    assert "num_warps=8" in text


def test_forked_fa2_tree_launcher_defaults_to_prefill_native_metrics_off() -> None:
    text = Path("scripts/fr13_launch_forked_fa2_tree_server.sh").read_text()

    assert "FR10_METRICS=${FR10_METRICS:-0}" in text
    assert "FR13_FA2_PREFILL_NATIVE=${FR13_FA2_PREFILL_NATIVE:-1}" in text
    assert "LUMO_MTP_DRAFT_TRACE_FILE=${LUMO_MTP_DRAFT_TRACE_FILE:-}" in text
    assert "LUMO_TREE_SAMPLER_DEBUG_LOG=${LUMO_TREE_SAMPLER_DEBUG_LOG:-}" in text
    assert "LUMO_TREE_PATH_LCP_LOG=${LUMO_TREE_PATH_LCP_LOG:-}" in text
    assert '-e FR10_METRICS="$FR10_METRICS"' in text
    assert '-e FR13_FA2_PREFILL_NATIVE="$FR13_FA2_PREFILL_NATIVE"' in text
    assert '-e LUMO_MTP_DRAFT_TRACE_FILE="$LUMO_MTP_DRAFT_TRACE_FILE"' in text
    assert '-e LUMO_TREE_SAMPLER_DEBUG_LOG="$LUMO_TREE_SAMPLER_DEBUG_LOG"' in text
    assert '-e LUMO_TREE_PATH_LCP_LOG="$LUMO_TREE_PATH_LCP_LOG"' in text
    assert "-e LUMO_MTP_DRAFT_TRACE_FILE=/logs/fr10_mtp_draft_trace.jsonl" not in text
    assert "-e LUMO_TREE_SAMPLER_DEBUG_LOG=/logs/tree_sampler_debug.jsonl" not in text
    assert "-e LUMO_TREE_PATH_LCP_LOG=/logs/tree_path_lcp.jsonl" not in text
    # FR13 Method-A BI allowlist: default OFF, plumbed through, boot needles.
    assert "FR13_BI_TREE_ATTN=${FR13_BI_TREE_ATTN:-0}" in text
    assert '-e FR13_BI_TREE_ATTN="$FR13_BI_TREE_ATTN"' in text
    assert "FR13_BI_TREE_ATTN allowlist patch missing" in text
    assert "FR13 BI decode num_splits expression missing" in text


def test_fr13_patcher_method_a_bi_allowlist_is_flag_gated() -> None:
    text = Path("scripts/fr13_patch_fa2_tree_bias.py").read_text()

    assert "def _patch_batch_invariant_guard" in text
    assert '"batch_invariant.py": _patch_batch_invariant_guard(' in text
    assert (
        'if os.environ.get("FR13_BI_TREE_ATTN", "0") == "1":' in text
    )
    assert (
        "decode_invariant_backends.append(AttentionBackendEnum.TREE_ATTN)" in text
    )
    # double gate raises unless both FR13_FA2_* flags are set
    assert "forked-FA2 tree decode + native-FA2 prefill paths" in text
    # EDIT 2: BI-forced non-split dispatch on the tree-bias decode call
    assert "num_splits=1 if envs.VLLM_BATCH_INVARIANT else 0," in text
    # swapped-mode hardening: bias only engages for multi-token decode rows
    assert "and decode_meta.max_query_len > 1" in text


def test_speed_launcher_defaults_to_nine_node_caterpillar() -> None:
    text = Path("scripts/fr10_launch_speed_server.sh").read_text()

    assert (
        "TREE=${TREE:-\"[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), "
        "(0, 0, 0, 0, 0), (0, 1), (0, 0, 1), (0, 0, 0, 1), "
        "(0, 0, 0, 0, 1)]\"}"
    ) in text
    assert "print(len(ast.literal_eval(os.environ[\"TREE\"])))" in text
    assert "ATTENTION_BACKEND=FLASH_ATTN" in text
    assert "--attention-backend '$ATTENTION_BACKEND'" in text
    assert "FR11_TREE_CONV_NATIVE_BF16_TAPS=${FR11_TREE_CONV_NATIVE_BF16_TAPS:-1}" in text
    assert '-e FR11_TREE_CONV_NATIVE_BF16_TAPS="$FR11_TREE_CONV_NATIVE_BF16_TAPS"' in text
    assert "LUMO_TREE_SAMPLER_DEBUG_LOG=${LUMO_TREE_SAMPLER_DEBUG_LOG:-}" in text
    assert "LUMO_TREE_PATH_LCP_LOG=${LUMO_TREE_PATH_LCP_LOG:-}" in text
    assert "FR13_PREFILL_GDN_CAPTURE=${FR13_PREFILL_GDN_CAPTURE:-}" in text
    assert (
        "FR13_PREFILL_GDN_CAPTURE_LAYER_PREFIX="
        "${FR13_PREFILL_GDN_CAPTURE_LAYER_PREFIX:-}"
    ) in text
    assert (
        "FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX="
        "${FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX:-1}"
    ) in text
    assert '-e LUMO_TREE_SAMPLER_DEBUG_LOG="$LUMO_TREE_SAMPLER_DEBUG_LOG"' in text
    assert '-e LUMO_TREE_PATH_LCP_LOG="$LUMO_TREE_PATH_LCP_LOG"' in text
    assert '-e FR13_PREFILL_GDN_CAPTURE="$FR13_PREFILL_GDN_CAPTURE"' in text
    assert (
        '-e FR13_PREFILL_GDN_CAPTURE_LAYER_PREFIX='
        '"$FR13_PREFILL_GDN_CAPTURE_LAYER_PREFIX"'
    ) in text
    assert (
        '-e FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX='
        '"$FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX"'
    ) in text


def test_tree_engagement_logs_are_available_with_metrics_off() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()

    assert 'or _fr10_lo.environ.get("LUMO_TREE_SAMPLER_DEBUG_LOG")' in text
    assert "or _fr10_lo.environ.get('LUMO_TREE_SAMPLER_DEBUG_LOG')" in text
    assert "or _fr10_lo.environ.get('LUMO_TREE_PATH_LCP_LOG')" in text
    assert "or _lcpo.environ.get('LUMO_TREE_PATH_LCP_LOG')" in text


def test_speed_launcher_recovers_host_memory_before_docker_run() -> None:
    text = Path("scripts/fr10_launch_speed_server.sh").read_text()

    recovery = "from lumo_flywheel_serving.model_server import recover_host_memory"
    docker_run = "docker run -d --name \"$CONTAINER\""
    assert recovery in text
    assert text.index(recovery) < text.index(docker_run)
    assert "recover_host_memory()" in text
    assert "MemFree>=100GiB and swap_used==0" in text
    assert "swap_used_kib != 0" in text

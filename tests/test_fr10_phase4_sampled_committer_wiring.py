from __future__ import annotations

from pathlib import Path


def test_phase4_patcher_routes_sampled_tree_to_verified_committer() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()

    assert "sample_multidraft_rejection_step as _fr10_sample_step" in text
    assert "sample_deterministic_multidraft_rejection_step as _fr10_sample_det_step" in text
    assert "if tree_parent_indices is not None and not sampling_metadata.all_greedy" in text
    assert "return _lumo_tree_canonical_multidraft_sample(" in text
    assert "if draft_probs_cpu is None:" in text


def test_phase4_tree_commit_uses_accepted_node_row_without_plus_one() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()

    assert "accepted_row = int(best_path[best_lcp - 1]) if best_lcp > 0 else 0" in text
    assert "accepted_row = int(current_parent)" in text
    assert "_LUMO_FA_LAST_ACCEPTED_TREE_LENS" in text
    assert "if _fr10_has_accept:" in text
    assert "accept_token_bias = _fr10_row" in text
    assert "and accept_token_bias > 0" not in text


def test_phase4_patcher_exports_src_native_handoff_payload() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()

    assert "_LUMO_FA_LAST_ACCEPTED_TREE_NODE_PATHS" in text
    assert "_LUMO_FA_LAST_ACCEPTED_TREE_TOKEN_IDS" in text
    assert "FR10_TREE_GDN_SRC_NATIVE_PAYLOAD" in text
    assert "next_read_ssm_state" in text
    assert "next_read_conv_state" in text


def test_phase4_tree_disengagement_raises_by_default() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()

    assert "FR10_ALLOW_LINEAR_FALLBACK" in text
    assert "FR10 tree metadata disengaged:" in text
    assert "FR10 tree causal-conv disengaged:" in text
    assert "FR10 tree scan disengaged:" in text
    assert "eligible_tree_spec_row_flat_fallback" in text


def test_phase4_handoff_logs_accepted_bank_row_alias() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()

    assert '"accepted_spec_state_bank_row":' in text
    assert '"accepted_bank_row":' in text
    assert '"value_spec": value_spec[\n                                            0, start:end' in text


def test_phase4_seeds_next_tree_read_base_from_accepted_row() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()

    assert "_FR10_TREE_READ_PREV" in text
    assert "spec_state_indices_tensor[\n                                    _fr10_seed_b, 0" in text
    assert "spec_state_indices_tensor[\n                                    fr10_b, 0" in text
    assert "_fr10_prev_read[\"conv_rows\"]" in text
    assert "_fr10_prev_read[\"tree_state\"]" in text


def test_speed_launcher_defaults_to_nine_node_caterpillar() -> None:
    text = Path("scripts/fr10_launch_speed_server.sh").read_text()

    assert (
        "TREE=${TREE:-\"[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), "
        "(0, 0, 0, 0, 0), (0, 1), (0, 0, 1), (0, 0, 0, 1), "
        "(0, 0, 0, 0, 1)]\"}"
    ) in text
    assert "print(len(ast.literal_eval(os.environ[\"TREE\"])))" in text

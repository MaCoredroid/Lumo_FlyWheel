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

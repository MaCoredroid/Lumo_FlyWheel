from __future__ import annotations

from pathlib import Path


def test_phase4_patcher_routes_sampled_tree_to_verified_committer() -> None:
    text = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").read_text()

    assert "sample_multidraft_rejection_step as _fr10_sample_step" in text
    assert "if tree_parent_indices is not None and not sampling_metadata.all_greedy" in text
    assert "return _lumo_tree_canonical_multidraft_sample(" in text
    assert "FR10 sampled tree committer requires draft_probs" in text

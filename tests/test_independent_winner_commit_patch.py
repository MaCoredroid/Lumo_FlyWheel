from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts/swe_x86_helpers/relaunch_qwen36_round.py"


def _winner_commit_patch() -> str:
    text = LAUNCHER.read_text()
    start = text.index("# LUMO_INDEPENDENT_ROWS_WINNER_COMMIT")
    end = text.index("GPUModelRunner._update_states_after_model_execute =", start)
    return text[start:end]


def test_independent_winner_commit_does_not_sync_full_token_matrix_to_cpu():
    patch = _winner_commit_patch()

    assert "output_token_ids.detach().cpu().tolist()" not in patch
    assert "output_token_ids.ge(0).sum(dim=1).detach().cpu().tolist()" in patch


def test_independent_winner_commit_remains_enabled_and_commits_gpu_rows():
    patch = _winner_commit_patch()

    assert 'LUMO_IR_WINNER_COMMIT", "1"' in patch
    assert "winner_row = output_token_ids[winner_idx].clone()" in patch
    assert "output_token_ids[idx].copy_(winner_row)" in patch

    mutation_pos = patch.index("output_token_ids[idx].copy_(winner_row)")
    native_update_pos = patch.index("_lumo_ir_orig_update_states_after_model_execute")
    assert mutation_pos > native_update_pos
    assert patch.index(
        "_lumo_ir_orig_update_states_after_model_execute",
        mutation_pos,
    ) > mutation_pos

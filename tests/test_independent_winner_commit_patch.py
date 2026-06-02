from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts/swe_x86_helpers/relaunch_qwen36_round.py"


def _winner_commit_patch() -> str:
    text = LAUNCHER.read_text()
    start = text.index("# LUMO_INDEPENDENT_ROWS_WINNER_COMMIT")
    end = text.index("GPUModelRunner._update_states_after_model_execute =", start)
    return text[start:end]


def _independent_rows_patch() -> str:
    text = LAUNCHER.read_text()
    start = text.index("# LUMO_INDEPENDENT_ROWS: hidden co-resident request rows")
    end = text.index("Scheduler.add_request = _lumo_ir_add_request", start)
    return text[start:end]


def test_independent_rows_syncs_hidden_sibling_scheduler_state():
    patch = _independent_rows_patch()

    assert "def _lumo_ir_sync_group_state(self):" in patch
    assert "clone_req._output_token_ids.clear()" in patch
    assert "clone_req._output_token_ids.extend(primary_req.output_token_ids)" in patch
    assert "clone_req._all_token_ids.extend(primary_req.all_token_ids)" in patch
    assert "clone_req.num_computed_tokens = int(primary_req.num_computed_tokens)" in patch

    native_update_pos = patch.index("_lumo_ir_orig_update_from_output")
    sync_pos = patch.index("_lumo_ir_sync_group_state(self)")
    filter_pos = patch.index("eco.outputs = [")
    assert native_update_pos < sync_pos < filter_pos


def test_independent_winner_commit_uses_native_accept_counts_not_extra_gpu_scan():
    patch = _winner_commit_patch()

    assert "output_token_ids.detach().cpu().tolist()" not in patch
    assert "output_token_ids.ge(0).sum(dim=1)" not in patch
    assert "self.input_batch.num_accepted_tokens_cpu[:num_rows]" in patch


def test_independent_winner_commit_remains_enabled_and_commits_gpu_rows():
    patch = _winner_commit_patch()

    assert 'LUMO_IR_WINNER_COMMIT", "1"' in patch
    assert "winner_row = output_token_ids[winner_idx].clone()" in patch
    assert "output_token_ids[idx].copy_(winner_row)" in patch

    mutation_pos = patch.index("output_token_ids[idx].copy_(winner_row)")
    native_update_pos = patch.index("_lumo_ir_orig_update_states_after_model_execute")
    native_call_pos = patch.index(
        "_lumo_ir_orig_update_states_after_model_execute",
        native_update_pos + 1,
    )
    assert native_call_pos < mutation_pos
    assert patch.find("_lumo_ir_orig_update_states_after_model_execute", mutation_pos) == -1


def test_independent_winner_commit_flushes_mamba_copy_only_when_buffer_nonempty():
    patch = _winner_commit_patch()

    assert "if copy_bufs.offset > 0:" in patch
    assert "do_mamba_copy_block(copy_bufs)" in patch

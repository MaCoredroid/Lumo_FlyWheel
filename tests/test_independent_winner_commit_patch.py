import subprocess
from pathlib import Path

import pytest

from scripts.swe_x86_helpers import relaunch_qwen36_round as relaunch


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


def test_independent_rows_upgrades_stale_scheduler_sentinel():
    text = LAUNCHER.read_text()

    assert "sync_sentinel = 'def _lumo_ir_sync_group_state(self):'" in text
    assert "elif sentinel in text:" in text
    assert "LUMO_INDEPENDENT_ROWS_SYNC_UPGRADE" in text
    assert "upgraded independent rows scheduler sync patch" in text
    assert "_lumo_ir_prev_update_from_output = Scheduler.update_from_output" in text


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


def test_independent_winner_commit_enforces_lossless_spine0_public_stream():
    patch = _winner_commit_patch()

    assert "def _lumo_ir_request_temperature(self, primary_req_id)" in patch
    assert "def _lumo_ir_commit_policy()" in patch
    assert "def _lumo_ir_select_commit_row(self, primary, req_ids, indices, accept_counts)" in patch
    assert 'LUMO_IR_PUBLIC_COMMIT_POLICY", "lossless"' in patch
    assert 'policy in ("best_of_spines", "unsafe_best_of_spines", "deterministic_best")' in patch
    assert 'commit_idx = primary_idx' in patch
    assert 'commit_acc = accept_counts[primary_idx]' in patch
    assert '"no_lossless_selector"' in patch
    assert '"policy": selection["policy"]' in patch
    assert '"selector_enabled": bool(selection["selector_enabled"])' in patch
    assert '"lossless_public_stream": bool(selection["lossless_public_stream"])' in patch
    assert '"candidate_winner_spine"' in patch
    assert '"hidden_winner_suppressed_reason": selection["suppressed_reason"]' in patch
    assert '"hidden_winner_public_policy"' not in patch


def test_independent_winner_commit_keeps_qwen_parser_protocol_guards():
    patch = _winner_commit_patch()
    text = LAUNCHER.read_text()

    assert "def _lumo_ir_allow_hidden_public_winner" not in patch
    assert 'return False, "stochastic_sampling"' not in patch
    assert "LUMO_QWEN_STRAY_REASONING_END_PUBLIC_GUARD" in text
    assert "LUMO_QWEN_PUBLIC_PROTOCOL_MARKER_GUARD" in text
    assert "LUMO_QWEN_RESPONSES_PUBLIC_ITEM_GUARD" in text
    assert "LUMO_QWEN_CHAT_HISTORY_ARGUMENTS_GUARD" in text
    assert "match = re.search(" in text
    assert "indent = match.group('indent')" in text
    assert "child_indent = indent + '    '" in text
    assert "def _lumo_repair_response_message_text" in text
    assert "def _lumo_repair_function_call_arguments" in text
    assert "def _lumo_extract_qwen_xml_arguments" in text
    assert "def _lumo_extract_json_arguments" in text
    assert "protocol_repaired = _lumo_repair_argument_protocol_text(value)" in text
    assert 'r"<parameter=([^>\\s]+)>(.*?)</parameter>"' in text
    assert '"__lumo_malformed_arguments__": content' in text
    assert "repaired.rsplit(\"</think>\", 1)[1]" in text
    assert "items = _lumo_repair_response_items_public(items)" in text
    assert "BaseThinkingReasoningParser.extract_reasoning_streaming" in text
    assert "BaseThinkingReasoningParser.extract_reasoning =" in text
    assert "OpenAIServingResponses._make_response_output_items =" in text
    assert "self.end_token_id in delta_token_ids and not has_start_state" in text
    assert '("<think>", "</think>", "<|host|>")' in text
    assert "_QWEN_REASONING_STREAM_BOUNDARY_BLOCK if independent_rows else" in text


def test_round_relaunch_launcher_has_no_top_level_patch_runtime_error():
    result = subprocess.run(
        ["python3", str(LAUNCHER), "--help"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "relaunch_qwen36_round.py" in result.stdout


def test_round_relaunch_rejects_legacy_policy_before_modelserver_start():
    result = subprocess.run(
        [
            "env",
            "LUMO_IR_PUBLIC_COMMIT_POLICY=best_of_spines",
            "python3",
            str(LAUNCHER),
            "--config",
            "Fb",
            "--row-mode",
            "independent",
            "--spines",
            "2",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    assert result.returncode != 0
    assert "forbidden" in result.stderr
    assert "ModelServer" not in result.stderr


def test_independent_winner_commit_flushes_mamba_copy_only_when_buffer_nonempty():
    patch = _winner_commit_patch()

    assert "if copy_bufs.offset > 0:" in patch
    assert "do_mamba_copy_block(copy_bufs)" in patch


def test_lossless_policy_default_allows_selector_off_independent_spines():
    env: dict[str, str] = {}

    policy = relaunch._lumo_ir_validate_public_commit_policy(
        independent_rows=True,
        spines=2,
        environ=env,
    )

    assert policy == "lossless"


@pytest.mark.parametrize(
    "policy",
    ["best_of_spines", "unsafe_best_of_spines", "deterministic_best"],
)
def test_legacy_independent_public_commit_policies_fail_closed(policy: str):
    env = {"LUMO_IR_PUBLIC_COMMIT_POLICY": policy}

    with pytest.raises(RuntimeError, match="forbidden"):
        relaunch._lumo_ir_validate_public_commit_policy(
            independent_rows=True,
            spines=2,
            environ=env,
        )


def test_unknown_independent_public_commit_policy_fails_closed():
    env = {"LUMO_IR_PUBLIC_COMMIT_POLICY": "longest_prefix"}

    with pytest.raises(RuntimeError, match="unknown"):
        relaunch._lumo_ir_validate_public_commit_policy(
            independent_rows=True,
            spines=2,
            environ=env,
        )


@pytest.mark.parametrize(
    "env_name",
    [
        "LUMO_IR_ALLOW_STOCHASTIC_HIDDEN_WINNER",
        "LUMO_IR_ALLOW_HIDDEN_PUBLIC_WINNER",
        "LUMO_IR_ENABLE_HIDDEN_PUBLICATION",
        "LUMO_IR_PUBLISH_HIDDEN_WINNER",
    ],
)
def test_hidden_publication_before_selector_fails_closed(env_name: str):
    env = {env_name: "1"}

    with pytest.raises(RuntimeError, match="hidden-spine public publication"):
        relaunch._lumo_ir_validate_public_commit_policy(
            independent_rows=True,
            spines=2,
            environ=env,
        )


def test_selector_enabled_before_implementation_fails_closed():
    env = {"LUMO_IR_LOSSLESS_SELECTOR_ENABLED": "1"}

    with pytest.raises(RuntimeError, match="not implemented"):
        relaunch._lumo_ir_validate_public_commit_policy(
            independent_rows=True,
            spines=2,
            environ=env,
        )


def test_winner_commit_disable_fails_closed_for_independent_rows():
    env = {"LUMO_IR_WINNER_COMMIT": "0"}

    with pytest.raises(RuntimeError, match="may not disable"):
        relaunch._lumo_ir_validate_public_commit_policy(
            independent_rows=True,
            spines=2,
            environ=env,
        )

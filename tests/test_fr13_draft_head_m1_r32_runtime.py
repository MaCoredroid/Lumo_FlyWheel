from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
LIVE_GATE = REPO / "scripts" / "fr13_run_b1_kernel_live_gate.sh"
RUNNER = REPO / "scripts" / "fr13_run_b1_draft_head_m1_r32.sh"
EXPECTED_SHA256 = (
    "c389bf5e01b942cfe73b2e4fc05db7b158f16b61205c9f3e9988cbd8a82474dd"
)


def _eagle_snippet() -> str:
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and (
            node.name == "_patch_eagle_tree_consumption_verify"
        ):
            for statement in ast.walk(node):
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "new"
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                    and "FR13_DRAFT_HEAD_M1_R32" in statement.value.value
                ):
                    return statement.value.value
    raise AssertionError("exact-order R32 draft-head replacement not found")


def _observed_runtime_snippet() -> str:
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_FR13_FIXED32_OBSERVED_RUNTIME_SOURCE"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise AssertionError("fixed32 observed-runtime replacement not found")


def test_r32_mode_is_default_off_and_exact_b1_k64_only() -> None:
    snippet = _eagle_snippet()

    assert '"FR13_DRAFT_HEAD_M1_R32", "0"' in snippet
    assert '_fr13_dh_m1_r32_raw not in ("0", "1")' in snippet
    assert '_fr13_dh_m1_r32_raw == "1"' in snippet
    assert "_fr13_dh_modes > 1" in snippet
    assert "or _fr13_dh_m1_r32" in snippet
    assert "not _fr13_is_fixed32" in snippet
    assert "not _fr13_dvk_root" in snippet
    assert "not _fr13_single_logits" in snippet
    assert "_fr13_dvk_configured != 65536" in snippet
    assert "(_fr13_dh_m1_r32_enabled and int(batch_size) != 1)" in snippet


def test_r32_live_ab_is_distinct_default_off_and_source_bound() -> None:
    snippet = _eagle_snippet()

    assert '"FR13_DRAFT_HEAD_M1_R32_LIVE_AB", "0"' in snippet
    assert '_fr13_dh_m1_r32_live_raw not in ("0", "1")' in snippet
    assert "_fr13_dh_m1_r32 or _fr13_dh_m1_r32_live" in snippet
    assert "_fr13_dh_m1_r32_live," in snippet
    assert "FR13_DRAFT_HEAD_M1_R32_RUNTIME_SOURCE_SHA256" in snippet
    assert "FR13_DRAFT_HEAD_M1_R32_CANDIDATE_SOURCE_SHA256" in snippet
    assert "FR13_DRAFT_HEAD_M1_R32_SOURCE_COMMIT" in snippet
    assert "astropy__astropy-12907" in snippet


def test_r32_setup_pins_binary_map_weight_and_capture_lifecycle() -> None:
    snippet = _eagle_snippet()

    assert '"/tmp/fr13_bf16_k64_head_r32.abi3.so"' in snippet
    assert '"FR13_DRAFT_HEAD_M1_R32_SHA256", ""' in snippet
    assert EXPECTED_SHA256 in snippet
    assert "_fr13_dh_m1_so.stat().st_size != 113648" in snippet
    assert "_fr13_dh_m1_digest.hexdigest()" in snippet
    assert "torch.ops.load_library(str(_fr13_dh_m1_so))" in snippet
    assert "torch.ops.fr13_bf16_k64_head.gemvx_m1_shuffle_r32_out" in snippet
    assert "tuple(_fr13_dh_m1_w.shape) != (65536, 5120)" in snippet
    assert "tuple(_fr13_dh_m1_w.stride()) != (5120, 1)" in snippet
    assert "_fr13_dh_m1_w.dtype != torch.bfloat16" in snippet
    assert "tuple(_fr13_dh_m1_map.shape) != (65536,)" in snippet
    assert "tuple(_fr13_dh_m1_map.stride()) != (1,)" in snippet
    assert "_fr13_dh_m1_map.dtype != torch.int64" in snippet
    assert "self._fr13_dh_m1_r32_output = torch.empty(" in snippet
    assert "self._fr13_dh_m1_r32_seen_eager = False" in snippet
    assert "CUDA graph capture before eager preparation" in snippet


def test_r32_runtime_uses_out_op_and_forbids_fallback() -> None:
    snippet = _eagle_snippet()
    helper_start = snippet.index("def _fr13_dvk_logits")
    helper_end = snippet.index("def _fr13_dvk_real_ids", helper_start)
    helper = snippet[helper_start:helper_end]

    assert helper.index("if _fr13_dh_m1_r32_on:") < helper.index(
        "elif _fr13_dh_fp8_on:"
    )
    assert "tuple(_h.shape) != (1, 5120)" in helper
    assert "tuple(_h.stride()) != (5120, 1)" in helper
    assert "self._fr13_dh_m1_r32_op(" in helper
    assert "_logits = self._fr13_dh_m1_r32_output" in helper
    assert '"selector=exact_order_r32 "' in helper
    assert '"eager_launch=1"' in helper
    assert "capture before one eager kernel launch" in helper
    failure = helper[helper.index("except Exception as _e:") :]
    assert failure.index('self, "_fr13_dh_m1_r32_active", False') < failure.index(
        'self, "_fr13_dh_fp8_active", False'
    )
    assert "strict runtime contract" in failure


def test_r32_live_ab_counts_every_bf16_value_per_site_and_serves_reference() -> None:
    snippet = _eagle_snippet()
    helper_start = snippet.index("def _fr13_dvk_logits")
    helper_end = snippet.index("def _fr13_dvk_real_ids", helper_start)
    helper = snippet[helper_start:helper_end]
    live_start = helper.index("if _fr13_dh_m1_r32_on:")
    live_end = helper.index("elif _fr13_dh_fp8_on:", live_start)
    live = helper[live_start:live_end]

    assert "_fr13_dh_m1_r32_note_live(_fr13_dh_capturing)" in live
    assert "self._fr13_dh_m1_r32_op(" in live
    assert "_sh.quant_method.apply(" in live
    assert "self._fr13_dh_m1_r32_output.view(" in live
    assert "_fr13_dh_m1_reference.view(" in live
    assert "torch.int16" in live
    assert "torch.count_nonzero(" in live
    assert "_fr13_dh_m1_r32_live_mismatches[" in live
    assert "_fr13_dh_m1_r32_live_compares[" in live
    assert "_logits = _fr13_dh_m1_reference" in live
    assert ".tolist()" not in live
    assert ".item()" not in live
    assert "sample_hidden_states, 0" in snippet
    assert snippet.count("last_hidden_states[:batch_size], token_index + 1") == 2


def test_r32_live_ab_attests_one_root_four_captured_heads_and_measured_replay() -> None:
    snippet = _eagle_snippet()
    observed = _observed_runtime_snippet()

    assert "_fr13_dh_m1_r32_live_selected_root_calls" in snippet
    assert "_fr13_dh_m1_r32_live_selected_capture_calls" in snippet
    assert "_fr13_dh_m1_r32_live_fallback_calls" in snippet
    assert "selected_capture_calls != 4" in snippet
    assert "measured_replays" in snippet
    assert snippet.count("_fr13_dh_m1_r32_note_live_replay(") == 3
    assert "_fr13_draft_head_m1_r32_live_register(" in observed
    assert "_fr13_draft_head_m1_r32_live_finalize(" in observed
    assert 'compares == (draft_events,) * 5' in observed
    assert 'mismatches == (0,) * 5' in observed
    assert 'per_site_compared_bf16_values' in observed
    assert 'device_counted_without_measured_host_sync' in observed
    finalize = observed[
        observed.index("def _fr13_draft_head_m1_r32_live_finalize") :
    ]
    assert 'state["compares"].tolist()' in finalize


def test_r32_graph_signature_is_derived_from_canonical_b1_manifest() -> None:
    manifest = {
        "schema": "fr13-fixed32-drafter-graph-manifest-v2",
        "batch_size": 1,
        "mtp_forward_calls": 4,
        "mtp_forward_rows": 4,
        "tree_attn_calls": 4,
        "tree_attn_rows": 4,
        "tree_attn_layer": "mtp.layers.0.self_attn.attn",
        "tree_attn_bias_shape": [1, 1],
    }
    canonical = json.dumps(
        manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    signature = hashlib.sha256(canonical).hexdigest()

    assert signature == (
        "d9a4ddece41d146e9949b9f8ff7c2603b8948d157b28ef69244e44469b36150c"
    )
    observed = _observed_runtime_snippet()
    eagle = _eagle_snippet()
    assert signature in observed
    assert signature[:32] in eagle
    assert signature[32:] in eagle


def test_launcher_pins_read_only_r32_binary_and_isolates_candidate() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "FR13_DRAFT_HEAD_M1_R32=${FR13_DRAFT_HEAD_M1_R32:-0}" in launcher
    assert "FR13_DRAFT_HEAD_M1_R32_SO=${FR13_DRAFT_HEAD_M1_R32_SO:-}" in launcher
    assert "FR13_DRAFT_HEAD_M1_R32=0 forbids candidate binary credentials" in launcher
    assert EXPECTED_SHA256 in launcher
    assert '"$(stat -c \'%s\' "$FR13_DRAFT_HEAD_M1_R32_SO")" == "113648"' in launcher
    assert '"$FR13_DRAFT_HEAD_M1_R32_SO" == /*' in launcher
    assert '! -L "$FR13_DRAFT_HEAD_M1_R32_SO"' in launcher
    assert (
        '-v "$FR13_DRAFT_HEAD_M1_R32_SO:'
        '/tmp/fr13_bf16_k64_head_r32.abi3.so:ro"'
    ) in launcher
    assert '"${FR13_DRAFT_HEAD_M1_R32_DOCKER_ARGS[@]}" \\' in launcher
    assert '|| "$_v" == "FR13_DRAFT_HEAD_M1_R32_SO" \\' in launcher
    assert '-e FR13_DRAFT_HEAD_M1_R32="$FR13_DRAFT_HEAD_M1_R32" \\' in launcher
    assert "draft-head mode must be the only kernel candidate" in launcher


def test_launcher_live_ab_reuses_pinned_binary_and_forwards_attested_sources() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "FR13_DRAFT_HEAD_M1_R32_LIVE_AB=${FR13_DRAFT_HEAD_M1_R32_LIVE_AB:-0}" in launcher
    assert "direct and live A/B modes are mutually exclusive" in launcher
    assert '&& "$FR13_DRAFT_HEAD_M1_R32_LIVE_AB" == "0"' in launcher
    assert "FR13 exact-order R32 live A/B requires the isolated canonical real B1 task" in launcher
    assert "sha256sum csrc/fr13_bf16_gemvx_k64_m1_shuffle.cu" in launcher
    assert '-e FR13_DRAFT_HEAD_M1_R32_LIVE_AB="$FR13_DRAFT_HEAD_M1_R32_LIVE_AB" \\' in launcher
    assert '-e FR13_DRAFT_HEAD_M1_R32_RUNTIME_SOURCE_SHA256=' in launcher
    assert '-e FR13_DRAFT_HEAD_M1_R32_CANDIDATE_SOURCE_SHA256=' in launcher
    assert '-e FR13_DRAFT_HEAD_M1_R32_SOURCE_COMMIT=' in launcher


def test_real_b1_runner_selects_only_r32_and_requires_markers() -> None:
    gate = LIVE_GATE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert "FR13_GATE_DRAFT_HEAD_M1_R32=${FR13_GATE_DRAFT_HEAD_M1_R32:-0}" in gate
    assert EXPECTED_SHA256 in gate
    assert 'FR13_DRAFT_HEAD_M1_R32="$FR13_GATE_DRAFT_HEAD_M1_R32" \\' in gate
    assert 'DRAFT_HEAD_M1_R32_RUNTIME_LOG="$RUNROOT/$ARM/docker_after_tasks.log"' in gate
    assert '[[ -f "$DRAFT_HEAD_M1_R32_RUNTIME_LOG" \\' in gate
    assert "ready selector=exact_order_r32" in gate
    assert "engaged selector=exact_order_r32 eager_launch=1" in gate
    assert gate.count('"$DRAFT_HEAD_M1_R32_RUNTIME_LOG" >/dev/null || {') == 2
    assert "must be the only enabled kernel candidate" in gate

    assert "set FR13_DRAFT_HEAD_M1_R32_SO" in runner
    assert "export FR13_B1_WORKLOAD_PROFILE=k64_root" in runner
    assert "export FR13_GATE_DRAFT_HEAD_M1_R32=1" in runner
    assert "export FR13_GATE_TAW_NATIVE=0" in runner
    assert "export FR13_GATE_DFWD_TOP3=0" in runner
    assert 'exec bash "$SCRIPT_DIR/fr13_run_b1_kernel_live_gate.sh"' in runner


def test_replacement_snippet_compiles_as_method_body() -> None:
    compile(
        "class _C:\n    def propose(self):\n" + _eagle_snippet(),
        "<fr13_draft_head_m1_r32_snippet>",
        "exec",
    )

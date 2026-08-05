from __future__ import annotations

import ast
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
    assert "(_fr13_dh_m1_r32 and int(batch_size) != 1)" in snippet


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

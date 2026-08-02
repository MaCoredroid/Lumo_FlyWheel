from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
LIVE_GATE = REPO / "scripts" / "fr13_run_b1_kernel_live_gate.sh"
PAIR8_RUNNER = REPO / "scripts" / "fr13_run_b1_draft_head_m1_vec.sh"


def _eagle_snippet() -> str:
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_patch_eagle_tree_consumption_verify"
        ):
            for statement in ast.walk(node):
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "new"
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                    and "FR13_DRAFT_HEAD_M1_VEC" in statement.value.value
                ):
                    return statement.value.value
    raise AssertionError("pair8bits draft-head replacement snippet not found")


def test_pair8bits_mode_is_default_off_and_exact_b1_k64_only() -> None:
    snippet = _eagle_snippet()

    assert '"FR13_DRAFT_HEAD_M1_VEC", "0"' in snippet
    assert '_fr13_dh_m1_vec_raw not in ("0", "pair8bits")' in snippet
    assert '_fr13_dh_m1_vec_raw == "pair8bits"' in snippet
    assert "_fr13_dh_modes > 1" in snippet
    assert "or _fr13_dh_m1_vec" in snippet
    assert "not _fr13_is_fixed32" in snippet
    assert "not _fr13_dvk_root" in snippet
    assert "not _fr13_single_logits" in snippet
    assert "_fr13_dvk_configured != 65536" in snippet


def test_pair8bits_setup_is_static_and_precedes_graph_capture() -> None:
    snippet = _eagle_snippet()

    assert '"/tmp/fr13_bf16_k64_head.abi3.so"' in snippet
    assert "not os.path.isfile(_fr13_dh_m1_so)" in snippet
    assert "os.path.islink(_fr13_dh_m1_so)" in snippet
    assert "torch.ops.load_library(_fr13_dh_m1_so)" in snippet
    assert "torch.ops.fr13_bf16_k64_head" in snippet
    assert '"gemvx_m1_warp32_r32_pair8bits_out"' in snippet
    assert "tuple(_fr13_dh_m1_w.shape) != (65536, 5120)" in snippet
    assert "tuple(_fr13_dh_m1_w.stride()) != (5120, 1)" in snippet
    assert "_fr13_dh_m1_w.dtype != torch.bfloat16" in snippet
    assert "tuple(_fr13_dh_m1_map.shape) != (65536,)" in snippet
    assert "tuple(_fr13_dh_m1_map.stride()) != (1,)" in snippet
    assert "_fr13_dh_m1_map.dtype != torch.int64" in snippet
    assert "_fr13_dh_m1_map.device" in snippet
    assert "not _fr13_dh_m1_map.is_contiguous()" in snippet
    assert "self._fr13_dh_m1_vec_output = torch.empty(" in snippet
    assert "self._fr13_dh_m1_vec_seen_eager = False" in snippet
    assert "setup reached CUDA " in snippet
    assert "graph capture before eager preparation" in snippet


def test_pair8bits_runtime_uses_out_op_and_never_silently_falls_back() -> None:
    snippet = _eagle_snippet()
    helper_start = snippet.index("def _fr13_dvk_logits")
    helper_end = snippet.index("def _fr13_dvk_real_ids", helper_start)
    helper = snippet[helper_start:helper_end]

    assert helper.index("if _fr13_dh_m1_vec_on:") < helper.index(
        "elif _fr13_dh_m32_live_on or _fr13_dh_m32_prod_on:"
    )
    assert "tuple(_h.shape) != (1, 5120)" in helper
    assert "tuple(_h.stride()) != (5120, 1)" in helper
    assert "tuple(_sh.weight.shape) != (65536, 5120)" in helper
    assert "self._fr13_dh_m1_vec_output.stride()" in helper
    assert "self._fr13_dh_m1_vec_op(" in helper
    assert "_logits = self._fr13_dh_m1_vec_output" in helper
    assert '"[FR13_DRAFT_HEAD_M1_VEC] engaged "' in helper
    assert '"selector=pair8bits eager_launch=1"' in helper
    assert "reached capture " in helper
    assert "before one eager kernel launch" in helper
    assert "torch.empty" not in helper
    failure_path = helper[helper.index("except Exception as _e:") :]
    assert failure_path.index(
        'self, "_fr13_dh_m1_vec_active", False'
    ) < failure_path.index('self, "_fr13_dh_m32_production_active", False')
    assert "pair8bits draft head failed its strict " in failure_path


def test_launcher_mounts_only_an_explicit_read_only_candidate() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "FR13_DRAFT_HEAD_M1_VEC=${FR13_DRAFT_HEAD_M1_VEC:-0}" in launcher
    assert "FR13_DRAFT_HEAD_M1_VEC_SO=${FR13_DRAFT_HEAD_M1_VEC_SO:-}" in launcher
    assert 'case "$FR13_DRAFT_HEAD_M1_VEC" in' in launcher
    assert "0|pair8bits" in launcher
    assert "FR13_DRAFT_HEAD_M1_VEC=0 forbids a candidate SO" in launcher
    assert '"$FR13_DRAFT_HEAD_M1_VEC_SO" == /*' in launcher
    assert '! -L "$FR13_DRAFT_HEAD_M1_VEC_SO"' in launcher
    assert "requires pinned gather-K64 single-logits real-B1 diagnostic mode" in launcher
    assert "FR13_DRAFT_HEAD_M32_TIMING_ARM" in launcher
    assert '== "/workspace/scripts/fr13_dvk_subset_blocks.json"' in launcher
    assert "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff" in launcher
    assert "6065985824bd2fd5353a2bd203ef2a77a205547171f2ffb89c19252e968fbce1" in launcher
    assert "FR13 pair8bits draft-head binary identity mismatch" in launcher
    assert (
        '-v "$FR13_DRAFT_HEAD_M1_VEC_SO:'
        '/tmp/fr13_bf16_k64_head.abi3.so:ro"'
    ) in launcher
    assert '"${FR13_DRAFT_HEAD_M1_VEC_DOCKER_ARGS[@]}" \\' in launcher
    assert '|| "$_v" == "FR13_DRAFT_HEAD_M1_VEC_SO" \\' in launcher
    assert '-e FR13_DRAFT_HEAD_M1_VEC="$FR13_DRAFT_HEAD_M1_VEC" \\' in launcher
    broad_guard = launcher[launcher.index("if [[ \"$FR13_DRAFT_HEAD_M32_LIVE_AB\"") :]
    assert '|| "$FR13_DRAFT_HEAD_M1_VEC" == "pair8bits"' in broad_guard
    assert '"$FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB" == "0"' in broad_guard
    assert '"$FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION" == "0"' in broad_guard
    assert "draft-head mode must be the only kernel candidate" in broad_guard


def test_replacement_snippet_still_compiles_as_a_method_body() -> None:
    compile(
        "class _C:\n    def propose(self):\n" + _eagle_snippet(),
        "<fr13_draft_head_m1_vec_snippet>",
        "exec",
    )


def test_real_b1_runner_selects_k64_floor_and_requires_execution() -> None:
    gate = LIVE_GATE.read_text(encoding="utf-8")
    runner = PAIR8_RUNNER.read_text(encoding="utf-8")

    assert "FR13_GATE_DRAFT_HEAD_M1_VEC=${FR13_GATE_DRAFT_HEAD_M1_VEC:-0}" in gate
    assert 'case "$FR13_GATE_DRAFT_HEAD_M1_VEC" in' in gate
    assert 'export FR13_DRAFT_VOCAB_ROOT=1' in gate
    assert 'export FR13_DRAFT_VOCAB_K=65536' in gate
    assert 'export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json' in gate
    assert 'unset FR13_NEEDS_ALLOW' in gate
    assert 'float(os.environ["FR13_WEIGHT_FLOOR_MS"]) * 1.15' in gate
    assert 'FR13_DRAFT_HEAD_M1_VEC="$FR13_GATE_DRAFT_HEAD_M1_VEC" \\' in gate
    assert 'DRAFT_HEAD_RUNTIME_LOG="$RUNROOT/$ARM/docker_after_tasks.log"' in gate
    assert '[[ -f "$DRAFT_HEAD_RUNTIME_LOG" && ! -L "$DRAFT_HEAD_RUNTIME_LOG" ]]' in gate
    assert "ready selector=pair8bits" in gate
    assert "engaged selector=pair8bits eager_launch=1" in gate
    assert gate.count('"$DRAFT_HEAD_RUNTIME_LOG" || {') == 2
    assert "must be the only enabled kernel candidate" in gate

    assert 'export FR13_GATE_DRAFT_HEAD_M1_VEC=pair8bits' in runner
    assert 'export FR13_GATE_TAW_NATIVE=0' in runner
    assert 'export FR13_GATE_GDN_BV=0' in runner
    assert "fr13_fixed32_dfwd_k64_m1_warp32_r32_pair8bits_build_20260802" in runner
    assert 'exec bash "$SCRIPT_DIR/fr13_run_b1_kernel_live_gate.sh"' in runner

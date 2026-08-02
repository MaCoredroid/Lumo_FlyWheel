from __future__ import annotations

import ast
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
LIVE_GATE = REPO / "scripts" / "fr13_run_b1_kernel_live_gate.sh"
M32_RUNNER = REPO / "scripts" / "fr13_run_b1_draft_head_m32_quality.sh"
RUNTIME_MANIFEST = REPO / "scripts" / "fr13_runtime_manifest.py"
PAD_ROWS = (32, 64, 128)


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
                    and "FR13_DRAFT_HEAD_PAD_ROWS" in statement.value.value
                ):
                    return statement.value.value
    raise AssertionError("draft-head padded-row replacement snippet not found")


def test_candidate_is_default_off_strict_and_b1_root64_only() -> None:
    snippet = _eagle_snippet()

    assert '"FR13_DRAFT_HEAD_PAD_ROWS", "0"' in snippet
    assert '"FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB", "0"' in snippet
    assert '_fr13_dh_rows_raw not in ("0", "32", "64", "128")' in snippet
    assert '_fr13_dh_ab_raw not in ("0", "1")' in snippet
    assert "_fr13_dh_modes > 1" in snippet
    assert "not _fr13_is_fixed32" in snippet
    assert "not _fr13_dvk_root" in snippet
    assert "not _fr13_single_logits" in snippet
    assert "_fr13_dvk_configured != 65536" in snippet
    assert "tuple(_h.shape) != (1, 5120)" in snippet


def test_static_buffers_and_gemm_use_only_selected_rows() -> None:
    snippet = _eagle_snippet()

    assert '!= "UnquantizedEmbeddingMethod"' in snippet
    assert "tuple(_fr13_dh_w.shape) != (65536, 5120)" in snippet
    assert "tuple(_fr13_dh_w.stride()) != (5120, 1)" in snippet
    assert "_fr13_dh_w.dtype != torch.bfloat16" in snippet
    assert "not _fr13_dh_w.is_contiguous()" in snippet
    assert "if _fr13_dh_ab" in snippet
    assert "else {" in snippet
    assert "(_fr13_dh_r, 5120)" in snippet
    assert "(_fr13_dh_r, 65536)" in snippet
    assert "_fr13_dh_in.copy_(_h.expand_as(_fr13_dh_in))" in snippet
    assert "torch.mm(_fr13_dh_in, _sh.weight.t(), out=_fr13_dh_out)" in snippet
    assert "return _fr13_dh_out[:1]" in snippet
    assert "self._fr13_dh_pad_seen_eager = False" in snippet
    assert "direct padded draft head reached capture before " in snippet
    assert '"[FR13_DRAFT_HEAD_PAD] engaged "' in snippet
    assert 'f"candidate_rows={_rows} eager_launch=1"' in snippet
    helper_start = snippet.index("def _fr13_dh_pad_logits")
    helper_end = snippet.index("def _fr13_dvk_logits", helper_start)
    helper = snippet[helper_start:helper_end]
    assert "torch.empty" not in helper
    assert helper.count("torch.cuda.is_current_stream_capturing()") == 1


def test_all_row_byte_ab_is_full_logit_graph_capturable_and_reference_returning() -> None:
    snippet = _eagle_snippet()

    assert "for _fr13_dh_i, _fr13_dh_r in enumerate(" in snippet
    assert "(32, 64, 128)" in snippet
    assert "_fr13_dh_candidate.view(torch.int16)" in snippet
    assert "_fr13_dh_reference.view(torch.int16)" in snippet
    assert "self._fr13_dh_ab_mismatches[_fr13_dh_i].add_(" in snippet
    assert "self._fr13_dh_ab_compares[_fr13_dh_i].add_(1)" in snippet
    assert "torch.count_nonzero(" in snippet
    assert "torch.cuda.is_current_stream_capturing()" in snippet
    assert "_logits = _fr13_dh_reference" in snippet
    assert "rows32_64_128={_fr13_dh_bad}" in snippet
    assert "raise RuntimeError(\n                            \"FR13 draft-head padding failed" in snippet


def test_replicated_row_mm_out_exposes_real_row_for_every_candidate() -> None:
    torch.manual_seed(0)
    hidden = torch.randn(1, 16, dtype=torch.bfloat16)
    weight = torch.randn(64, 16, dtype=torch.bfloat16)
    reference = torch.nn.functional.linear(hidden, weight)

    for rows in PAD_ROWS:
        static_input = torch.empty(rows, 16, dtype=torch.bfloat16)
        static_output = torch.empty(rows, 64, dtype=torch.bfloat16)
        static_input.copy_(hidden.expand_as(static_input))
        torch.mm(static_input, weight.t(), out=static_output)

        assert torch.equal(static_output[0], static_output[-1])
        assert torch.equal(static_output[:1], reference)


def test_launcher_passes_only_strict_candidate_modes() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "FR13_DRAFT_HEAD_PAD_ROWS=${FR13_DRAFT_HEAD_PAD_ROWS:-0}" in launcher
    assert (
        "FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=${FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB:-0}"
        in launcher
    )
    assert 'case "$FR13_DRAFT_HEAD_PAD_ROWS" in' in launcher
    assert "0|32|64|128" in launcher
    assert 'case "$FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB" in' in launcher
    assert "FR13 draft-head padding requires fixed32 B1 root64" in launcher
    assert (
        '-e FR13_DRAFT_HEAD_PAD_ROWS="$FR13_DRAFT_HEAD_PAD_ROWS" \\'
        in launcher
    )
    assert (
        "-e FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB="
        '"$FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB" \\'
        in launcher
    )


def test_real_b1_m32_quality_runner_is_fixed_k64_and_requires_execution() -> None:
    gate = LIVE_GATE.read_text(encoding="utf-8")
    runner = M32_RUNNER.read_text(encoding="utf-8")
    manifest = RUNTIME_MANIFEST.read_text(encoding="utf-8")

    assert "FR13_GATE_DRAFT_HEAD_PAD_ROWS=${FR13_GATE_DRAFT_HEAD_PAD_ROWS:-0}" in gate
    assert 'case "$FR13_GATE_DRAFT_HEAD_PAD_ROWS" in' in gate
    assert "FR13_GATE_DRAFT_HEAD_PAD_ROWS must be 0 or 32" in gate
    assert 'export FR13_DRAFT_VOCAB_K=65536' in gate
    assert 'FR13_DRAFT_HEAD_PAD_ROWS="$FR13_GATE_DRAFT_HEAD_PAD_ROWS" \\' in gate
    assert "static buffers ready candidate_rows=$FR13_GATE_DRAFT_HEAD_PAD_ROWS" in gate
    assert "engaged candidate_rows=$FR13_GATE_DRAFT_HEAD_PAD_ROWS eager_launch=1" in gate
    assert 'DRAFT_HEAD_RUNTIME_LOG="$RUNROOT/$ARM/docker_after_tasks.log"' in gate

    assert "FR13_GATE_DRAFT_HEAD_PAD_ROWS=32" in runner
    assert "FR13_GATE_DRAFT_HEAD_PAD=0" in runner
    assert "FR13_GATE_DRAFT_HEAD_M32=0" in runner
    assert "FR13_GATE_DRAFT_HEAD_M1_VEC=0" in runner
    assert 'exec bash "$SCRIPT_DIR/fr13_run_b1_kernel_live_gate.sh"' in runner
    assert '"scripts/fr13_run_b1_draft_head_m32_quality.sh"' in manifest


def test_exact_bytes_flops_buffers_and_roofline() -> None:
    vocab = 65_536
    hidden = 5_120
    passes = 5
    bandwidth_bytes_per_s = 273_000_000_000
    dense_bf16_flops_per_s = 125_000_000_000_000

    weight_bytes_per_pass = vocab * hidden * 2
    assert weight_bytes_per_pass == 671_088_640
    assert passes * weight_bytes_per_pass == 3_355_443_200

    expected = {
        32: (21_474_836_480, 4_521_984, 31.7858182171, 8.736),
        64: (42_949_672_960, 9_043_968, 63.1489689728, 17.472),
        128: (85_899_345_920, 18_087_936, 124.6405477368, 34.944),
    }
    for rows, (flops, persistent, intensity, floor_tflops) in expected.items():
        input_bytes = rows * hidden * 2
        output_bytes = rows * vocab * 2
        assert 2 * rows * vocab * hidden == flops
        assert input_bytes + output_bytes == persistent
        minimum_bytes = weight_bytes_per_pass + input_bytes + output_bytes
        assert abs(flops / minimum_bytes - intensity) < 1e-9
        floor_s = weight_bytes_per_pass / bandwidth_bytes_per_s
        assert abs(flops / floor_s / 1e12 - floor_tflops) < 1e-12
        assert flops / dense_bf16_flops_per_s < floor_s

    all_row_ab_persistent = sum(value[1] for value in expected.values())
    assert all_row_ab_persistent == 31_653_888

from __future__ import annotations

import ast
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"


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
                    and "FR13_DRAFT_HEAD_M32" in statement.value.value
                ):
                    return statement.value.value
    raise AssertionError("draft-head M32 replacement snippet not found")


def test_candidate_is_default_off_strict_and_b1_root64_only() -> None:
    snippet = _eagle_snippet()

    assert '"FR13_DRAFT_HEAD_M32", "0"' in snippet
    assert '"FR13_DRAFT_HEAD_M32_BYTE_AB", "0"' in snippet
    assert '_fr13_dh_m32_raw not in ("0", "1")' in snippet
    assert '_fr13_dh_ab_raw not in ("0", "1")' in snippet
    assert "_fr13_dh_m32 and _fr13_dh_ab" in snippet
    assert "not _fr13_is_fixed32" in snippet
    assert "not _fr13_dvk_root" in snippet
    assert "not _fr13_single_logits" in snippet
    assert "_fr13_dvk_configured != 65536" in snippet
    assert "tuple(_h.shape) != (1, 5120)" in snippet


def test_static_buffers_and_m32_gemm_have_fixed_geometry() -> None:
    snippet = _eagle_snippet()

    assert '!= "UnquantizedLinearMethod"' in snippet
    assert "tuple(_fr13_dh_w.shape) != (65536, 5120)" in snippet
    assert "_fr13_dh_w.dtype != torch.bfloat16" in snippet
    assert "not _fr13_dh_w.is_contiguous()" in snippet
    assert "self._fr13_dh_m32_input = torch.empty(\n                        (32, 5120)" in snippet
    assert "self._fr13_dh_m32_output = torch.empty(\n                        (32, 65536)" in snippet
    assert "_fr13_dh_in.copy_(_h.expand_as(_fr13_dh_in))" in snippet
    assert "torch.mm(_fr13_dh_in, _sh.weight.t(), out=_fr13_dh_out)" in snippet
    assert "return _fr13_dh_out[:1]" in snippet
    helper_start = snippet.index("def _fr13_dh_m32_logits")
    helper_end = snippet.index("def _fr13_dvk_logits", helper_start)
    assert "torch.empty" not in snippet[helper_start:helper_end]


def test_byte_ab_is_full_logit_bitwise_graph_capturable_and_fail_loud() -> None:
    snippet = _eagle_snippet()

    assert "_fr13_dh_candidate.view(torch.int16)" in snippet
    assert "_fr13_dh_reference.view(torch.int16)" in snippet
    assert "self._fr13_dh_ab_mismatches.add_(" in snippet
    assert "torch.count_nonzero(" in snippet
    assert "torch.cuda.is_current_stream_capturing()" in snippet
    assert "_logits = _fr13_dh_reference" in snippet
    assert "FR13_DRAFT_HEAD_M32_BYTE_AB full-logit " in snippet
    assert 'f"mismatch count={_fr13_dh_bad}"' in snippet
    assert "raise RuntimeError(\n                            \"FR13 draft-head M32 failed" in snippet


def test_replicated_row_mm_out_exposes_the_real_row() -> None:
    torch.manual_seed(0)
    hidden = torch.randn(1, 16, dtype=torch.bfloat16)
    weight = torch.randn(64, 16, dtype=torch.bfloat16)
    static_input = torch.empty(32, 16, dtype=torch.bfloat16)
    static_output = torch.empty(32, 64, dtype=torch.bfloat16)

    static_input.copy_(hidden.expand_as(static_input))
    torch.mm(static_input, weight.t(), out=static_output)

    assert torch.equal(static_output[0], static_output[-1])
    assert torch.equal(
        static_output[:1], torch.nn.functional.linear(hidden, weight)
    )


def test_launcher_passes_only_strict_candidate_modes() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "FR13_DRAFT_HEAD_M32=${FR13_DRAFT_HEAD_M32:-0}" in launcher
    assert "FR13_DRAFT_HEAD_M32_BYTE_AB=${FR13_DRAFT_HEAD_M32_BYTE_AB:-0}" in launcher
    assert 'case "$FR13_DRAFT_HEAD_M32" in' in launcher
    assert 'case "$FR13_DRAFT_HEAD_M32_BYTE_AB" in' in launcher
    assert "FR13 draft-head M32 requires fixed32 B1 root64" in launcher
    assert '-e FR13_DRAFT_HEAD_M32="$FR13_DRAFT_HEAD_M32" \\' in launcher
    assert (
        '-e FR13_DRAFT_HEAD_M32_BYTE_AB="$FR13_DRAFT_HEAD_M32_BYTE_AB" \\'
        in launcher
    )


def test_exact_bytes_flops_and_measured_projection() -> None:
    vocab = 65536
    hidden = 5120
    passes = 5
    rows = 32
    bandwidth_bytes_per_s = 273_000_000_000
    measured_gemvx_ms_per_event = 26.227316
    measured_verifier_head_bytes = 2_542_796_800
    measured_verifier_head_ms = 12.153933

    weight_bytes_per_pass = vocab * hidden * 2
    assert weight_bytes_per_pass == 671_088_640
    assert passes * weight_bytes_per_pass == 3_355_443_200
    assert 2 * vocab * hidden == 671_088_640
    assert rows * 2 * vocab * hidden == 21_474_836_480
    assert passes * rows * 2 * vocab * hidden == 107_374_182_400
    floor_ms_per_pass = weight_bytes_per_pass / bandwidth_bytes_per_s * 1000
    assert abs(floor_ms_per_pass - 2.4582001465201464) < 1e-12

    current_ms_per_pass = measured_gemvx_ms_per_event / passes
    verifier_effective_bytes_per_ms = (
        measured_verifier_head_bytes / measured_verifier_head_ms
    )
    projected_ms_per_pass = weight_bytes_per_pass / verifier_effective_bytes_per_ms
    projected_savings_ms_per_event = passes * (
        current_ms_per_pass - projected_ms_per_pass
    )
    assert 5.24 < current_ms_per_pass < 5.25
    assert 3.20 < projected_ms_per_pass < 3.22
    assert 10.1 < projected_savings_ms_per_event < 10.3

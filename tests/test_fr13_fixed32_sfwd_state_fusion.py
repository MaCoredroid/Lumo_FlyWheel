from __future__ import annotations

import ast
import json
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py"
PATCHER_PATH = ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER_PATH = ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
RUNNER_PATH = ROOT / "scripts" / "fr13_run_b1_sfwd_state_fusion_gate.sh"

sys.path.insert(0, str(ROOT / "src"))
try:
    import triton  # noqa: F401
except ModuleNotFoundError:
    triton_stub = types.ModuleType("triton")

    def _jit(function=None, **_kwargs):
        return (lambda decorated: decorated) if function is None else function

    triton_stub.jit = _jit
    triton_stub.cdiv = lambda left, right: (left + right - 1) // right
    triton_stub.next_power_of_2 = lambda value: 1 << (value - 1).bit_length()
    language_stub = types.ModuleType("triton.language")
    triton_stub.language = language_stub
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = language_stub

from lumo_flywheel_serving import fr10_gdn_tree_kernel as kernel  # noqa: E402


def _function_source(name: str) -> str:
    source = KERNEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def test_contract_is_closed_and_launch_invariant_for_b1_b4() -> None:
    for batch in (1, 2, 3, 4):
        contract = kernel.fixed32_sfwd_state_fusion_contract(
            batch, tree_rows=32, conv_width=4, conv_state_len=12
        )
        assert contract["physical_rows_per_request"] == 32
        assert contract["logical_rows"] == batch * 32
        assert contract["source_rows_per_request"] == 36
        assert contract["source_rows"] == batch * 36
        assert contract["channels"] == 10240
        assert contract["conv_state_launches_per_layer"] == 1
        assert contract["conv_rows_per_program"] == 8
        assert contract["conv_row_groups_per_request"] == 4
        assert contract["gdn_level_path_programs"] == (batch, 11 * batch)
        assert contract["gdn_physical_launches_per_layer"] == 2
        assert contract["gdn_ring_export"] is True
        assert contract["gdn_flags_export"] is True
        assert contract["reference_always_served"] is True

    invalid = (
        (0, 32, 4, 12),
        (5, 32, 4, 12),
        (1, 31, 4, 12),
        (1, 32, 3, 12),
        (1, 32, 4, 11),
    )
    for batch, rows, width, state_len in invalid:
        with pytest.raises(ValueError):
            kernel.fixed32_sfwd_state_fusion_contract(
                batch,
                tree_rows=rows,
                conv_width=width,
                conv_state_len=state_len,
            )


def test_source_descriptor_is_the_exact_fixed32_window_mapping() -> None:
    parent = tuple(int(value) for value in kernel._FR13_FIXED32_PARENT)
    actual = kernel._fr13_fixed32_conv_source_flat_expected(4)
    assert len(parent) == 32
    assert len(actual) == 32 * 4

    expected: list[int] = []
    for node in range(32):
        path = []
        cursor = node
        while cursor >= 0:
            path.append(cursor)
            cursor = parent[cursor]
        path.reverse()
        source = [0, 1, 2] + [3 + path_node for path_node in path]
        expected.extend(source[-4:])
    assert actual == tuple(expected)
    assert actual[:4] == (0, 1, 2, 3)
    assert min(actual) == 0
    assert max(actual) == 34


def test_cpu_reference_matches_direct_fused_indexing_for_b1_b4() -> None:
    torch.manual_seed(20260801)
    channels = 8
    width = 4
    state_len = 12
    rows = 32
    source_flat = torch.tensor(
        kernel._fr13_fixed32_conv_source_flat_expected(width),
        dtype=torch.long,
    ).view(rows, width)
    conv_state = torch.randn(19, channels, state_len).to(torch.bfloat16)
    weights = torch.randn(channels, width).to(torch.bfloat16)
    bias = torch.randn(channels).to(torch.bfloat16)

    for batch in (1, 2, 3, 4):
        x = torch.randn(batch, rows, channels).to(torch.bfloat16)
        bank_rows = torch.tensor([2, 5, 11, 17][:batch], dtype=torch.long)
        reference_out = []
        reference_stage = []
        direct_out = torch.empty_like(x)
        direct_stage = torch.empty(
            batch, width - 1 + rows + 1, channels, dtype=torch.bfloat16
        )

        for request in range(batch):
            prior = conv_state[bank_rows[request], :, : width - 1]
            source = torch.cat(
                (
                    prior.transpose(0, 1),
                    x[request],
                    torch.zeros(1, channels, dtype=torch.bfloat16),
                ),
                dim=0,
            )
            reference_stage.append(source)
            window = source.index_select(0, source_flat.reshape(-1)).view(
                rows, width, channels
            )
            products = (
                (window * weights.t().unsqueeze(0)).to(torch.bfloat16).to(torch.float32)
            )
            acc = bias.to(torch.float32).view(1, -1) + products[:, 0]
            for tap in range(1, width):
                acc = acc + products[:, tap]
            reference_out.append(torch.nn.functional.silu(acc).to(torch.bfloat16))

            direct_stage[request, : width - 1] = prior.transpose(0, 1)
            direct_stage[request, width - 1 : width - 1 + rows] = x[request]
            direct_stage[request, -1].zero_()
            for node in range(rows):
                direct_acc = bias.to(torch.float32)
                for tap in range(width):
                    source_row = int(source_flat[node, tap])
                    if source_row < width - 1:
                        value = prior[:, source_row]
                    else:
                        value = x[request, source_row - (width - 1)]
                    product = (
                        (value * weights[:, tap]).to(torch.bfloat16).to(torch.float32)
                    )
                    direct_acc = direct_acc + product
                direct_out[request, node] = torch.nn.functional.silu(direct_acc).to(
                    torch.bfloat16
                )

        assert torch.equal(direct_stage, torch.stack(reference_stage, dim=0))
        assert torch.equal(direct_out, torch.stack(reference_out, dim=0))


def test_gate_control_requires_an_authenticated_real_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    enabled = tmp_path / "enabled"
    event = tmp_path / "event"
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "tail6_fixed32")

    assert kernel.fixed32_sfwd_state_fusion_gate_control(
        environ={}, enabled_path=str(enabled), event_path=str(event)
    ) == (False, None)
    enabled.write_text("1\n", encoding="ascii")
    assert kernel.fixed32_sfwd_state_fusion_gate_control(
        environ={}, enabled_path=str(enabled), event_path=str(event)
    ) == (True, None)

    event.write_text("probe:synthetic\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="swe_verified:<task_id>"):
        kernel.fixed32_sfwd_state_fusion_gate_control(
            environ={}, enabled_path=str(enabled), event_path=str(event)
        )
    event.write_text("swe_verified:astropy__astropy-12907\n", encoding="ascii")
    assert kernel.fixed32_sfwd_state_fusion_gate_control(
        environ={}, enabled_path=str(enabled), event_path=str(event)
    ) == (True, "swe_verified:astropy__astropy-12907")


def test_byte_gate_is_strict_and_never_marks_candidate_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "gate.jsonl"
    pass_path = tmp_path / "pass.json"
    monkeypatch.setenv("FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB_PATH", str(log_path))
    monkeypatch.setenv("FR13_FIXED32_SFWD_STATE_FUSION_PASS_PATH", str(pass_path))
    monkeypatch.setenv("FR13_DRAFT_VOCAB_K", "65536")
    monkeypatch.setenv("FR13_DRAFT_VOCAB_ROOT", "1")
    monkeypatch.setenv(
        "FR13_DRAFT_VOCAB_BLOCKS",
        str(ROOT / "scripts" / "fr13_dvk_subset_blocks.json"),
    )
    monkeypatch.setattr(
        kernel,
        "_FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB_STATE",
        {"task_marker": None, "batch": None, "passed": set(), "attempts": {}},
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    reference_out = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    reference_stage = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)
    equal = kernel.fixed32_sfwd_state_fusion_byte_gate(
        task_marker="swe_verified:astropy__astropy-12907",
        layer_key=17,
        batch_size=1,
        reference_out=reference_out,
        candidate_out=reference_out.clone(),
        reference_source_stage=reference_stage,
        candidate_source_stage=reference_stage.clone(),
    )
    assert equal["status"] == "pass"
    assert equal["zero_diff"] is True
    assert equal["reference_always_served"] is True
    assert equal["production_eligible"] is False

    mismatch_stage = reference_stage.clone()
    mismatch_stage[0, 0] = 5.0
    mismatch = kernel.fixed32_sfwd_state_fusion_byte_gate(
        task_marker="swe_verified:astropy__astropy-12907",
        layer_key=18,
        batch_size=1,
        reference_out=reference_out,
        candidate_out=reference_out.clone(),
        reference_source_stage=reference_stage,
        candidate_source_stage=mismatch_stage,
    )
    assert mismatch["status"] == "mismatch_reference_served"
    assert mismatch["zero_diff"] is False
    assert mismatch["first_nonzero"]["name"] == "commit_source_stage"
    assert mismatch["reference_always_served"] is True
    assert 18 not in kernel._FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB_STATE["passed"]

    kernel._fr13_fixed32_sfwd_state_fusion_pass_emit(
        task_marker="swe_verified:astropy__astropy-12907",
        batch=1,
        layer_keys=set(range(48)),
    )
    payload = json.loads(pass_path.read_text(encoding="ascii"))
    assert payload["status"] == "byte_pass_source_only"
    assert payload["run_classification"] == (
        "one_real_swe_verified_k64_root_b1_byte_diagnostic"
    )
    assert payload["draft_vocab_k"] == 65536
    assert payload["draft_vocab_root"] == 1
    assert payload["draft_vocab_blocks_sha256"] == (
        "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
    )
    assert payload["layer_count"] == 48
    assert payload["reference_always_served"] is True
    assert payload["real_task_authenticated"] is True
    assert payload["timing_eligible"] is False
    assert payload["floor_acceptance_eligible"] is False
    assert payload["production_eligible"] is False
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [record["status"] for record in records] == [
        "pass",
        "mismatch_reference_served",
    ]


def test_live_pass_rejects_non_k64_root_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_STATE_FUSION_PASS_PATH", str(tmp_path / "pass.json")
    )
    monkeypatch.setenv("FR13_DRAFT_VOCAB_K", "0")
    monkeypatch.setenv("FR13_DRAFT_VOCAB_ROOT", "0")
    monkeypatch.setenv(
        "FR13_DRAFT_VOCAB_BLOCKS",
        str(ROOT / "scripts" / "fr13_dvk_subset_blocks.json"),
    )
    with pytest.raises(RuntimeError, match="audited K64/root1 block map"):
        kernel._fr13_fixed32_sfwd_state_fusion_pass_emit(
            task_marker="swe_verified:astropy__astropy-12907",
            batch=1,
            layer_keys=set(range(48)),
        )


def test_kernel_and_wiring_preserve_order_and_reference_serving() -> None:
    candidate = _function_source("_fr13_fixed32_sfwd_state_fusion_kernel")
    launcher = _function_source("launch_fixed32_sfwd_state_fusion")
    patcher = PATCHER_PATH.read_text(encoding="utf-8")

    assert "for tap in tl.static_range(0, WIDTH):" in candidate
    assert "product = (value * weight).to(tl.bfloat16).to(tl.float32)" in candidate
    assert "acc = acc + product" in candidate
    assert "tl.sum" not in candidate
    assert "tl.dot" not in candidate
    assert "ROWS_PER_PROGRAM: tl.constexpr" in candidate
    assert "tl.arange(0, ROWS_PER_PROGRAM)[:, None]" in candidate
    assert "row_groups = N // ROWS_PER_PROGRAM" in candidate
    assert "n_mask" not in candidate
    assert "c_mask" not in candidate
    assert "channels != _FR13_FIXED32_SFWD_CHANNELS" in launcher
    assert "source_edge_writer = pid_n_base == 0" in candidate
    assert "source_stage" in candidate
    assert "rows_per_program = _FR13_FIXED32_SFWD_ROWS_PER_PROGRAM" in launcher
    assert "num_warps=8" in launcher
    assert "FR13_FIXED32_SFWD_STATE_FUSION source candidate is eager" in launcher
    assert "actual_source_flat" in launcher
    assert "source_flat.detach().cpu().tolist()" in launcher
    assert "launch_fixed32_sfwd_state_fusion(" in patcher
    assert "fixed32_sfwd_state_fusion_byte_gate(" in patcher
    assert "No candidate bytes become model inputs in this arm." in patcher
    assert "mixed_qkv_spec = _fr10_tree_conv_out" in patcher
    assert "mixed_qkv_spec = _fr13_sfwd_candidate_out" not in patcher

    tree = ast.parse(patcher)
    fragment = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "launch_fixed32_sfwd_state_fusion(" in node.value
    )
    ast.parse(textwrap.dedent(fragment))


def test_rowgroup8_covers_each_b1_b4_physical_row_once() -> None:
    rows = 32
    rows_per_program = kernel._FR13_FIXED32_SFWD_ROWS_PER_PROGRAM
    assert rows_per_program == 8
    for batch in (1, 2, 3, 4):
        row_groups = rows // rows_per_program
        covered: list[tuple[int, int]] = []
        edge_writers: list[tuple[int, int]] = []
        for linear_group in range(batch * row_groups):
            request = linear_group // row_groups
            group = linear_group - request * row_groups
            base = group * rows_per_program
            for offset in range(rows_per_program):
                covered.append((request, base + offset))
                if base + offset == 0:
                    edge_writers.append((request, base + offset))
        assert covered == [
            (request, row)
            for request in range(batch)
            for row in range(rows)
        ]
        assert edge_writers == [(request, 0) for request in range(batch)]


def test_b1_k64_root_runner_is_reference_returning_and_nonacceptance() -> None:
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "subset_b1_diagnostic_one.json" in runner
    assert "subset_b4_four.json" not in runner
    assert "subset_b16" not in runner
    assert "FR13_DRAFT_VOCAB_ROOT=1" in runner
    assert "FR13_DRAFT_VOCAB_K=65536" in runner
    assert "FR13_DRAFT_VOCAB_BLOCKS=\"$DRAFT_VOCAB_BLOCKS_CONTAINER\"" in runner
    assert "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff" in runner
    assert "K64_ROOT_WEIGHT_BYTES=32666638208" in runner
    assert "K64_ROOT_FLOOR_MS=119.658015414" in runner
    assert "K64_ROOT_CAP_MS=137.6067177261" in runner
    assert 'docker_after_tasks_path = arm_dir / "docker_after_tasks.log"' in runner
    assert 'shim_prefix = "[FR13_DRAFT_VOCAB] shim built K=65536 "' in runner
    assert 'root_prefix = "[FR13_DRAFT_VOCAB_ROOT] engaged K=65536 "' in runner
    assert 'disabled_prefix = "[FR13_DRAFT_VOCAB] DISABLED"' in runner
    assert 'len(shim_lines) != 1 or "mode=gather" not in shim_lines[0]' in runner
    assert 'len(root_lines) != 1 or "mode=gather" not in root_lines[0]' in runner
    assert "draft-vocabulary runtime fallback to full vocabulary engaged" in runner
    assert 'run_root / "runtime_manifest.at_launch.json"' in runner
    assert 'run_root / "runtime_manifest.at_end.json"' in runner
    assert 'run_root / "external_manifest.at_launch.json"' in runner
    assert 'run_root / "external_manifest.at_end.json"' in runner
    assert "runtime_manifest_launch_raw != runtime_manifest_end_raw" in runner
    assert "external_manifest_launch_raw != external_manifest_end_raw" in runner
    assert "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=1" in runner
    assert "FR13_CONV_WB_BATCHED=1" in runner
    assert "FR13_TREE_CONV_FUSED=1" in runner
    assert "timing_eligible=false" in runner
    assert "floor_acceptance_eligible=false" in runner
    assert "reference_returned=true" in runner
    assert "physical_rows_per_request=32" in runner
    assert "gdn_level_path_programs=1,11" in runner
    assert "PROBE_ONLY" not in runner
    assert "ACCEPT_SPEED_PROBE" not in runner

    assert (
        "FR13_FIXED32_SFWD_STATE_FUSION_REAL_EVENT_PATH=/logs/"
        "fr13_fixed32_sfwd_state_fusion.real_event.arm"
    ) in launcher
    assert "must be the only kernel candidate" in launcher
    assert "requires exact K64/root1 eager fixed32 B1" in launcher
    assert "fr13_fixed32_sfwd_state_fusion_byte_ab.enabled" in launcher

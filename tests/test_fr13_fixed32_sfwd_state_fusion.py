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
from lumo_flywheel_serving import (  # noqa: E402
    fr13_sfwd_state_fusion_production as production,
)


LIVE_PASS_PATH = (
    ROOT
    / "results"
    / "fr13_fixed32_sfwd_b1_real_task_byte_pass_20260801"
    / "run_evidence"
    / "fr13_fixed32_sfwd_state_fusion.live_pass.json"
)


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
            batch, tree_rows=32, conv_width=4, conv_state_len=34
        )
        assert contract["physical_rows_per_request"] == 32
        assert contract["logical_rows"] == batch * 32
        assert contract["source_rows_per_request"] == 36
        assert contract["source_rows"] == batch * 36
        assert contract["conv_state_len"] == 34
        assert contract["conv_state_launches_per_layer"] == 1
        assert contract["gdn_level_path_programs"] == (batch, 11 * batch)
        assert contract["gdn_physical_launches_per_layer"] == 2
        assert contract["gdn_ring_export"] is True
        assert contract["gdn_flags_export"] is True
        assert contract["reference_always_served"] is True

    invalid = (
        (0, 32, 4, 34),
        (5, 32, 4, 34),
        (1, 31, 4, 34),
        (1, 32, 3, 34),
        (1, 32, 4, 12),
        (1, 32, 4, 33),
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
    state_len = 34
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
    event.chmod(0o444)
    with pytest.raises(RuntimeError, match="canonical exact4"):
        kernel.fixed32_sfwd_state_fusion_gate_control(
            environ={}, enabled_path=str(enabled), event_path=str(event)
        )
    event.chmod(0o600)
    markers = kernel._FR13_FIXED32_SFWD_STATE_FUSION_EXACT4_TASK_MARKERS
    event.write_text("\n".join(markers) + "\n", encoding="ascii")
    event.chmod(0o444)
    assert kernel.fixed32_sfwd_state_fusion_gate_control(
        environ={}, enabled_path=str(enabled), event_path=str(event)
    ) == (True, markers)


def test_byte_gate_is_strict_and_never_marks_candidate_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "gate.jsonl"
    pass_path = tmp_path / "pass.json"
    monkeypatch.setenv("FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB_PATH", str(log_path))
    monkeypatch.setenv("FR13_FIXED32_SFWD_STATE_FUSION_PASS_PATH", str(pass_path))
    monkeypatch.setattr(
        kernel,
        "_FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB_STATE",
        {
            "task_markers": None,
            "batch": None,
            "passed": set(),
            "attempts": {},
            "failed": False,
        },
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    reference_out = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    reference_stage = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)
    equal = kernel.fixed32_sfwd_state_fusion_byte_gate(
        task_markers=(
            kernel._FR13_FIXED32_SFWD_STATE_FUSION_EXACT4_TASK_MARKERS
        ),
        layer_key=17,
        batch_size=4,
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
        task_markers=(
            kernel._FR13_FIXED32_SFWD_STATE_FUSION_EXACT4_TASK_MARKERS
        ),
        layer_key=18,
        batch_size=4,
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
        task_markers=(
            kernel._FR13_FIXED32_SFWD_STATE_FUSION_EXACT4_TASK_MARKERS
        ),
        batch=4,
        layer_keys=set(range(48)),
    )
    payload = json.loads(pass_path.read_text(encoding="ascii"))
    assert payload["status"] == "byte_pass_source_only"
    assert payload["run_classification"] == (
        "real_swe_verified_exact4_b4_byte_diagnostic"
    )
    assert payload["task_markers"] == list(
        kernel._FR13_FIXED32_SFWD_STATE_FUSION_EXACT4_TASK_MARKERS
    )
    assert payload["batch_size"] == 4
    assert payload["concurrency"] == 4
    assert payload["layer_count"] == 48
    assert payload["reference_always_served"] is True
    assert payload["production_eligible"] is False
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [record["status"] for record in records] == [
        "pass",
        "mismatch_reference_served",
    ]


def test_kernel_and_wiring_preserve_order_and_reference_serving() -> None:
    candidate = _function_source("_fr13_fixed32_sfwd_state_fusion_kernel")
    launcher = _function_source("launch_fixed32_sfwd_state_fusion")
    patcher = PATCHER_PATH.read_text(encoding="utf-8")

    assert "for tap in tl.static_range(0, WIDTH):" in candidate
    assert "product = (value * weight).to(tl.bfloat16).to(tl.float32)" in candidate
    assert "acc = acc + product" in candidate
    assert "tl.sum" not in candidate
    assert "tl.dot" not in candidate
    assert "source_stage" in candidate
    assert "x_stride_row" in candidate
    assert "* x_stride_row" in candidate
    assert "FR13_FIXED32_SFWD_STATE_FUSION source candidate is eager" in launcher
    assert "actual_source_flat" in launcher
    assert "source_flat.detach().cpu().tolist()" in launcher
    assert "int(x.stride(1)) != 1" in launcher
    assert "int(x.stride(0)) < channels" in launcher
    assert "not x.is_contiguous()" not in launcher
    assert "launch_fixed32_sfwd_state_fusion(" in patcher
    assert "fixed32_sfwd_state_fusion_byte_gate(" in patcher
    assert "task_markers=_fr13_sfwd_task_markers" in patcher
    assert "int(attn_metadata.num_spec_decodes) == 4" in patcher
    assert patcher.count("int(conv_state.size(2)) != 34") == 2
    assert "int(conv_state.size(2)) != 12" not in patcher
    assert "No candidate bytes become model inputs in this arm." in patcher
    assert "mixed_qkv_spec = _fr10_tree_conv_out" in patcher
    assert "mixed_qkv_spec = _fr13_sfwd_candidate_out" not in patcher
    assert "fixed32_sfwd_state_fusion_production_control()" in patcher
    assert "fixed32_sfwd_state_fusion_production_engagement(" in patcher
    assert "0\n                        if _fr13_sfwd_production is not None" in patcher

    tree = ast.parse(patcher)
    fragment = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "launch_fixed32_sfwd_state_fusion(" in node.value
    )
    ast.parse(textwrap.dedent(fragment))


def _production_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
) -> dict[str, object]:
    live = tmp_path / "live-pass.json"
    live.write_bytes(LIVE_PASS_PATH.read_bytes())
    digest = __import__("hashlib").sha256(live.read_bytes()).hexdigest()
    digest_path = tmp_path / "live-pass.sha256"
    digest_path.write_text(digest + "\n", encoding="ascii")
    arm = tmp_path / "production.arm"
    arm.write_text("1\n", encoding="ascii")
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", mode)
    production._CREDENTIAL_IDS.clear()
    monkeypatch.setattr(
        production,
        "_STATE",
        {
            "live_pass_sha256": None,
            "source_sha256": None,
            "closure_sha256": None,
            "layers": set(),
            "launches": 0,
            "emitted": False,
        },
    )
    credential = production.fixed32_sfwd_state_fusion_production_control(
        environ={
            "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION": "1",
            "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB": "0",
            "FR13_DRAFT_VOCAB_ROOT": "1",
            "FR13_DRAFT_VOCAB_K": "65536",
            "FR13_FIXED32_SFWD_STATE_FUSION_LIVE_PASS_SHA256": digest,
            "FR13_FIXED32_SFWD_STATE_FUSION_ENABLED_PATH": str(
                tmp_path / "absent-byte-gate"
            ),
        },
        arm_path=str(arm),
        pass_path=str(live),
        pass_sha256_path=str(digest_path),
    )
    assert credential is not None
    return credential


def test_production_closure_matches_the_byte_qualified_candidate() -> None:
    closure = production.fixed32_sfwd_state_fusion_candidate_closure()
    assert closure["sha256"] == production.QUALIFIED_CLOSURE_SHA256
    assert tuple(closure["members"]) == production._CLOSURE_NAMES
    payload = json.loads(LIVE_PASS_PATH.read_text(encoding="ascii"))
    assert payload["source_sha256"] == production.QUALIFIED_KERNEL_SOURCE_SHA256


@pytest.mark.parametrize("mode", ("tail6_fixed32", "hydra27_fixed32"))
def test_production_requires_k64_root1_and_attests_all_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    credential = _production_credential(tmp_path, monkeypatch, mode=mode)
    assert credential["candidate_closure_sha256"] == (
        production.QUALIFIED_CLOSURE_SHA256
    )
    assert credential["qualified_source_sha256"] == (
        production.QUALIFIED_KERNEL_SOURCE_SHA256
    )

    engagement_path = tmp_path / "engagement.json"
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION_ENGAGEMENT_PATH",
        str(engagement_path),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for layer_key in range(48):
        record = production.fixed32_sfwd_state_fusion_production_engagement(
            credential=credential,
            layer_key=layer_key,
            batch_size=1,
        )
    assert record["layer_count"] == 48
    payload = json.loads(engagement_path.read_text(encoding="ascii"))
    assert payload["status"] == "engaged"
    assert payload["run_classification"] == (
        "real_swe_verified_exact4_k64_b1_kernel_stack"
    )
    assert payload["physical_rows_per_request"] == 32
    assert payload["draft_vocab_root"] == 1
    assert payload["draft_vocab_k"] == 65536
    assert payload["candidate_conv_launches_per_layer"] == 1
    assert payload["incumbent_conv_launches_per_layer"] == 0
    assert payload["timing_eligible"] is True
    assert payload["floor_acceptance_eligible"] is False


def test_production_rejects_non_k64_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = tmp_path / "production.arm"
    arm.write_text("1\n", encoding="ascii")
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "hydra27_fixed32")
    with pytest.raises(RuntimeError, match="ROOT=1 K=65536"):
        production.fixed32_sfwd_state_fusion_production_control(
            environ={
                "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION": "1",
                "FR13_DRAFT_VOCAB_ROOT": "0",
                "FR13_DRAFT_VOCAB_K": "65536",
            },
            arm_path=str(arm),
        )

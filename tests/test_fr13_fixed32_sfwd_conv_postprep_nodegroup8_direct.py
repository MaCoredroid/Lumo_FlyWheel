from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "src"
    / "lumo_flywheel_serving"
    / "fr13_sfwd_conv_postprep_fusion.py"
)
KERNEL_PATH = (
    ROOT
    / "src"
    / "lumo_flywheel_serving"
    / "fr13_sfwd_conv_postprep_fusion_kernel.py"
)
GENERATOR_PATH = (
    ROOT / "scripts" / "fr13_generate_sfwd_conv_postprep_fusion_kernel.py"
)
INCUMBENT_KERNEL_SHA256 = (
    "0384e4947e605846c9ed995bc73fa1252a6f5f815d1bc905685527fbf7f8d8ff"
)
PATCHER_PATH = ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
GATE_PATH = ROOT / "scripts" / "fr13_sfwd_conv_postprep_gate.py"
LAUNCHER_PATH = ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
RUNNER_PATH = ROOT / "scripts" / "fr13_run_b1_sfwd_conv_postprep_gate.sh"

sys.path.insert(0, str(ROOT / "src"))
try:
    import triton  # noqa: F401
except ModuleNotFoundError:
    triton_stub = types.ModuleType("triton")

    def _jit(function=None, **_kwargs):
        return (lambda decorated: decorated) if function is None else function

    triton_stub.jit = _jit
    triton_stub.cdiv = lambda left, right: (left + right - 1) // right
    language_stub = types.ModuleType("triton.language")
    triton_stub.language = language_stub
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = language_stub

from lumo_flywheel_serving import (  # noqa: E402
    fr13_sfwd_conv_postprep_fusion as candidate,
)
from lumo_flywheel_serving.fr13_sfwd_prior_reuse_descriptorless import (  # noqa: E402
    fixed32_descriptorless_sources,
)


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    segment = ast.get_source_segment(source, function)
    assert segment is not None
    return segment


def _serial_oracle(
    x: torch.Tensor,
    prior: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, rows, channels = x.shape
    output = torch.empty_like(x)
    stage = torch.zeros((batch, rows + 4, channels), dtype=torch.bfloat16)
    sources = fixed32_descriptorless_sources()
    for request in range(batch):
        all_sources = torch.cat((prior[request], x[request]), dim=0)
        for node, historical in enumerate(sources):
            acc = bias.clone()
            for tap, source_row in enumerate((*historical, node + 3)):
                product = (
                    all_sources[source_row] * weights[:, tap]
                ).to(torch.bfloat16).to(torch.float32)
                acc = acc + product
            output[request, node] = (
                acc / (1.0 + torch.exp(0.0 - acc))
            ).to(torch.bfloat16)
        stage[request] = torch.cat(
            (prior[request], x[request], torch.zeros((1, channels))), dim=0
        ).to(torch.bfloat16)
    return output, stage


def _nodepair16_oracle(
    x: torch.Tensor,
    prior: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, rows, channels = x.shape
    output = torch.empty_like(x)
    stage = torch.full(
        (batch, rows + 4, channels),
        float("nan"),
        dtype=torch.bfloat16,
    )
    sources = fixed32_descriptorless_sources()
    for request in range(batch):
        for pair in range(2):
            for group in range(pair * 2, pair * 2 + 2):
                nodes = range(group * 8, (group + 1) * 8)
                unique_x = {
                    row - 3
                    for node in nodes
                    for row in sources[node]
                    if row >= 3
                } | set(nodes)
                x_values = {row: x[request, row] for row in unique_x}
                for node in nodes:
                    operands = tuple(
                        prior[request, row]
                        if row < 3
                        else x_values[row - 3]
                        for row in sources[node]
                    ) + (x_values[node],)
                    product_0 = (operands[0] * weights[:, 0]).to(
                        torch.bfloat16
                    ).to(torch.float32)
                    acc = bias + product_0
                    for tap in range(1, 4):
                        product = (operands[tap] * weights[:, tap]).to(
                            torch.bfloat16
                        ).to(torch.float32)
                        acc = acc + product
                    output[request, node] = (
                        acc / (1.0 + torch.exp(0.0 - acc))
                    ).to(torch.bfloat16)
                    stage[request, node + 3] = x_values[node]
                if group == 0:
                    stage[request, :3] = prior[request]
                    stage[request, -1] = 0.0
    return output, stage


@pytest.mark.parametrize("batch", (1, 4))
def test_nodepair16_cpu_oracle_matches_serial_bytes(batch: int) -> None:
    generator = torch.Generator().manual_seed(20260805 + batch)
    channels = 24
    x = torch.randn((batch, 32, channels), generator=generator).to(
        torch.bfloat16
    )
    prior = torch.randn((batch, 3, channels), generator=generator).to(
        torch.bfloat16
    )
    weights = torch.randn((channels, 4), generator=generator).to(
        torch.bfloat16
    )
    bias = torch.randn((channels,), generator=generator, dtype=torch.float32)
    serial_output, serial_stage = _serial_oracle(x, prior, weights, bias)
    grouped_output, grouped_stage = _nodepair16_oracle(x, prior, weights, bias)
    assert torch.equal(
        serial_output.contiguous().view(torch.uint8),
        grouped_output.contiguous().view(torch.uint8),
    )
    assert torch.equal(
        serial_stage.contiguous().view(torch.uint8),
        grouped_stage.contiguous().view(torch.uint8),
    )


def test_nodepair16_source_is_direct_and_mechanically_covers_fixed32() -> None:
    source = _function_source(
        KERNEL_PATH,
        "_fr13_fixed32_sfwd_conv_postprep_nodepair16_direct_kernel",
    )
    expected_x = (
        {0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        {1, 2, 3, 4, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23},
        {9, 13, 14, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31},
    )
    for group, expected in enumerate(expected_x):
        observed = {
            int(line.split(f"x_g{group}_", 1)[1].split(" ", 1)[0])
            for line in source.splitlines()
            if f"x_g{group}_" in line and " = tl.load(" in line
        }
        assert observed == expected
    assert source.count("if pid_pair == 0:") == 1
    assert source.count("elif pid_pair == 1:") == 1
    assert source.count("# Serial logical nodegroup") == 4
    assert source.count("_fr13_store_fixed32_conv_outputs(") == 32
    assert source.count(".to(tl.bfloat16).to(tl.float32)") == 128
    assert source.count("stage_batch + ((WIDTH - 1) +") == 32
    assert source.count("tl.store(stage_batch + offs_c, prior_0)") == 1
    assert source.count("tl.store(stage_batch + C + offs_c, prior_1)") == 1
    assert source.count("stage_batch + 2 * C + offs_c, prior_2") == 1
    assert source.count("stage_batch + (SOURCE_ROWS - 1) * C + offs_c") == 1
    for forbidden in (
        "tl.gather",
        "tl.shared",
        "tl.sum",
        "tl.dot",
        "debug_barrier",
        "ROWS_PER_PROGRAM",
    ):
        assert forbidden not in source


def test_nodepair16_generator_is_idempotent_and_preserves_incumbent() -> None:
    spec = importlib.util.spec_from_file_location(
        "sfwd_nodegroup8_generator", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generated_once = generator.generate()
    generated_twice = generator.generate()
    assert generated_once == generated_twice
    assert generated_once == KERNEL_PATH.read_text(encoding="utf-8")
    assert generator._descriptorless_sources() == fixed32_descriptorless_sources()

    incumbent = _function_source(
        KERNEL_PATH, "_fr13_fixed32_sfwd_conv_postprep_fusion_kernel"
    )
    assert hashlib.sha256(incumbent.encode("utf-8")).hexdigest() == (
        INCUMBENT_KERNEL_SHA256
    )


def _contract(batch: int, **kwargs) -> dict[str, object]:
    return candidate.fixed32_sfwd_conv_postprep_fusion_contract(
        batch,
        fixed32_mode="hydra27_fixed32",
        tree_parent=candidate.FIXED32_PARENT,
        qualification_profile="k64_root",
        draft_vocab_k=65536,
        draft_vocab_root=1,
        **kwargs,
    )


def test_nodepair16_contract_is_default_off_and_halves_program_geometry() -> None:
    assert _contract(1) == _contract(1, direct_nodegroup8=False)
    for batch in (1, 4):
        standalone = _contract(batch, direct_nodegroup8=True)
        embedded = _contract(
            batch, direct_nodegroup8=True, embed_gate_cta=True
        )
        assert standalone["candidate"] == candidate.DIRECT_NODEGROUP8_CANDIDATE
        assert standalone["channel_programs_per_request"] == 80
        assert standalone["programs_per_request"] == 84
        assert embedded["candidate"] == (
            candidate.DIRECT_NODEGROUP8_EMBEDDED_GATE_CANDIDATE
        )
        assert embedded["channel_programs_per_request"] == 80
        assert embedded["programs_per_request"] == 80
        assert embedded["embedded_gating_channel_programs_per_request"] == 4
        assert standalone["node_groups_per_request"] == 4
        assert standalone["nodegroups_per_channel_program"] == 2
        assert standalone["nodes_per_channel_program"] == 16
        assert standalone["channel_program_pairs_per_request"] == 2
        assert standalone["node_group_unique_x_loads"] == (8, 14, 18, 14)
        assert standalone["node_group_peak_live_x"] == (4, 9, 10, 7)
        assert standalone["has_gather"] is False
        assert standalone["algorithmic_shared_bytes"] == 0
        assert standalone["has_reduction"] is False
        assert standalone["has_barrier"] is False
        assert standalone["candidate_codegen_registers_per_thread"] is None
        assert standalone["source_register_ceiling_per_thread"] == 48
        assert standalone["offline_codegen_stack_bytes"] is None
        assert standalone["offline_codegen_local_bytes"] is None
        assert standalone["offline_codegen_shared_bytes"] is None
        assert standalone["codegen_registers_verified"] is False
        assert standalone["codegen_status"] == "pending_sm121a_offline_codegen"
        assert embedded["candidate_codegen_registers_per_thread"] is None
        assert standalone["full_graph_qualified"] is False
        assert standalone["timing_claim"] is False


def test_nodepair16_launcher_selector_keeps_one_launch_per_arm() -> None:
    source = _function_source(
        MODULE_PATH, "launch_fixed32_sfwd_conv_postprep_fusion"
    )
    tree = ast.parse(source)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    defaults = dict(
        zip(
            (argument.arg for argument in function.args.kwonlyargs),
            function.args.kw_defaults,
            strict=True,
        )
    )
    direct_default = defaults["direct_nodegroup8"]
    assert isinstance(direct_default, ast.Constant)
    assert direct_default.value is False
    assert source.count(
        "_fr13_fixed32_sfwd_conv_postprep_nodepair16_direct_kernel[grid]("
    ) == 1
    assert source.count(
        "_fr13_fixed32_sfwd_conv_postprep_fusion_kernel[grid]("
    ) == 1
    assert "if direct_nodegroup8:\n        channel_tasks *= 2" in source
    assert "channel_tasks if embed_gate_cta else channel_tasks + gate_tasks" in source


def _reset_byte_gate_state() -> None:
    candidate._BYTE_GATE_STATE.clear()
    candidate._BYTE_GATE_STATE.update(
        task_marker=None,
        batch_size=None,
        embedded_gate_cta=None,
        direct_nodegroup8=None,
        source_binding=None,
        passed={},
        attempts={},
        failed=False,
    )


def _direct_manifest(tmp_path: Path, commit: str) -> tuple[Path, str]:
    path = tmp_path / "direct.source_manifest.json"
    payload = {
        "schema": candidate.SOURCE_MANIFEST_SCHEMA,
        "candidate": candidate.DIRECT_NODEGROUP8_CANDIDATE,
        "source_commit": commit,
        "files": {
            relative: {
                "bytes": len((ROOT / relative).read_bytes()),
                "sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
            }
            for relative in candidate.SOURCE_FILES
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    path.chmod(0o400)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_nodepair16_real_b1_control_is_default_off_and_fail_closed(
    tmp_path: Path,
) -> None:
    enabled = tmp_path / "enabled"
    enabled.write_bytes(b"1\n")
    enabled.chmod(0o400)
    event = tmp_path / "event"
    event.write_text(candidate.TASK_MARKER + "\n", encoding="ascii")
    event.chmod(0o444)
    assert candidate.fixed32_sfwd_conv_postprep_gate_control(
        fixed32_mode="hydra27_fixed32",
        environ={},
        enabled_path=str(enabled),
        event_path=str(event),
    ) == (False, None)
    exact = {
        "FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB": "1",
        "FR13_FIXED32_SFWD_NODEGROUP8_DIRECT": "1",
        "FR13_DRAFT_VOCAB_ROOT": "1",
        "FR13_DRAFT_VOCAB_K": "65536",
        "MAX_NUM_SEQS": "1",
    }
    assert candidate.fixed32_sfwd_conv_postprep_gate_control(
        fixed32_mode="hydra27_fixed32",
        environ=exact,
        enabled_path=str(enabled),
        event_path=str(event),
    ) == (True, candidate.TASK_MARKER)
    with pytest.raises(RuntimeError, match="Hydra27 physical32 K64/root1 B1"):
        candidate.fixed32_sfwd_conv_postprep_gate_control(
            fixed32_mode="tail6_fixed32",
            environ=exact,
            enabled_path=str(enabled),
            event_path=str(event),
        )


@pytest.mark.parametrize("embedded_gate_cta", (False, True))
def test_nodepair16_real_b1_byte_gate_binds_candidate_and_serves_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    embedded_gate_cta: bool,
) -> None:
    commit = "8" * 40
    manifest, manifest_sha256 = _direct_manifest(tmp_path, commit)
    records = tmp_path / "records.jsonl"
    live_pass = tmp_path / "live_pass.json"
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB_PATH", str(records)
    )
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_PASS_PATH", str(live_pass)
    )
    monkeypatch.setenv(
        "FR13_DRAFT_VOCAB_BLOCKS",
        str(ROOT / "scripts" / "fr13_dvk_subset_blocks.json"),
    )
    monkeypatch.setenv("FR13_DRAFT_VOCAB_ROOT", "1")
    monkeypatch.setenv("FR13_DRAFT_VOCAB_K", "65536")
    monkeypatch.setenv("FR13_FA2_QROW16_PRODUCTION", "1")
    monkeypatch.setenv("FR13_FA2_QROW16_SO_SHA256", candidate.QROW16_FA2_SHA256)
    monkeypatch.setenv(
        "FR13_FA2_QROW16_LIVE_PASS_SHA256", candidate.QROW16_LIVE_PASS_SHA256
    )
    reference = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
    _reset_byte_gate_state()
    for layer_key in range(1, candidate.LAYERS + 1):
        record = candidate.fixed32_sfwd_conv_postprep_byte_gate(
            fixed32_mode="hydra27_fixed32",
            task_marker=candidate.TASK_MARKER,
            layer_prefix=f"model.layers.{layer_key}",
            layer_key=layer_key,
            batch_size=1,
            reference_query=reference,
            candidate_query=reference.clone(),
            reference_key=reference,
            candidate_key=reference.clone(),
            reference_value_spec=reference,
            candidate_value_spec=reference.clone(),
            reference_value_tree=reference,
            candidate_value_tree=reference.clone(),
            reference_g=reference,
            candidate_g=reference.clone(),
            reference_beta=reference,
            candidate_beta=reference.clone(),
            reference_source_stage=reference,
            candidate_source_stage=reference.clone(),
            source_manifest_path=str(manifest),
            expected_source_manifest_sha256=manifest_sha256,
            expected_source_commit=commit,
            embedded_gate_cta=embedded_gate_cta,
            direct_nodegroup8=True,
        )
        assert record["zero_diff"] is True
        assert record["reference_always_served"] is True
        assert record["candidate_returned"] is False
    expected_candidate = (
        candidate.DIRECT_NODEGROUP8_EMBEDDED_GATE_CANDIDATE
        if embedded_gate_cta
        else candidate.DIRECT_NODEGROUP8_CANDIDATE
    )
    expected_programs = 80 if embedded_gate_cta else 84
    assert record["candidate"] == expected_candidate
    assert record["direct_nodegroup8"] is True
    assert record["programs_per_request"] == expected_programs
    payload = json.loads(live_pass.read_text(encoding="ascii"))
    assert payload["candidate"] == expected_candidate
    assert payload["direct_nodegroup8"] is True
    assert payload["programs_per_request"] == expected_programs
    assert payload["reference_always_served"] is True
    assert payload["candidate_returned"] is False
    _reset_byte_gate_state()


def test_nodepair16_selector_reaches_authenticated_real_b1_route() -> None:
    patcher = PATCHER_PATH.read_text(encoding="utf-8")
    gate = GATE_PATH.read_text(encoding="utf-8")
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    selector = "FR13_FIXED32_SFWD_NODEGROUP8_DIRECT"
    assert f'{selector}", "0"' in patcher
    assert "direct_nodegroup8=(" in patcher
    assert patcher.count("_FR13_FIXED32_SFWD_NODEGROUP8_DIRECT") >= 8
    assert f"{selector}=${{{selector}:-0}}" in launcher
    assert f'-e {selector}="${selector}"' in launcher
    assert "direct nodegroup8 byte gate requires exact Hydra27 B1" in launcher
    assert f"{selector}=${{{selector}:-0}}" in runner
    assert "DIRECT_ARGS+=(--direct-nodegroup8)" in runner
    assert runner.count('"${DIRECT_ARGS[@]}"') == 5
    assert gate.count('add_argument("--direct-nodegroup8"') == 4
    assert "DIRECT_NODEGROUP8_CANDIDATE" in gate
    assert f'"{candidate.DIRECT_NODEGROUP8_CANDIDATE}"' in gate
    assert f'"{candidate.DIRECT_NODEGROUP8_CANDIDATE}"' in patcher
    assert "DIRECT_CHANNEL_PROGRAMS_PER_REQUEST = 80" in gate

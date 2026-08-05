from __future__ import annotations

import ast
import hashlib
import importlib.util
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


def _nodegroup8_oracle(
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
        for group in range(4):
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
def test_nodegroup8_cpu_oracle_matches_serial_bytes(batch: int) -> None:
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
    grouped_output, grouped_stage = _nodegroup8_oracle(x, prior, weights, bias)
    assert torch.equal(
        serial_output.contiguous().view(torch.uint8),
        grouped_output.contiguous().view(torch.uint8),
    )
    assert torch.equal(
        serial_stage.contiguous().view(torch.uint8),
        grouped_stage.contiguous().view(torch.uint8),
    )


def test_nodegroup8_source_is_direct_and_mechanically_covers_fixed32() -> None:
    source = _function_source(
        KERNEL_PATH,
        "_fr13_fixed32_sfwd_conv_postprep_nodegroup8_direct_kernel",
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
    assert source.count("if pid_group == 0:") == 1
    assert sum(source.count(f"elif pid_group == {group}:") for group in (1, 2, 3)) == 3
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


def test_nodegroup8_generator_is_idempotent_and_preserves_incumbent() -> None:
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


def test_nodegroup8_contract_is_default_off_and_has_exact_program_geometry() -> None:
    assert _contract(1) == _contract(1, direct_nodegroup8=False)
    for batch in (1, 4):
        standalone = _contract(batch, direct_nodegroup8=True)
        embedded = _contract(
            batch, direct_nodegroup8=True, embed_gate_cta=True
        )
        assert standalone["candidate"] == candidate.DIRECT_NODEGROUP8_CANDIDATE
        assert standalone["channel_programs_per_request"] == 160
        assert standalone["programs_per_request"] == 164
        assert embedded["candidate"] == (
            candidate.DIRECT_NODEGROUP8_EMBEDDED_GATE_CANDIDATE
        )
        assert embedded["channel_programs_per_request"] == 160
        assert embedded["programs_per_request"] == 160
        assert embedded["embedded_gating_channel_programs_per_request"] == 4
        assert standalone["node_groups_per_request"] == 4
        assert standalone["nodes_per_channel_program"] == 8
        assert standalone["node_group_unique_x_loads"] == (8, 14, 18, 14)
        assert standalone["node_group_peak_live_x"] == (4, 9, 10, 7)
        assert standalone["has_gather"] is False
        assert standalone["algorithmic_shared_bytes"] == 0
        assert standalone["has_reduction"] is False
        assert standalone["has_barrier"] is False
        assert standalone["candidate_codegen_registers_per_thread"] == 46
        assert standalone["source_register_ceiling_per_thread"] == 48
        assert standalone["offline_codegen_stack_bytes"] == 0
        assert standalone["offline_codegen_local_bytes"] == 0
        assert standalone["offline_codegen_shared_bytes"] == 0
        assert standalone["codegen_registers_verified"] is True
        assert embedded["candidate_codegen_registers_per_thread"] == 48
        assert standalone["full_graph_qualified"] is False
        assert standalone["timing_claim"] is False


def test_nodegroup8_launcher_selector_keeps_one_launch_per_arm() -> None:
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
        "_fr13_fixed32_sfwd_conv_postprep_nodegroup8_direct_kernel[grid]("
    ) == 1
    assert source.count(
        "_fr13_fixed32_sfwd_conv_postprep_fusion_kernel[grid]("
    ) == 1
    assert "if direct_nodegroup8:\n        channel_tasks *= 4" in source
    assert "channel_tasks if embed_gate_cta else channel_tasks + gate_tasks" in source

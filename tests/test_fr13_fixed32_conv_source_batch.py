"""CPU gate for the default-off fixed32 batched conv-source builder."""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lumo_flywheel_serving.fr13_tree_conv_fused import (  # noqa: E402
    fused_tree_conv_source,
    fused_tree_conv_sources_batched,
)

PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
SEQUENCE = REPO / "scripts" / "fr13_fixed32_floor_timers_seq.sh"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"


def _conv_replacement() -> str:
    tree = ast.parse(PATCHER.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "conv_replacement"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise AssertionError("conv_replacement string not found")


def _bits(value: torch.Tensor) -> torch.Tensor:
    view_dtype = torch.int16 if value.dtype == torch.bfloat16 else torch.int32
    return value.contiguous().view(view_dtype)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("batch", (1, 2, 3, 4))
def test_b1_b4_sources_are_byte_exact_and_stage_direct(
    dtype: torch.dtype,
    batch: int,
) -> None:
    tree_n = 32
    channels = 19
    prior_cols = torch.tensor((9, 2, 13), dtype=torch.long)
    generator = torch.Generator().manual_seed(1700 + batch)
    prior_bank = torch.randn(
        (batch, channels, 17), generator=generator, dtype=dtype
    )
    x = torch.randn(
        (batch * tree_n, channels), generator=generator, dtype=dtype
    )
    zero_row = torch.zeros((1, channels), dtype=dtype)
    zero_row[0, 0] = -0.0
    source_rows = int(prior_cols.numel()) + tree_n + 1
    staging = torch.full(
        (batch * source_rows + 7, channels), 11.0, dtype=dtype
    )
    untouched_tail = staging[batch * source_rows :].clone()

    expected_priors = torch.stack(
        [
            prior_bank[request].index_select(1, prior_cols)
            for request in range(batch)
        ]
    )
    expected_sources = torch.stack(
        [
            fused_tree_conv_source(
                prior_window=expected_priors[request],
                x=x[request * tree_n : (request + 1) * tree_n],
                zero_row=zero_row,
            )
            for request in range(batch)
        ]
    )

    sources, prior_windows = fused_tree_conv_sources_batched(
        prior_bank=prior_bank,
        prior_cols=prior_cols,
        x=x,
        zero_row=zero_row,
        staging=staging,
        batch=batch,
        tree_n=tree_n,
    )

    assert sources.data_ptr() == staging.data_ptr()
    assert sources.shape == (batch, source_rows, channels)
    assert torch.equal(_bits(sources), _bits(expected_sources))
    assert torch.equal(_bits(prior_windows), _bits(expected_priors))
    assert torch.equal(
        _bits(staging[batch * source_rows :]), _bits(untouched_tail)
    )


@pytest.mark.parametrize("batch,tree_n", ((0, 32), (5, 32), (1, 31)))
def test_fixed32_geometry_fails_loud(batch: int, tree_n: int) -> None:
    with pytest.raises(ValueError, match="requires B=1..4 and tree_n=32"):
        fused_tree_conv_sources_batched(
            prior_bank=torch.zeros((4, 8, 5)),
            prior_cols=torch.tensor((0, 1), dtype=torch.long),
            x=torch.zeros((128, 8)),
            zero_row=torch.zeros((1, 8)),
            staging=torch.zeros((140, 8)),
            batch=batch,
            tree_n=tree_n,
        )


def test_source_contract_drift_fails_loud() -> None:
    common = {
        "prior_bank": torch.zeros((2, 8, 5)),
        "x": torch.zeros((64, 8)),
        "zero_row": torch.zeros((1, 8)),
        "batch": 2,
        "tree_n": 32,
    }
    with pytest.raises(RuntimeError, match="tensor geometry drift"):
        fused_tree_conv_sources_batched(
            **common,
            prior_cols=torch.tensor((0, 1), dtype=torch.int32),
            staging=torch.zeros((70, 8)),
        )
    with pytest.raises(RuntimeError, match="source contract drift"):
        fused_tree_conv_sources_batched(
            **common,
            prior_cols=torch.tensor((0, 1), dtype=torch.long),
            staging=torch.zeros((69, 8)),
        )


def test_candidate_has_constant_source_op_count_and_preserves_legacy_arm() -> None:
    helper = inspect.getsource(fused_tree_conv_sources_batched)
    assert helper.count(".index_select(") == 1
    assert helper.count("torch.cat(") == 1
    assert "out=sources" in helper

    conv = _conv_replacement()
    candidate = conv.index("fused_tree_conv_sources_batched(")
    request_loop = conv.index(
        "for _fr10_b in range(attn_metadata.num_spec_decodes):"
    )
    assert candidate < request_loop
    assert "if not _FR13_FIXED32_CONV_SOURCE_BATCH:" in conv
    assert (
        "_fr10_prior_conv_state_bank[\n"
        "                                _fr10_b\n"
        "                            ].index_select(1, _fr10_prior_cols)"
    ) in conv
    assert (
        "_fr13_wbb_stage[\n"
        "                                    _fr10_b * _fr13_wbb_srows:\n"
        "                                    (_fr10_b + 1) * _fr13_wbb_srows\n"
        "                                ].copy_(_fr10_source)"
    ) in conv


def test_candidate_is_default_off_and_forwarded_end_to_end() -> None:
    patcher = PATCHER.read_text()
    conv = _conv_replacement()
    sequence = SEQUENCE.read_text()
    launcher = LAUNCHER.read_text()
    assert (
        'os.environ.get("FR13_FIXED32_CONV_SOURCE_BATCH", "0") == "1"'
        in patcher
    )
    assert (
        "FR13_FIXED32_CONV_SOURCE_BATCH=${FR13_FIXED32_CONV_SOURCE_BATCH:-0}"
        in sequence
    )
    assert '0|1) export FR13_FIXED32_CONV_SOURCE_BATCH ;;' in sequence
    assert (
        '-e FR13_FIXED32_CONV_SOURCE_BATCH="${FR13_FIXED32_CONV_SOURCE_BATCH:-0}"'
        in launcher
    )
    assert '"FR13_FIXED32_CONV_SOURCE_BATCH requires the "' in conv
    assert '"fixed32 batched-writeback staging route"' in conv

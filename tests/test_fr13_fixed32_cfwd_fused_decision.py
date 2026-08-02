from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import pytest
import torch


KERNEL_PATH = Path("scripts/fr13_cfwd_fused_decision_kernel.py")
KERNEL_SPEC = importlib.util.spec_from_file_location(
    "fr13_cfwd_fused_decision_kernel_test",
    KERNEL_PATH,
)
assert KERNEL_SPEC is not None and KERNEL_SPEC.loader is not None
kernel = importlib.util.module_from_spec(KERNEL_SPEC)
KERNEL_SPEC.loader.exec_module(kernel)

TAW_PATH = Path("scripts/fr13_device_multidraft_kernel.py")
TAW_SPEC = importlib.util.spec_from_file_location(
    "fr13_cfwd_fused_decision_taw_test",
    TAW_PATH,
)
assert TAW_SPEC is not None and TAW_SPEC.loader is not None
taw = importlib.util.module_from_spec(TAW_SPEC)
TAW_SPEC.loader.exec_module(taw)

CENSUS_PATH = Path("scripts/fr13_fixed32_work_census.py")
CENSUS_SPEC = importlib.util.spec_from_file_location(
    "fr13_cfwd_fused_decision_census_test",
    CENSUS_PATH,
)
assert CENSUS_SPEC is not None and CENSUS_SPEC.loader is not None
census = importlib.util.module_from_spec(CENSUS_SPEC)
sys.path.insert(0, str(CENSUS_PATH.parent.resolve()))
sys.modules[CENSUS_SPEC.name] = census
CENSUS_SPEC.loader.exec_module(census)


def _dense_residual(target_probability, kid_tokens, kid_mask):
    normalized = target_probability / target_probability.sum(
        dim=-1, keepdim=True
    )
    overlaps = torch.gather(normalized, -1, kid_tokens) * kid_mask
    overlap_mass = overlaps.sum(dim=-1, keepdim=True)
    q_weights = overlaps / overlap_mass.clamp(min=1.0e-30)
    q_mix = torch.zeros_like(normalized)
    q_mix.scatter_add_(-1, kid_tokens, q_weights * kid_mask)
    residual = (normalized - q_mix).clamp(min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = torch.where(
        residual_mass > 0.0,
        residual / residual_mass.clamp(min=1.0e-30),
        normalized,
    )
    zero_mass = kid_mask.any(dim=-1, keepdim=True) & (overlap_mass <= 0.0)
    residual = torch.where(zero_mass, normalized, residual)
    return normalized, overlaps, q_weights, q_mix, residual


def _inverse_cdf(weights, uniform):
    cumulative = torch.cumsum(weights, dim=-1)
    threshold = uniform.unsqueeze(-1) * cumulative[:, -1:]
    return (cumulative <= threshold).sum(dim=-1).clamp(
        max=weights.shape[-1] - 1
    )


def _blockwise_inverse_cdf(weights, uniform, block_size: int):
    rows, vocab = weights.shape
    block_count = (vocab + block_size - 1) // block_size
    padded = torch.nn.functional.pad(
        weights, (0, block_count * block_size - vocab)
    )
    blocks = padded.reshape(rows, block_count, block_size)
    block_sums = blocks.sum(dim=-1)
    block_cdf = torch.cumsum(block_sums, dim=-1)
    threshold = uniform * block_cdf[:, -1]
    selected_block = (block_cdf <= threshold.unsqueeze(-1)).sum(dim=-1)
    selected_block.clamp_(max=block_count - 1)
    prefix = torch.where(
        torch.arange(block_count).unsqueeze(0) < selected_block.unsqueeze(-1),
        block_sums,
        torch.zeros_like(block_sums),
    ).sum(dim=-1)
    selected_values = blocks[torch.arange(rows), selected_block]
    local_cdf = torch.cumsum(selected_values, dim=-1)
    selected_local = (
        local_cdf <= (threshold - prefix).unsqueeze(-1)
    ).sum(dim=-1)
    valid_count = (vocab - selected_block * block_size).clamp(max=block_size)
    selected_local = torch.minimum(selected_local, valid_count - 1)
    return selected_block * block_size + selected_local


@pytest.mark.parametrize("batch_size", (1, 2, 3, 4))
def test_workspace_is_fixed_for_every_serving_batch(batch_size: int) -> None:
    spec = kernel.workspace_spec(batch_size)
    assert spec["cfwd_fused_probability_block_sums"] == (
        (batch_size * 30, 1024),
        torch.float32,
    )
    assert spec["cfwd_fused_residual_block_sums"] == (
        (batch_size * 17, 1024),
        torch.float32,
    )
    assert spec["cfwd_fused_kid_tokens"] == (
        (batch_size * 17, 3),
        torch.long,
    )
    assert spec["cfwd_fused_self_token"] == (
        (batch_size, 13),
        torch.long,
    )
    assert spec["cfwd_fused_accepted"] == (
        (batch_size, 17),
        torch.bool,
    )


def test_selector_is_strict_and_default_off() -> None:
    assert not taw._fr13_fixed32_cfwd_fused_decisions_enabled(environ={})
    assert not taw._fr13_fixed32_cfwd_fused_decisions_enabled(
        environ={"FR13_FIXED32_CFWD_FUSED_DECISIONS": "0"}
    )
    assert taw._fr13_fixed32_cfwd_fused_decisions_enabled(
        environ={"FR13_FIXED32_CFWD_FUSED_DECISIONS": "1"}
    )
    with pytest.raises(RuntimeError, match="exactly 0 or 1"):
        taw._fr13_fixed32_cfwd_fused_decisions_enabled(
            environ={"FR13_FIXED32_CFWD_FUSED_DECISIONS": "true"}
        )


@pytest.mark.parametrize("vocab_size", (1, 7, 31, 97, 257))
def test_block_then_in_block_scan_matches_full_inverse_cdf(
    vocab_size: int,
) -> None:
    generator = torch.Generator().manual_seed(7000 + vocab_size)
    weights = torch.rand((64, vocab_size), generator=generator, dtype=torch.float64)
    uniforms = torch.linspace(0.0, 0.999, 64, dtype=torch.float64)
    torch.testing.assert_close(
        _blockwise_inverse_cdf(weights, uniforms, block_size=8),
        _inverse_cdf(weights, uniforms),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("batch_size", (1, 4))
@pytest.mark.parametrize("vocab_size", (7, 31, 97))
def test_sparse_duplicate_math_matches_dense_scatter(
    batch_size: int,
    vocab_size: int,
) -> None:
    for seed in range(32):
        generator = torch.Generator().manual_seed(
            1000 * batch_size + 100 * vocab_size + seed
        )
        probability = torch.rand(
            (batch_size * 17, vocab_size),
            generator=generator,
            dtype=torch.float64,
        )
        kid_tokens = torch.randint(
            0,
            vocab_size,
            (batch_size * 17, 3),
            generator=generator,
        )
        if seed % 2 == 0:
            kid_tokens[:, 1] = kid_tokens[:, 0]
        if seed % 5 == 0:
            kid_tokens[:, 2] = kid_tokens[:, 0]
        counts = torch.randint(
            1,
            4,
            (batch_size * 17, 1),
            generator=generator,
        )
        kid_mask = torch.arange(3).unsqueeze(0) < counts

        expected = _dense_residual(probability, kid_tokens, kid_mask)
        actual = kernel.sparse_residual_oracle(
            probability,
            kid_tokens,
            kid_mask,
        )
        for dense, sparse in zip(
            (expected[0], expected[1], expected[2], expected[4]),
            actual,
            strict=True,
        ):
            torch.testing.assert_close(sparse, dense, rtol=0.0, atol=1.0e-15)


def test_distinct_drafts_are_removed_from_the_residual() -> None:
    probability = torch.tensor(
        [[0.02, 0.10, 0.18, 0.30, 0.40]], dtype=torch.float64
    )
    kids = torch.tensor([[1, 2, 4]])
    mask = torch.ones_like(kids, dtype=torch.bool)
    _, _, _, residual = kernel.sparse_residual_oracle(
        probability, kids, mask
    )
    assert residual[0, [1, 2, 4]].tolist() == [0.0, 0.0, 0.0]
    torch.testing.assert_close(
        residual.sum(dim=-1), torch.ones(1, dtype=torch.float64)
    )


def test_duplicate_mass_above_one_preserves_singleton_candidate_residual() -> None:
    probability = torch.tensor([[0.45, 0.35, 0.20]], dtype=torch.float64)
    kids = torch.tensor([[0, 0, 1]])
    mask = torch.ones_like(kids, dtype=torch.bool)
    normalized, overlaps, q_weights, residual = (
        kernel.sparse_residual_oracle(probability, kids, mask)
    )
    overlap_mass = overlaps.sum()
    assert overlap_mass > 1.0
    expected_singleton_unnormalized = normalized[0, 1] * (
        1.0 - 1.0 / overlap_mass
    )
    q_mix_singleton = q_weights[0, 2]
    actual_singleton_unnormalized = (
        normalized[0, 1] - q_mix_singleton
    ).clamp(min=0.0)
    torch.testing.assert_close(
        actual_singleton_unnormalized,
        expected_singleton_unnormalized,
    )
    assert residual[0, 1] > 0.0
    assert residual[0, 0] == 0.0


def test_zero_residual_falls_back_to_target_distribution() -> None:
    probability = torch.tensor([[0.25, 0.75, 0.0]], dtype=torch.float64)
    kids = torch.tensor([[0, 1, 2]])
    mask = torch.tensor([[True, True, False]])
    normalized, _, _, residual = kernel.sparse_residual_oracle(
        probability, kids, mask
    )
    torch.testing.assert_close(residual, normalized)


def test_acceptance_probability_keeps_duplicate_multiplicity() -> None:
    generator = torch.Generator().manual_seed(9841)
    for _ in range(128):
        probability = torch.rand((1, 19), generator=generator)
        normalized = probability / probability.sum(dim=-1, keepdim=True)
        kids = torch.randint(0, 19, (1, 3), generator=generator)
        kids[0, 1] = kids[0, 0]
        overlaps = torch.gather(normalized, -1, kids)
        overlap_mass = overlaps.sum(dim=-1)
        q_weights = overlaps / overlap_mass.unsqueeze(-1)
        source = int(torch.randint(0, 3, (), generator=generator))
        selected = kids[:, source]
        multiplicity = (kids == selected.unsqueeze(-1)).sum(dim=-1)
        q_mix_selected = (
            q_weights * (kids == selected.unsqueeze(-1))
        ).sum(dim=-1)
        direct = (
            torch.gather(normalized, -1, selected.unsqueeze(-1)).squeeze(-1)
            / q_mix_selected
        ).clamp(max=1.0)
        derived = (overlap_mass / multiplicity).clamp(max=1.0)
        torch.testing.assert_close(direct, derived)


def test_source_contains_four_stage_sparse_pipeline() -> None:
    source = KERNEL_PATH.read_text(encoding="ascii")
    for name in (
        "_fr13_cfwd_probability_block_sums_kernel",
        "_fr13_cfwd_parent_setup_kernel",
        "_fr13_cfwd_residual_block_sums_kernel",
        "_fr13_cfwd_inverse_cdf_kernel",
    ):
        assert source.count(f"def {name}(") == 1
        assert f"{name}[" in inspect.getsource(kernel.launch)
    launch_source = inspect.getsource(kernel.launch)
    assert "torch.zeros_like" not in launch_source
    assert "scatter_add_" not in launch_source
    assert "torch.cumsum" not in launch_source
    assert kernel.MAX_VOCAB == 262144
    assert kernel.MAX_BLOCKS == 1024


def test_work_census_and_runtime_route_are_source_bound() -> None:
    assert census.TAW_SOURCE_CONTRACT_SCHEMA == taw._FR13_FIXED32_TAW_SOURCE_SCHEMA
    assert census.TAW_SOURCE_CONTRACT_SHA256 == taw._FR13_FIXED32_TAW_SOURCE_SHA256
    assert census.TAW_CFWD_FUSED_DIAGNOSTIC_TENSOR_CALL_CENSUS == (
        taw._FR13_FIXED32_CFWD_FUSED_DIAGNOSTIC_TENSOR_CALL_CENSUS
    )
    assert census.TAW_CFWD_FUSED_PRODUCTION_TENSOR_CALL_CENSUS == (
        taw._FR13_FIXED32_CFWD_FUSED_PRODUCTION_TENSOR_CALL_CENSUS
    )
    launcher = Path("scripts/fr13_launch_forked_fa2_tree_server.sh").read_text(
        encoding="utf-8"
    )
    runner = Path("scripts/fr13_run_b4_tail23_all_parent_live_gate.sh").read_text(
        encoding="utf-8"
    )
    manifest = Path("scripts/fr13_runtime_manifest.py").read_text(
        encoding="utf-8"
    )
    assert (
        "FR13_FIXED32_CFWD_FUSED_DECISIONS=${"
        "FR13_FIXED32_CFWD_FUSED_DECISIONS:-0}"
    ) in launcher
    assert '-e FR13_FIXED32_CFWD_FUSED_DECISIONS="$' in launcher
    assert "FR13_FIXED32_CFWD_FUSED_DECISIONS=1" in runner
    assert kernel.CANDIDATE in runner
    assert KERNEL_PATH.as_posix() in manifest


def test_main_loader_binds_complete_candidate_source() -> None:
    loaded = taw._fr13_fixed32_cfwd_fused_decision_module()
    assert loaded.CANDIDATE == kernel.CANDIDATE
    assert loaded.SOURCE_SCHEMA == kernel.SOURCE_SCHEMA
    assert loaded.MAX_VOCAB == kernel.MAX_VOCAB


def test_preseed_allocates_candidate_workspace_only_when_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = taw._fr13_fixed32_topology()
    mode = "hydra27_fixed32"
    valid_mask = int(topology.VALID_MASK_BY_MODE[mode])
    monkeypatch.setenv("FR13_FIXED32_CFWD_FUSED_DECISIONS", "1")
    taw.fr13_fixed32_taw_preseed(
        torch.device("cpu"),
        mode=mode,
        valid_mask=valid_mask,
    )
    for batch_size in (1, 4):
        key = taw.fr13_fixed32_taw_cache_key(
            mode,
            valid_mask,
            batch_size,
            torch.device("cpu"),
        )
        candidate_entry = taw._FR13_FIXED32_TAW_CACHE[key]["native_ab_entry"]
        for name, (shape, dtype) in kernel.workspace_spec(batch_size).items():
            value = candidate_entry[name]
            assert tuple(value.shape) == shape
            assert value.dtype == dtype
            assert value.is_contiguous()

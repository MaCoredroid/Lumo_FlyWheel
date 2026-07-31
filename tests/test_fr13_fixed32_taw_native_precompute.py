from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path("scripts/fr13_device_multidraft_kernel.py")
SPEC = importlib.util.spec_from_file_location(
    "fr13_fixed32_taw_native_precompute",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
taw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(taw)

CENSUS_PATH = Path("scripts/fr13_fixed32_work_census.py")
sys.path.insert(0, str(CENSUS_PATH.parent.resolve()))
CENSUS_SPEC = importlib.util.spec_from_file_location(
    "fr13_fixed32_taw_native_precompute_census",
    CENSUS_PATH,
)
assert CENSUS_SPEC is not None and CENSUS_SPEC.loader is not None
census_validator = importlib.util.module_from_spec(CENSUS_SPEC)
sys.modules[CENSUS_SPEC.name] = census_validator
CENSUS_SPEC.loader.exec_module(census_validator)


def _set_fixed_env(
    monkeypatch: pytest.MonkeyPatch,
    topology,
    mode: str,
) -> None:
    monkeypatch.setenv("FR13_FIXED32_MODE", mode)
    monkeypatch.setenv(
        "FR13_FIXED32_VALID_MASK",
        hex(int(topology.VALID_MASK_BY_MODE[mode])),
    )
    active = taw._fr13_fixed32_expected_active(topology, mode)
    monkeypatch.setenv("FR13_FIXED32_ACTIVE_NODES", str(active))
    monkeypatch.setenv("FR13_FIXED32_TAW_WALK_CAP", str(topology.WALK_CAP))
    monkeypatch.setenv("FR13_TAW", "1")
    monkeypatch.setenv("FR13_FIXED32_WORK_CENSUS", "1")


def _fixture(topology, mode: str, batch_size: int, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fixture = taw._fr13_fixed32_test_fixture(
        topology,
        mode,
        batch_size,
        vocab_size=97,
    )
    fixture["target"].copy_(
        torch.randn(fixture["target"].shape, generator=generator)
    )
    fixture["self"].copy_(
        torch.randn(fixture["self"].shape, generator=generator)
    )
    fixture["uniforms"].copy_(
        torch.rand(fixture["uniforms"].shape, generator=generator)
    )
    return fixture


@pytest.mark.parametrize("mode", ("tail6_fixed32", "hydra27_fixed32"))
@pytest.mark.parametrize("batch_size", (1, 4))
def test_native_precompute_is_full_byte_equivalent_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    batch_size: int,
) -> None:
    topology = taw._fr13_fixed32_topology()
    _set_fixed_env(monkeypatch, topology, mode)
    monkeypatch.delenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE", raising=False)
    valid_mask = int(topology.VALID_MASK_BY_MODE[mode])
    taw.fr13_fixed32_taw_preseed(
        torch.device("cpu"),
        mode=mode,
        valid_mask=valid_mask,
    )
    callback_rows = []
    taw.fr13_fixed32_taw_set_work_callback(callback_rows.append)

    for seed in range(4):
        fixture = _fixture(topology, mode, batch_size, seed)
        baseline = tuple(
            tensor.clone()
            for tensor in taw._fr13_fixed32_test_call(topology, mode, fixture)
        )
        monkeypatch.setenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE", "1")
        candidate = tuple(
            tensor.clone()
            for tensor in taw._fr13_fixed32_test_call(topology, mode, fixture)
        )
        for expected, actual in zip(baseline, candidate, strict=True):
            assert expected.numpy().tobytes() == actual.numpy().tobytes()
        key = taw.fr13_fixed32_taw_cache_key(
            mode,
            valid_mask,
            batch_size,
            torch.device("cpu"),
        )
        entry = taw._FR13_FIXED32_TAW_CACHE[key]
        assert entry["native_ab_probability_mismatches"].item() == 0
        assert entry["native_ab_product_mismatches"].item() == 0
        monkeypatch.setenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE", "0")

    baseline_work, candidate_work = callback_rows[-2:]
    assert baseline_work["taw"]["route"] == (
        "fixed32_pytorch_exact_float_triton_integer_commit"
    )
    assert candidate_work["taw"]["route"] == (
        "fixed32_native_precompute_byte_ab_reference_return"
    )


def test_native_precompute_uses_one_fixed_row_union_for_both_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = taw._fr13_fixed32_topology()
    sources = {}
    for mode in ("tail6_fixed32", "hydra27_fixed32"):
        _set_fixed_env(monkeypatch, topology, mode)
        valid_mask = int(topology.VALID_MASK_BY_MODE[mode])
        taw.fr13_fixed32_taw_preseed(
            torch.device("cpu"),
            mode=mode,
            valid_mask=valid_mask,
        )
        key = taw.fr13_fixed32_taw_cache_key(
            mode,
            valid_mask,
            1,
            torch.device("cpu"),
        )
        entry = taw._FR13_FIXED32_TAW_CACHE[key]
        assert entry["native_self_rows_per_request"] == 13
        assert entry["native_target_rows_per_request"] == 17
        sources[mode] = (
            entry["native_self_source_indices"].tolist(),
            entry["native_target_source_indices"].tolist(),
        )
    assert sources["tail6_fixed32"] == sources["hydra27_fixed32"]


def test_native_precompute_work_census_is_pinned() -> None:
    event = census_validator.reference_event(
        "tail6_fixed32",
        1,
        "native-precompute-contract",
    )
    event["taw"]["route"] = census_validator.TAW_NATIVE_PRECOMPUTE_ROUTE
    event["taw"]["tensor_call_census"] = dict(
        census_validator.TAW_NATIVE_PRECOMPUTE_TENSOR_CALL_CENSUS
    )
    for name in (
        "child_lanes",
        "target_rows",
        "self_rows",
        "self_cdf_rows",
        "source_cdf_rows",
        "residual_cdf_rows",
        "qmix_rows",
        "residual_rows",
        "row_scatter_slots",
        "path_scatter_slots",
        "exact_commit_launches",
        "exact_commit_programs",
    ):
        event["taw"][name] *= 2
    validated = census_validator.validate_event(
        event,
        source="native-precompute-contract",
    )
    assert validated.normalized_work["taw"]["tensor_call_census"][
        "full_vocab_softmax_calls"
    ] == 26


def test_native_precompute_flag_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE", "yes")
    with pytest.raises(RuntimeError, match="must be unset, 0, or 1"):
        taw._fr13_fixed32_taw_native_precompute_enabled()

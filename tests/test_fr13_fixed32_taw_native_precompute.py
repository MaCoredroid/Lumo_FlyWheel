from __future__ import annotations

import importlib.util
import json
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


def _write_live_pass(path: Path, *, mode: str, batch_size: int) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "fr13.fixed32.taw_native_precompute.live_pass.v1",
                "status": "pass",
                "candidate": taw._FR13_FIXED32_TAW_NATIVE_CANDIDATE,
                "source_contract_schema": taw._FR13_FIXED32_TAW_SOURCE_SCHEMA,
                "source_contract_sha256": taw._FR13_FIXED32_TAW_SOURCE_SHA256,
                "task_marker": "swe_verified:django__django-12345",
                "mode": mode,
                "batch_size": batch_size,
                "covered_batches": list(range(1, batch_size + 1)),
                "geometry": taw._FR13_FIXED32_TAW_GEOMETRY,
                "probability_mismatches": 0,
                "product_mismatches": 0,
                "evidence_route": "full_graph_replay",
                "reference_returned": True,
                "candidate_returned": False,
            }
        ),
        encoding="ascii",
    )


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

    event = census_validator.reference_event(
        "tail6_fixed32",
        1,
        "native-precompute-production-contract",
    )
    event["taw"]["route"] = (
        census_validator.TAW_NATIVE_PRECOMPUTE_PRODUCTION_ROUTE
    )
    event["taw"]["tensor_call_census"] = dict(
        census_validator.TAW_NATIVE_PRECOMPUTE_PRODUCTION_TENSOR_CALL_CENSUS
    )
    validated = census_validator.validate_event(
        event,
        source="native-precompute-production-contract",
    )
    assert validated.normalized_work["taw"]["tensor_call_census"][
        "full_vocab_softmax_calls"
    ] == 2


def test_native_precompute_flag_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE", "yes")
    with pytest.raises(RuntimeError, match="must be unset, 0, or 1"):
        taw._fr13_fixed32_taw_native_precompute_enabled()


def test_native_production_is_default_off_source_bound_and_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = taw._fr13_fixed32_topology()
    _set_fixed_env(monkeypatch, topology, "tail6_fixed32")
    diagnostic = tmp_path / "diagnostic.arm"
    production = tmp_path / "production.arm"
    live_pass = tmp_path / "pass.json"
    monkeypatch.delenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE", raising=False)
    monkeypatch.delenv(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION", raising=False
    )
    monkeypatch.setattr(
        taw,
        "_FR13_FIXED32_TAW_NATIVE_DIAGNOSTIC_SIDECARS",
        (str(diagnostic),),
    )
    monkeypatch.setattr(
        taw,
        "_FR13_FIXED32_TAW_NATIVE_PRODUCTION_SIDECARS",
        (str(production),),
    )
    monkeypatch.setenv(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_PATH", str(live_pass)
    )

    assert taw._fr13_fixed32_taw_native_selector() == "reference"
    production.write_text("1\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="requires a regular live PASS"):
        taw._fr13_fixed32_taw_native_selector()

    _write_live_pass(live_pass, mode="tail6_fixed32", batch_size=1)
    assert taw._fr13_fixed32_taw_native_selector() == "production"

    diagnostic.write_text("1\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        taw._fr13_fixed32_taw_native_selector()

    diagnostic.unlink()
    payload = json.loads(live_pass.read_text(encoding="ascii"))
    payload["source_contract_sha256"] = "0" * 64
    live_pass.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(RuntimeError, match="different candidate/source"):
        taw._fr13_fixed32_taw_native_selector()


def test_native_production_returns_candidate_products(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = taw._fr13_fixed32_topology()
    mode = "tail6_fixed32"
    batch_size = 1
    _set_fixed_env(monkeypatch, topology, mode)
    monkeypatch.delenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE", raising=False)
    monkeypatch.delenv(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION", raising=False
    )
    monkeypatch.setattr(
        taw, "_FR13_FIXED32_TAW_NATIVE_DIAGNOSTIC_SIDECARS", ()
    )
    monkeypatch.setattr(
        taw, "_FR13_FIXED32_TAW_NATIVE_PRODUCTION_SIDECARS", ()
    )
    valid_mask = int(topology.VALID_MASK_BY_MODE[mode])
    taw.fr13_fixed32_taw_preseed(
        torch.device("cpu"), mode=mode, valid_mask=valid_mask
    )
    fixture = _fixture(topology, mode, batch_size, 991)
    reference = tuple(
        tensor.clone()
        for tensor in taw._fr13_fixed32_test_call(topology, mode, fixture)
    )

    live_pass = tmp_path / "pass.json"
    _write_live_pass(live_pass, mode=mode, batch_size=batch_size)
    monkeypatch.setenv(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION", "1"
    )
    monkeypatch.setenv(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_PATH", str(live_pass)
    )
    candidate = tuple(
        tensor.clone()
        for tensor in taw._fr13_fixed32_test_call(topology, mode, fixture)
    )
    assert all(
        expected.numpy().tobytes() == actual.numpy().tobytes()
        for expected, actual in zip(reference, candidate, strict=True)
    )
    work = taw.fr13_fixed32_taw_last_work()
    assert work is not None
    assert work["taw"]["route"] == (
        "fixed32_native_precompute_production_candidate_return"
    )
    assert work["taw"]["tensor_call_census"]["full_vocab_softmax_calls"] == 2


def test_native_production_b4_pass_warms_exact_b1_through_b4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = taw._fr13_fixed32_topology()
    mode = "hydra27_fixed32"
    valid_mask = int(topology.VALID_MASK_BY_MODE[mode])
    _set_fixed_env(monkeypatch, topology, mode)
    monkeypatch.setattr(
        taw, "_FR13_FIXED32_TAW_NATIVE_DIAGNOSTIC_SIDECARS", ()
    )
    monkeypatch.setattr(
        taw, "_FR13_FIXED32_TAW_NATIVE_PRODUCTION_SIDECARS", ()
    )
    live_pass = tmp_path / "pass.json"
    _write_live_pass(live_pass, mode=mode, batch_size=4)
    monkeypatch.setenv(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION", "1"
    )
    monkeypatch.setenv(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_PATH", str(live_pass)
    )
    taw._FR13_FIXED32_TAW_WARMUPS.clear()
    taw.fr13_fixed32_taw_preseed(
        torch.device("cpu"), mode=mode, valid_mask=valid_mask
    )

    evidence = taw.fr13_fixed32_taw_warm_execute(
        torch.device("cpu"),
        mode=mode,
        valid_mask=valid_mask,
        max_batch_size=4,
        vocab_size=101,
    )

    assert evidence["ready"] is True
    assert evidence["batches"] == (1, 2, 3, 4)
    assert evidence["executions"] == 4
    assert evidence["staging_state_restored"] is True

    _write_live_pass(live_pass, mode=mode, batch_size=1)
    with pytest.raises(RuntimeError, match="batch is not covered by PASS"):
        taw._fr13_fixed32_taw_native_production_pass(
            path=str(live_pass), expected_mode=mode, expected_batch=4
        )


def test_native_live_pass_emitter_binds_source_and_real_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "live.json"
    monkeypatch.setenv(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_LIVE_JSON", str(path)
    )
    taw._fr13_fixed32_taw_native_live_pass_emit(
        mode="hydra27_fixed32",
        batch_size=4,
        task_marker="swe_verified:astropy__astropy-12907",
        evidence_route="full_graph_replay",
    )
    payload = json.loads(path.read_text(encoding="ascii"))
    assert payload["source_contract_sha256"] == (
        taw._FR13_FIXED32_TAW_SOURCE_SHA256
    )
    assert payload["candidate"] == taw._FR13_FIXED32_TAW_NATIVE_CANDIDATE
    assert payload["batch_size"] == 4
    assert payload["covered_batches"] == [1, 2, 3, 4]
    assert payload["reference_returned"] is True
    assert payload["candidate_returned"] is False


def test_native_live_gate_binds_and_checks_one_graph_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mode = "tail6_fixed32"
    topology = taw._fr13_fixed32_topology()
    _set_fixed_env(monkeypatch, topology, mode)
    valid_mask = int(topology.VALID_MASK_BY_MODE[mode])
    entry = {
        "native_ab_probability_mismatches": torch.ones((), dtype=torch.int64),
        "native_ab_product_mismatches": torch.ones((), dtype=torch.int64),
        "native_ab_live_marker": None,
        "native_ab_live_gate_pending": False,
        "native_ab_live_pass_emitted": False,
    }
    monkeypatch.setitem(
        taw._FR13_FIXED32_TAW_CACHE,
        (mode, valid_mask, 4, ("cuda", 0)),
        entry,
    )
    monkeypatch.setattr(
        taw, "_fr13_fixed32_taw_native_selector", lambda: "diagnostic"
    )
    monkeypatch.setattr(
        taw,
        "_fr13_fixed32_taw_native_real_event_marker",
        lambda: "swe_verified:django__django-12345",
    )
    emitted = []
    monkeypatch.setattr(
        taw,
        "_fr13_fixed32_taw_native_live_pass_emit",
        lambda **payload: emitted.append(payload),
    )

    armed = taw.fr13_fixed32_taw_native_live_gate_begin(
        mode=mode, batch_size=4
    )
    assert armed == {"status": "armed", "batch_size": 4}
    assert entry["native_ab_probability_mismatches"].item() == 0
    assert entry["native_ab_product_mismatches"].item() == 0

    passed = taw.fr13_fixed32_taw_native_live_gate_on_replay(
        mode=mode, batch_size=4
    )
    assert passed == {"status": "passed", "batch_size": 4}
    assert emitted == [
        {
            "mode": mode,
            "batch_size": 4,
            "task_marker": "swe_verified:django__django-12345",
            "evidence_route": "full_graph_replay",
        }
    ]


def test_launcher_keeps_taw_native_production_default_off_and_source_gated() -> None:
    launcher = Path(
        "scripts/fr13_launch_forked_fa2_tree_server.sh"
    ).read_text(encoding="utf-8")
    patcher = Path(
        "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
    ).read_text(encoding="utf-8")
    assert (
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=${"
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION:-0}"
    ) in launcher
    assert "TAW native diagnostic and production are mutually exclusive" in launcher
    assert "TAW native production requires fixed32 and a regular live PASS JSON" in launcher
    assert "fr13_fixed32_taw_native_precompute_production.arm" in launcher
    assert "fr13_fixed32_taw_native_precompute.production_pass.json" in launcher
    assert "fr13_fixed32_taw_native_live_gate_begin(" in patcher
    assert "fr13_fixed32_taw_native_live_gate_on_replay(" in patcher

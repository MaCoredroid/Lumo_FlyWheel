"""Shape-pinned all-parent candidate and the widened TAW byte gate.

Evidence these tests encode (device probe, results/fr13_taw_sampler_lever_
shape_pin_20260815/):

  * ``torch.softmax`` is bit-identical at every row count, so batching the
    softmax rows is free.
  * ``torch.cumsum`` is bit-identical for rows >= 2 and non-reproducible at
    rows == 1, so the batched walk can only exist at B >= 2.
  * ``sum(dim=-1)`` drifts up to 2 ULP across row counts.  Computing it as
    ``cat([X[i:i+W].sum(-1) for i in range(0, N, W)])`` with W equal to the
    served batch reproduces the reference bitwise; the fused reduction does
    not.  A 2-ULP drift there flips ~0.16% of inverse-CDF indices, which the
    pre-widening gate could not see because it only compared the
    PRE-normalization softmax rows.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path("scripts/fr13_device_multidraft_kernel.py")
SPEC = importlib.util.spec_from_file_location(
    "fr13_fixed32_taw_widegate_shape_pin",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
taw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(taw)

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(
    not CUDA,
    reason="bitwise reduction-shape claims are device behaviour; needs CUDA",
)


# ---------------------------------------------------------------------------
# shared fixture plumbing
# ---------------------------------------------------------------------------
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
    monkeypatch.setenv(
        "FR13_FIXED32_ACTIVE_NODES",
        str(taw._fr13_fixed32_expected_active(topology, mode)),
    )
    monkeypatch.setenv("FR13_FIXED32_TAW_WALK_CAP", str(topology.WALK_CAP))
    monkeypatch.setenv("FR13_TAW", "1")
    monkeypatch.delenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE", raising=False)
    monkeypatch.delenv(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION", raising=False
    )


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


def _diagnostic_route(
    topology,
    mode: str,
    batch_size: int,
    fixture,
    *,
    normalization_ulp_drift: int = 0,
    decision_flip: bool = False,
):
    """Replay the diagnostic A/B route with an injectable candidate defect.

    Mirrors ``fr13_fixed32_taw_commit``'s diagnostic branch exactly: build the
    row-union softmax caches, normalize them at the served width, decide every
    fixed parent once, run the reference walk with the byte gate armed, then
    walk the candidate from those same decisions.
    """
    valid_mask = int(topology.VALID_MASK_BY_MODE[mode])
    entry = taw._FR13_FIXED32_TAW_CACHE[
        taw.fr13_fixed32_taw_cache_key(
            mode, valid_mask, batch_size, torch.device("cpu")
        )
    ]
    drafts, bonus_flat = taw._fr13_fixed32_validate_inputs(
        topology,
        entry,
        fixture["counts"],
        fixture["drafts"],
        fixture["parents"],
        fixture["target"],
        fixture["self"],
        fixture["bonus"],
        int(topology.PHYSICAL_DRAFTS),
    )
    uniforms, _route = taw._fr13_fixed32_fill_uniforms(
        entry,
        uniforms=fixture["uniforms"],
    )
    probability_caches = taw._fr13_fixed32_taw_probability_caches(
        entry,
        fixture["target"],
        fixture["self"],
        native_precompute=True,
    )
    self_cache, target_cache = probability_caches
    if normalization_ulp_drift:
        # Exactly the failure the probe found: the normalization SUM lands a
        # couple of ULP away when it is not launched at the served width.
        # Nothing about the pre-normalization softmax rows changes.
        def _drift(cache):
            total = taw._fr13_fixed32_taw_pinned_row_sum(
                cache, width=batch_size
            )
            drifted = (
                total.view(torch.int32) + normalization_ulp_drift
            ).view(torch.float32)
            return cache / drifted.unsqueeze(1)

        normalized = (_drift(self_cache), _drift(target_cache))
    else:
        normalized = taw._fr13_fixed32_taw_pinned_normalized_caches(
            probability_caches,
            batch_size=batch_size,
        )

    gate = {"self_raw": self_cache, "target_raw": target_cache}
    decisions = taw._fr13_fixed32_taw_all_parent_decisions(
        topology,
        entry["native_ab_entry"],
        drafts,
        uniforms,
        probability_caches,
        normalized_probability_caches=normalized,
        gate_capture=gate,
    )
    if decision_flip:
        flipped = gate["accepted"].clone()
        flipped[0, 0] = ~flipped[0, 0]
        gate["accepted"] = flipped

    probability_mismatches = torch.zeros((), dtype=torch.int64)
    accept_decision_mismatches = torch.zeros((), dtype=torch.int64)
    reference = taw._fr13_fixed32_taw_execute_torch(
        topology,
        entry,
        drafts,
        fixture["target"],
        fixture["self"],
        bonus_flat,
        uniforms,
        walk_cap=int(topology.WALK_CAP),
        native_precompute=False,
        comparison_probability_caches=probability_caches,
        comparison_gate=gate,
        probability_mismatches=probability_mismatches,
        accept_decision_mismatches=accept_decision_mismatches,
    )
    reference_products = tuple(tensor.clone() for tensor in reference[:5])
    candidate = taw._fr13_fixed32_taw_execute_all_parent(
        topology,
        entry["native_ab_entry"],
        drafts,
        bonus_flat,
        uniforms,
        walk_cap=int(topology.WALK_CAP),
        probability_caches=probability_caches,
        decisions=decisions,
    )
    product_mismatches = sum(
        int(torch.count_nonzero(expected != actual))
        for expected, actual in zip(reference_products, candidate[:5])
    )
    return {
        "probability_mismatches": int(probability_mismatches.item()),
        "accept_decision_mismatches": int(accept_decision_mismatches.item()),
        "product_mismatches": product_mismatches,
    }


def _narrow_gate_level(
    entry,
    gate,
    *,
    request_rows,
    self_slots,
    target_slots,
    leaf,
    has_kids,
    reference_self,
    reference_target,
    probability_mismatches,
    accept_decision_mismatches,
) -> None:
    """The pre-widening gate: PRE-normalization int-view rows and nothing else."""
    self_indices = entry["native_self_request_offsets"] + self_slots
    target_indices = entry["native_target_request_offsets"] + target_slots
    probability_mismatches.add_(
        torch.count_nonzero(
            (
                reference_self["pre_normalization"].view(torch.int32)
                != gate["self_raw"][self_indices].view(torch.int32)
            )
            & leaf.unsqueeze(1)
        )
    )
    probability_mismatches.add_(
        torch.count_nonzero(
            (
                reference_target["pre_normalization"].view(torch.int32)
                != gate["target_raw"][target_indices].view(torch.int32)
            )
            & has_kids.unsqueeze(1)
        )
    )


@pytest.fixture
def preseeded(monkeypatch: pytest.MonkeyPatch):
    topology = taw._fr13_fixed32_topology()
    mode = "hydra27_fixed32"
    _set_fixed_env(monkeypatch, topology, mode)
    taw.fr13_fixed32_taw_preseed(
        torch.device("cpu"),
        mode=mode,
        valid_mask=int(topology.VALID_MASK_BY_MODE[mode]),
    )
    taw.fr13_fixed32_taw_set_work_callback(lambda _payload: None)
    yield topology, mode
    taw.fr13_fixed32_taw_set_work_callback(None)


# ---------------------------------------------------------------------------
# 1. the shape pin itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width", (2, 3, 4))
def test_pinned_row_sum_matches_the_served_width_reference(width: int) -> None:
    rows = torch.softmax(
        torch.randn(
            width * 17,
            257,
            generator=torch.Generator(device="cpu").manual_seed(20260815),
        ),
        dim=-1,
    )
    reference = torch.empty(rows.shape[0], dtype=torch.float32)
    for start in range(0, rows.shape[0], width):
        reference[start:start + width] = rows[start:start + width].sum(dim=-1)

    pinned = taw._fr13_fixed32_taw_pinned_row_sum(rows, width=width)

    assert tuple(pinned.shape) == (rows.shape[0],)
    assert pinned.numpy().tobytes() == reference.numpy().tobytes()


@requires_cuda
@pytest.mark.parametrize("width", (2, 3, 4))
def test_pinned_row_sum_is_bitwise_where_the_fused_reduction_is_not(
    width: int,
) -> None:
    rows = torch.softmax(
        torch.randn(
            width * 17,
            8192,
            generator=torch.Generator(device="cpu").manual_seed(20260815),
        ),
        dim=-1,
    ).cuda()
    reference = torch.empty(rows.shape[0], dtype=torch.float32, device="cuda")
    for start in range(0, rows.shape[0], width):
        reference[start:start + width] = rows[start:start + width].sum(dim=-1)

    pinned = taw._fr13_fixed32_taw_pinned_row_sum(rows, width=width)

    assert torch.equal(pinned.view(torch.int32), reference.view(torch.int32))


def test_pinned_row_sum_refuses_batch_one_and_ragged_widths() -> None:
    rows = torch.rand(6, 8)
    with pytest.raises(RuntimeError, match="served batch width"):
        taw._fr13_fixed32_taw_pinned_row_sum(rows, width=1)
    with pytest.raises(RuntimeError, match="served batch width"):
        taw._fr13_fixed32_taw_pinned_row_sum(rows, width=4)
    with pytest.raises(RuntimeError, match=r"\[N, V\] tensor"):
        taw._fr13_fixed32_taw_pinned_row_sum(torch.rand(6), width=2)


@pytest.mark.parametrize("batch_size", (2, 3, 4))
def test_pinned_normalization_reproduces_the_reference_walk_bitwise(
    preseeded,
    batch_size: int,
) -> None:
    topology, mode = preseeded
    fixture = _fixture(topology, mode, batch_size, 20260815)
    verdict = _diagnostic_route(topology, mode, batch_size, fixture)
    assert verdict == {
        "probability_mismatches": 0,
        "accept_decision_mismatches": 0,
        "product_mismatches": 0,
    }


# ---------------------------------------------------------------------------
# 2. B=1 refusal
# ---------------------------------------------------------------------------
def test_all_parent_candidate_refuses_batch_one(preseeded) -> None:
    topology, mode = preseeded
    valid_mask = int(topology.VALID_MASK_BY_MODE[mode])
    entry = taw._FR13_FIXED32_TAW_CACHE[
        taw.fr13_fixed32_taw_cache_key(
            mode, valid_mask, 1, torch.device("cpu")
        )
    ]
    fixture = _fixture(topology, mode, 1, 7)
    drafts, bonus_flat = taw._fr13_fixed32_validate_inputs(
        topology,
        entry,
        fixture["counts"],
        fixture["drafts"],
        fixture["parents"],
        fixture["target"],
        fixture["self"],
        fixture["bonus"],
        int(topology.PHYSICAL_DRAFTS),
    )
    uniforms, _route = taw._fr13_fixed32_fill_uniforms(
        entry, uniforms=fixture["uniforms"]
    )
    caches = taw._fr13_fixed32_taw_probability_caches(
        entry, fixture["target"], fixture["self"], native_precompute=True
    )
    with pytest.raises(RuntimeError, match="refuses batch 1"):
        taw._fr13_fixed32_taw_all_parent_decisions(
            topology,
            entry["native_ab_entry"],
            drafts,
            uniforms,
            caches,
        )
    with pytest.raises(RuntimeError, match="refuses batch 1"):
        taw._fr13_fixed32_taw_execute_all_parent(
            topology,
            entry["native_ab_entry"],
            drafts,
            bonus_flat,
            uniforms,
            walk_cap=int(topology.WALK_CAP),
            probability_caches=caches,
        )


def test_batch_one_serves_the_reference_walk_in_both_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = taw._fr13_fixed32_topology()
    mode = "tail6_fixed32"
    _set_fixed_env(monkeypatch, topology, mode)
    diagnostic = tmp_path / "diagnostic.arm"
    monkeypatch.setattr(
        taw,
        "_FR13_FIXED32_TAW_NATIVE_DIAGNOSTIC_SIDECARS",
        (str(diagnostic),),
    )
    monkeypatch.setattr(
        taw, "_FR13_FIXED32_TAW_NATIVE_PRODUCTION_SIDECARS", ()
    )
    diagnostic.write_text("1\n", encoding="ascii")

    assert taw._fr13_fixed32_taw_native_selector() == "diagnostic"
    assert taw._fr13_fixed32_taw_native_selector(batch_size=1) == "reference"
    for batch_size in (2, 3, 4):
        assert (
            taw._fr13_fixed32_taw_native_selector(batch_size=batch_size)
            == "diagnostic"
        )


def test_live_gate_and_pass_emitter_refuse_batch_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = taw._fr13_fixed32_topology()
    mode = "tail6_fixed32"
    _set_fixed_env(monkeypatch, topology, mode)
    monkeypatch.setattr(
        taw, "_fr13_fixed32_taw_native_selector", lambda **_kwargs: "diagnostic"
    )
    with pytest.raises(RuntimeError, match="refuses to serve below B=2"):
        taw.fr13_fixed32_taw_native_live_gate_begin(mode=mode, batch_size=1)
    with pytest.raises(RuntimeError, match="refuses to serve below B=2"):
        taw._fr13_fixed32_taw_native_live_pass_emit(
            mode=mode,
            batch_size=1,
            task_marker="swe_verified:campaign4_" + "a" * 64,
            evidence_route="full_graph_replay",
        )


def test_required_production_batches_cover_only_served_batches() -> None:
    required = taw._FR13_FIXED32_TAW_REQUIRED_PRODUCTION_BATCHES
    assert required == (4,)
    assert taw._FR13_FIXED32_TAW_PINNED_MIN_BATCH == 2
    assert all(
        batch >= taw._FR13_FIXED32_TAW_PINNED_MIN_BATCH for batch in required
    )
    assert set(required).issubset(taw._FR13_FIXED32_BATCHES)


# ---------------------------------------------------------------------------
# 3. THE CENTREPIECE: the same 2-ULP normalization drift, both gate widths
# ---------------------------------------------------------------------------
def test_two_ulp_normalization_drift_passes_the_old_narrow_gate(
    preseeded,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-widening gate is blind to a drifted normalization sum.

    It only int-view compared PRE-normalization softmax rows, which the drift
    leaves untouched, and the five committed products, which a 2-ULP drift
    does not move on this fixture.
    """
    topology, mode = preseeded
    monkeypatch.setattr(taw, "_fr13_fixed32_taw_gate_level", _narrow_gate_level)

    verdict = _diagnostic_route(
        topology,
        mode,
        4,
        _fixture(topology, mode, 4, 20260815),
        normalization_ulp_drift=2,
    )

    assert verdict == {
        "probability_mismatches": 0,
        "accept_decision_mismatches": 0,
        "product_mismatches": 0,
    }


def test_two_ulp_normalization_drift_fails_the_widened_gate(
    preseeded,
) -> None:
    """The widened gate catches the identical drift the narrow gate passed."""
    topology, mode = preseeded

    verdict = _diagnostic_route(
        topology,
        mode,
        4,
        _fixture(topology, mode, 4, 20260815),
        normalization_ulp_drift=2,
    )

    # The committed products are still byte-identical and no accept decision
    # moved -- exactly the state the old gate called a PASS.  The widened gate
    # refuses the drift before it becomes a token flip.
    assert verdict["probability_mismatches"] > 0
    assert verdict["product_mismatches"] == 0
    assert verdict["accept_decision_mismatches"] == 0


# ---------------------------------------------------------------------------
# 4. the rest of the widened surface, and the verdict wiring
# ---------------------------------------------------------------------------
def test_widened_gate_fails_loudly_on_a_flipped_accept_decision(
    preseeded,
) -> None:
    topology, mode = preseeded
    verdict = _diagnostic_route(
        topology,
        mode,
        4,
        _fixture(topology, mode, 4, 20260815),
        decision_flip=True,
    )
    assert verdict["accept_decision_mismatches"] > 0
    assert verdict["probability_mismatches"] == 0


def test_gate_byte_comparator_refuses_shape_or_dtype_drift() -> None:
    mask = torch.ones(2, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="dtype drift"):
        taw._fr13_fixed32_taw_gate_bytes(
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.int64),
            mask,
        )
    with pytest.raises(RuntimeError, match="shape drift"):
        taw._fr13_fixed32_taw_gate_bytes(
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(3, dtype=torch.float32),
            mask,
        )


def test_half_wired_byte_gate_is_refused() -> None:
    counter = torch.zeros((), dtype=torch.int64)
    assert (
        taw._fr13_fixed32_taw_gate_is_complete(None, None, None, None) is False
    )
    assert (
        taw._fr13_fixed32_taw_gate_is_complete(
            (None, None), {}, counter, counter
        )
        is True
    )
    with pytest.raises(RuntimeError, match="byte gate is incomplete"):
        taw._fr13_fixed32_taw_gate_is_complete((None, None), None, counter, None)


def test_accept_decision_drift_fails_the_final_census(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = {
        "native_ab_candidate_census_events": torch.tensor(2, dtype=torch.int64),
        "native_ab_probability_mismatches": torch.zeros((), dtype=torch.int64),
        "native_ab_product_mismatches": torch.zeros((), dtype=torch.int64),
        "native_ab_accept_decision_mismatches": torch.tensor(
            1, dtype=torch.int64
        ),
        "native_ab_live_marker": "swe_verified:campaign4_" + "a" * 64,
    }
    monkeypatch.setattr(
        taw, "_fr13_fixed32_taw_native_selector", lambda **_kwargs: "diagnostic"
    )
    monkeypatch.setattr(
        taw, "_fr13_fixed32_taw_native_live_entry", lambda **_kwargs: entry
    )
    monkeypatch.setattr(
        taw,
        "_fr13_fixed32_taw_source_contract",
        lambda *_args, **_kwargs: {
            "source_contract_schema": taw._FR13_FIXED32_TAW_SOURCE_SCHEMA,
            "source_contract_sha256": taw._FR13_FIXED32_TAW_SOURCE_SHA256,
            "tensor_call_census": {},
        },
    )
    binding = {
        "operation": "fr13_bf16_k64_head::gemvx_m1_shuffle_r64_u8_out",
        "candidate_so_sha256": "b" * 64,
        "candidate_source_sha256": "c" * 64,
        "task_ids": ["astropy__astropy-12907"],
    }
    with pytest.raises(RuntimeError, match="final census mismatch"):
        taw.fr13_fixed32_taw_candidate_acceptance_census(
            mode="hydra27_fixed32",
            batch_size=4,
            completed_events=2,
            events_sha256="d" * 64,
            candidate_binding=binding,
        )
    entry["native_ab_accept_decision_mismatches"].zero_()
    record = taw.fr13_fixed32_taw_candidate_acceptance_census(
        mode="hydra27_fixed32",
        batch_size=4,
        completed_events=2,
        events_sha256="d" * 64,
        candidate_binding=binding,
    )
    assert record["status"] == "PASS"
    assert record["accept_decision_mismatches"] == 0


# ---------------------------------------------------------------------------
# 5. nothing armed, census unchanged
# ---------------------------------------------------------------------------
def test_shape_pin_does_not_move_the_tensor_call_census() -> None:
    # The pin changes reduction LAUNCH shapes, not the number of full-vocab
    # rows the route touches, so every census row stays where it was.
    assert taw._FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_TENSOR_CALL_CENSUS == {
        **taw._FR13_FIXED32_TAW_TENSOR_CALL_CENSUS,
        "walk_levels": 13,
        "full_vocab_row_gathers": 54,
        "full_vocab_fp32_casts": 26,
        "full_vocab_softmax_calls": 26,
        "full_vocab_normalizations": 83,
        "full_vocab_cdf_calls": 54,
        "source_cdf_calls": 29,
        "qmix_zero_fills": 29,
        "qmix_scatter_add_calls": 29,
        "residual_subtract_calls": 29,
        "residual_clamp_calls": 29,
        "residual_where_calls": 58,
        "exact_commit_launches": 13,
        "exact_commit_programs_per_request": 13,
    }
    assert (
        taw._FR13_FIXED32_TAW_NATIVE_PRODUCTION_TENSOR_CALL_CENSUS[
            "full_vocab_normalizations"
        ]
        == 47
    )


def test_nothing_is_armed_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    topology = taw._fr13_fixed32_topology()
    _set_fixed_env(monkeypatch, topology, "tail6_fixed32")
    monkeypatch.setattr(
        taw, "_FR13_FIXED32_TAW_NATIVE_DIAGNOSTIC_SIDECARS", ()
    )
    monkeypatch.setattr(
        taw, "_FR13_FIXED32_TAW_NATIVE_PRODUCTION_SIDECARS", ()
    )
    for batch_size in (None, 1, 2, 3, 4):
        assert (
            taw._fr13_fixed32_taw_native_selector(batch_size=batch_size)
            == "reference"
        )
    assert not Path(taw._FR13_FIXED32_TAW_NATIVE_PRODUCTION_PASS).exists()


def test_selector_fail_closed_behaviour_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = taw._fr13_fixed32_topology()
    _set_fixed_env(monkeypatch, topology, "tail6_fixed32")
    diagnostic = tmp_path / "diagnostic.arm"
    production = tmp_path / "production.arm"
    live_pass = tmp_path / "pass.json"
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
    diagnostic.write_text("1\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        taw._fr13_fixed32_taw_native_selector()
    production.unlink()
    assert taw._fr13_fixed32_taw_native_selector() == "diagnostic"
    monkeypatch.setenv("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE", "yes")
    with pytest.raises(RuntimeError, match="must be unset, 0, or 1"):
        taw._fr13_fixed32_taw_native_selector()

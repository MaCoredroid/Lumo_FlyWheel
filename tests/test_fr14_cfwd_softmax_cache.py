"""FR14 lane 3 -- the TAW softmax cache flag and its refusal to arm unwired.

The lever's arithmetic is proven on GPU by
``scripts/fr14_cfwd_softmax_batching_probe.py`` (gates G1/G3/G4). What this file
pins is the part a GPU probe cannot see:

  * the flag is strict and default-OFF, so an unarmed serve is byte-for-byte the
    serve it is today;
  * arming it does NOT drag in the all-parent candidate's B>=2 floor, which
    belongs to a different lever;
  * arming it routes the walk to the batched cache at all three arm points,
    and leaves the all-parent candidate's B>=2 selector exactly where it was;
  * the census that the wired lever would publish is recorded and honest --
    including the two extra row gathers the lever costs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path("scripts/fr13_device_multidraft_kernel.py")
SPEC = importlib.util.spec_from_file_location(
    "fr14_cfwd_softmax_cache_module", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
taw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(taw)

FLAG = "FR13_FIXED32_TAW_SOFTMAX_CACHE"


@pytest.fixture(autouse=True)
def _scratch_sidecars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the sidecars at a scratch dir so a stray /tmp arm cannot leak in."""
    monkeypatch.setattr(
        taw,
        "_FR13_FIXED32_TAW_SOFTMAX_CACHE_SIDECARS",
        (str(tmp_path / "a.arm"), str(tmp_path / "b.arm")),
    )
    monkeypatch.delenv(FLAG, raising=False)
    return tmp_path


def test_default_is_off() -> None:
    """Unset env, no sidecar -> OFF. The lever ships dark."""
    assert taw._fr13_fixed32_taw_softmax_cache_requested(environ={}) is False
    assert taw._fr13_fixed32_taw_softmax_cache_requested() is False


def test_explicit_zero_is_off_and_one_is_on() -> None:
    assert taw._fr13_fixed32_taw_softmax_cache_requested(environ={FLAG: "0"}) is False
    assert taw._fr13_fixed32_taw_softmax_cache_requested(environ={FLAG: "1"}) is True
    assert taw._fr13_fixed32_taw_softmax_cache_requested(environ={FLAG: " 1 "}) is True


@pytest.mark.parametrize("value", ["", "true", "yes", "on", "2", "01", "1.0", "-1"])
def test_malformed_value_raises_rather_than_reading_as_off(value: str) -> None:
    """A typo must not silently produce a candidate arm that is the stock path."""
    with pytest.raises(RuntimeError, match="must be exactly 0 or 1"):
        taw._fr13_fixed32_taw_softmax_cache_requested(environ={FLAG: value})


def test_sidecar_arms_the_lever(_scratch_sidecars: Path) -> None:
    """EngineCore worker curation drops env; the sidecar is the deployable arm."""
    assert taw._fr13_fixed32_taw_softmax_cache_requested(environ={}) is False
    (_scratch_sidecars / "b.arm").write_text("", encoding="utf-8")
    assert taw._fr13_fixed32_taw_softmax_cache_requested(environ={}) is True


def test_explicit_zero_does_not_defeat_an_armed_sidecar(
    _scratch_sidecars: Path,
) -> None:
    """Documents the precedence actually implemented: sidecar OR env, not AND."""
    (_scratch_sidecars / "a.arm").write_text("", encoding="utf-8")
    assert taw._fr13_fixed32_taw_softmax_cache_requested(environ={FLAG: "0"}) is True


def test_unarmed_resolution_is_identical_to_the_pre_lane3_expression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default OFF must be byte-for-byte the incumbent resolution.

    With the flag clear the shared resolver has to reduce EXACTLY to
    _fr13_fixed32_taw_native_precompute_enabled(), which is what the three walk
    arm points evaluated before this lane. Checked in both states of the native
    selector so the reduction is proven, not sampled once.
    """
    monkeypatch.setenv(FLAG, "0")
    for native in (False, True):
        monkeypatch.setattr(
            taw, "_fr13_fixed32_taw_native_precompute_enabled", lambda n=native: n
        )
        assert taw._fr13_fixed32_taw_probability_cache_requested() is native


def test_arming_requests_the_cache_without_the_native_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lever must reach the walk on its own, with the candidate unarmed."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(
        taw, "_fr13_fixed32_taw_native_precompute_enabled", lambda: False
    )
    assert taw._fr13_fixed32_taw_probability_cache_requested() is True


def test_recorded_census_for_the_wired_lever_is_honest() -> None:
    """The lever costs two extra gathers; the census must not hide that."""
    base = taw._FR13_FIXED32_TAW_TENSOR_CALL_CENSUS
    wired = taw._FR13_FIXED32_TAW_SOFTMAX_CACHE_TENSOR_CALL_CENSUS

    assert base["full_vocab_softmax_calls"] == 24
    assert wired["full_vocab_softmax_calls"] == 2
    assert wired["full_vocab_fp32_casts"] == 2
    # Gathers RISE. The lever is a trade, and the census says so.
    assert wired["full_vocab_row_gathers"] == base["full_vocab_row_gathers"] + 2

    for key in (
        "walk_levels",
        "full_vocab_normalizations",
        "full_vocab_cdf_calls",
        "source_cdf_calls",
        "qmix_scatter_add_calls",
        "qmix_zero_fills",
        "residual_subtract_calls",
        "residual_clamp_calls",
        "residual_where_calls",
        "exact_commit_launches",
        "exact_commit_programs_per_request",
    ):
        assert wired[key] == base[key], key
    assert wired["floating_sampling_reimplementation"] is False


def test_lever_does_not_move_the_all_parent_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The B>=2 floor belongs to the all-parent candidate, not to this lever.

    The all-parent candidate batches reductions (row sums, cumsum), measured
    shape-sensitive by up to 2 ULP (probe gate G2). This lever batches only the
    softmax. Arming the lane-3 flag must leave the native selector exactly where
    it was, at every batch width.
    """
    monkeypatch.setenv(FLAG, "1")
    for batch in (1, 2, 3, 4):
        assert taw._fr13_fixed32_taw_native_selector(batch_size=batch) == "reference"
    assert taw._fr13_fixed32_taw_native_precompute_enabled() is False
    assert taw._FR13_FIXED32_TAW_PINNED_MIN_BATCH == 2


def test_published_census_follows_the_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unarmed publishes the stock census; armed publishes the cache census."""
    monkeypatch.setenv(FLAG, "0")
    published = taw._fr13_fixed32_taw_tensor_call_census(batch_size=1)
    assert published == dict(taw._FR13_FIXED32_TAW_TENSOR_CALL_CENSUS)
    assert published["full_vocab_softmax_calls"] == 24

    monkeypatch.setenv(FLAG, "1")
    published = taw._fr13_fixed32_taw_tensor_call_census(batch_size=1)
    assert published == dict(taw._FR13_FIXED32_TAW_SOFTMAX_CACHE_TENSOR_CALL_CENSUS)
    assert published["full_vocab_softmax_calls"] == 2


def test_all_three_walk_arm_points_are_wired() -> None:
    """A walk left on the old resolver would silently ignore the flag."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    wired = source.count(
        "native_precompute = _fr13_fixed32_taw_probability_cache_requested()"
    )
    stale = source.count(
        "native_precompute = _fr13_fixed32_taw_native_precompute_enabled()"
    )
    assert wired == 3, f"expected 3 wired arm points, found {wired}"
    assert stale == 0, "a walk still resolves through the pre-lane-3 helper"


def test_source_digest_is_re_attested_to_the_wired_source() -> None:
    """The pinned digest must match what the wired module actually computes.

    A stale pin here is exactly the failure mode this lane spent its first hour
    attributing (a5110fe71, 2026-08-01): the credential silently describing a
    source that no longer exists.
    """
    import importlib

    topology_path = MODULE_PATH.parent / "fr13_fixed32_topology.py"
    assert topology_path.is_file()
    contract = taw._fr13_fixed32_taw_source_contract(taw._fr13_fixed32_topology())
    assert contract["source_contract_sha256"] == taw._FR13_FIXED32_TAW_SOURCE_SHA256
    assert taw._FR13_FIXED32_TAW_SOURCE_SHA256 == (
        "68b289aee5773edf1134f184c37551a90ec8543430d768a05066bc1341473c6d"
    )

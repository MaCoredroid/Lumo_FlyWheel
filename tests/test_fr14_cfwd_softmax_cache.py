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
        "6ffe57287e768bfee5e2e72f10de0dfea6fb3e6c0fa50f32b6c099c63fa916a2"
    )


# ---------------------------------------------------------------------------
# Inventory guard for the mirror-miss defect class.
#
# Pass 44 re-attested _FR13_FIXED32_TAW_SOURCE_SHA256 in the emitter and missed
# scripts/fr13_fixed32_work_census.py, which self-asserts on every boot: serves
# completed but their terminal audits died, and every fixed32 credential re-earn
# at that HEAD was refused. The campaign found the mirror that fired first; a
# sweep found ELEVEN more files carrying the same retyped literal.
#
# The fix for a defect class is not to retype twelve literals correctly once. It
# is to make the twelfth impossible to get wrong silently -- so this scans the
# tree for every retyped TAW source digest and requires it to equal the value the
# emitter actually publishes. A future re-attestation that misses a mirror fails
# here, in a unit test, instead of in a terminal audit after a serve.
# ---------------------------------------------------------------------------

import re

_TAW_DIGEST_ASSIGNMENT = re.compile(
    r"""(?P<name>[A-Za-z_]*TAW_SOURCE(?:_CONTRACT)?_SHA256)"""
    r"""\s*(?:=|:)\s*\(?\s*["']?(?P<digest>[0-9a-f]{64})["']?""",
    re.MULTILINE,
)

_MIRROR_SCAN_DIRS = ("scripts", "tests")
_MIRROR_SCAN_SUFFIXES = (".py", ".sh")


def _taw_digest_mirrors() -> dict[str, list[tuple[int, str, str]]]:
    """Every retyped TAW source digest under scripts/ and tests/."""
    root = MODULE_PATH.parent.parent
    found: dict[str, list[tuple[int, str, str]]] = {}
    for directory in _MIRROR_SCAN_DIRS:
        for path in sorted((root / directory).rglob("*")):
            if path.suffix not in _MIRROR_SCAN_SUFFIXES or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            hits = []
            for match in _TAW_DIGEST_ASSIGNMENT.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append((line, match.group("name"), match.group("digest")))
            if hits:
                found[str(path.relative_to(root))] = hits
    return found


def test_every_taw_source_digest_mirror_matches_the_emitter() -> None:
    """No retyped TAW digest may disagree with the value the emitter publishes."""
    current = taw._FR13_FIXED32_TAW_SOURCE_SHA256
    mirrors = _taw_digest_mirrors()

    # The scan must actually be finding things; a regex that silently matches
    # nothing would make this test a no-op that passes forever.
    assert len(mirrors) >= 8, f"mirror scan found too few files: {sorted(mirrors)}"

    stale = {
        path: [hit for hit in hits if hit[2] != current]
        for path, hits in mirrors.items()
    }
    stale = {path: hits for path, hits in stale.items() if hits}
    assert not stale, (
        "stale TAW source digest mirror(s) -- these refuse serves and credential "
        f"re-earns at HEAD. Emitter publishes {current}. Stale: {stale}"
    )


def test_work_census_mirror_tracks_the_emitter() -> None:
    """The mirror whose staleness blocked the promotion gate re-earn.

    Named separately from the scan above because this one is load-bearing on
    every boot: the census self-asserts, so a stale value here does not fail a
    test, it fails a completed serve's terminal audit.
    """
    import importlib.util as _ilu
    import sys

    census_path = MODULE_PATH.parent / "fr13_fixed32_work_census.py"
    # The census imports its sibling topology contract by bare module name.
    scripts_dir = str(census_path.parent.resolve())
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = _ilu.spec_from_file_location("fr14_lane3_census_mirror", census_path)
    assert spec is not None and spec.loader is not None
    census = _ilu.module_from_spec(spec)
    # Register before exec: the census defines dataclasses, and dataclasses
    # resolves cls.__module__ through sys.modules during class creation.
    sys.modules[spec.name] = census
    try:
        spec.loader.exec_module(census)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    assert census.TAW_SOURCE_CONTRACT_SHA256 == taw._FR13_FIXED32_TAW_SOURCE_SHA256
    assert census.TAW_SOURCE_CONTRACT_SCHEMA == taw._FR13_FIXED32_TAW_SOURCE_SCHEMA

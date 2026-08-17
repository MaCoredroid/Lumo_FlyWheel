"""The B1 nsys attribution harness describes the config it claims to describe.

FR14 ruling (A). The harness was pinned to the FR13 production stack in three
independent ways -- tail6 topology, K64/root1 draft vocabulary, and the qrow16
incumbent kernel. Those pins were RIGHT: an unpinned profiler re-measures whatever
the defaults select and annotates it with another arm's floor. So the promoted
FR14 config (hydra27 + full_vocab + gqa_pair) is served by RE-PINNING to a
declared identity, never by deleting the checks.

Every parameter defaults to its FR13 value, so an unparameterised invocation
reproduces every banked FR13 profile exactly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROFILER = REPO / "scripts" / "fr13_fixed32_b1_nsys_profile.sh"
SEQUENCE = REPO / "scripts" / "fr13_fixed32_floor_timers_seq.sh"
REDUCER = REPO / "scripts" / "fr13_fixed32_nsys_reduce.py"
PROFILER_TEXT = PROFILER.read_text(encoding="utf-8")
SEQUENCE_TEXT = SEQUENCE.read_text(encoding="utf-8")

sys.path.insert(0, str(REPO / "scripts"))
import fr13_fixed32_nsys_reduce as reduce_mod  # noqa: E402
import fr13_fixed32_topology as topology  # noqa: E402

HYDRA27_XFLAGS = (
    "FR13_FIXED32_MODE=hydra27_fixed32 "
    "FR13_FIXED32_VALID_MASK=0x7abdffff "
    "FR13_FIXED32_ACTIVE_NODES=27"
)
TAIL6_XFLAGS = (
    "FR13_FIXED32_MODE=tail6_fixed32 "
    "FR13_FIXED32_VALID_MASK=0x7a9ce7ff "
    "FR13_FIXED32_ACTIVE_NODES=23"
)


def _runlog(tmp_path: Path, xflags: str) -> Path:
    path = tmp_path / "arm.runlog"
    path.write_text(f"=== ARM ... xflags=[{xflags}] subset=... ===\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# the reducer verifies a DECLARATION; it no longer forbids a parameter value
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "xflags"),
    [("tail6_fixed32", TAIL6_XFLAGS), ("hydra27_fixed32", HYDRA27_XFLAGS)],
)
def test_a_truthful_declaration_is_accepted(
    tmp_path: Path, mode: str, xflags: str
) -> None:
    identity = reduce_mod._verify_declared_topology(mode, _runlog(tmp_path, xflags))
    assert identity["mode"] == mode
    assert int(identity["valid_mask"], 16) == topology.VALID_MASK_BY_MODE[mode]


@pytest.mark.parametrize(
    ("declared", "xflags"),
    [
        ("tail6_fixed32", HYDRA27_XFLAGS),
        ("hydra27_fixed32", TAIL6_XFLAGS),
    ],
)
def test_a_mislabelled_profile_is_refused(
    tmp_path: Path, declared: str, xflags: str
) -> None:
    """The failure the old gate could not see.

    Hardcoding `mode != tail6_fixed32` forbade one parameter VALUE while
    verifying nothing about the run, so it could not tell a truthful tail6
    profile from a tail6-labelled profile of some other topology.
    """
    with pytest.raises(reduce_mod.ReductionError, match="but the run served"):
        reduce_mod._verify_declared_topology(declared, _runlog(tmp_path, xflags))


def test_geometry_that_contradicts_the_canonical_table_is_refused(
    tmp_path: Path,
) -> None:
    """Agreeing labels are not enough: the served mask/active must be canonical."""
    bogus = (
        "FR13_FIXED32_MODE=hydra27_fixed32 "
        "FR13_FIXED32_VALID_MASK=0xdeadbeef "
        "FR13_FIXED32_ACTIVE_NODES=27"
    )
    with pytest.raises(reduce_mod.ReductionError, match="geometry disagrees"):
        reduce_mod._verify_declared_topology(
            "hydra27_fixed32", _runlog(tmp_path, bogus)
        )


def test_a_runlog_without_topology_xflags_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "arm.runlog"
    path.write_text("nothing useful here\n", encoding="utf-8")
    with pytest.raises(reduce_mod.ReductionError, match="no fixed32 topology"):
        reduce_mod._verify_declared_topology("hydra27_fixed32", path)


def test_a_missing_runlog_is_refused(tmp_path: Path) -> None:
    with pytest.raises(reduce_mod.ReductionError, match="unreadable"):
        reduce_mod._verify_declared_topology(
            "hydra27_fixed32", tmp_path / "absent.runlog"
        )


def test_an_unsupported_mode_is_still_refused(tmp_path: Path) -> None:
    with pytest.raises(reduce_mod.ReductionError, match="must be one of"):
        reduce_mod._verify_declared_topology(
            "b4_fixed32", _runlog(tmp_path, HYDRA27_XFLAGS)
        )


def test_the_verified_topology_is_published_in_the_provenance() -> None:
    """A verification nobody can read from the artifact is not evidence."""
    text = REDUCER.read_text(encoding="utf-8")
    assert '"topology_identity": topology_identity,' in text
    assert "--variant-runlog" in text


# --------------------------------------------------------------------------
# the profiler's declared identity
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("var", "fr13_default"),
    [
        ("FIXED32_MODE", "tail6_fixed32"),
        ("FA2_SELECTOR", "qrow16"),
        ("DRAFT_VOCAB_PROFILE", "k64_root"),
    ],
)
def test_every_identity_parameter_defaults_to_the_fr13_value(
    var: str, fr13_default: str
) -> None:
    """Banked FR13 profiles must reproduce byte-for-byte without new flags."""
    assert f"{var}=${{{var}:-{fr13_default}}}" in PROFILER_TEXT


def test_the_profiler_refuses_an_incoherent_selector_topology_pairing() -> None:
    """gqa_pair is legal only in hydra27; catch it before an 8-minute boot."""
    assert (
        '[[ "$FA2_SELECTOR" != "gqa_pair" || "$FIXED32_MODE" == "hydra27_fixed32" ]]'
        in PROFILER_TEXT
    )


def test_the_arm_name_and_declared_mode_cannot_disagree() -> None:
    assert "ARM=${FIXED32_MODE}_${TAG}" in PROFILER_TEXT
    assert "NSYS_EXPECTED_VARIANT_KIND=$FIXED32_MODE" in PROFILER_TEXT
    assert '--mode "$FIXED32_MODE"' in PROFILER_TEXT


def test_the_sequence_profiles_the_declared_topology() -> None:
    assert 'run_variant "${_FR13_ATTRIBUTION_MODE}_${TAG}" "$_FR13_ATTRIBUTION_MODE"' in (
        SEQUENCE_TEXT
    )
    assert (
        "_FR13_ATTRIBUTION_MODE=${FR13_FIXED32_ATTRIBUTION_MODE:-tail6_fixed32}"
        in SEQUENCE_TEXT
    )
    assert "FR13_FIXED32_ATTRIBUTION_MODE must be" in SEQUENCE_TEXT


def test_the_promoted_binary_is_pinned_as_tightly_as_the_incumbent() -> None:
    """Re-pin, not relax: each selection still asserts sha AND size."""
    assert (
        "GQA_PAIR_SO_SHA256=3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae"
        in PROFILER_TEXT
    )
    assert "GQA_PAIR_SO_BYTES=299815552" in PROFILER_TEXT
    assert '"$(stat -c \'%s\' "$FORKED_FA2_SO")" == "$EXPECTED_FA2_SO_BYTES"' in (
        PROFILER_TEXT
    )
    assert '== "$EXPECTED_FA2_SO_SHA256"' in PROFILER_TEXT


def test_the_full_vocab_profile_carries_the_sanctioned_override() -> None:
    assert 'export FR13_NEEDS_ALLOW="FR13_DRAFT_VOCAB_K=0"' in PROFILER_TEXT
    assert "export FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE=full_vocab" in PROFILER_TEXT


def test_the_gqa_pair_selection_does_not_pin_the_incumbent_as_production() -> None:
    """FR13_FA2_QROW16_PRODUCTION=1 would block the promoted-default path, whose
    scope guard requires it to be 0."""
    assert "export FR13_FA2_QROW16_PRODUCTION=0" in PROFILER_TEXT


def test_the_floor_is_derived_not_hardcoded() -> None:
    """A hardcoded floor is how a profile gets labelled with another arm's floor."""
    for pinned in ("FR13_MANDATORY_WEIGHT_BYTES=", "FR13_WEIGHT_FLOOR_MS="):
        assert pinned not in PROFILER_TEXT, (
            f"profiler hardcodes {pinned}; the sequence must derive it from the "
            "draft-vocabulary identity instead"
        )


def test_the_profiler_is_syntactically_valid() -> None:
    for script in (PROFILER, SEQUENCE):
        assert (
            subprocess.run(
                ["bash", "-n", str(script)], capture_output=True, check=False
            ).returncode
            == 0
        ), f"{script.name} does not parse"

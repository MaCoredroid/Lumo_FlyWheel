"""FR14 — draft-vocabulary qualification profiles, and the promoted-default fix.

Mark's K0 ruling made full-vocab drafting the production config. Every
credentialed lever had the K64/root1 draft-vocabulary identity hard-coded into
its own legality predicate, so the ruling parked the ARMING PATH for the whole
lever portfolio at once
(results/fr14_nvfp4_port_20260816/b1_lever_armability_under_k0.md).

The fix follows the shape the CUTLASS wave lever already used: a lever declares
WHICH draft-vocabulary shape its credential was earned in, and the launcher
checks the serving shape agrees. The safety property is unchanged -- a K64
credential still cannot authorize a K0 serve, and vice versa -- it is simply no
longer hard-coded to one of the two shapes.

These tests source the REAL helper out of the launcher rather than a copy, so
they cannot drift away from the thing they are testing.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"

K64_ENV = {
    "FR13_DRAFT_VOCAB_ROOT": "1",
    "FR13_DRAFT_VOCAB_K": "65536",
    "FR13_DRAFT_VOCAB_BLOCKS": "/workspace/scripts/fr13_dvk_subset_blocks.json",
    "FR13_NEEDS_ALLOW": "",
}
K0_ENV = {
    "FR13_DRAFT_VOCAB_ROOT": "0",
    "FR13_DRAFT_VOCAB_K": "0",
    "FR13_DRAFT_VOCAB_BLOCKS": "",
    "FR13_NEEDS_ALLOW": "FR13_DRAFT_VOCAB_K=0",
}


def _helper_source() -> str:
    """Extract the real helper from the launcher, so a copy cannot drift."""
    text = LAUNCHER.read_text(encoding="utf-8")
    start = text.index("_fr13_assert_draft_vocab_profile() {")
    end = text.index("\n}\n", start) + len("\n}\n")
    body = text[start:end]
    assert "k64_root" in body and "full_vocab" in body
    return body


def _run(profile: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    script = (
        "set -uo pipefail\n"
        + _helper_source()
        + f'\n_fr13_assert_draft_vocab_profile "{profile}" "TESTLEVER"\n'
        "echo rc=$?\n"
    )
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **env},
    )


def _accepted(result: subprocess.CompletedProcess[str]) -> bool:
    return "rc=0" in result.stdout


def test_k64_profile_accepts_the_k64_shape() -> None:
    assert _accepted(_run("k64_root", K64_ENV))


def test_full_vocab_profile_accepts_the_k0_shape() -> None:
    """The whole point: the K0 production shape is now a sanctioned identity."""
    assert _accepted(_run("full_vocab", K0_ENV))


def test_profiles_do_not_accept_each_others_shape() -> None:
    """The safety property. A credential earned in one shape must not
    authorize a serve in the other -- that is the only thing the old hard-coded
    clause was actually protecting, and it survives intact."""
    crossed_k64 = _run("k64_root", K0_ENV)
    assert not _accepted(crossed_k64)
    assert "k64_root draft-vocabulary identity" in crossed_k64.stderr

    crossed_k0 = _run("full_vocab", K64_ENV)
    assert not _accepted(crossed_k0)
    assert "full_vocab draft-vocabulary identity" in crossed_k0.stderr


def test_unknown_profile_is_refused() -> None:
    for profile in ("", "K64", "full-vocab", "anything"):
        result = _run(profile, K64_ENV)
        assert not _accepted(result), profile
        assert "must be exactly k64_root or full_vocab" in result.stderr


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("FR13_DRAFT_VOCAB_ROOT", "1"),
        ("FR13_DRAFT_VOCAB_K", "65536"),
        ("FR13_NEEDS_ALLOW", ""),
        ("FR13_NEEDS_ALLOW", "FR13_DRAFT_VOCAB_K=0 FR13_OTHER=1"),
    ),
    ids=("root-not-0", "k-not-0", "override-absent", "override-not-exact"),
)
def test_full_vocab_is_fail_closed_on_every_field(field: str, value: str) -> None:
    """FR13_NEEDS_ALLOW is part of the IDENTITY, not an escape hatch.

    Under full_vocab it must be EXACTLY the sanctioned K0 override -- not
    absent, and not that override plus something else. Under k64_root it must be
    absent. Either way a diagnostic override cannot smuggle a different shape
    past a lever's credential.
    """
    env = dict(K0_ENV)
    env[field] = value
    assert not _accepted(_run("full_vocab", env))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("FR13_DRAFT_VOCAB_ROOT", "0"),
        ("FR13_DRAFT_VOCAB_K", "0"),
        ("FR13_DRAFT_VOCAB_BLOCKS", "/workspace/scripts/other.json"),
        ("FR13_NEEDS_ALLOW", "FR13_DRAFT_VOCAB_K=0"),
    ),
    ids=("root-not-1", "k-not-65536", "wrong-block-map", "override-present"),
)
def test_k64_root_is_fail_closed_on_every_field(field: str, value: str) -> None:
    env = dict(K64_ENV)
    env[field] = value
    assert not _accepted(_run("k64_root", env))


def test_every_lever_declares_a_profile_and_defaults_to_k64_root() -> None:
    """Backward compatibility: every existing caller and banked K64 credential
    keeps its exact previous meaning, because the default is k64_root."""
    text = LAUNCHER.read_text(encoding="utf-8")
    for var in (
        "FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE",
        "FR13_FA2_QROW32_B4_QUALIFICATION_PROFILE",
        "FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE",
        "FR13_FIXED32_GDN_GQA_GROUP3_QUALIFICATION_PROFILE",
    ):
        assert f"{var}=${{{var}:-k64_root}}" in text, var
        assert f'"${var}" "FR13 ' in text, f"{var} is declared but never asserted"


def test_converted_levers_no_longer_hard_code_the_draft_vocabulary() -> None:
    """The four levers this unit converted must delegate to the helper.

    An inline `K == 65536` inside a lever predicate is a lever that silently
    cannot be armed under K0 -- that is the regression that parked the whole
    portfolio.
    """
    lines = LAUNCHER.read_text(encoding="utf-8").split("\n")
    inline = re.compile(
        r'^\s*&& "\$\{?FR13_DRAFT_VOCAB_(?:K|ROOT|BLOCKS)[^"]*"?\}?" == "[^"]*" \\$'
    )
    for msg in (
        "FR13 qrow32 B1 selector requires Hydra27 B1",
        "FR13 qrow32 B4 GQA-pair timing or production requires",
        "FR13 GDN GQA-group3 production requires exact credentialed",
        "FR13 GDN single-launch production requires exact credentialed",
    ):
        err = next(i for i, line in enumerate(lines) if msg in line)
        start = next(
            i for i in range(err, err - 60, -1) if re.match(r"^\s*\[\[ ", lines[i])
        )
        offenders = [lines[i].strip() for i in range(start, err) if inline.match(lines[i])]
        assert offenders == [], (msg, offenders)
        # ...and the profile assert must sit immediately above the predicate.
        assert "_fr13_assert_draft_vocab_profile" in "\n".join(
            lines[max(start - 4, 0):start]
        ), msg


# Inline `K == 65536` clauses still present in the launcher. Pinned as a raw
# line count because it is unambiguous -- attributing each clause to a lever by
# nearest error message mis-fires, since predicates do not always end in their
# own echo.
#
# Breakdown at the time of this unit (12 total):
#   1  the helper's own k64_root branch          (correct -- it IS the profile)
#   2  CUTLASS's two k64_root profile branches   (correct -- same pattern)
#   9  levers not converted by this unit: packed-walk node trust / active depth
#      / node-trust production, draft-head U8, draft-head M4 U8, DFWD K64 top3,
#      qrow16 live A/B, and two further predicate chains.
#
# CORRECTION (found by running the gate): "none is on the B1/B4 chain" was
# wrong for a DIFFERENT clause shape. The B1 live-A/B gate predicate bound the
# K64 shape via a bare `-z FR13_NEEDS_ALLOW` rather than a K==65536 line, so
# this line-count pin never saw it. It is now dropped as redundant -- the B1
# selector predicate asserts the profile, and every live-A/B arm passes through
# it. Shape-hardcoding can hide in a NEEDS_ALLOW clause too; grep for both.
# Converting them is future work; the pin makes that a deliberate edit.
_INLINE_K64_CLAUSE_LINES = 12  # unchanged: that clause was a bare NEEDS_ALLOW, not a K==65536 line


def test_remaining_inline_k64_clauses_are_pinned() -> None:
    lines = LAUNCHER.read_text(encoding="utf-8").split("\n")
    inline = re.compile(r'^\s*&& "\$\{?FR13_DRAFT_VOCAB_K[^"]*"?\}?" == "65536" \\$')
    found = [line.strip() for line in lines if inline.match(line)]
    assert len(found) == _INLINE_K64_CLAUSE_LINES, len(found)


# --------------------------------------------------------------------------
# The promoted-default fix.
# --------------------------------------------------------------------------


def test_promoted_default_requires_a_presented_credential() -> None:
    """Arm 2's rc=2, root-caused.

    The promoted default used to arm gqa_pair on ANY Hydra27 B1 launch, checking
    only that no other arm was named -- never that a credential existed. The
    selector predicate then demanded SOURCE_COMMIT == HEAD, which is empty on an
    ordinary campaign launch, so the launcher exited 2. Arm 1 (tail6) was
    unaffected because the default is Hydra27-only, which is exactly the
    asymmetry observed: tail6 served clean, hydra27 died, every campaign.

    A promotion says which arm to prefer WHEN AVAILABLE; it is never a reason to
    refuse the boot.
    """
    text = LAUNCHER.read_text(encoding="utf-8")
    guard = (
        'if [[ -n "${FR13_FA2_QROW32_B1_PRODUCTION_PASS_SIDECAR:-}" \\\n'
        '        && -n "${FR13_FA2_QROW32_B1_SOURCE_COMMIT:-}" ]]; then'
    )
    assert guard in text

    # The arming assignment must live inside that guard, with an else branch
    # that falls back to the incumbent rather than exiting.
    guard_at = text.index(guard)
    arm_at = text.index(
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM=${FR13_FA2_QROW32_B1_PRODUCTION_ARM_DEFAULT",
        guard_at,
    )
    else_at = text.index("serving the INCUMBENT, not the promoted default", guard_at)
    assert guard_at < arm_at < else_at

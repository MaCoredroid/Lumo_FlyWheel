"""Launcher wiring for lane 4's split-K FA2 arm (`gqa_pair_splitk`).

The kernel, its numbers and its build credentials are lane 4's
(`results/fr14_nvfp4_port_20260816/splitk_fa2.md`). What is tested here is only
the launcher side: that the arm exists, that it is reachable ONLY as a live arm,
that the promoted `gqa_pair` default is untouched, and that arming it is
impossible without the pinned binary actually being on disk.

The load-bearing test is `test_pins_are_derived_from_the_build_attestation`: the
literals in the launcher are checked against
`fr14_splitk_fa2_build_attestation.json` rather than being retyped constants that
could drift away from the artifact they claim to pin.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
# every launcher family, from the single roster -- never re-enumerated here,
# because "both families" was wrong by one for six rounds
# `scripts` is not a package, so this module only imported when some OTHER
# test file had already inserted it on sys.path -- i.e. it passed in a full run
# and failed when run alone. A test whose result depends on collection order is
# a test that will eventually pass for the wrong reason.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fr14_mode_table_parity as _parity  # noqa: E402

LAUNCHERS = _parity.LAUNCHER_FAMILIES
ATTESTATION = (
    REPO
    / "results/fr14_nvfp4_port_20260816/fr14_splitk_fa2_build_attestation.json"
)
STAGED_SO = Path(
    "/home/mark/fr14_splitk_build_20260818/"
    "_vllm_fa2_qrow32_gqa_pair_splitk_b1_sm121a.abi3.so"
)


@pytest.fixture(params=LAUNCHERS)
def launcher(request):
    return (REPO / request.param).read_text()


@pytest.fixture(scope="module")
def attestation():
    if not ATTESTATION.exists():
        pytest.skip("split-K build attestation not present")
    return json.loads(ATTESTATION.read_text())


# ---------------------------------------------------------------------------
# Reachability: live arm yes, production arm no.
# ---------------------------------------------------------------------------

def test_arm_is_selectable_as_a_live_arm(launcher):
    assert '""|nosplit|split2|visibility|gqa_pair|gqa_pair_splitk) ;;' in launcher
    assert (
        "FR13_FA2_QROW32_B1_LIVE_AB_ARM must be empty, nosplit, split2, "
        "visibility, gqa_pair, or gqa_pair_splitk" in launcher
    )


def test_arm_is_never_a_production_arm(launcher):
    """Tier-B: the raw-byte gate structurally refuses this arm, so it is gate-only."""
    assert '  ""|nosplit|gqa_pair) ;;' in launcher
    assert (
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM must be empty, nosplit, or gqa_pair"
        in launcher
    )
    # the promoted default is untouched
    assert "FR13_FA2_QROW32_B1_PRODUCTION_ARM_DEFAULT:-gqa_pair" in launcher


def test_arm_has_its_own_pin_branch(launcher):
    assert "    gqa_pair_splitk)\n" in launcher
    assert "FR13 qrow32 B1 split-K binary/source provenance drifted" in launcher


# ---------------------------------------------------------------------------
# The credential.
# ---------------------------------------------------------------------------

def _splitk_branch(text):
    start = text.index("    gqa_pair_splitk)\n")
    end = text.index("\n      ;;\n", start)
    return text[start:end]


def test_pins_are_derived_from_the_build_attestation(launcher, attestation):
    """The launcher literals must equal the artifact's own attested identity."""
    branch = _splitk_branch(launcher)
    ident = attestation["candidate_so_identity"]
    assert f'== "{ident["so_sha256"]}"' in branch
    assert f'== "{ident["so_size"]}"' in branch
    assert (
        f'== "{attestation["compile_env"]["source_commit"]}"' in branch
    ), "FA2 head must be the attested source commit"
    baseline = attestation["baseline_sass_digest"]["baseline_sass_digest_pinned"]
    assert f'== "{baseline}"' in branch, "sealed baseline SASS digest"


def test_sass_credentials_are_pinned(launcher):
    """The .so sha is not rebuild-reproducible; the SASS digests are the kernel's."""
    branch = _splitk_branch(launcher)
    assert "FR13_FA2_QROW32_B1_SPLITK_SASS_DIGEST" in branch
    assert "FR13_FA2_QROW32_B1_SPLITK_BASELINE_SASS_DIGEST" in branch
    assert (
        "3f24d70dce2ff70ad9209bad5af2a93cc39453df529cb298e4476cbfbfd80b9e"
        in branch
    )
    assert "split-K SASS credential drifted" in branch


def test_refuses_to_arm_without_the_binary_present_and_matching(launcher):
    """Presence, non-symlink, and a real re-hash -- not just the declared pin."""
    branch = _splitk_branch(launcher)
    assert '[[ -f "$FORKED_FA2_SO"' in branch
    assert '! -L "$FORKED_FA2_SO"' in branch
    assert (
        '"$(sha256sum "$FORKED_FA2_SO" | cut -d\' \' -f1)" '
        '== "$FR13_FA2_QROW32_B1_SO_SHA256"' in branch
    )


def test_the_rehash_is_not_redundant_with_the_size_check(launcher):
    """Two known links of this arm share a size and differ in sha.

    The generic B1 selector only compares `stat -c '%s'`, so without the re-hash
    the wrong one of those two binaries would arm silently.
    """
    assert '"$(stat -c \'%s\' "$FORKED_FA2_SO")" == "$FR13_FA2_QROW32_B1_SO_SIZE"' in launcher
    branch = _splitk_branch(launcher)
    assert "sha256sum" in branch


def test_staged_binary_matches_the_pin_if_present(attestation):
    if not STAGED_SO.exists():
        pytest.skip("staged split-K binary not on this host")
    import hashlib

    digest = hashlib.sha256(STAGED_SO.read_bytes()).hexdigest()
    assert digest == attestation["candidate_so_identity"]["so_sha256"]
    assert STAGED_SO.stat().st_size == int(
        attestation["candidate_so_identity"]["so_size"]
    )


# ---------------------------------------------------------------------------
# The guard machinery must treat the new arm like its siblings.
# ---------------------------------------------------------------------------

def test_named_splitk_live_arm_arms_the_local_env_guard(launcher):
    """Every other B1 live arm arms it; omitting this one would silently weaken."""
    assert (
        '"${_FR13_CALLER_M32_GUARD[FR13_FA2_QROW32_B1_LIVE_AB_ARM]}" '
        '== "set:gqa_pair_splitk"' in launcher
    )


def test_new_pin_names_cannot_be_moved_by_local_env(launcher):
    """They must be inside _FR13_M32_GUARD_NAMES, like every other pin var."""
    block = launcher[
        launcher.index("_FR13_M32_GUARD_NAMES") : launcher.index(
            "declare -A _FR13_CALLER_M32_GUARD"
        )
    ]
    assert "FR13_FA2_QROW32_B1_SPLITK_SASS_DIGEST" in block
    assert "FR13_FA2_QROW32_B1_SPLITK_BASELINE_SASS_DIGEST" in block


def test_off_path_is_untouched(launcher):
    """The incumbent pins are untouched by the promotion.

    THIS TEST'S PREMISE CHANGED WITH THE PROMOTION (Mark, pass 100), and the
    change is the point rather than a regression: split-K is now the production
    default under hydra27_fixed32, so a launch that names no arm DOES reach it.
    What must still hold is that promotion did not disturb the arms it
    displaced -- the incumbent and gqa_pair pins are byte-for-byte what they
    were, and split-K's literals appear in exactly two places: the pin-arm
    branch that validates a named arm, and the promoted-default block that
    arms an unnamed one.
    """
    assert (
        "3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae"
        in launcher
    ), "gqa_pair .so pin"
    assert "299815552" in launcher, "gqa_pair .so size"
    assert (
        "a9d8a6887b8b27b3a83af60bba7945eb66caff174ba710c2ee2aea92b8e7081a"
        in launcher
    ), "incumbent .so pin"
    # the split-K literals appear ONLY inside the new branch
    branch = _splitk_branch(launcher)
    default_block = launcher[
        launcher.index("# ---------------------------------------------------------------- split-K"):
    ]
    default_block = default_block[: default_block.index("\n  fi\n")]
    for literal in (
        "28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857",
        "4ed00909cef7ea83849f897018ea4f6a14119b8d160927af426938920c170878",
        "3f24d70dce2ff70ad9209bad5af2a93cc39453df529cb298e4476cbfbfd80b9e",
    ):
        # Exactly two: the validating pin-arm branch, and the promoted default.
        assert launcher.count(literal) == 2, literal
        assert literal in branch
        assert literal in default_block


def test_every_b1_pin_arm_has_a_distinct_binary(launcher):
    """No two arms may pin the same .so, or the pin proves nothing."""
    shas = re.findall(
        r'"\$FR13_FA2_QROW32_B1_SO_SHA256" == "([0-9a-f]{64})"', launcher
    )
    assert len(shas) == 4, f"expected visibility/gqa_pair/splitk/incumbent, got {shas}"
    assert len(set(shas)) == 4


# ---------------------------------------------------------------------------
# Behaviour, not text: extract the branch and actually run it.
# ---------------------------------------------------------------------------

GOOD = {
    "FR13_FA2_QROW32_B1_SO_SHA256": (
        "28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857"
    ),
    "FR13_FA2_QROW32_B1_SO_SIZE": "300123792",
    "FR13_FA2_QROW32_B1_FA2_HEAD": "29210221863736a08f71a866459e368ad1ac4a95",
    "FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256": (
        "4ed00909cef7ea83849f897018ea4f6a14119b8d160927af426938920c170878"
    ),
    "FR13_FA2_QROW32_B1_SPLITK_SASS_DIGEST": (
        "3f24d70dce2ff70ad9209bad5af2a93cc39453df529cb298e4476cbfbfd80b9e"
    ),
    "FR13_FA2_QROW32_B1_SPLITK_BASELINE_SASS_DIGEST": (
        "fa01f98840420b9c0177d06297aacabb0ed5e00c674511fdaa4aa618c3473470"
    ),
}


def _run_branch(launcher_text, env_overrides, tmpdir):
    import os
    import subprocess

    # drop the `gqa_pair_splitk)` case label -- the body alone is a valid script
    body = _splitk_branch(launcher_text).split("\n", 1)[1]
    script = Path(tmpdir) / "branch.sh"
    script.write_text("set -uo pipefail\n" + body + "\n")
    env = dict(os.environ)
    env.update(GOOD)
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, env=env
    )


@pytest.fixture(scope="module")
def workdir():
    import tempfile

    return tempfile.mkdtemp()


def test_branch_accepts_the_real_pinned_binary(launcher, workdir):
    if not STAGED_SO.exists():
        pytest.skip("staged split-K binary not on this host")
    r = _run_branch(launcher, {"FORKED_FA2_SO": str(STAGED_SO)}, workdir)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize(
    "override,expect",
    [
        ({"FORKED_FA2_SO": "/nonexistent.so"}, "present and byte-identical"),
        ({"FR13_FA2_QROW32_B1_SPLITK_SASS_DIGEST": "0" * 64}, "SASS credential"),
        (
            {"FR13_FA2_QROW32_B1_SPLITK_BASELINE_SASS_DIGEST": "1" * 64},
            "SASS credential",
        ),
        (
            {
                "FR13_FA2_QROW32_B1_SO_SHA256": (
                    "3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae"
                )
            },
            "provenance drifted",
        ),
        ({"FR13_FA2_QROW32_B1_SO_SIZE": "299815552"}, "provenance drifted"),
    ],
)
def test_branch_refuses_every_way_of_getting_it_wrong(
    launcher, workdir, override, expect
):
    env = {"FORKED_FA2_SO": str(STAGED_SO)}
    env.update(override)
    r = _run_branch(launcher, env, workdir)
    assert r.returncode == 2, r.stdout + r.stderr
    assert expect in r.stderr


def test_branch_refuses_a_symlink_to_the_right_binary(launcher, workdir):
    """A symlink is refused even when it resolves to the pinned bytes."""
    if not STAGED_SO.exists():
        pytest.skip("staged split-K binary not on this host")
    link = Path(workdir) / "link.so"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(STAGED_SO)
    r = _run_branch(launcher, {"FORKED_FA2_SO": str(link)}, workdir)
    assert r.returncode == 2
    assert "present and byte-identical" in r.stderr


def test_branch_refuses_a_wrong_binary_at_the_right_path(launcher, workdir):
    wrong = Path(workdir) / "wrong.so"
    wrong.write_bytes(b"not the kernel")
    r = _run_branch(launcher, {"FORKED_FA2_SO": str(wrong)}, workdir)
    assert r.returncode == 2
    assert "present and byte-identical" in r.stderr

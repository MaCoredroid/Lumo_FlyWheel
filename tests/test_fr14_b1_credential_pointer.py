"""The run-local B1 gqa_pair credential pointer (FR14 ruling C).

Ordinary serves should be able to carry the promoted gqa_pair arm without every
caller hand-typing the whole credential. The pointer is a convenience for TYPING
ONLY: the full validation chain still runs at boot, so a wrong, stale or hostile
pointer cannot manufacture a credential -- it can only fail those checks or, on
the unnamed path, degrade to the incumbent loudly.

The property these tests exist to protect is the WHITELIST. A pointer able to set
FR13_FA2_QROW32_B1_PRODUCTION_PASS_SIDECAR would smuggle back the launcher-private
trap the promoted-default train had just removed -- the caller would be handing
the launcher a pre-minted credential through a side door.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
GATE_RUNNER = REPO / "scripts" / "fr14_run_b1_k0_qrow32_live_gate.sh"
LAUNCHER_TEXT = LAUNCHER.read_text(encoding="utf-8")
GATE_TEXT = GATE_RUNNER.read_text(encoding="utf-8")

LOADER = "_fr13_b1_load_credential_pointer"

# Exactly the names a caller may present for the gqa_pair B1 production arm.
CALLER_FACING = (
    "FR13_FA2_QROW32_B1_GQA_PAIR_GATE_JSON",
    "FR13_FA2_QROW32_B1_GQA_PAIR_GATE_SHA256",
    "FR13_FA2_QROW32_B1_GQA_PAIR_LIVE_RESULT_JSON",
    "FR13_FA2_QROW32_B1_SOURCE_COMMIT",
    "FR13_FA2_QROW32_B1_SO_SHA256",
    "FR13_FA2_QROW32_B1_SO_SIZE",
    "FR13_FA2_QROW32_B1_FA2_HEAD",
    "FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256",
    "FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256",
    "FR13_FA2_QROW32_B1_EXACT4_TASK_IDS",
    "FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256",
    "FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE",
)

LAUNCHER_PRIVATE = (
    "FR13_FA2_QROW32_B1_PRODUCTION_PASS_SIDECAR",
    "FR13_FA2_QROW32_B1_PRODUCTION_PASS_SIDECAR_SHA256",
    "FR13_FA2_QROW32_B4_PRODUCTION_PASS_SIDECAR",
    "FR13_FA2_QROW32_B4_PRODUCTION_PASS_SIDECAR_SHA256",
)


def _loader_source() -> str:
    match = re.search(
        rf"^{re.escape(LOADER)}\(\) \{{.*?^\}}", LAUNCHER_TEXT, re.S | re.M
    )
    assert match, "credential pointer loader not found in the launcher"
    return match.group(0)


def _run_loader(
    tmp_path: Path, pointer_body: str, preset: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Exercise the real loader text, extracted from the real launcher."""
    pointer = tmp_path / "pointer.env"
    pointer.write_text(pointer_body, encoding="utf-8")
    script = tmp_path / "drive.sh"
    script.write_text(
        "set -euo pipefail\n"
        + _loader_source()
        + f'\n{LOADER} "$1"\n'
        + "".join(
            f'printf "%s=%s\\n" {name} "${{{name}:-}}"\n' for name in CALLER_FACING
        ),
        encoding="utf-8",
    )
    env = {"PATH": "/usr/bin:/bin"}
    env.update(preset or {})
    return subprocess.run(
        ["bash", str(script), str(pointer)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        check=False,
    )


def _parsed(result: subprocess.CompletedProcess[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


# --------------------------------------------------------------------------
# the whitelist -- the property that matters
# --------------------------------------------------------------------------


@pytest.mark.parametrize("private", LAUNCHER_PRIVATE)
def test_the_pointer_may_never_set_a_launcher_private_credential(
    tmp_path: Path, private: str
) -> None:
    """The side door the promoted-default train just closed stays closed."""
    result = _run_loader(tmp_path, f"{private}=/tmp/forged.json\n")
    assert result.returncode != 0, f"pointer accepted the private {private}"
    assert f"may not set {private}" in result.stderr


def test_the_whitelist_is_exactly_the_caller_facing_contract() -> None:
    loader = _loader_source()
    for name in CALLER_FACING:
        assert name in loader, f"{name} is not accepted by the pointer"
    for name in LAUNCHER_PRIVATE:
        assert name not in loader, f"{name} appears in the pointer whitelist"


def test_an_unknown_name_refuses_rather_than_being_ignored(tmp_path: Path) -> None:
    """Silently dropping an unknown key would hide a typo'd credential."""
    result = _run_loader(tmp_path, "PATH=/tmp/evil\n")
    assert result.returncode != 0
    assert "may not set PATH" in result.stderr


def test_a_non_assignment_line_refuses(tmp_path: Path) -> None:
    result = _run_loader(tmp_path, "this is not an assignment\n")
    assert result.returncode != 0
    assert "non-assignment line" in result.stderr


# --------------------------------------------------------------------------
# precedence and absence
# --------------------------------------------------------------------------


def test_an_explicit_caller_value_always_beats_the_pointer(tmp_path: Path) -> None:
    """The pointer saves typing; it never overrides an explicit decision."""
    result = _run_loader(
        tmp_path,
        "FR13_FA2_QROW32_B1_SOURCE_COMMIT=frompointer\n"
        "FR13_FA2_QROW32_B1_GQA_PAIR_GATE_JSON=/x/gate.json\n",
        preset={"FR13_FA2_QROW32_B1_SOURCE_COMMIT": "fromcaller"},
    )
    assert result.returncode == 0, result.stderr
    parsed = _parsed(result)
    assert parsed["FR13_FA2_QROW32_B1_SOURCE_COMMIT"] == "fromcaller"
    # and the key the caller did NOT set is still supplied
    assert parsed["FR13_FA2_QROW32_B1_GQA_PAIR_GATE_JSON"] == "/x/gate.json"


def test_a_missing_pointer_is_a_silent_no_op(tmp_path: Path) -> None:
    """No pointer is the ordinary case, not an error: it degrades to incumbent."""
    script = tmp_path / "drive.sh"
    script.write_text(
        "set -euo pipefail\n" + _loader_source() + f'\n{LOADER} "$1"\necho OK\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(script), str(tmp_path / "absent.env")],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_comments_and_blank_lines_are_tolerated(tmp_path: Path) -> None:
    result = _run_loader(
        tmp_path,
        "# written by the gate runner\n"
        "\n"
        "   # indented comment\n"
        "FR13_FA2_QROW32_B1_SOURCE_COMMIT=abc123\n",
    )
    assert result.returncode == 0, result.stderr
    assert _parsed(result)["FR13_FA2_QROW32_B1_SOURCE_COMMIT"] == "abc123"


def test_a_value_containing_an_equals_sign_survives(tmp_path: Path) -> None:
    """Split on the FIRST '=' only -- task id lists and paths may contain more."""
    result = _run_loader(
        tmp_path, "FR13_FA2_QROW32_B1_EXACT4_TASK_IDS=a=b,c\n"
    )
    assert result.returncode == 0, result.stderr
    assert _parsed(result)["FR13_FA2_QROW32_B1_EXACT4_TASK_IDS"] == "a=b,c"


def test_a_symlinked_pointer_refuses(tmp_path: Path) -> None:
    """Every other run artifact in this repo refuses symlinks; so does this."""
    real = tmp_path / "real.env"
    real.write_text("FR13_FA2_QROW32_B1_SOURCE_COMMIT=abc\n", encoding="utf-8")
    link = tmp_path / "link.env"
    link.symlink_to(real)
    script = tmp_path / "drive.sh"
    script.write_text(
        "set -euo pipefail\n" + _loader_source() + f'\n{LOADER} "$1"\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(script), str(link)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert result.returncode != 0
    assert "must be a regular file" in result.stderr


# --------------------------------------------------------------------------
# wiring: the launcher consumes it, the gate runner produces it
# --------------------------------------------------------------------------


def test_the_launcher_defaults_the_pointer_under_the_ignored_output_tree() -> None:
    """A TRACKED pointer would invalidate itself: it records SOURCE_COMMIT, and
    committing it moves HEAD."""
    assert (
        "FR13_B1_CREDENTIAL_POINTER=${FR13_B1_CREDENTIAL_POINTER:-"
        "$REPO/output/fr13_b1_gqa_pair_credential.env}" in LAUNCHER_TEXT
    )


def test_a_malformed_pointer_refuses_the_launch(tmp_path: Path) -> None:
    """Unreadable provenance must never be served past."""
    assert "credential pointer is malformed; refusing rather than serving an" in (
        LAUNCHER_TEXT
    )


def test_the_gate_runner_publishes_the_pointer_only_for_gqa_pair() -> None:
    assert '"$LIVE_ARM" == "gqa_pair" && -f "$VERIFICATION_JSON"' in GATE_TEXT
    assert "FR13_FA2_QROW32_B1_GQA_PAIR_GATE_SHA256=%s" in GATE_TEXT


def test_the_gate_runner_refuses_to_write_a_tracked_pointer() -> None:
    """The untracked property is enforced, not assumed."""
    assert 'git check-ignore -q "$POINTER"' in GATE_TEXT
    assert "credential pointer destination is not Git-ignored" in GATE_TEXT


def test_the_pointer_carries_the_production_identity_not_the_gates_own() -> None:
    """The gate runs ONE diagnostic task; the credential is used by four-task
    production serves. Publishing the gate's own subset would present an identity
    the launcher rejects."""
    assert "PRODUCTION_SUBSET_SHA256=" in GATE_TEXT
    assert (
        "EXACT4_TASK_IDS=astropy__astropy-12907,astropy__astropy-13033,"
        "astropy__astropy-13236,astropy__astropy-13398" in GATE_TEXT
    )
    # and it is verified against the checked-in subset rather than trusted
    assert 'sha256sum "$PRODUCTION_SUBSET"' in GATE_TEXT


def test_the_production_identity_matches_what_the_launcher_demands() -> None:
    """Two hardcodings that must agree, asserted against each other."""
    launcher_ids = re.search(
        r'"\$FR13_FA2_QROW32_B1_EXACT4_TASK_IDS" == "([^"]+)"', LAUNCHER_TEXT
    )
    launcher_sha = re.search(
        r'"\$FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256" == "([0-9a-f]{64})"',
        LAUNCHER_TEXT,
    )
    assert launcher_ids and launcher_sha
    assert f"EXACT4_TASK_IDS={launcher_ids.group(1)}" in GATE_TEXT
    assert f"PRODUCTION_SUBSET_SHA256={launcher_sha.group(1)}" in GATE_TEXT

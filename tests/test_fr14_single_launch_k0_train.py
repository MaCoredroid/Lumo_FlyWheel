"""The ordered-GDN single_launch gate can be earned in the shape it will SERVE.

FR14. Under Mark's K0 production ruling the served B1 config is full_vocab, but
the ordered-GDN live gate was pinned to K64/root1 in two places -- a launcher
clause and the shared runner. A credential earned in a shape production never
serves is the mislabelled-evidence class this campaign keeps closing, so both
sites are RE-PINNED to a declared identity rather than unpinned.

Both default to k64_root, so all four banked ordered-GDN entrypoints
(fr13_run_b1_/b4_hydra27_/b4_tail23_/…) reproduce byte-for-byte.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
RUNNER = REPO / "scripts" / "fr13_run_gdn_single_launch_live_gate.sh"
B1_SHIM = REPO / "scripts" / "fr13_run_b1_gdn_single_launch_live_gate.sh"
LAUNCHER_TEXT = LAUNCHER.read_text(encoding="utf-8")
RUNNER_TEXT = RUNNER.read_text(encoding="utf-8")

PROFILE_BLOCK_RE = re.compile(
    r"^FR13_GDN_GATE_DRAFT_VOCAB_PROFILE=.*?"
    r"^export FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE=[^\n]*\n",
    re.S | re.M,
)


def _resolve(profile: str | None) -> dict[str, str]:
    """Execute the runner's real identity block and report what it exports."""
    match = PROFILE_BLOCK_RE.search(RUNNER_TEXT)
    assert match, "gate draft-vocabulary profile block not found"
    script = (
        "set -uo pipefail\n"
        + match.group(0)
        + 'printf "K=%s\\nROOT=%s\\nBLOCKS=%s\\nNEEDS=%s\\nBYTES=%s\\nFLOOR=%s\\n" '
        '"$FR13_DRAFT_VOCAB_K" "$FR13_DRAFT_VOCAB_ROOT" "$FR13_DRAFT_VOCAB_BLOCKS" '
        '"${FR13_NEEDS_ALLOW:-}" "$GATE_EXPECTED_WEIGHT_BYTES" "$GATE_EXPECTED_FLOOR_MS"\n'
    )
    env = {"PATH": "/usr/bin:/bin"}
    if profile is not None:
        env["FR13_GDN_GATE_DRAFT_VOCAB_PROFILE"] = profile
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env, check=False
    )
    if proc.returncode != 0:
        return {"_rc": str(proc.returncode), "_err": proc.stderr.strip()}
    out = dict(
        line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line
    )
    out["_rc"] = "0"
    return out


# --------------------------------------------------------------------------
# regression safety: the banked entrypoints must not move
# --------------------------------------------------------------------------


@pytest.mark.parametrize("profile", [None, "k64_root"])
def test_the_default_reproduces_the_banked_k64_gate_exactly(profile) -> None:
    got = _resolve(profile)
    assert got["_rc"] == "0", got
    assert got["K"] == "65536"
    assert got["ROOT"] == "1"
    assert got["NEEDS"] == ""
    # the exact constants the runner asserted before this train
    assert got["BYTES"] == "25210209416"
    assert got["FLOOR"] == "92.345089436"


def test_full_vocab_serves_the_k0_identity_and_the_arm_b_floor() -> None:
    got = _resolve("full_vocab")
    assert got["_rc"] == "0", got
    assert got["K"] == "0"
    assert got["ROOT"] == "0"
    assert got["NEEDS"] == "FR13_DRAFT_VOCAB_K=0", "sanctioned override missing"
    assert got["BYTES"] == "25430574256"
    assert got["FLOOR"] == "93.15228665201465"


def test_an_unknown_profile_is_refused() -> None:
    got = _resolve("k64")  # plausible typo
    assert got["_rc"] != "0"
    assert "must be k64_root or full_vocab" in got["_err"]


def test_the_block_map_is_canonical_in_both_profiles() -> None:
    """Production carries the canonical map; a gate that dropped it would not be
    earned in the production shape."""
    for profile in ("k64_root", "full_vocab"):
        got = _resolve(profile)
        assert got["BLOCKS"] == "/workspace/scripts/fr13_dvk_subset_blocks.json"


# --------------------------------------------------------------------------
# the launcher clause
# --------------------------------------------------------------------------


def test_the_launcher_gate_clause_no_longer_hardcodes_k64() -> None:
    assert "FR13 ordered GDN live gate requires exact K64/root1" not in LAUNCHER_TEXT


def test_the_launcher_gate_clause_asserts_the_whole_identity_via_the_helper() -> None:
    """Two of four fields is not an identity: the helper checks root, K, the
    block map and the sanctioned override, and is the same one every other lever
    uses -- so runner and launcher cannot disagree about what a profile means."""
    assert "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE" in LAUNCHER_TEXT
    assert (
        '_fr13_assert_draft_vocab_profile \\\n'
        '\t      "$FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE" \\\n'
        '\t      "FR13 ordered GDN live gate" || exit 2' in LAUNCHER_TEXT
    )


def test_the_launcher_gate_profile_defaults_to_k64_root() -> None:
    assert (
        "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE="
        "${FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE:-k64_root}"
        in LAUNCHER_TEXT
    )


# --------------------------------------------------------------------------
# the runner reports what it served
# --------------------------------------------------------------------------


def test_the_launcher_meta_reports_the_served_identity_not_a_literal() -> None:
    """A K0 gate that announced draft_vocab_k=65536 is exactly the credential
    self-misdescription the qrow32 chain already had to re-run to fix."""
    assert "draft_vocab_k=65536\\ndraft_vocab_root=1" not in RUNNER_TEXT
    assert 'draft_vocab_k=\'"$FR13_DRAFT_VOCAB_K"\'' in RUNNER_TEXT
    assert 'draft_vocab_root=\'"$FR13_DRAFT_VOCAB_ROOT"\'' in RUNNER_TEXT


def test_the_serve_env_carries_the_declared_identity_and_the_override() -> None:
    assert (
        'FR13_DRAFT_VOCAB_K="$FR13_DRAFT_VOCAB_K" '
        'FR13_DRAFT_VOCAB_ROOT="$FR13_DRAFT_VOCAB_ROOT"' in RUNNER_TEXT
    )
    assert 'FR13_NEEDS_ALLOW="${FR13_NEEDS_ALLOW:-}"' in RUNNER_TEXT


def test_the_runner_propagates_the_profile_to_both_launcher_gates() -> None:
    """The gate path and the production selector must qualify under the SAME
    declared identity, or a gate could be earned under one and spent under
    another."""
    for var in (
        "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE",
        "FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE",
    ):
        assert f'export {var}="$FR13_GDN_GATE_DRAFT_VOCAB_PROFILE"' in RUNNER_TEXT


def test_the_b1_entrypoint_needs_no_fork_for_k0() -> None:
    """The K0 B1 gate is the existing B1 shim plus one env var -- no new runner,
    so there is no second copy to drift."""
    shim = B1_SHIM.read_text(encoding="utf-8")
    assert "FR13_GDN_GATE_MODE=hydra27_fixed32" in shim
    assert "FR13_GDN_GATE_BATCH=1" in shim
    assert "exec bash" in shim
    assert "FR13_GDN_GATE_DRAFT_VOCAB_PROFILE" not in shim, (
        "the shim must not pin a profile; the caller declares it"
    )


def test_both_scripts_parse() -> None:
    for script in (LAUNCHER, RUNNER, B1_SHIM):
        assert (
            subprocess.run(
                ["bash", "-n", str(script)], capture_output=True, check=False
            ).returncode
            == 0
        ), f"{script.name} does not parse"


# --------------------------------------------------------------------------
# the fourth site: the in-container patcher's env contract
# --------------------------------------------------------------------------

PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
PATCHER_TEXT = PATCHER.read_text(encoding="utf-8")


def test_the_patcher_gate_contract_no_longer_hardcodes_k64() -> None:
    """The fourth site, found by a 70-second boot refusal rather than by reading.

    The launcher clause, the gate runner and the credential validator all carried
    the declared identity, and the gate STILL refused -- inside the container, at
    the patcher's own env contract, which pinned ROOT=1/K=65536 for every
    ordered-GDN candidate. Every layer of this stack checks the identity
    independently, which is why re-pointing one is never finished until a real
    boot says so.
    """
    # Both single_launch contracts -- the GATE's and the PRODUCTION arm's -- are
    # separate sites and both had to move. Other levers' K64 pin sets
    # (gqa_group3, SFWD fusion, the draft-head U8 bridge) are deliberately left
    # alone: they are correct for their own arms.
    for marker in ("exact_single_launch = {", "exact_single_launch_production = {"):
        idx = PATCHER_TEXT.index(marker)
        block = PATCHER_TEXT[idx : idx + 400]
        assert '"FR13_DRAFT_VOCAB_K": "65536"' not in block, (
            f"{marker} still hardcodes K64"
        )
        assert "**_" in block, f"{marker} does not splice a declared profile"
    assert "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE" in PATCHER_TEXT
    assert "FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE" in PATCHER_TEXT


def test_the_patcher_gate_contract_offers_both_profiles_and_refuses_others() -> None:
    assert '"k64_root": {"FR13_DRAFT_VOCAB_ROOT": "1", "FR13_DRAFT_VOCAB_K": "65536"}' in (
        PATCHER_TEXT
    )
    assert '"full_vocab": {"FR13_DRAFT_VOCAB_ROOT": "0", "FR13_DRAFT_VOCAB_K": "0"}' in (
        PATCHER_TEXT
    )
    assert "must be \\nk64_root or full_vocab" in PATCHER_TEXT.replace(
        '"\n                f"', "\\n"
    ) or "k64_root or full_vocab" in PATCHER_TEXT


def test_the_patcher_gate_contract_defaults_to_k64_root() -> None:
    """Banked ordered-GDN gates must be unaffected."""
    assert (
        'os.environ.get(\n            "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE", "k64_root"\n        )'
        in PATCHER_TEXT
    )


def test_all_four_sites_are_covered() -> None:
    """One list, so the next person can see the whole train at once."""
    sites = {
        "launcher gate clause": "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE"
        in LAUNCHER_TEXT,
        "shared gate runner": "FR13_GDN_GATE_DRAFT_VOCAB_PROFILE" in RUNNER_TEXT,
        "credential validator": "DRAFT_VOCAB_PROFILES"
        in (REPO / "scripts" / "fr13_gdn_single_launch_production_credential.py").read_text(
            encoding="utf-8"
        ),
        "in-container patcher": "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE"
        in PATCHER_TEXT,
    }
    missing = [name for name, ok in sites.items() if not ok]
    assert not missing, f"draft-vocabulary profile missing from: {missing}"

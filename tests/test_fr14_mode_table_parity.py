"""Every place that decides "which fixed32 profile is this?" must know all of them.

The same question is asked in three kinds of place, and each has now refused a
serve on its own: the consumers (rounds <=12), the vehicle dispatch (round 13),
and the launcher's in-container preflight (round 14).

Round 14 taught the sharper half: fixing a mode TABLE is not enough. Three lines
below the table, the same block compared the dispatched tree against
`topology.FIXED32_CHOICES` unconditionally -- and hydra31's tree genuinely
differs (ancestry 90873d81 -> 5b33c46a), so teaching only the table would have
moved the refusal three lines down. A third profile-varying comparison, the walk
cap, sat nine lines further on.

`scripts/fr14_mode_table_parity.py` answers both halves from
`fr13_fixed32_topology.PROFILES`, so it fails on the NEXT profile rather than on
the next boot.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import fr13_fixed32_topology as topo  # noqa: E402
import fr14_mode_table_parity as parity  # noqa: E402

LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
VEHICLE = REPO / "scripts" / "fr13_bigdenom_swe_serve_variant.sh"


def _tmp(text):
    p = Path(tempfile.mkdtemp()) / "x.sh"
    p.write_text(text)
    return p


# ---------------------------------------------------------------------------
# The inventory at HEAD.
# ---------------------------------------------------------------------------

def test_every_site_admits_every_profile():
    stale = parity.sweep()
    assert not stale, "\n".join(f"{r['kind']}: {r['detail']}" for r in stale)


def test_the_scan_covers_all_four_shell_sites():
    for rel in parity.SHELL_SITES:
        assert (REPO / rel).exists(), f"{rel} moved"
    # the preflight block is mirrored in three launchers -- all must be scanned
    mirrored = [
        rel for rel in parity.SHELL_SITES
        if "unsupported fixed32 mode" in (REPO / rel).read_text()
    ]
    assert len(mirrored) == 3, mirrored


def test_profile_varying_and_invariant_sets_do_not_overlap():
    assert not (parity.PROFILE_VARYING & parity.PROFILE_INVARIANT)
    # and the names really do differ / agree between profiles
    a = topo.profile(topo.PROFILE_HYDRA27)
    b = topo.profile(topo.PROFILE_HYDRA31)
    assert a["walk_cap"] != b["walk_cap"]
    assert tuple(a["choices"]) != tuple(b["choices"])
    assert topo.PHYSICAL_DRAFTS == 31 and topo.PHYSICAL_ROWS == 32


# ---------------------------------------------------------------------------
# Mutation coverage: the detector must fire on each real round-14-class defect.
# ---------------------------------------------------------------------------

def test_detector_catches_a_mode_table_missing_a_profile():
    broken = LAUNCHER.read_text().replace(
        '    "hydra31_fixed32": (\n'
        "        topology.HYDRA31_VALID_MASK,\n"
        "        topology.HYDRA31_ACTIVE_DRAFTS,\n"
        "    ),\n",
        "",
    )
    assert broken != LAUNCHER.read_text(), "anchor moved"
    hits = parity.scan_mode_tables(_tmp(broken))
    assert hits and "hydra31_fixed32" in hits[0]


def test_detector_catches_the_unconditional_tree_comparison():
    """THE round-14 second refusal -- found by reading, not by booting."""
    broken = LAUNCHER.read_text().replace(
        'if tree != tuple(_profile["choices"]):',
        "if tree != topology.FIXED32_CHOICES:",
    )
    assert broken != LAUNCHER.read_text(), "anchor moved"
    hits = parity.scan_unconditional_comparisons(_tmp(broken))
    assert hits and "FIXED32_CHOICES" in hits[0]


def test_detector_catches_the_unconditional_walk_cap_comparison():
    """The third profile-varying compare in the same block."""
    broken = LAUNCHER.read_text().replace(
        'or walk_cap != int(_profile["walk_cap"])',
        "or walk_cap != topology.WALK_CAP",
    )
    assert broken != LAUNCHER.read_text(), "anchor moved"
    hits = parity.scan_unconditional_comparisons(_tmp(broken))
    assert hits and "WALK_CAP" in hits[0]


@pytest.mark.parametrize(
    "path,old,new",
    [
        ("launcher", '""|tail6_fixed32|hydra27_fixed32|hydra31_fixed32)',
         '""|tail6_fixed32|hydra27_fixed32)'),
        ("vehicle", '|| "$KIND" == "hydra31_fixed32"', ""),
    ],
)
def test_detector_catches_a_whitelist_dropping_a_profile(path, old, new):
    src = (LAUNCHER if path == "launcher" else VEHICLE).read_text()
    broken = src.replace(old, new, 1)
    assert broken != src, "anchor moved"
    hits = parity.scan_shell_whitelists(_tmp(broken))
    assert hits and "hydra31_fixed32" in hits[0]


def test_detector_leaves_single_mode_lever_preconditions_alone():
    """hydra27-qualified levers legitimately refuse hydra31; not a whitelist."""
    probe = _tmp(
        'if [[ "$FIXED32_MODE" != "hydra27_fixed32" ]]; then exit 2; fi\n'
    )
    assert parity.scan_shell_whitelists(probe) == []


# ---------------------------------------------------------------------------
# Executed: the real preflight accepts every profile and refuses the traps.
# ---------------------------------------------------------------------------

def _preflight_source():
    text = LAUNCHER.read_text()
    start = text.index("import ast\nimport json\nimport os\nimport sys")
    return text[start: text.index("\nPY\n", start)]


def _env_for(mode, tree_mode=None, walk_cap=None):
    import json as _json

    prof = topo.profile(
        topo.PROFILE_HYDRA31 if mode == "hydra31_fixed32"
        else topo.PROFILE_HYDRA27
    )
    tree_prof = topo.profile(
        topo.PROFILE_HYDRA31 if (tree_mode or mode) == "hydra31_fixed32"
        else topo.PROFILE_HYDRA27
    )
    mask = {
        "tail6_fixed32": topo.TAIL6_VALID_MASK,
        "hydra27_fixed32": topo.HYDRA27_VALID_MASK,
        "hydra31_fixed32": topo.HYDRA31_VALID_MASK,
    }[mode]
    active = {
        "tail6_fixed32": topo.TAIL6_ACTIVE_DRAFTS,
        "hydra27_fixed32": topo.HYDRA27_ACTIVE_DRAFTS,
        "hydra31_fixed32": topo.HYDRA31_ACTIVE_DRAFTS,
    }[mode]
    tree = repr(list(tree_prof["choices"]))
    return {
        "FR13_FIXED32_MODE": mode,
        "FR13_FIXED32_VALID_MASK": f"{mask:#x}",
        "FR13_FIXED32_ACTIVE_NODES": str(active),
        "FR13_FIXED32_PHYSICAL_DRAFTS": str(topo.PHYSICAL_DRAFTS),
        "FR13_FIXED32_TAW_WALK_CAP": str(
            walk_cap if walk_cap is not None else prof["walk_cap"]
        ),
        "NUM_SPECULATIVE_TOKENS": str(topo.PHYSICAL_DRAFTS),
        "TREE": tree,
        "SPEC_CONFIG": _json.dumps({
            "method": "qwen3_5_mtp",
            "num_speculative_tokens": topo.PHYSICAL_DRAFTS,
            "speculative_token_tree": tree,
        }),
    }


def _run_preflight(env):
    import os

    py = REPO / ".venv" / "bin" / "python"
    if not py.exists():
        pytest.skip("preflight needs .venv/bin/python")
    src = Path(tempfile.mkdtemp()) / "preflight.py"
    src.write_text(_preflight_source())
    full = dict(os.environ)
    full.update(env)
    return subprocess.run(
        [str(py), str(src)], capture_output=True, text=True, cwd=REPO, env=full
    )


@pytest.mark.parametrize(
    "mode", ["tail6_fixed32", "hydra27_fixed32", "hydra31_fixed32"]
)
def test_preflight_accepts_every_mode(mode):
    r = _run_preflight(_env_for(mode))
    assert r.returncode == 0, r.stdout + r.stderr


def test_preflight_refuses_hydra31_carrying_hydra27s_tree():
    """The exact trap: right mask, wrong tree. It would otherwise boot."""
    r = _run_preflight(_env_for("hydra31_fixed32", tree_mode="hydra27_fixed32"))
    assert r.returncode != 0
    assert "TREE differs" in r.stdout + r.stderr


def test_preflight_refuses_hydra31_with_the_hydra27_walk_cap():
    r = _run_preflight(_env_for("hydra31_fixed32", walk_cap=12))
    assert r.returncode != 0
    assert "shape mismatch" in r.stdout + r.stderr


def test_preflight_refuses_an_unknown_mode():
    env = _env_for("hydra27_fixed32")
    env["FR13_FIXED32_MODE"] = "hydra99_fixed32"
    r = _run_preflight(env)
    assert r.returncode != 0
    assert "unsupported fixed32 mode" in r.stdout + r.stderr


def test_vehicle_supplies_the_walk_cap_the_preflight_demands():
    """The arm has to carry it: the runner's env says 12 for every arm."""
    text = VEHICLE.read_text()
    for mode in sorted(topo.PROFILES):
        start = text.index(f"\n  {mode})\n")
        block = text[start: text.index("\n    ;;", start)]
        assert "FR13_FIXED32_TAW_WALK_CAP=$" in block, (
            f"{mode} kind block does not export a derived walk cap"
        )


# ---------------------------------------------------------------------------
# LAUNCHER-FAMILY PARITY.
#
# "both launcher families" was wrong by one for six rounds. fr14_leg3 is a live
# serving path (arm B's profile-chain legs) and carried none of the FR14 work --
# including the PROMOTED fused-topk default, so it would have served the unfused
# kernel silently while the other two served the promoted one.
#
# The roster now lives in exactly one place and every detector imports it.
# ---------------------------------------------------------------------------

def test_the_roster_is_three_and_is_the_only_enumeration():
    assert len(parity.LAUNCHER_FAMILIES) == 3
    for rel in parity.LAUNCHER_FAMILIES:
        assert (REPO / rel).exists(), rel
    # nothing else may re-enumerate the families by hand
    for rel in (
        "scripts/fr14_paired_contract_sweep.py",
        "tests/test_fr14_suffix_pass_gate.py",
        "tests/test_fr14_splitk_arm_launcher_wiring.py",
    ):
        text = (REPO / rel).read_text()
        assert "LAUNCHER_FAMILIES" in text, (
            f"{rel} does not import the roster"
        )
        assert '"scripts/fr14_armb_leg3_launch_nomiddleware.sh",\n)' not in text


def test_family_parity_is_clean_at_head():
    assert parity.scan_family_parity() == []


@pytest.mark.parametrize("marker", parity.FAMILY_PARITY_MARKERS)
def test_every_parity_marker_is_present_in_every_family(marker):
    for rel in parity.LAUNCHER_FAMILIES:
        assert marker in (REPO / rel).read_text(), (
            f"{marker} missing from {Path(rel).name}"
        )


@pytest.mark.parametrize(
    "marker",
    ["FR14_FUSED_DRAFT_TOPK", "_fr14_gate_incompat", "FR14_GATE_SPLIT_GRAPH"],
)
def test_detector_fires_when_one_family_omits_a_marker(monkeypatch, marker):
    """Exactly the defect that happened: two families promoted, one not."""
    third = parity.LAUNCHER_FAMILIES[2]
    stripped = (REPO / third).read_text().replace(marker, "FR14_REMOVED")
    tmp = Path(tempfile.mkdtemp())
    (tmp / "scripts").mkdir()
    for rel in parity.LAUNCHER_FAMILIES:
        target = tmp / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            stripped if rel == third else (REPO / rel).read_text()
        )
    monkeypatch.setattr(parity, "REPO", tmp)
    bad = parity.scan_family_parity()
    assert bad and any(marker in b for b in bad), bad
    assert any("fr14_leg3" in b for b in bad), bad


def test_the_promoted_default_is_ON_in_every_family():
    """A promoted default that is ON in two families and absent in a third is
    not promoted -- it is a silent A/B."""
    for rel in parity.LAUNCHER_FAMILIES:
        text = (REPO / rel).read_text()
        assert '-e FR14_FUSED_DRAFT_TOPK="${FR14_FUSED_DRAFT_TOPK:-1}"' in text
        assert 'case "${FR14_FUSED_DRAFT_TOPK:-1}" in' in text
        assert "PROMOTED-ON but its pinned .so is missing" in text


def test_the_refused_gate_is_guarded_in_every_family():
    """The gate is refused-final; the guards must still exist so a stale env
    cannot arm it in a family that never learned about it."""
    for rel in parity.LAUNCHER_FAMILIES:
        text = (REPO / rel).read_text()
        assert '-e FR14_SUFFIX_PASS_GATE="${FR14_SUFFIX_PASS_GATE:-0}"' in text
        assert 'echo "FR14_SUFFIX_PASS_GATE must be 0 or 1" >&2; exit 2' in text
        assert (
            "FR14_SUFFIX_PASS_GATE=1 requires FR13_TAIL_MODE=1 and "
            "FR13_DRAFT_SOURCE=merged" in text
        )
        assert 'grep -q "FR14_GATE_SPLIT_GRAPH"' in text

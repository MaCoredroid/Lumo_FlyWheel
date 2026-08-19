"""Every topology profile must have a dispatch kind in the serve VEHICLE.

Round 13 refused pre-boot at zero GPU: stage 2 taught seven files the hydra31
profile and did not teach `fr13_bigdenom_swe_serve_variant.sh`, which is what
actually launches an arm. External env could not bypass it either -- the kind
block hardcodes `FR13_FIXED32_MODE` into `XFLAGS`, which the vehicle exports
AFTER the caller's environment.

That was the fifth instance of one shape: consumers taught, selector not. This
test closes it by construction. It is a PURE SOURCE test -- no GPU, no serve --
and it fails on the NEXT profile anyone adds to `fr13_fixed32_topology.PROFILES`
without giving the vehicle a kind for it.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fr13_fixed32_topology as topo  # noqa: E402

VEHICLE = SCRIPTS / "fr13_bigdenom_swe_serve_variant.sh"


def _kind_block(kind: str) -> str:
    """The `case` arm for one KIND, as source text."""
    text = VEHICLE.read_text()
    marker = f"\n  {kind})\n"
    if marker not in text:
        pytest.fail(f"the vehicle has no dispatch kind for {kind!r}")
    start = text.index(marker)
    end = text.index("\n    ;;", start)
    return text[start:end]


def _exported(block: str, name: str) -> str:
    m = re.search(rf'{name}=(\$?[\w./"$]+)', block)
    assert m, f"{name} not exported by the kind block"
    return m.group(1).strip('"')


# ---------------------------------------------------------------------------
# THE PARITY DETECTOR
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", sorted(topo.PROFILES))
def test_every_profile_has_a_vehicle_kind(mode):
    """A profile the vehicle cannot launch is a profile that does not exist."""
    block = _kind_block(mode)
    assert f"FR13_FIXED32_MODE={mode}" in block, (
        f"{mode} kind block does not export FR13_FIXED32_MODE={mode}"
    )
    assert f"FIXED32_MODE={mode}" in block


@pytest.mark.parametrize("mode", sorted(topo.PROFILES))
def test_vehicle_kind_derives_its_constants(mode):
    """Mask/nodes/drafts must come from shell vars, never be retyped literals."""
    block = _kind_block(mode)
    for field in (
        "FR13_FIXED32_VALID_MASK",
        "FR13_FIXED32_ACTIVE_NODES",
        "FR13_FIXED32_PHYSICAL_DRAFTS",
    ):
        value = _exported(block, field)
        assert value.startswith("$"), (
            f"{mode}: {field}={value} is a hand-copied literal; it must come "
            "from the topology authority block"
        )


@pytest.mark.parametrize("mode", sorted(topo.PROFILES))
def test_vehicle_kind_passes_its_OWN_tree(mode):
    """hydra31 is a different TREE, not hydra27 with a wider mask.

    14 of 31 draft ids carry different paths. Passing the wrong tree under the
    right mask arms four rank-2 side branches as if they were spine -- and it
    boots.
    """
    block = _kind_block(mode)
    treearg = _exported(block, "TREEARG")
    assert treearg.startswith("$"), f"{mode}: TREEARG must be a derived variable"
    if mode == topo.PROFILE_HYDRA31:
        assert "HYDRA31" in treearg, (
            f"{mode} must not reuse hydra27's tree: TREEARG={treearg}"
        )


def test_the_authority_block_emits_one_field_per_consumer():
    """The python authority and the bash unpack must agree on the field count."""
    text = VEHICLE.read_text()
    block = text[text.index("mapfile -t _FIXED32_CONTRACT"):]
    block = block[: block.index("unset _FIXED32_CONTRACT")]
    emitted = block.count("\nprint(")
    m = re.search(r"\$\{#_FIXED32_CONTRACT\[@\]\} == (\d+)", block)
    assert m, "the field-count guard is gone"
    assert int(m.group(1)) == emitted, (
        f"authority emits {emitted} fields, guard expects {m.group(1)}"
    )
    indices = {int(i) for i in re.findall(r"_FIXED32_CONTRACT\[(\d+)\]", block)}
    assert indices == set(range(emitted)), (
        f"unpacked indices {sorted(indices)} do not cover 0..{emitted - 1}"
    )


def test_the_authority_validates_every_contract_before_emitting():
    text = VEHICLE.read_text()
    block = text[text.index("mapfile -t _FIXED32_CONTRACT"):]
    block = block[: block.index("unset _FIXED32_CONTRACT")]
    for check in (
        "validate_contract()",
        "validate_gate_contract()",
        "validate_tail10_contract()",
    ):
        assert check in block, f"authority does not call {check}"


# ---------------------------------------------------------------------------
# Executed, not asserted: run the vehicle's own dispatch.
# ---------------------------------------------------------------------------

def _run_dispatch(kind):
    text = VEHICLE.read_text().split("\n")
    i = next(k for k, l in enumerate(text)
             if l.startswith("mapfile -t _FIXED32_CONTRACT"))
    j = next(k for k, l in enumerate(text[i:], i)
             if "hydra31 contract drifted" in l) + 2
    k0 = next(k for k, l in enumerate(text) if l.strip() == 'case "$KIND" in')
    k1 = next(k for k, l in enumerate(text[k0:], k0) if "unknown KIND" in l) + 1
    script = "\n".join(
        ["set -uo pipefail", *text[i:j], 'KIND=${1:?}', *text[k0:k1], "esac",
         'echo "MODE=$FIXED32_MODE"',
         'for kv in "${XFLAGS[@]:-}"; do echo "X:$kv"; done',
         'echo "TREE:$TREEARG"']
    )
    import tempfile
    p = Path(tempfile.mkdtemp()) / "d.sh"
    p.write_text(script)
    return subprocess.run(["bash", str(p), kind], capture_output=True,
                          text=True, cwd=REPO)


@pytest.mark.parametrize("mode", sorted(topo.PROFILES))
def test_dispatch_exports_match_the_profile(mode):
    if not (REPO / ".venv" / "bin" / "python").exists():
        pytest.skip("vehicle needs .venv/bin/python")
    r = _run_dispatch(mode)
    assert r.returncode == 0, r.stdout + r.stderr
    p = topo.profile(mode)
    out = r.stdout
    assert f"X:FR13_FIXED32_MODE={mode}" in out
    assert f"X:FR13_FIXED32_VALID_MASK={p['valid_mask']:#x}" in out
    assert f"X:FR13_FIXED32_ACTIVE_NODES={p['active_drafts']}" in out
    assert f"X:FR13_FIXED32_PHYSICAL_DRAFTS={topo.PHYSICAL_DRAFTS}" in out
    tree = ast.literal_eval(
        out.split("TREE:", 1)[1].strip().splitlines()[0]
    )
    assert [tuple(x) for x in tree] == list(p["choices"]), (
        f"{mode} dispatch passes the wrong tree"
    )


def test_the_two_profiles_dispatch_different_trees():
    if not (REPO / ".venv" / "bin" / "python").exists():
        pytest.skip("vehicle needs .venv/bin/python")
    trees = {}
    for mode in (topo.PROFILE_HYDRA27, topo.PROFILE_HYDRA31):
        out = _run_dispatch(mode).stdout
        trees[mode] = out.split("TREE:", 1)[1].strip().splitlines()[0]
    assert trees[topo.PROFILE_HYDRA27] != trees[topo.PROFILE_HYDRA31]


def test_unknown_kind_is_still_refused():
    if not (REPO / ".venv" / "bin" / "python").exists():
        pytest.skip("vehicle needs .venv/bin/python")
    r = _run_dispatch("hydra99_fixed32")
    assert "unknown KIND" in r.stdout + r.stderr


# ---------------------------------------------------------------------------
# The generic fixed32 preconditions must admit every profile.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", sorted(topo.PROFILES))
def test_generic_fixed32_gates_admit_every_profile(mode):
    """OFFLOAD_AGENT, private arm dirs and the layer-batch gate are generic.

    A profile the vehicle dispatches but then refuses at a generic fixed32 gate
    is refused just as hard as one with no kind at all.
    """
    text = VEHICLE.read_text()
    generic = [
        line for line in text.split("\n")
        if '"$KIND" == "tail6_fixed32"' in line
    ]
    assert generic, "the generic fixed32 whitelists moved"
    for line in generic:
        assert f'"$KIND" == "{mode}"' in line, (
            f"{mode} missing from a generic fixed32 gate: {line.strip()[:90]}"
        )


# ---------------------------------------------------------------------------
# The detector must be able to FAIL. A parity test that cannot is worse than
# none, because it reads like coverage.
# ---------------------------------------------------------------------------

def test_detector_fails_on_a_profile_the_vehicle_does_not_know(monkeypatch):
    """Simulate the NEXT profile: registered in topology, absent from the vehicle."""
    fake = dict(topo.PROFILES)
    fake["hydra99_fixed32"] = dict(topo.PROFILES[topo.PROFILE_HYDRA31])
    monkeypatch.setattr(topo, "PROFILES", fake)
    # pytest.fail raises Failed, which derives from BaseException, not Exception
    with pytest.raises(BaseException):
        test_every_profile_has_a_vehicle_kind("hydra99_fixed32")


def test_detector_fails_on_a_hand_copied_constant(monkeypatch, tmp_path=None):
    """Re-literalise the mask in the hydra31 block and the detector must catch it."""
    import tempfile

    src = VEHICLE.read_text()
    broken = src.replace(
        '"FR13_FIXED32_VALID_MASK=$FIXED32_HYDRA31_MASK"',
        "FR13_FIXED32_VALID_MASK=0x7fffffff",
    )
    assert broken != src, "anchor moved"
    p = Path(tempfile.mkdtemp()) / "vehicle.sh"
    p.write_text(broken)
    monkeypatch.setattr(sys.modules[__name__], "VEHICLE", p)
    with pytest.raises(Exception):
        test_vehicle_kind_derives_its_constants(topo.PROFILE_HYDRA31)


def test_detector_fails_when_a_kind_reuses_the_wrong_tree(monkeypatch):
    """The exact bug the hydra31 block was written to avoid."""
    import tempfile

    src = VEHICLE.read_text()
    broken = src.replace(
        'TREEARG="$FIXED32_HYDRA31_TREE"', 'TREEARG="$FIXED32_TREE"'
    )
    assert broken != src, "anchor moved"
    p = Path(tempfile.mkdtemp()) / "vehicle.sh"
    p.write_text(broken)
    monkeypatch.setattr(sys.modules[__name__], "VEHICLE", p)
    with pytest.raises(Exception):
        test_vehicle_kind_passes_its_OWN_tree(topo.PROFILE_HYDRA31)


def test_detector_fails_when_a_generic_gate_forgets_a_profile(monkeypatch):
    import tempfile

    src = VEHICLE.read_text()
    broken = src.replace(
        '|| "$KIND" == "hydra31_fixed32"', "", 1
    )
    assert broken != src, "anchor moved"
    p = Path(tempfile.mkdtemp()) / "vehicle.sh"
    p.write_text(broken)
    monkeypatch.setattr(sys.modules[__name__], "VEHICLE", p)
    with pytest.raises(Exception):
        test_generic_fixed32_gates_admit_every_profile(topo.PROFILE_HYDRA31)

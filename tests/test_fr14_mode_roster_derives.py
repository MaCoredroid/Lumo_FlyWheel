"""THE FOURTH KIND: a membership roster.

Boot seven reached HEALTH -- the first hydra31 boot ever, 306s, all nine
engine-side pins dead -- and then died HOST-SIDE on a list of who is allowed:

    fr13_fixed32_flush_protocol.py:465 FlushConfigurationError:
      mode must be one of ['hydra27_fixed32', 'tail6_fixed32'],
      got 'hydra31_fixed32'

It killed the generation-zero ready ack AND the terminal flush, so a stale
roster blocks serve and teardown alike.

A ROSTER IS NOT A VALUE, A CONTRACT OR AN AGGREGATE. It is the MEMBERSHIP
version of the same defect and it takes the same cure: admissibility derives
from the authority instead of being written down. A roster that could have
derived is a defect while it is still correct -- it is simply a defect whose
boot has not come yet.

THE AUTHORITY ALREADY EXISTED AND SAID SO. fr13_fixed32_contract.FIXED32_MODES
carries this comment above it: "The serving mode vocabulary. Consumers validate
an arm's mode against this rather than hardcoding a list of their own." Three
consumers hardcoded lists of their own anyway.

THE DISTINCTION THAT MATTERS, and it cost me a wrong edit before I checked:
a two-element tuple of mode names is not always a roster. fr13_floor_gate's
`expected_modes` READS like one and is actually a PAIR SHAPE -- it is unpacked
as `tail_mode, hydra_mode` and every exact_keys call demands exactly those two
report keys. Widening it raises ValueError before it admits anything. Verify
the use before deriving the list.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fr13_fixed32_contract as contract  # noqa: E402
import fr13_fixed32_flush_protocol as flush  # noqa: E402
import fr13_fixed32_topology as topology  # noqa: E402


def test_the_three_rosters_agree_with_the_authority() -> None:
    """One vocabulary, three holders, no drift."""
    assert set(flush.FIXED32_MODES) == set(contract.FIXED32_MODES)
    assert set(contract.FIXED32_MODES) == set(topology.SERVING_MODES)
    assert "hydra31_fixed32" in flush.FIXED32_MODES


def test_the_flush_protocol_admits_every_registered_mode() -> None:
    """Both halves of what boot seven died on: the ready ack and the flush."""
    for mode in sorted(topology.SERVING_MODES):
        record = flush.ready_ack(mode=mode, producer_pid=4312)
        assert record["mode"] == mode
        assert record["schema"] == flush.ACK_SCHEMA


def test_the_flush_protocol_still_refuses_an_unregistered_mode() -> None:
    """The roster keeps its teeth; it just stopped being written down."""
    with pytest.raises(flush.FlushConfigurationError) as refusal:
        flush.ready_ack(mode="hydra99_fixed32", producer_pid=4312)
    assert "unsupported fixed32 mode" in str(refusal.value)


def test_the_flush_mirror_matches_the_authority() -> None:
    """The module is stdlib-only by design, so it carries a fallback mirror.

    The mirror is only safe while something compares it.
    """
    assert set(flush._FIXED32_MODES_MIRROR) == set(topology.SERVING_MODES)


def test_the_campaign_driver_cli_derives_its_choices() -> None:
    """argparse would have rejected --fixed32-mode hydra31_fixed32 outright."""
    source = (SCRIPTS / "run_swe_bench_q36_a.py").read_text(encoding="utf-8")
    assert 'choices=("tail6_fixed32", "hydra27_fixed32"),' not in source
    assert "choices=fixed32_contract.FIXED32_MODES," in source


def test_the_floor_gate_pair_shape_is_classified_not_widened() -> None:
    """The wrong edit I nearly landed, recorded so nobody re-makes it."""
    source = (SCRIPTS / "fr13_floor_gate.py").read_text(encoding="utf-8")
    assert "NOT A ROSTER -- a PAIR SHAPE" in source
    assert 'expected_modes = ("tail6_fixed32", "hydra27_fixed32")' in source
    # the evidence: it is unpacked as a pair
    assert "tail_mode, hydra_mode = expected_modes" in source


#: Rosters that are single-mode BY DECLARATION, each with its adjudication in
#: the source. These refuse hydra31 on purpose: a default-off candidate
#: qualified on hydra27's tree must re-qualify, and the refusal IS the answer.
DECLARED_SINGLE_MODE_ROSTERS = {
    "src/lumo_flywheel_serving/fr13_fixed32_commit_slot_scatter.py": (
        "round-18 adjudication: hydra31 arms four more drafts and a deeper "
        "spine, so the scatter plan is a different plan"
    ),
    "src/lumo_flywheel_serving/fr13_gdn_gqa_group3.py": (
        "the grouping was qualified against hydra27's subtree decomposition; "
        "hydra31's second level runs 11 rows deep instead of 7"
    ),
    "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion.py": (
        "default-off fusion candidate, byte-AB qualified on hydra27's tree"
    ),
    "src/lumo_flywheel_serving/fr13_host_tail_prep.py": (
        "position literals BAKED from one tree; hydra31 needs its own bake"
    ),
}


def test_every_declared_single_mode_roster_states_its_reason() -> None:
    """A roster that refuses on purpose must say so where it refuses."""
    for rel, reason in DECLARED_SINGLE_MODE_ROSTERS.items():
        source = (REPO / rel).read_text(encoding="utf-8")
        assert 'FIXED32_MODES = frozenset(("tail6_fixed32", "hydra27_fixed32"))' in (
            source
        ) or '_FIXED32_MODES = ("hydra27_fixed32", "tail6_fixed32")' in source, rel
        assert len(reason) > 40, rel
        # the adjudication is written next to the roster, not in a ledger
        assert "hydra31" in source, rel


def test_no_unclassified_roster_survives_in_the_serve_closure() -> None:
    """THE STANDING CHECK. A new two-mode roster in the closure fails here.

    Scope: files in the serve execution closure that name hydra27 and never
    hydra31, carrying a container of two or more mode names. Every one must be
    either fixed (it now derives) or classified as declared-single-mode/pair.
    """
    import fr14_mode_table_parity as parity

    closure = set(parity.serve_execution_closure())
    with_h27 = set(
        subprocess.run(
            ["grep", "-rl", "hydra27_fixed32", "scripts/", "src/"],
            capture_output=True,
            text=True,
            cwd=REPO,
        ).stdout.split()
    )
    with_h31 = set(
        subprocess.run(
            ["grep", "-rl", "hydra31_fixed32", "scripts/", "src/"],
            capture_output=True,
            text=True,
            cwd=REPO,
        ).stdout.split()
    )
    known_modes = {"hydra27_fixed32", "tail6_fixed32"}
    unclassified: dict[str, list[int]] = {}
    for rel in sorted((with_h27 - with_h31) & closure):
        if rel in DECLARED_SINGLE_MODE_ROSTERS:
            continue
        path = REPO / rel
        if not path.is_file() or not rel.endswith(".py"):
            continue
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        lines = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Set, ast.Tuple, ast.List, ast.Call)):
                continue
            values = {
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            }
            if len(values & known_modes) >= 2:
                lines.append(node.lineno)
        if lines:
            unclassified[rel] = sorted(set(lines))
    # Everything left is a hydra27-era gate, credential or reduce that never
    # runs on a hydra31 serve; the ones that DO run are fixed or classified.
    allowed = {
        "scripts/fr13_b4_floor_gate_reduce.py",
        "scripts/fr13_b4_gqa_width4_pair_reduce.py",
        "scripts/fr13_b4_taw_width4_pair_reduce.py",
        "scripts/fr13_cfwd_logit_direct_decision_kernel.py",
        "scripts/fr13_cfwd_packed_walk_active_depth_kernel.py",
        "scripts/fr13_cfwd_packed_walk_node_trust_kernel.py",
        "scripts/fr13_cutlass_b4_pass.py",
        "scripts/fr13_cutlass_wave_binary.py",
        "scripts/fr13_depth_acceptance.py",
        "scripts/fr13_fa2_qrow32_gate.py",
        "scripts/fr13_fa2_qrow32_gqa_pair_gate.py",
        "scripts/fr13_fixed32_nsys_reduce.py",
        "scripts/fr13_floor_gate.py",
    }
    surprises = {k: v for k, v in unclassified.items() if k not in allowed}
    assert not surprises, (
        "unclassified two-mode rosters in the serve execution closure: "
        + repr(surprises)
    )

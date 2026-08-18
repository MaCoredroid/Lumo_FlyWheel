"""The hydra31_fixed32 (tail10) profile, and the ways it must NOT disturb hydra27.

tail10 respends the four slots hydra27 disarms as spine continuations 0^12..0^15.
Because choices sort by (len, path), that moves the paths of draft ids >= 17 --
the PHYSICAL TREE differs, not just the mask -- so hydra27 is left exactly as it
was and tail10 is added as a new named profile.

Stage 1 touches no serving path. These tests are what make that claim checkable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fr13_fixed32_topology as topo  # noqa: E402


def test_all_three_contracts_validate():
    topo.validate_contract()
    topo.validate_gate_contract()
    topo.validate_tail10_contract()


# --- hydra27 must be byte-identical -----------------------------------------

def test_hydra27_identity_is_untouched():
    assert topo.HYDRA27_VALID_MASK == 0x7ABDFFFF
    assert topo.HYDRA27_ACTIVE_DRAFTS == 27
    assert topo.HYDRA27_INACTIVE_DRAFT_IDS == (17, 22, 24, 26)
    assert topo.PHYSICAL_PARENT_SHA256 == (
        "7abd25e38323d6c088eb627785b5c190b2e878b0a710bb349e2d690852a06ddd"
    )
    assert topo.TREE_ANCESTRY_SHA256 == (
        "90873d81e83ce1644ee4701e043b7e9d26e83b7a7ca752d538a0e6eed1946dad"
    )
    # the module-level constants every existing consumer imports
    assert topo.ARCTIC_MAIN_TAIL_LENGTH == 6
    assert topo.ARCTIC_LOOKUP_TOKENS_PER_REQUEST == 12
    assert topo.RESCUE_CARRY_SLOTS_PER_REQUEST == 4
    assert topo.MAX_PHYSICAL_DEPTH == 11
    assert topo.WALK_CAP == 12


def test_the_two_profiles_are_distinct_artifacts():
    a = topo.profile(topo.PROFILE_HYDRA27)
    b = topo.profile(topo.PROFILE_HYDRA31)
    assert a["physical_parent_sha256"] != b["physical_parent_sha256"]
    assert a["tree_ancestry_sha256"] != b["tree_ancestry_sha256"]
    assert a["valid_mask"] != b["valid_mask"]


# --- what tail10 actually is -------------------------------------------------

def test_tail10_respends_exactly_the_four_disarmed_slots():
    added = [p for p in topo.TAIL10_CHOICES if p not in set(topo.FIXED32_CHOICES)]
    assert len(added) == len(topo.HYDRA27_INACTIVE_DRAFT_IDS) == 4
    assert sorted(len(p) for p in added) == [12, 13, 14, 15]
    assert all(set(p) == {0} for p in added), "they must be SPINE, not branches"
    dropped = [p for p in topo.FIXED32_CHOICES if p not in set(topo.TAIL10_CHOICES)]
    assert all(p[0] == 2 for p in dropped), "only deep rank-2 branches are spent"


def test_tail10_arms_every_physical_draft():
    assert topo.HYDRA31_ACTIVE_DRAFTS == topo.PHYSICAL_DRAFTS == 31
    assert topo.HYDRA31_VALID_MASK == (1 << 31) - 1
    assert topo.HYDRA31_INACTIVE_DRAFT_IDS == ()


def test_the_32_row_geometry_is_unchanged():
    """This is what keeps tail10 host-side: no kernel geometry moves."""
    assert len(topo.TAIL10_CHOICES) == topo.PHYSICAL_DRAFTS == 31
    assert topo.PHYSICAL_ROWS == 32
    children = {}
    for node, parent in enumerate(topo.TAIL10_DRAFT_PARENT):
        children.setdefault(parent, []).append(node)
    assert children[-1] == [0, 1, 2]
    assert max(len(v) for v in children.values()) == topo.SAMPLER_MAX_FANOUT == 3


def test_tail10_is_a_well_formed_tree():
    for node, (path, parent) in enumerate(
        zip(topo.TAIL10_CHOICES, topo.TAIL10_DRAFT_PARENT, strict=True)
    ):
        if len(path) > 1:
            assert parent < node, f"choice {node} is not parent-ordered"
            assert topo.TAIL10_CHOICES[parent] == path[:-1]


def test_ids_from_17_up_really_do_move():
    """The reason this cannot be an in-place edit to hydra27."""
    moved = [
        i for i in range(topo.PHYSICAL_DRAFTS)
        if topo.FIXED32_CHOICES[i] != topo.TAIL10_CHOICES[i]
    ]
    assert moved and min(moved) == 17, moved
    assert len(moved) == 14


# --- the committer must be able to walk it -----------------------------------

def test_tail10_sits_exactly_at_the_commit_path_capacity():
    assert topo.TAIL10_WALK_CAP == topo.TAIL10_MAX_PHYSICAL_DEPTH + 1 == 16
    assert topo.TAIL10_WALK_CAP == topo.COMMIT_PATH_CAP, (
        "tail10 is the deepest tail this committer can walk without widening "
        "COMMIT_PATH_CAP -- n=3+tail14 (depth 17, walk 18) does NOT fit"
    )


def test_the_sequenced_follow_on_would_not_fit():
    """Recorded so the sequencing decision is made with this in hand."""
    tail14_depth = topo.GATED_MTP_K + 14
    assert tail14_depth + 1 > topo.COMMIT_PATH_CAP, (
        "if this ever passes, COMMIT_PATH_CAP was widened and n=3+tail14 became "
        "a host-side change"
    )


# --- Arctic feed and the gate interplay --------------------------------------

def test_tail10_arctic_feed():
    p = topo.profile(topo.PROFILE_HYDRA31)
    assert p["main_tail_length"] == 10
    assert p["arctic_requested_tokens"] == 16          # 10 + rank1 4 + rank2 2
    assert p["rescue_carry_slots"] == 0                # physical == logical now
    assert sum(l for _r, l in p["physical_branch_chains"]) == 6


def test_pack_is_31_columns_in_both_gate_shapes():
    head = topo.N_MTP_HEAD_DEPTHS * (1 + topo.BRANCHES_PER_HEAD_DEPTH)
    rescue = sum(l for _r, l in topo.TAIL10_PHYSICAL_BRANCH_CHAINS)
    for hd, cols in (
        (topo.N_MTP_HEAD_DEPTHS, topo.TAIL10_ARCTIC_MAIN_TAIL_LENGTH),
        (topo.GATED_MTP_K, topo.TAIL10_GATED_ARCTIC_MAIN_TAIL_LENGTH),
    ):
        in_head = topo.N_MTP_HEAD_DEPTHS - hd
        assert head + (cols - in_head) + rescue == topo.PHYSICAL_DRAFTS


def test_gate_handoff_shapes_move_with_the_tail():
    assert topo.TAIL10_LEGAL_HANDOFF_SHAPES == ((4, 10), (2, 12))
    # and the ungated contract for hydra27 is untouched
    assert topo.LEGAL_HANDOFF_SHAPES == ((4, 6), (2, 8))


def test_profile_lookup_rejects_an_unknown_mode():
    with pytest.raises(KeyError):
        topo.profile("hydra99_fixed32")

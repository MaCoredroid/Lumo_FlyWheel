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


# ===========================================================================
# STAGE 2 -- the four consumers.
# ===========================================================================

import json  # noqa: E402

import fr13_fixed32_work_census as census  # noqa: E402
import fr13_merged_drafter as md  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def test_derived_subtree_schedule_reproduces_the_shipped_hydra27_literal():
    """The rule is only trustworthy for hydra31 if it rebuilds hydra27 exactly."""
    rederived = topo.derive_subtree_levels(topo.PHYSICAL_PARENT)
    assert rederived[0] == topo.SUBTREE_LEVELS[0]
    assert set(rederived[1]) == set(topo.SUBTREE_LEVELS[1])


def test_hydra31_gdn_schedule_is_well_formed():
    p = topo.profile(topo.PROFILE_HYDRA31)
    assert p["gdn_level_path_counts"] == (1, 11)
    assert p["gdn_level_max_lengths"] == (5, 11)
    assert sum(p["gdn_level_max_lengths"]) == p["walk_cap"] == 16
    covered = set()
    for level in p["subtree_levels"]:
        for path, _parent in level:
            covered |= set(path)
    assert covered == set(range(topo.PHYSICAL_ROWS))
    # hydra27's schedule is untouched
    assert topo.GDN_LEVEL_MAX_LENGTHS == (5, 7)


# --- consumer 1: decide_fixed32 ---------------------------------------------

def test_drafter_defaults_to_hydra27_without_a_sidecar():
    md._FIXED32_MODE_CACHE = None
    assert md.fixed32_mode() == "hydra27_fixed32"


def test_drafter_takes_every_width_from_the_profile():
    src = (REPO / "scripts" / "fr13_merged_drafter.py").read_text()
    body = src[src.index("def decide_fixed32"):src.index("def get_fixed32_drafter_last_work")]
    for gone in (
        "ARCTIC_MAIN_TAIL_LENGTH,",
        "GATED_ARCTIC_MAIN_TAIL_LENGTH,",
        "ARCTIC_LOOKUP_TOKENS_PER_REQUEST,",
    ):
        assert gone not in body, f"decide_fixed32 still imports {gone}"
    assert 'profile(fixed32_mode())' in body


# --- consumer 2: census ------------------------------------------------------

def test_census_knows_the_profile_and_maps_tail6_to_hydra27():
    assert "hydra31_fixed32" in census.MODE_SEMANTICS
    assert census.MODE_SEMANTICS["hydra31_fixed32"]["active_nodes"] == 31
    assert census.MODE_SEMANTICS["hydra31_fixed32"]["valid_mask"] == 0x7FFFFFFF
    # tail6 and hydra27 are the same physical tree
    assert (
        census.shape_profile("tail6_fixed32")["physical_parent_sha256"]
        == census.shape_profile("hydra27_fixed32")["physical_parent_sha256"]
    )
    assert (
        census.shape_profile("hydra31_fixed32")["physical_parent_sha256"]
        != census.shape_profile("hydra27_fixed32")["physical_parent_sha256"]
    )


def test_taw_call_table_is_derived_and_reproduces_the_shipped_literal():
    assert census.taw_tensor_call_census(12) == census.TAW_TENSOR_CALL_CENSUS
    at16 = census.taw_tensor_call_census(16)
    assert at16["walk_levels"] == 16
    assert at16["full_vocab_row_gathers"] == 32
    assert at16["full_vocab_normalizations"] == 48


def _banked_or_skip():
    p = (
        REPO / "output/fr14_b1_stock_20260817T054447Z/tail6_fixed32_b1radix"
        / "logs/fr13_fixed32_work_census.jsonl"
    )
    if not p.exists():
        pytest.skip("banked census not present")
    with p.open() as fh:
        ev = json.loads(fh.readline())
    # isolate lane 3's in-flight TAW pin re-attestation, which is not this lane's
    ev["taw"]["source_contract_sha256"] = census.TAW_SOURCE_CONTRACT_SHA256
    return ev


def _morph(ev, mode):
    p = topo.profile(mode)
    b = ev["batch_size"]
    rescue = sum(l for _r, l in p["physical_branch_chains"])
    e = json.loads(json.dumps(ev))
    e["mode"] = mode
    e["active_nodes"] = p["active_drafts"]
    e["valid_mask"] = p["valid_mask"]
    d = e["drafter"]
    d["main_tail_length"] = p["main_tail_length"]
    d["arctic_requested_tokens"] = p["arctic_requested_tokens"] * b
    d["carry_fill_slots"] = p["rescue_carry_slots"] * b
    rt = e["drafter_runtime"]
    rt["arctic_requested_tokens"] = p["arctic_requested_tokens"] * b
    rt["merge_fill_columns"] = p["main_tail_length"] + rescue
    rt["merge_fill_rows"] = rt["merge_fill_columns"] * b
    rt["rescue_carry_slots"] = p["rescue_carry_slots"] * b
    rt["physical_parent_sha256"] = p["physical_parent_sha256"]
    rt["arctic_ledger"] = [
        dict(r, tokens=p["main_tail_length"] * b) if r["kind"] == "main" else r
        for r in rt["arctic_ledger"]
    ]
    # SITE 13's neighbour: the tree-attention digests IDENTIFY THE TREE. The
    # validator used to compare them against the module scalars, so a hydra31
    # event carrying hydra27's bias digest passed -- the one field that
    # distinguishes the profiles was the one field nothing checked per mode.
    ta = e["tree_attn"]
    ta["physical_parent_digest"] = p["physical_parent_sha256"]
    ta["bias_digest"] = p["tree_ancestry_sha256"]
    g = e["gdn"]
    g["critical_path"] = p["walk_cap"]
    g["grid_z"] = list(p["gdn_level_path_counts"])
    g["max_path_lengths"] = list(p["gdn_level_max_lengths"])
    g["path_programs"] = g["scan_calls"] * p["gdn_path_programs"]
    g["padded_slots"] = g["scan_calls"] * p["gdn_padded_slots"]
    # the TAW walk is level-proportional; scale every walk-derived field
    old_walk = topo.WALK_CAP
    new_walk = p["walk_cap"]
    taw = e["taw"]
    for key in (
        "loop_iterations", "walk_levels", "uniform_slots", "child_lanes",
        "target_rows", "self_rows", "self_cdf_rows", "source_cdf_rows",
        "residual_cdf_rows", "qmix_rows", "residual_rows",
        "row_scatter_slots", "path_scatter_slots",
        "exact_commit_launches", "exact_commit_programs",
    ):
        if key in taw and isinstance(taw[key], int) and not isinstance(taw[key], bool):
            taw[key] = taw[key] * new_walk // old_walk
    # ...and the shapes that carry the walk in a dimension, not a count.
    if isinstance(taw.get("uniform_shape"), list) and len(taw["uniform_shape"]) == 3:
        taw["uniform_shape"] = [
            taw["uniform_shape"][0], new_walk, taw["uniform_shape"][2]
        ]
    if isinstance(taw.get("uniform_stride"), list) and len(taw["uniform_stride"]) == 3:
        taw["uniform_stride"] = [
            new_walk * taw["uniform_shape"][2], taw["uniform_shape"][2], 1
        ]
    pack = e.get("accepted_path_pack")
    if isinstance(pack, dict) and isinstance(pack.get("source_walk_slots"), int):
        pack["source_walk_slots"] = (
            pack["source_walk_slots"] * new_walk // old_walk
        )
    calls = taw.get("tensor_call_census")
    if isinstance(calls, dict):
        taw["tensor_call_census"] = {
            k: (
                v * new_walk // old_walk
                if isinstance(v, int) and not isinstance(v, bool)
                else v
            )
            for k, v in calls.items()
        }
    return e


def test_census_validates_a_hydra27_event_unchanged():
    census.validate_event(_banked_or_skip(), source="h27")


def test_census_validates_a_hydra31_event():
    census.validate_event(_morph(_banked_or_skip(), "hydra31_fixed32"), source="h31")


@pytest.mark.parametrize(
    "field,value",
    [
        ("active_nodes", 27),
        ("valid_mask", 0x7ABDFFFF),
    ],
)
def test_census_refuses_hydra27_shape_under_the_hydra31_mode(field, value):
    e = _morph(_banked_or_skip(), "hydra31_fixed32")
    e[field] = value
    with pytest.raises(census.CensusError):
        census.validate_event(e, source="bad")


def test_census_refuses_a_hydra27_tail_under_hydra31():
    e = _morph(_banked_or_skip(), "hydra31_fixed32")
    e["drafter"]["main_tail_length"] = 6
    with pytest.raises(census.CensusError, match="main_tail_length"):
        census.validate_event(e, source="bad")


# --- consumer 3: patcher mode table -----------------------------------------

def test_patcher_mode_table_matches_the_profile_table():
    src = (REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py").read_text()
    assert '"hydra31_fixed32": (0x7FFFFFFF, 31)' in src
    assert '"hydra31_fixed32": 0x7FFFFFFF' in src, "topology needle mask map"
    p = topo.profile(topo.PROFILE_HYDRA31)
    assert p["valid_mask"] == 0x7FFFFFFF and p["active_drafts"] == 31


# --- consumer 4: launcher ----------------------------------------------------

@pytest.mark.parametrize(
    "launcher",
    [
        "scripts/fr13_launch_forked_fa2_tree_server.sh",
        "scripts/fr14_armb_leg3_launch_nomiddleware.sh",
    ],
)
def test_launcher_accepts_hydra31_and_refuses_hydra27_qualified_levers(launcher):
    text = (REPO / launcher).read_text()
    assert '""|tail6_fixed32|hydra27_fixed32|hydra31_fixed32) ;;' in text
    assert (
        "FR13_FIXED32_MODE must be empty, tail6_fixed32, hydra27_fixed32 or "
        "hydra31_fixed32" in text
    )
    assert "is incompatible with $_fr14_h31_incompat (qualified on the hydra27 tree)" in text
    for lever in (
        "FR13_CFWD_LOGIT_DIRECT_PRODUCTION",
        "FR13_DRAFT_HEAD_FP8",
        "FR13_CFWD_PACKED_WALK_NODE_TRUST_PRODUCTION",
    ):
        block = text[text.index("_fr14_h31_incompat in"):text.index("unset _fr14_h31_incompat")]
        assert lever in block, f"{lever} not refused under hydra31"

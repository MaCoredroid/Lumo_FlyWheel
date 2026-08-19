#!/usr/bin/env python3
"""Pure topology contract for the fixed-32 Tail6/Hydra27 execution bucket.

Both logical arms use the same 31 physical draft nodes plus the implicit root.
The arm changes only the validity mask applied before the sampler builds its
child tables. Invalid nodes must never enter TAW source/q_mix/rejection math.
No serving path imports this module until the fixed-32 integration lands.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable


Mode = str
Path = tuple[int, ...]

# ---------------------------------------------------------------------------
# THE TWO ROSTERS (round 19)
# ---------------------------------------------------------------------------
# This module had THREE key sets for what looked like one idea: PROFILES held
# hydra27+hydra31, VALID_BY_MODE and VALID_MASK_BY_MODE held hydra27+tail6, and
# twenty-nine HYDRA31_*/TAIL10_* constants sat fully described but unindexed. A
# consumer that did exactly what rounds 13-18 prescribed -- delegate to the
# authority instead of carrying its own table -- was failed BY the authority:
# fr13_device_multidraft_kernel's preseed asks VALID_MASK_BY_MODE for the mode's
# mask and has no hydra31 mention anywhere, because it should not need one.
# Routing consumers through the indices is right; it also means every gap in an
# index is a gap in every consumer at once.
#
# The adjudication, and the module now says it:
#
#   A serving MODE is what an arm is launched as. A topology PROFILE is a
#   physical tree. They are not the same set and never were. tail6_fixed32 and
#   hydra27_fixed32 dispatch the SAME 31 physical paths and differ only in which
#   drafts the sampler treats as valid, so they are two MODES over one PROFILE.
#   hydra31_fixed32 is a different tree, so it is both.
#
# Therefore PROFILES is keyed by PROFILE (tail6 is absent BY CONSTRUCTION -- it
# has no tree of its own, it borrows hydra27's, and TREE_PROFILE_BY_MODE below
# is where that borrowing is written down), while every *_BY_MODE index is keyed
# by MODE and must carry all three. The key-set invariant in the contract tests
# derives both rosters from these constants -- never from PROFILES, which is the
# mapping that was incomplete.
PROFILE_HYDRA27: Mode = "hydra27_fixed32"
PROFILE_HYDRA31: Mode = "hydra31_fixed32"
MODE_TAIL6: Mode = "tail6_fixed32"

#: Physical trees. The key set of PROFILES.
TOPOLOGY_PROFILES: tuple[Mode, ...] = (PROFILE_HYDRA27, PROFILE_HYDRA31)
#: Serving modes. The key set of every *_BY_MODE index.
SERVING_MODES: tuple[Mode, ...] = (MODE_TAIL6, PROFILE_HYDRA27, PROFILE_HYDRA31)
#: Which physical tree each serving mode dispatches.
TREE_PROFILE_BY_MODE: dict[Mode, Mode] = {
    MODE_TAIL6: PROFILE_HYDRA27,
    PROFILE_HYDRA27: PROFILE_HYDRA27,
    PROFILE_HYDRA31: PROFILE_HYDRA31,
}

TAIL6_CHOICES: tuple[Path, ...] = (
    (0,),
    (1,),
    (2,),
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 0, 0),
    (0, 0, 1),
    (0, 0, 2),
    (0, 0, 0, 0),
    (0, 0, 0, 1),
    (0, 0, 0, 2),
    (0, 0, 0, 0, 0),
    (0, 0, 0, 0, 1),
    (0, 0, 0, 0, 2),
    (0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
)

# The fixed-32 drafter already produces one Arctic continuation for each of
# the two root runner-ups. Tail6 used to mask both rows out even though every
# downstream kernel still executed the full 31-draft physical bucket. Keep
# the legacy Tail6 shape above intact for historical tooling, but spend those
# two otherwise-idle fixed slots as additive depth-1 rescue tails.
TAIL6_FIXED32_RESCUE_CHOICES: tuple[Path, ...] = ((1, 0), (2, 0))
TAIL6_FIXED32_CHOICES = frozenset(TAIL6_CHOICES) | frozenset(
    TAIL6_FIXED32_RESCUE_CHOICES
)

# Hydra27 logically uses rank-1 length 4 and rank-2 length 2. The physical
# rank-2 suffix continues to length 6 so every arm fills exactly 31 drafts.
PHYSICAL_BRANCH_CHAINS: tuple[tuple[int, int], ...] = ((1, 4), (2, 6))
HYDRA27_BRANCH_CHAINS: tuple[tuple[int, int], ...] = ((1, 4), (2, 2))
ARCTIC_LOOKUP_CHAINS: tuple[tuple[int, int], ...] = HYDRA27_BRANCH_CHAINS
ARCTIC_MAIN_TAIL_LENGTH = 6
MTP_FORWARD_CALLS = 4
ARCTIC_LOOKUP_CALLS_PER_REQUEST = 1 + len(ARCTIC_LOOKUP_CHAINS)
ARCTIC_LOOKUP_TOKENS_PER_REQUEST = ARCTIC_MAIN_TAIL_LENGTH + sum(
    length for _rank, length in ARCTIC_LOOKUP_CHAINS
)
RESCUE_CARRY_SLOTS_PER_REQUEST = 4


def branch_paths(chains: Iterable[tuple[int, int]]) -> tuple[Path, ...]:
    paths = []
    for raw_rank, raw_length in chains:
        rank = int(raw_rank)
        length = int(raw_length)
        if rank <= 0 or length < 0:
            raise ValueError(f"invalid branch chain {(rank, length)!r}")
        paths.extend((rank,) + (0,) * (offset + 1) for offset in range(length))
    return tuple(paths)


FIXED32_CHOICES: tuple[Path, ...] = tuple(
    sorted(
        set(TAIL6_CHOICES) | set(branch_paths(PHYSICAL_BRANCH_CHAINS)),
        key=lambda path: (len(path), path),
    )
)
HYDRA27_CHOICES = frozenset(TAIL6_CHOICES) | frozenset(
    branch_paths(HYDRA27_BRANCH_CHAINS)
)


def _draft_parents(choices: tuple[Path, ...]) -> tuple[int, ...]:
    index = {path: node for node, path in enumerate(choices)}
    parents = []
    for path in choices:
        if len(path) == 1:
            parents.append(-1)
            continue
        try:
            parents.append(index[path[:-1]])
        except KeyError as exc:
            raise ValueError(f"missing physical parent for {path!r}") from exc
    return tuple(parents)


DRAFT_PARENT: tuple[int, ...] = _draft_parents(FIXED32_CHOICES)
PHYSICAL_PARENT: tuple[int, ...] = (-1,) + tuple(
    0 if parent < 0 else parent + 1 for parent in DRAFT_PARENT
)
EXPECTED_DRAFT_PARENT = (
    -1,
    -1,
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    3,
    3,
    6,
    7,
    8,
    8,
    8,
    11,
    12,
    13,
    13,
    13,
    16,
    17,
    18,
    22,
    23,
    24,
    25,
    27,
    28,
    29,
)
EXPECTED_PHYSICAL_PARENT = (
    -1,
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    3,
    4,
    4,
    4,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    14,
    14,
    17,
    18,
    19,
    23,
    24,
    25,
    26,
    28,
    29,
    30,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _ancestor_matrix(parent: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    rows = []
    for node in range(len(parent)):
        visible = [0] * len(parent)
        cursor = node
        while cursor >= 0:
            visible[cursor] = 1
            cursor = parent[cursor]
        rows.append(tuple(visible))
    return tuple(rows)


PHYSICAL_PARENT_SHA256 = _canonical_sha256(PHYSICAL_PARENT)
TREE_ANCESTRY_SHA256 = _canonical_sha256(_ancestor_matrix(PHYSICAL_PARENT))

TAIL6_VALID: tuple[bool, ...] = tuple(
    path in TAIL6_FIXED32_CHOICES for path in FIXED32_CHOICES
)
HYDRA27_VALID: tuple[bool, ...] = tuple(
    path in HYDRA27_CHOICES for path in FIXED32_CHOICES
)


def bit_mask(valid: Iterable[bool]) -> int:
    return sum(1 << node for node, enabled in enumerate(valid) if enabled)


# Mask bit n is draft-local choice n; its root-inclusive physical row is n+1.
TAIL6_VALID_MASK = bit_mask(TAIL6_VALID)
HYDRA27_VALID_MASK = bit_mask(HYDRA27_VALID)
TAIL6_INACTIVE_DRAFT_IDS = (11, 12, 16, 17, 21, 22, 24, 26)
HYDRA27_INACTIVE_DRAFT_IDS = (17, 22, 24, 26)

# VALID_BY_MODE / VALID_MASK_BY_MODE are built further down, once HYDRA31_VALID
# exists. Defining an index before every one of its keys can be spelled is what
# produced the eighth site: the two modes describable here got indexed, the
# third was described 670 lines later and never added.

# Node ids here include the implicit root at physical row zero.
SUBTREE_LEVELS: tuple[tuple[tuple[tuple[int, ...], int], ...], ...] = (
    (((0, 1, 4, 9, 14), -1),),
    (
        ((19, 24, 26, 28, 29, 30, 31), 14),
        ((2, 7, 12, 17, 22), 0),
        ((3, 8, 13, 18, 23, 25, 27), 0),
        ((5,), 1),
        ((6,), 1),
        ((10,), 4),
        ((11,), 4),
        ((15,), 9),
        ((16,), 9),
        ((20,), 14),
        ((21,), 14),
    ),
)

# --- FR14 lever 2: suffix-aware MTP pass gating -----------------------------
# On a GATED step the drafter runs 2 post-root MTP forwards instead of 4, so
# head depths 4 and 5 (0-indexed draft positions 3 and 4) have no MTP token.
# The two SPINE nodes at those depths are Arctic-filled -- they must be, because
# the whole depth-6..11 tail hangs off the depth-5 spine node and padding it
# would orphan six live suffix nodes.  The four RUNNER-UP nodes at those depths
# are duplicate-sibling padded (repeat the parent's spine token), which is the
# already-deployed last-resort pad of fr13_mtp_suffix_assembly and whose
# committer tie convention is proven on device by fr13_greedy_pointmass_dup_gate.
#
# Consequences that keep the blast radius small, all asserted in
# validate_gate_contract():
#   * the validity MASK does not change -- a gated step is still hydra27/27
#     active nodes, so no sampler child table is rebuilt and no committer graph
#     is invalidated;
#   * the 31-column pack does not change -- 15 head + 6 tail + 10 rescue -- only
#     the FILL SOURCE of 6 head columns changes;
#   * both shapes reach the same maximum draft position (11 = MAX_PHYSICAL_DEPTH).
N_MTP_HEAD_DEPTHS = 5
BRANCHES_PER_HEAD_DEPTH = 2
GATED_MTP_K = 3
GATED_MTP_FORWARD_CALLS = GATED_MTP_K - 1
GATED_ARCTIC_MAIN_TAIL_LENGTH = 8
GATED_ARCTIC_LOOKUP_TOKENS_PER_REQUEST = GATED_ARCTIC_MAIN_TAIL_LENGTH + sum(
    length for _rank, length in ARCTIC_LOOKUP_CHAINS
)
# head depths (1-indexed, as in the path length) whose spine is Arctic-filled
GATED_SUFFIX_HEAD_DEPTHS = (4, 5)
# draft ids of the runner-up nodes those depths lose -- duplicate-sibling padded
GATED_PADDED_DRAFT_IDS = (14, 15, 19, 20)
# draft ids of the spine nodes those depths keep -- Arctic-filled, never padded
GATED_SUFFIX_SPINE_DRAFT_IDS = (13, 18)

# ---- THE per-step shape contract, in ONE place -----------------------------
# Every consumer of a decode step's drafter shape must agree on these, and the
# three campaign blockers to date were all one defect: a paired structure
# updated on one side only. So the pair members import this rather than each
# carrying its own literal, and tests/test_fr14_paired_contract_lint.py fails
# when any side drifts -- including the patcher's injected blob, which cannot
# import and therefore has its literal compared textually.
#
#   (mtp_forward_calls, graph_replays) per step:
#     (4, 1)  the single shipped 4-pass graph, replayed once   [gate OFF]
#     (4, 2)  split capture, UNGATED step: lo then hi          [gate ON]
#     (2, 1)  split capture, GATED step: lo alone              [gate ON]
LEGAL_STEP_SHAPES = ((4, 1), (4, 2), (2, 1))
# a split capture opens two capture scopes inside one proposal
LEGAL_GRAPH_CAPTURES = (0, 1, 2)
# (mtp_forward_calls, main_tail_length): a 2-forward step MUST have handed off
# to an 8-token Arctic chain and a 4-forward step MUST NOT have
LEGAL_HANDOFF_SHAPES = (
    (MTP_FORWARD_CALLS, ARCTIC_MAIN_TAIL_LENGTH),
    (GATED_MTP_FORWARD_CALLS, GATED_ARCTIC_MAIN_TAIL_LENGTH),
)

PHYSICAL_DRAFTS = 31
PHYSICAL_ROWS = 32
TAIL6_ACTIVE_DRAFTS = 23
HYDRA27_ACTIVE_DRAFTS = 27
MODEL_LAYERS = 64
TREE_ATTENTION_LAYERS = 16
GDN_LAYERS = 48
MAX_PHYSICAL_DEPTH = 11
WALK_CAP = MAX_PHYSICAL_DEPTH + 1
COMMIT_PATH_CAP = 16
SAMPLER_MAX_FANOUT = 3
SAMPLER_TABLE_SHAPE = (PHYSICAL_ROWS, SAMPLER_MAX_FANOUT)
GDN_LEVEL_PATH_COUNTS = (1, 11)
GDN_LEVEL_MAX_LENGTHS = (5, 7)
GDN_LAUNCHES = len(GDN_LEVEL_PATH_COUNTS)
GDN_PATH_PROGRAMS = sum(GDN_LEVEL_PATH_COUNTS)
GDN_PADDED_SLOTS = sum(
    paths * length
    for paths, length in zip(GDN_LEVEL_PATH_COUNTS, GDN_LEVEL_MAX_LENGTHS, strict=True)
)
TAW_PARENT_SCANS = PHYSICAL_DRAFTS
TAW_TABLE_INIT_SLOTS = PHYSICAL_ROWS * SAMPLER_MAX_FANOUT
TAW_CHILD_LANES = WALK_CAP * SAMPLER_MAX_FANOUT
TAW_UNIFORM_SLOTS = WALK_CAP * 3
TAW_ROW_SCATTER_SLOTS = WALK_CAP * 2
TAW_PATH_SCATTER_SLOTS = WALK_CAP
COMMITTER_RING_GATHERS = 4
OUTPUT_PUBLISH_CAPACITY = PHYSICAL_ROWS
ACCEPTED_PATH_CAPACITY = COMMIT_PATH_CAP
REQUEST_KEY_PATH_CAPACITY = COMMIT_PATH_CAP
KV_REMAP_PATH_CAPACITY = COMMIT_PATH_CAP
KV_REMAP_GROUPS = 1
KV_REMAP_TARGET_CACHE_TENSORS = TREE_ATTENTION_LAYERS
KV_REMAP_DRAFTER_CACHE_TENSORS = 1
KV_REMAP_CACHE_TENSORS = (
    KV_REMAP_TARGET_CACHE_TENSORS + KV_REMAP_DRAFTER_CACHE_TENSORS
)
KV_REMAP_PLANES = 2
KV_REMAP_TARGET_PAIR_SLOTS = KV_REMAP_PATH_CAPACITY
KV_REMAP_DRAFTER_PAIR_SLOTS = KV_REMAP_PATH_CAPACITY
KV_REMAP_PAIR_SLOTS = (
    KV_REMAP_TARGET_PAIR_SLOTS + KV_REMAP_DRAFTER_PAIR_SLOTS
)
KV_REMAP_TARGET_PREPARE_CALLS = 1
KV_REMAP_DRAFTER_PREPARE_CALLS = 1
KV_REMAP_TARGET_APPLY_CACHE_CALLS = KV_REMAP_TARGET_CACHE_TENSORS
KV_REMAP_DRAFTER_APPLY_CACHE_CALLS = KV_REMAP_DRAFTER_CACHE_TENSORS
CONV_COMMIT_LAYERS = GDN_LAYERS
CONV_PREGATHER_LAYERS = GDN_LAYERS
CONV_PREGATHER_BLOCK = 1024
GDN_CONV_KEY_HEADS = 16
GDN_CONV_VALUE_HEADS = 48
GDN_CONV_KEY_HEAD_DIM = 128
GDN_CONV_VALUE_HEAD_DIM = 128
GDN_CONV_KERNEL_SIZE = 4
GDN_CONV_CHANNELS = (
    2 * GDN_CONV_KEY_HEADS * GDN_CONV_KEY_HEAD_DIM
    + GDN_CONV_VALUE_HEADS * GDN_CONV_VALUE_HEAD_DIM
)
GDN_CONV_STATE_LENGTH = GDN_CONV_KERNEL_SIZE - 1 + PHYSICAL_DRAFTS
CONV_PREGATHER_ROW_ELEMS = GDN_CONV_CHANNELS * GDN_CONV_STATE_LENGTH
COMMITTER_NEUTRALIZE_OPS = 5
COMMITTER_RING_LAYER_PATH_ROWS = COMMITTER_RING_GATHERS * GDN_LAYERS * COMMIT_PATH_CAP
FIXED_EXECUTION_SIGNATURE = {
    "physical_drafts": PHYSICAL_DRAFTS,
    "physical_parent_sha256": PHYSICAL_PARENT_SHA256,
    "tree_ancestry_sha256": TREE_ANCESTRY_SHA256,
    "model_layers": MODEL_LAYERS,
    "tree_attention_layers": TREE_ATTENTION_LAYERS,
    "gdn_layers": GDN_LAYERS,
    "mtp_forward_calls": MTP_FORWARD_CALLS,
    "arctic_main_tail_length": ARCTIC_MAIN_TAIL_LENGTH,
    "arctic_lookup_calls_per_request": ARCTIC_LOOKUP_CALLS_PER_REQUEST,
    "arctic_lookup_tokens_per_request": ARCTIC_LOOKUP_TOKENS_PER_REQUEST,
    "rescue_carry_slots_per_request": RESCUE_CARRY_SLOTS_PER_REQUEST,
    "target_rows": PHYSICAL_ROWS,
    "tree_attention_rows": PHYSICAL_ROWS,
    "gdn_rows": PHYSICAL_ROWS,
    "gdn_level_path_counts": GDN_LEVEL_PATH_COUNTS,
    "gdn_level_max_lengths": GDN_LEVEL_MAX_LENGTHS,
    "gdn_launches": GDN_LAUNCHES,
    "gdn_path_programs": GDN_PATH_PROGRAMS,
    "gdn_padded_slots": GDN_PADDED_SLOTS,
    "sampler_table_shape": SAMPLER_TABLE_SHAPE,
    "sampler_parent_scans": TAW_PARENT_SCANS,
    "sampler_table_init_slots": TAW_TABLE_INIT_SLOTS,
    "sampler_walk_iterations": WALK_CAP,
    "sampler_child_lanes": TAW_CHILD_LANES,
    "sampler_uniform_slots": TAW_UNIFORM_SLOTS,
    "sampler_row_scatter_slots": TAW_ROW_SCATTER_SLOTS,
    "sampler_path_scatter_slots": TAW_PATH_SCATTER_SLOTS,
    "output_publish_capacity": OUTPUT_PUBLISH_CAPACITY,
    "accepted_path_capacity": ACCEPTED_PATH_CAPACITY,
    "request_key_path_capacity": REQUEST_KEY_PATH_CAPACITY,
    "kv_remap_path_capacity": KV_REMAP_PATH_CAPACITY,
    "kv_remap_groups": KV_REMAP_GROUPS,
    "kv_remap_target_cache_tensors": KV_REMAP_TARGET_CACHE_TENSORS,
    "kv_remap_drafter_cache_tensors": KV_REMAP_DRAFTER_CACHE_TENSORS,
    "kv_remap_cache_tensors": KV_REMAP_CACHE_TENSORS,
    "kv_remap_planes": KV_REMAP_PLANES,
    "kv_remap_target_pair_slots": KV_REMAP_TARGET_PAIR_SLOTS,
    "kv_remap_drafter_pair_slots": KV_REMAP_DRAFTER_PAIR_SLOTS,
    "kv_remap_pair_slots": KV_REMAP_PAIR_SLOTS,
    "kv_remap_target_prepare_calls": KV_REMAP_TARGET_PREPARE_CALLS,
    "kv_remap_drafter_prepare_calls": KV_REMAP_DRAFTER_PREPARE_CALLS,
    "kv_remap_target_apply_cache_calls": KV_REMAP_TARGET_APPLY_CACHE_CALLS,
    "kv_remap_drafter_apply_cache_calls": KV_REMAP_DRAFTER_APPLY_CACHE_CALLS,
    "conv_commit_layers": CONV_COMMIT_LAYERS,
    "conv_pregather_layers": CONV_PREGATHER_LAYERS,
    "conv_pregather_block": CONV_PREGATHER_BLOCK,
    "conv_pregather_row_elems": CONV_PREGATHER_ROW_ELEMS,
    "committer_path_capacity": COMMIT_PATH_CAP,
    "committer_ring_gathers": COMMITTER_RING_GATHERS,
    "committer_neutralize_ops": COMMITTER_NEUTRALIZE_OPS,
    "committer_ring_layer_path_rows": COMMITTER_RING_LAYER_PATH_ROWS,
    "arctic_lookup_chains": ARCTIC_LOOKUP_CHAINS,
    "physical_pack_width": PHYSICAL_DRAFTS,
}


def valid_for_mode(mode: Mode) -> tuple[bool, ...]:
    try:
        return VALID_BY_MODE[mode]
    except KeyError as exc:
        raise ValueError(f"unknown fixed-32 mode {mode!r}") from exc


def choices_for_mode(mode: Mode) -> tuple[Path, ...]:
    """The PHYSICAL paths a mode dispatches -- its profile's, not hydra27's.

    Round 19: active_choices and active_child_lists read the bare
    FIXED32_CHOICES / DRAFT_PARENT, which are hydra27's. Filling VALID_BY_MODE
    with hydra31 without this would have built hydra31's sampler tables out of
    HYDRA27's 31 paths -- every bit valid, every shape check satisfied, wrong
    tree. The completion of an index is only safe once the functions it feeds
    stop assuming which tree they are indexing.

    For tail6 and hydra27 this returns FIXED32_CHOICES itself, so their
    behaviour is unchanged by identity, not by resemblance.
    """
    return tuple(profile(TREE_PROFILE_BY_MODE[mode])["choices"])


def draft_parent_for_mode(mode: Mode) -> tuple[int, ...]:
    """Draft-local parents of the paths a mode dispatches."""
    if TREE_PROFILE_BY_MODE[mode] == PROFILE_HYDRA27:
        return DRAFT_PARENT
    return _draft_parents(choices_for_mode(mode))


def active_choices(mode: Mode) -> tuple[Path, ...]:
    return tuple(
        path
        for path, enabled in zip(
            choices_for_mode(mode), valid_for_mode(mode), strict=True
        )
        if enabled
    )


def active_child_lists(mode: Mode) -> dict[int, tuple[int, ...]]:
    """Return draft-local sampler children after validity filtering."""
    children: dict[int, list[int]] = {}
    valid = valid_for_mode(mode)
    for node, (parent, enabled) in enumerate(
        zip(draft_parent_for_mode(mode), valid, strict=True)
    ):
        if not enabled:
            continue
        if parent >= 0 and not valid[parent]:
            raise ValueError(f"{mode}: active node {node} has inactive parent {parent}")
        children.setdefault(parent, []).append(node)
    return {parent: tuple(nodes) for parent, nodes in children.items()}


def sampler_child_table(
    mode: Mode,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Build fixed-shape child ids/counts after applying the mode validity mask."""
    table = [[-1] * SAMPLER_MAX_FANOUT for _ in range(PHYSICAL_ROWS)]
    counts = [0] * PHYSICAL_ROWS
    for parent, children in active_child_lists(mode).items():
        slot = parent + 1
        if len(children) > SAMPLER_MAX_FANOUT:
            raise ValueError(f"{mode}: parent {parent} exceeds fixed sampler fanout")
        table[slot][: len(children)] = children
        counts[slot] = len(children)
    return tuple(tuple(row) for row in table), tuple(counts)


def validate_contract() -> None:
    if FIXED32_CHOICES != tuple(
        sorted(FIXED32_CHOICES, key=lambda path: (len(path), path))
    ):
        raise AssertionError("fixed choices are not in canonical sorted order")
    if len(set(FIXED32_CHOICES)) != len(FIXED32_CHOICES):
        raise AssertionError("fixed choices contain duplicates")
    if len(FIXED32_CHOICES) != PHYSICAL_DRAFTS:
        raise AssertionError("fixed tree must contain 31 physical drafts")
    if len(PHYSICAL_PARENT) != PHYSICAL_ROWS:
        raise AssertionError("physical parent vector must contain 32 rows")
    if PHYSICAL_ROWS & (PHYSICAL_ROWS - 1):
        raise AssertionError("physical row count must be a power of two")
    if max(map(len, FIXED32_CHOICES)) != MAX_PHYSICAL_DEPTH:
        raise AssertionError("fixed tree depth drifted")
    if DRAFT_PARENT != EXPECTED_DRAFT_PARENT:
        raise AssertionError("draft-local parent vector drifted")
    if PHYSICAL_PARENT != EXPECTED_PHYSICAL_PARENT:
        raise AssertionError("root-inclusive physical parent vector drifted")
    if tuple(node for node, parent in enumerate(DRAFT_PARENT) if parent < 0) != (
        0,
        1,
        2,
    ):
        raise AssertionError("the physical tree must have exactly three root children")
    for node, (path, parent) in enumerate(
        zip(FIXED32_CHOICES, DRAFT_PARENT, strict=True)
    ):
        if len(path) > 1 and parent >= node:
            raise AssertionError(f"choice {node} is not parent-before-child ordered")
    if sum(TAIL6_VALID) != TAIL6_ACTIVE_DRAFTS:
        raise AssertionError("Tail6 validity count drifted")
    if sum(HYDRA27_VALID) != HYDRA27_ACTIVE_DRAFTS:
        raise AssertionError("Hydra27 validity count drifted")
    if set(active_choices("tail6_fixed32")) != set(TAIL6_FIXED32_CHOICES):
        raise AssertionError("Tail6 mask does not recover the Tail6 topology")
    if set(active_choices("hydra27_fixed32")) != set(HYDRA27_CHOICES):
        raise AssertionError("Hydra27 mask does not recover the logical topology")
    if not set(TAIL6_FIXED32_CHOICES) < set(HYDRA27_CHOICES):
        raise AssertionError("Hydra27 must be a strict candidate superset of Tail6")
    if TAIL6_VALID_MASK != 0x7A9CE7FF:
        raise AssertionError("Tail6 validity mask drifted")
    if HYDRA27_VALID_MASK != 0x7ABDFFFF:
        raise AssertionError("Hydra27 validity mask drifted")
    if TAIL6_VALID_MASK & ~HYDRA27_VALID_MASK:
        raise AssertionError("Tail6 validity must be a subset of Hydra27 validity")
    if tuple(node for node, valid in enumerate(TAIL6_VALID) if not valid) != (
        TAIL6_INACTIVE_DRAFT_IDS
    ):
        raise AssertionError("Tail6 inactive draft ids drifted")
    if tuple(node for node, valid in enumerate(HYDRA27_VALID) if not valid) != (
        HYDRA27_INACTIVE_DRAFT_IDS
    ):
        raise AssertionError("Hydra27 inactive draft ids drifted")
    if WALK_CAP > COMMIT_PATH_CAP:
        raise AssertionError("walk can overflow the fixed committer path capacity")

    for mode in VALID_BY_MODE:
        child_lists = active_child_lists(mode)
        if child_lists.get(-1) != (0, 1, 2):
            raise AssertionError(f"{mode}: root child order/width drifted")
        if max(map(len, child_lists.values())) != SAMPLER_MAX_FANOUT:
            raise AssertionError(f"{mode}: sampler fanout drifted")
        table, child_counts = sampler_child_table(mode)
        if (len(table), len(table[0])) != SAMPLER_TABLE_SHAPE:
            raise AssertionError(f"{mode}: sampler child-table shape drifted")
        if len(child_counts) != PHYSICAL_ROWS:
            raise AssertionError(f"{mode}: sampler child-count shape drifted")
        if sum(child_counts) != sum(valid_for_mode(mode)):
            raise AssertionError(f"{mode}: sampler child table lost active nodes")
        inactive = {
            node for node, enabled in enumerate(valid_for_mode(mode)) if not enabled
        }
        if inactive.intersection(node for row in table for node in row if node >= 0):
            raise AssertionError(f"{mode}: inactive sampler child leaked into a table")

    seen: set[int] = set()
    earlier: set[int] = set()
    for level_index, level in enumerate(SUBTREE_LEVELS):
        current: set[int] = set()
        for path, parent in level:
            if not path:
                raise AssertionError(f"empty schedule path at level {level_index}")
            root = path[0]
            if PHYSICAL_PARENT[root] != parent:
                raise AssertionError(
                    f"schedule root {root} parent {parent} != {PHYSICAL_PARENT[root]}"
                )
            if parent >= 0 and parent not in earlier:
                raise AssertionError(
                    f"schedule parent {parent} is not in an earlier level"
                )
            for previous, node in zip(path, path[1:]):
                if PHYSICAL_PARENT[node] != previous:
                    raise AssertionError(f"schedule edge {previous}->{node} is invalid")
            overlap = seen | current
            if overlap.intersection(path):
                raise AssertionError(f"schedule repeats nodes {path!r}")
            current.update(path)
        seen.update(current)
        earlier.update(current)
    if seen != set(range(PHYSICAL_ROWS)):
        raise AssertionError("subtree schedule does not cover all physical rows")

    level_counts = tuple(len(level) for level in SUBTREE_LEVELS)
    max_lengths = tuple(max(len(path) for path, _ in level) for level in SUBTREE_LEVELS)
    if level_counts != GDN_LEVEL_PATH_COUNTS or max_lengths != GDN_LEVEL_MAX_LENGTHS:
        raise AssertionError("fixed subtree schedule geometry drifted")
    if sum(max_lengths) != WALK_CAP:
        raise AssertionError("subtree critical path and walk cap must both be 12")
    if FIXED_EXECUTION_SIGNATURE["arctic_lookup_chains"] != ((1, 4), (2, 2)):
        raise AssertionError("fixed Arctic work must be one rank1/4 and rank2/2 lookup")
    if set(FIXED_EXECUTION_SIGNATURE) != {
        "physical_drafts",
        "physical_parent_sha256",
        "tree_ancestry_sha256",
        "model_layers",
        "tree_attention_layers",
        "gdn_layers",
        "mtp_forward_calls",
        "arctic_main_tail_length",
        "arctic_lookup_calls_per_request",
        "arctic_lookup_tokens_per_request",
        "rescue_carry_slots_per_request",
        "target_rows",
        "tree_attention_rows",
        "gdn_rows",
        "gdn_level_path_counts",
        "gdn_level_max_lengths",
        "gdn_launches",
        "gdn_path_programs",
        "gdn_padded_slots",
        "sampler_table_shape",
        "sampler_parent_scans",
        "sampler_table_init_slots",
        "sampler_walk_iterations",
        "sampler_child_lanes",
        "sampler_uniform_slots",
        "sampler_row_scatter_slots",
        "sampler_path_scatter_slots",
        "output_publish_capacity",
        "accepted_path_capacity",
        "request_key_path_capacity",
        "kv_remap_path_capacity",
        "kv_remap_groups",
        "kv_remap_target_cache_tensors",
        "kv_remap_drafter_cache_tensors",
        "kv_remap_cache_tensors",
        "kv_remap_planes",
        "kv_remap_target_pair_slots",
        "kv_remap_drafter_pair_slots",
        "kv_remap_pair_slots",
        "kv_remap_target_prepare_calls",
        "kv_remap_drafter_prepare_calls",
        "kv_remap_target_apply_cache_calls",
        "kv_remap_drafter_apply_cache_calls",
        "conv_commit_layers",
        "conv_pregather_layers",
        "conv_pregather_block",
        "conv_pregather_row_elems",
        "committer_path_capacity",
        "committer_ring_gathers",
        "committer_neutralize_ops",
        "committer_ring_layer_path_rows",
        "arctic_lookup_chains",
        "physical_pack_width",
    }:
        raise AssertionError("fixed execution signature fields drifted")
    if (
        MODEL_LAYERS,
        TREE_ATTENTION_LAYERS,
        GDN_LAYERS,
        MTP_FORWARD_CALLS,
        ARCTIC_MAIN_TAIL_LENGTH,
        ARCTIC_LOOKUP_CALLS_PER_REQUEST,
        ARCTIC_LOOKUP_TOKENS_PER_REQUEST,
        RESCUE_CARRY_SLOTS_PER_REQUEST,
        GDN_LAUNCHES,
        GDN_PATH_PROGRAMS,
        GDN_PADDED_SLOTS,
        TAW_PARENT_SCANS,
        TAW_TABLE_INIT_SLOTS,
        TAW_CHILD_LANES,
        TAW_UNIFORM_SLOTS,
        TAW_ROW_SCATTER_SLOTS,
        TAW_PATH_SCATTER_SLOTS,
        OUTPUT_PUBLISH_CAPACITY,
        ACCEPTED_PATH_CAPACITY,
        REQUEST_KEY_PATH_CAPACITY,
        KV_REMAP_PATH_CAPACITY,
        KV_REMAP_GROUPS,
        KV_REMAP_TARGET_CACHE_TENSORS,
        KV_REMAP_DRAFTER_CACHE_TENSORS,
        KV_REMAP_CACHE_TENSORS,
        KV_REMAP_PLANES,
        KV_REMAP_TARGET_PAIR_SLOTS,
        KV_REMAP_DRAFTER_PAIR_SLOTS,
        KV_REMAP_PAIR_SLOTS,
        KV_REMAP_TARGET_PREPARE_CALLS,
        KV_REMAP_DRAFTER_PREPARE_CALLS,
        KV_REMAP_TARGET_APPLY_CACHE_CALLS,
        KV_REMAP_DRAFTER_APPLY_CACHE_CALLS,
        CONV_COMMIT_LAYERS,
        CONV_PREGATHER_LAYERS,
        CONV_PREGATHER_BLOCK,
        CONV_PREGATHER_ROW_ELEMS,
        COMMITTER_RING_GATHERS,
        COMMITTER_NEUTRALIZE_OPS,
        COMMITTER_RING_LAYER_PATH_ROWS,
    ) != (
        64,
        16,
        48,
        4,
        6,
        3,
        12,
        4,
        2,
        12,
        82,
        31,
        96,
        36,
        36,
        24,
        12,
        32,
        16,
        16,
        16,
        1,
        16,
        1,
        17,
        2,
        16,
        16,
        32,
        1,
        1,
        16,
        1,
        48,
        48,
        1024,
        348160,
        4,
        5,
        3072,
    ):
        raise AssertionError("fixed per-request work counts drifted")
    if (
        GDN_CONV_KEY_HEADS,
        GDN_CONV_VALUE_HEADS,
        GDN_CONV_KEY_HEAD_DIM,
        GDN_CONV_VALUE_HEAD_DIM,
        GDN_CONV_KERNEL_SIZE,
        GDN_CONV_CHANNELS,
        GDN_CONV_STATE_LENGTH,
    ) != (16, 48, 128, 128, 4, 10240, 34):
        raise AssertionError("fixed GDN conv geometry drifted")
    if TREE_ATTENTION_LAYERS + GDN_LAYERS != MODEL_LAYERS:
        raise AssertionError("hybrid layer partition drifted")
    if (
        PHYSICAL_PARENT_SHA256
        != "7abd25e38323d6c088eb627785b5c190b2e878b0a710bb349e2d690852a06ddd"
        or TREE_ANCESTRY_SHA256
        != "90873d81e83ce1644ee4701e043b7e9d26e83b7a7ca752d538a0e6eed1946dad"
    ):
        raise AssertionError("physical topology digest drifted")


def validate_gate_contract() -> None:
    """Contract for a GATED step (FR14_SUFFIX_PASS_GATE).

    Kept separate from validate_contract() on purpose: the ungated contract and
    its pinned literals must remain byte-identical whether or not the gate
    exists, so nothing here may weaken anything there.
    """
    # both shapes reach the same deepest draft position
    if GATED_MTP_K + GATED_ARCTIC_MAIN_TAIL_LENGTH != MAX_PHYSICAL_DEPTH:
        raise AssertionError("gated shape does not reach the physical tree depth")
    if N_MTP_HEAD_DEPTHS + ARCTIC_MAIN_TAIL_LENGTH != MAX_PHYSICAL_DEPTH:
        raise AssertionError("ungated shape does not reach the physical tree depth")
    if GATED_MTP_FORWARD_CALLS != GATED_MTP_K - 1:
        raise AssertionError("gated post-root forward count drifted")
    if GATED_MTP_FORWARD_CALLS * 2 != MTP_FORWARD_CALLS:
        raise AssertionError(
            "the gated graph must be exactly half the ungated one -- the split "
            "capture depends on lo and hi holding the same number of passes"
        )
    if GATED_ARCTIC_LOOKUP_TOKENS_PER_REQUEST != 14:
        raise AssertionError("gated Arctic request width drifted")

    # the head columns the gate re-sources: depths 4 and 5, spine + 2 runner-ups
    expected_padded = tuple(
        node
        for node, path in enumerate(FIXED32_CHOICES)
        if len(path) in GATED_SUFFIX_HEAD_DEPTHS and path[0] == 0 and path[-1] != 0
    )
    if expected_padded != GATED_PADDED_DRAFT_IDS:
        raise AssertionError(
            f"gated padded draft ids drifted: {expected_padded!r}"
        )
    expected_spine = tuple(
        node
        for node, path in enumerate(FIXED32_CHOICES)
        if len(path) in GATED_SUFFIX_HEAD_DEPTHS and set(path) == {0}
    )
    if expected_spine != GATED_SUFFIX_SPINE_DRAFT_IDS:
        raise AssertionError(
            f"gated suffix spine draft ids drifted: {expected_spine!r}"
        )

    # THE well-formedness invariant: every node the gate stops MTP-feeding is
    # either Arctic-filled or duplicate-padded, and no node is orphaned.  A
    # padded node must have no valid descendants; a spine node must keep them.
    child_of = {}
    for node, parent in enumerate(DRAFT_PARENT):
        child_of.setdefault(parent, []).append(node)
    for node in GATED_PADDED_DRAFT_IDS:
        if child_of.get(node):
            raise AssertionError(
                f"padded draft {node} has children {child_of[node]} -- padding it "
                "would orphan live nodes; it must be Arctic-filled instead"
            )
    for node in GATED_SUFFIX_SPINE_DRAFT_IDS:
        if not child_of.get(node):
            raise AssertionError(f"spine draft {node} lost its subtree")
        if not HYDRA27_VALID[node]:
            raise AssertionError(f"spine draft {node} is not active in hydra27")

    # the mask is NOT changed by gating: same active set, same committer tables
    for node in GATED_PADDED_DRAFT_IDS:
        if not HYDRA27_VALID[node]:
            raise AssertionError(
                f"draft {node} is already inactive -- the gate pads FILLED nodes"
            )

    # the shape contract must stay internally consistent
    if set(LEGAL_HANDOFF_SHAPES) != {
        (MTP_FORWARD_CALLS, ARCTIC_MAIN_TAIL_LENGTH),
        (GATED_MTP_FORWARD_CALLS, GATED_ARCTIC_MAIN_TAIL_LENGTH),
    }:
        raise AssertionError("handoff shape contract drifted")
    for _calls, _replays in LEGAL_STEP_SHAPES:
        if _calls not in (MTP_FORWARD_CALLS, GATED_MTP_FORWARD_CALLS):
            raise AssertionError(f"illegal forward count in contract: {_calls}")
        if _calls != _replays * (_calls // _replays) or _replays < 1:
            raise AssertionError("step shape is not a whole number of replays")
    if max(_r for _c, _r in LEGAL_STEP_SHAPES) + 1 != len(LEGAL_GRAPH_CAPTURES):
        raise AssertionError("capture count contract disagrees with replays")

    # the 31-column pack is unchanged: 15 head + 6 tail + 10 rescue
    gated_tail_columns = GATED_ARCTIC_MAIN_TAIL_LENGTH - len(
        GATED_SUFFIX_SPINE_DRAFT_IDS
    )
    if gated_tail_columns != ARCTIC_MAIN_TAIL_LENGTH:
        raise AssertionError("gated tail column count drifted from the pack width")
    head_columns = N_MTP_HEAD_DEPTHS * (1 + BRANCHES_PER_HEAD_DEPTH)
    rescue_columns = sum(length for _rank, length in PHYSICAL_BRANCH_CHAINS)
    if head_columns + gated_tail_columns + rescue_columns != PHYSICAL_DRAFTS:
        raise AssertionError("gated pack is not exactly 31 columns")


# ===========================================================================
# TAIL10 / hydra31_fixed32 -- a SECOND PHYSICAL PROFILE, added beside hydra27.
# ===========================================================================
# tail10 spends the four slots hydra27 disarms. In hydra27 they are the deep
# rank-2 side-branches (2,0,0,0) .. (2,0,0,0,0,0,0) at depths 4-7, masked off by
# 0x7ABDFFFF. tail10 redefines them as SPINE continuations 0^12..0^15 and arms
# them, so the Arctic tail runs 6 -> 10 and every one of the 31 physical drafts
# is live.
#
# THIS IS NOT A MASK CHANGE. Sorting by (len, path) puts 0^12..0^15 at the end,
# so draft ids >= 17 acquire different paths: the PHYSICAL TREE ITSELF differs,
# and with it DRAFT_PARENT, PHYSICAL_PARENT and both digests. hydra27 therefore
# cannot be edited in place without silently changing what every banked hydra27
# credential attests. It is added as a NEW NAMED PROFILE, the K0-train
# qualification-profile pattern, and every module-level constant above stays
# bound to hydra27 so no existing consumer moves.
#
# What does NOT change, and is why this stays host-side: PHYSICAL_DRAFTS 31,
# PHYSICAL_ROWS 32, SAMPLER_MAX_FANOUT 3, root fan-out 3. The kernels see the
# same 32-row geometry; only the ancestry TABLE and the walk DEPTH differ.
TAIL10_SPINE_MAX_DEPTH = 15
TAIL10_BRANCH_CHAINS: tuple[tuple[int, int], ...] = ((1, 4), (2, 2))
# physical == logical here, so nothing is carried into dead slots (hydra27
# carries 4, because its physical rank-2 chain is 6 long and its logical one 2)
TAIL10_PHYSICAL_BRANCH_CHAINS = TAIL10_BRANCH_CHAINS
TAIL10_RESCUE_CARRY_SLOTS_PER_REQUEST = 0
TAIL10_ARCTIC_MAIN_TAIL_LENGTH = TAIL10_SPINE_MAX_DEPTH - N_MTP_HEAD_DEPTHS
TAIL10_ARCTIC_LOOKUP_CHAINS = TAIL10_BRANCH_CHAINS
TAIL10_ARCTIC_LOOKUP_CALLS_PER_REQUEST = 1 + len(TAIL10_ARCTIC_LOOKUP_CHAINS)
TAIL10_ARCTIC_LOOKUP_TOKENS_PER_REQUEST = TAIL10_ARCTIC_MAIN_TAIL_LENGTH + sum(
    length for _rank, length in TAIL10_ARCTIC_LOOKUP_CHAINS
)


def _spine_head_choices(head_depth: int, branches: int, spine_max: int):
    head = [
        (0,) * (depth - 1) + (rank,)
        for depth in range(1, head_depth + 1)
        for rank in range(1 + branches)
    ]
    spine = [(0,) * depth for depth in range(head_depth + 1, spine_max + 1)]
    return set(head) | set(spine)


TAIL10_CHOICES: tuple[Path, ...] = tuple(
    sorted(
        _spine_head_choices(
            N_MTP_HEAD_DEPTHS, BRANCHES_PER_HEAD_DEPTH, TAIL10_SPINE_MAX_DEPTH
        )
        | set(branch_paths(TAIL10_PHYSICAL_BRANCH_CHAINS)),
        key=lambda path: (len(path), path),
    )
)
TAIL10_DRAFT_PARENT: tuple[int, ...] = _draft_parents(TAIL10_CHOICES)
TAIL10_PHYSICAL_PARENT: tuple[int, ...] = (-1,) + tuple(
    0 if parent < 0 else parent + 1 for parent in TAIL10_DRAFT_PARENT
)
TAIL10_PHYSICAL_PARENT_SHA256 = _canonical_sha256(TAIL10_PHYSICAL_PARENT)
TAIL10_TREE_ANCESTRY_SHA256 = _canonical_sha256(
    _ancestor_matrix(TAIL10_PHYSICAL_PARENT)
)
HYDRA31_VALID: tuple[bool, ...] = (True,) * len(TAIL10_CHOICES)
HYDRA31_VALID_MASK = bit_mask(HYDRA31_VALID)
HYDRA31_ACTIVE_DRAFTS = len(TAIL10_CHOICES)
HYDRA31_INACTIVE_DRAFT_IDS: tuple[int, ...] = ()
TAIL10_MAX_PHYSICAL_DEPTH = TAIL10_SPINE_MAX_DEPTH
TAIL10_WALK_CAP = TAIL10_MAX_PHYSICAL_DEPTH + 1

# The gate (FR14_SUFFIX_PASS_GATE) composes: a gated step still hands off two
# head depths earlier, so its Arctic chain is two longer than the ungated one.
TAIL10_GATED_ARCTIC_MAIN_TAIL_LENGTH = (
    TAIL10_SPINE_MAX_DEPTH - GATED_MTP_K
)
TAIL10_GATED_ARCTIC_LOOKUP_TOKENS_PER_REQUEST = (
    TAIL10_GATED_ARCTIC_MAIN_TAIL_LENGTH
    + sum(length for _rank, length in TAIL10_ARCTIC_LOOKUP_CHAINS)
)
TAIL10_LEGAL_HANDOFF_SHAPES = (
    (MTP_FORWARD_CALLS, TAIL10_ARCTIC_MAIN_TAIL_LENGTH),
    (GATED_MTP_FORWARD_CALLS, TAIL10_GATED_ARCTIC_MAIN_TAIL_LENGTH),
)


# ---- GDN subtree schedule, DERIVED (stage 2) --------------------------------
# hydra27 ships SUBTREE_LEVELS as a hand-written literal. hydra31 cannot reuse
# it (different parents), and hand-writing a second one would be the same
# hardcode twice. So the rule is written down instead, and
# validate_tail10_contract() proves it reproduces the shipped hydra27 table
# EXACTLY before trusting it for hydra31.
#
# The rule: level 0 is the spine prefix of N_MTP_HEAD_DEPTHS rows (root plus the
# head spine); level 1 is, for every child of a level-0 node that is not itself
# in level 0, the maximal descending chain from that child. The head runner-ups
# force the level-0 length: a runner-up at depth d hangs off the spine at d-1,
# and every level-1 chain root's parent must live in an earlier level.
SUBTREE_LEVEL0_ROWS = N_MTP_HEAD_DEPTHS


def derive_subtree_levels(physical_parent, level0_rows=SUBTREE_LEVEL0_ROWS):
    children: dict[int, list[int]] = {}
    for node, parent in enumerate(physical_parent):
        children.setdefault(parent, []).append(node)

    def chain_from(start: int) -> tuple[int, ...]:
        out = [start]
        while children.get(out[-1]):
            out.append(min(children[out[-1]]))
        return tuple(out)

    spine = chain_from(0)
    level0 = spine[:level0_rows]
    level0_set = set(level0)
    level1 = [
        (chain_from(child), node)
        for node in level0
        for child in sorted(children.get(node, ()))
        if child not in level0_set
    ]
    # longest first, then by root -- the shipped hydra27 ordering
    level1.sort(key=lambda entry: (-len(entry[0]), entry[0][0]))
    return ((level0, -1),), tuple(level1)


TAIL10_SUBTREE_LEVELS = derive_subtree_levels(TAIL10_PHYSICAL_PARENT)
TAIL10_GDN_LEVEL_PATH_COUNTS = tuple(
    len(level) for level in TAIL10_SUBTREE_LEVELS
)
TAIL10_GDN_LEVEL_MAX_LENGTHS = tuple(
    max(len(path) for path, _parent in level) for level in TAIL10_SUBTREE_LEVELS
)
TAIL10_GDN_LAUNCHES = len(TAIL10_GDN_LEVEL_PATH_COUNTS)
TAIL10_GDN_PATH_PROGRAMS = sum(TAIL10_GDN_LEVEL_PATH_COUNTS)
TAIL10_GDN_PADDED_SLOTS = sum(
    paths * length
    for paths, length in zip(
        TAIL10_GDN_LEVEL_PATH_COUNTS, TAIL10_GDN_LEVEL_MAX_LENGTHS, strict=True
    )
)

# --- THE MODE INDICES -------------------------------------------------------
# Keyed by SERVING_MODES, built here because HYDRA31_VALID is only defined
# above this point. Values are taken from the per-profile constants, never
# retyped: a retyped mask is a mask that can disagree with its own tree.
# Complete as of round 19's ruling: the blocking equality check in
# fr13_device_multidraft_kernel._fr13_fixed32_taw_topology_binding was
# re-scoped to a coverage check and the TAW source-closure digest re-attested
# through the established machinery. A consumer depends on the authority
# COVERING what it needs, never on the authority knowing nothing more.
VALID_BY_MODE: dict[Mode, tuple[bool, ...]] = {
    MODE_TAIL6: TAIL6_VALID,
    PROFILE_HYDRA27: HYDRA27_VALID,
    PROFILE_HYDRA31: HYDRA31_VALID,
}
VALID_MASK_BY_MODE: dict[Mode, int] = {
    MODE_TAIL6: TAIL6_VALID_MASK,
    PROFILE_HYDRA27: HYDRA27_VALID_MASK,
    PROFILE_HYDRA31: HYDRA31_VALID_MASK,
}

#: Physical trees, keyed by PROFILE. tail6_fixed32 is deliberately absent: it is
#: a MODE, not a tree, and TREE_PROFILE_BY_MODE maps it onto hydra27's entry.
#: Asking PROFILES for a mode is a category error the key-set invariant catches.
PROFILES: dict[str, dict[str, object]] = {
    PROFILE_HYDRA27: {
        "choices": FIXED32_CHOICES,
        "valid": HYDRA27_VALID,
        "valid_mask": HYDRA27_VALID_MASK,
        "active_drafts": HYDRA27_ACTIVE_DRAFTS,
        "inactive_draft_ids": HYDRA27_INACTIVE_DRAFT_IDS,
        "physical_parent": PHYSICAL_PARENT,
        "physical_parent_sha256": PHYSICAL_PARENT_SHA256,
        "tree_ancestry_sha256": TREE_ANCESTRY_SHA256,
        "max_physical_depth": MAX_PHYSICAL_DEPTH,
        "walk_cap": WALK_CAP,
        "main_tail_length": ARCTIC_MAIN_TAIL_LENGTH,
        "arctic_requested_tokens": ARCTIC_LOOKUP_TOKENS_PER_REQUEST,
        "gated_main_tail_length": GATED_ARCTIC_MAIN_TAIL_LENGTH,
        "gated_arctic_requested_tokens": GATED_ARCTIC_LOOKUP_TOKENS_PER_REQUEST,
        "physical_branch_chains": PHYSICAL_BRANCH_CHAINS,
        "rescue_carry_slots": RESCUE_CARRY_SLOTS_PER_REQUEST,
        "arctic_lookup_chains": ARCTIC_LOOKUP_CHAINS,
        "arctic_lookup_calls": ARCTIC_LOOKUP_CALLS_PER_REQUEST,
        "subtree_levels": SUBTREE_LEVELS,
        "gdn_level_path_counts": GDN_LEVEL_PATH_COUNTS,
        "gdn_level_max_lengths": GDN_LEVEL_MAX_LENGTHS,
        "gdn_launches": GDN_LAUNCHES,
        "gdn_path_programs": GDN_PATH_PROGRAMS,
        "gdn_padded_slots": GDN_PADDED_SLOTS,
    },
    PROFILE_HYDRA31: {
        "choices": TAIL10_CHOICES,
        "valid": HYDRA31_VALID,
        "valid_mask": HYDRA31_VALID_MASK,
        "active_drafts": HYDRA31_ACTIVE_DRAFTS,
        "inactive_draft_ids": HYDRA31_INACTIVE_DRAFT_IDS,
        "physical_parent": TAIL10_PHYSICAL_PARENT,
        "physical_parent_sha256": TAIL10_PHYSICAL_PARENT_SHA256,
        "tree_ancestry_sha256": TAIL10_TREE_ANCESTRY_SHA256,
        "max_physical_depth": TAIL10_MAX_PHYSICAL_DEPTH,
        "walk_cap": TAIL10_WALK_CAP,
        "main_tail_length": TAIL10_ARCTIC_MAIN_TAIL_LENGTH,
        "arctic_requested_tokens": TAIL10_ARCTIC_LOOKUP_TOKENS_PER_REQUEST,
        "gated_main_tail_length": TAIL10_GATED_ARCTIC_MAIN_TAIL_LENGTH,
        "gated_arctic_requested_tokens":
            TAIL10_GATED_ARCTIC_LOOKUP_TOKENS_PER_REQUEST,
        "physical_branch_chains": TAIL10_PHYSICAL_BRANCH_CHAINS,
        "rescue_carry_slots": TAIL10_RESCUE_CARRY_SLOTS_PER_REQUEST,
        "arctic_lookup_chains": TAIL10_ARCTIC_LOOKUP_CHAINS,
        "arctic_lookup_calls": TAIL10_ARCTIC_LOOKUP_CALLS_PER_REQUEST,
        "subtree_levels": TAIL10_SUBTREE_LEVELS,
        "gdn_level_path_counts": TAIL10_GDN_LEVEL_PATH_COUNTS,
        "gdn_level_max_lengths": TAIL10_GDN_LEVEL_MAX_LENGTHS,
        "gdn_launches": TAIL10_GDN_LAUNCHES,
        "gdn_path_programs": TAIL10_GDN_PATH_PROGRAMS,
        "gdn_padded_slots": TAIL10_GDN_PADDED_SLOTS,
    },
}

# --- SITE 13: the walk cap, mode-keyed ---------------------------------------
# WALK_CAP below is MAX_PHYSICAL_DEPTH + 1 and has always been hydra27's 12. It
# is a MODULE-LEVEL SCALAR, so it never became per-mode when hydra31 arrived,
# and a guard that interpolated {mode} into its message while comparing that
# scalar read as mode-aware without being it. The key-set invariant from round
# 19 polices dicts; a scalar has no key set to be wrong about, so it escaped.
#
# This index is DERIVED from PROFILES through TREE_PROFILE_BY_MODE, so it
# cannot disagree with the authority it summarises, and being a dict it falls
# under the key-set invariant automatically -- the same detector that would
# have caught this had the quantity been shaped this way to begin with.
WALK_CAP_BY_MODE: dict[Mode, int] = {
    serving_mode: int(PROFILES[TREE_PROFILE_BY_MODE[serving_mode]]["walk_cap"])
    for serving_mode in SERVING_MODES
}


def walk_cap_for_mode(mode: Mode) -> int:
    """The SERVED mode's TAW walk cap. Refuses an unknown mode.

    Every execution-path reader goes through this. A reader that wants
    hydra27's 12 specifically must say so at its own site and explain why,
    because "the walk cap" without a mode is the bug this closes.
    """
    try:
        return WALK_CAP_BY_MODE[mode]
    except KeyError:
        raise KeyError(
            f"unknown fixed32 mode {mode!r} for walk cap; known modes are "
            f"{list(SERVING_MODES)}"
        ) from None


def profile(mode: str) -> dict[str, object]:
    """Return a TOPOLOGY PROFILE. Not an index by serving mode -- see below.

    tail6_fixed32 is a mode, not a tree: it dispatches hydra27's 31 physical
    paths under a narrower sampler mask. Resolving it to hydra27's profile here
    would hand the caller hydra27's valid_mask for a tail6 arm, so this refuses
    and says where the mode-keyed answer lives instead.
    """
    try:
        return PROFILES[mode]
    except KeyError:
        if mode in TREE_PROFILE_BY_MODE:
            raise KeyError(
                f"{mode!r} is a serving MODE, not a topology PROFILE: it "
                f"dispatches {TREE_PROFILE_BY_MODE[mode]}'s tree under its own "
                "sampler mask. Use TREE_PROFILE_BY_MODE[mode] for the tree, "
                "VALID_MASK_BY_MODE[mode] for the mask."
            ) from None
        raise KeyError(
            f"unknown fixed32 profile {mode!r}; profiles are "
            f"{list(TOPOLOGY_PROFILES)}, serving modes are {list(SERVING_MODES)}"
        ) from None


def validate_tail10_contract() -> None:
    """The hydra31 contract, and the ways it must NOT disturb hydra27."""
    # hydra27 is untouched by construction -- assert it, do not assume it
    if PHYSICAL_PARENT_SHA256 != (
        "7abd25e38323d6c088eb627785b5c190b2e878b0a710bb349e2d690852a06ddd"
    ):
        raise AssertionError("hydra27 parent digest moved -- tail10 must not")
    if HYDRA27_VALID_MASK != 0x7ABDFFFF or HYDRA27_ACTIVE_DRAFTS != 27:
        raise AssertionError("hydra27 identity moved -- tail10 must not")

    # the four slots hydra27 disarms are exactly the ones tail10 respends
    if len(HYDRA27_INACTIVE_DRAFT_IDS) != 4:
        raise AssertionError("hydra27 does not have four spare slots")
    respent = [
        path for path in TAIL10_CHOICES if path not in set(FIXED32_CHOICES)
    ]
    if len(respent) != 4 or any(set(p) != {0} for p in respent):
        raise AssertionError(f"tail10 must add four SPINE nodes, got {respent}")
    if sorted(len(p) for p in respent) != [12, 13, 14, 15]:
        raise AssertionError("tail10's new nodes must be 0^12..0^15")

    # same physical budget -- this is what keeps it host-side
    if len(TAIL10_CHOICES) != PHYSICAL_DRAFTS:
        raise AssertionError("tail10 must still be exactly 31 physical drafts")
    if HYDRA31_ACTIVE_DRAFTS != PHYSICAL_DRAFTS:
        raise AssertionError("tail10 arms every physical draft")
    if HYDRA31_VALID_MASK != (1 << PHYSICAL_DRAFTS) - 1:
        raise AssertionError("tail10 mask must arm all 31 bits")

    # a well-formed tree: parent-before-child, one root fan-out of three,
    # fan-out never exceeds what the sampler tables hold
    children: dict[int, list[int]] = {}
    for node, parent in enumerate(TAIL10_DRAFT_PARENT):
        children.setdefault(parent, []).append(node)
    if children.get(-1) != [0, 1, 2]:
        raise AssertionError("tail10 root fan-out drifted")
    if max(len(v) for v in children.values()) != SAMPLER_MAX_FANOUT:
        raise AssertionError("tail10 exceeds the sampler fan-out")
    for node, (path, parent) in enumerate(
        zip(TAIL10_CHOICES, TAIL10_DRAFT_PARENT, strict=True)
    ):
        if len(path) > 1 and parent >= node:
            raise AssertionError(f"tail10 choice {node} is not parent-ordered")

    # the committer must be able to walk it. tail10 sits EXACTLY at the cap.
    if TAIL10_WALK_CAP > COMMIT_PATH_CAP:
        raise AssertionError(
            f"tail10 walk {TAIL10_WALK_CAP} exceeds commit path capacity "
            f"{COMMIT_PATH_CAP}"
        )

    # the pack is still 15 head + tail + rescue = 31, ungated and gated
    head_columns = N_MTP_HEAD_DEPTHS * (1 + BRANCHES_PER_HEAD_DEPTH)
    rescue = sum(l for _r, l in TAIL10_PHYSICAL_BRANCH_CHAINS)
    for hd, cols in (
        (N_MTP_HEAD_DEPTHS, TAIL10_ARCTIC_MAIN_TAIL_LENGTH),
        (GATED_MTP_K, TAIL10_GATED_ARCTIC_MAIN_TAIL_LENGTH),
    ):
        in_head = N_MTP_HEAD_DEPTHS - hd
        if head_columns + (cols - in_head) + rescue != PHYSICAL_DRAFTS:
            raise AssertionError("tail10 pack is not 31 columns")

    # nothing is carried: physical and logical rescue chains are equal now
    if TAIL10_PHYSICAL_BRANCH_CHAINS != TAIL10_BRANCH_CHAINS:
        raise AssertionError("tail10 rescue chains diverged")
    if TAIL10_RESCUE_CARRY_SLOTS_PER_REQUEST != 0:
        raise AssertionError("tail10 carries nothing")

    # the schedule RULE must reproduce the shipped hydra27 literal exactly --
    # only then is it trustworthy for hydra31
    rederived = derive_subtree_levels(PHYSICAL_PARENT)
    if rederived[0] != SUBTREE_LEVELS[0] or set(rederived[1]) != set(
        SUBTREE_LEVELS[1]
    ):
        raise AssertionError(
            "derive_subtree_levels does not reproduce the shipped hydra27 "
            "schedule -- do not trust it for hydra31"
        )
    # hydra31's schedule must satisfy the same contract validate_contract()
    # applies to hydra27's
    covered: set[int] = set()
    earlier: set[int] = set()
    for level in TAIL10_SUBTREE_LEVELS:
        current: set[int] = set()
        for path, parent in level:
            if not path or TAIL10_PHYSICAL_PARENT[path[0]] != parent:
                raise AssertionError(f"tail10 schedule root {path!r} misparented")
            if parent >= 0 and parent not in earlier:
                raise AssertionError("tail10 schedule parent is not earlier")
            for previous, node in zip(path, path[1:]):
                if TAIL10_PHYSICAL_PARENT[node] != previous:
                    raise AssertionError("tail10 schedule edge is invalid")
            if (covered | current).intersection(path):
                raise AssertionError("tail10 schedule repeats nodes")
            current.update(path)
        covered.update(current)
        earlier.update(current)
    if covered != set(range(PHYSICAL_ROWS)):
        raise AssertionError("tail10 schedule does not cover all physical rows")
    if sum(TAIL10_GDN_LEVEL_MAX_LENGTHS) != TAIL10_WALK_CAP:
        raise AssertionError("tail10 critical path must equal its walk cap")

    # the two profiles must not collide
    if TAIL10_PHYSICAL_PARENT_SHA256 == PHYSICAL_PARENT_SHA256:
        raise AssertionError("profiles must have distinct parent digests")
    if set(PROFILES) != {PROFILE_HYDRA27, PROFILE_HYDRA31}:
        raise AssertionError("profile table drifted")


if __name__ == "__main__":
    validate_contract()
    validate_gate_contract()
    validate_tail10_contract()
    print(
        "PASS fixed32 topology: physical=31/32 active=23/27 "
        f"masks={TAIL6_VALID_MASK:#x}/{HYDRA27_VALID_MASK:#x} "
        "schedule=[1,11]/[5,7] walk_cap=12"
    )

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

VALID_BY_MODE: dict[Mode, tuple[bool, ...]] = {
    "tail6_fixed32": TAIL6_VALID,
    "hydra27_fixed32": HYDRA27_VALID,
}
VALID_MASK_BY_MODE: dict[Mode, int] = {
    "tail6_fixed32": TAIL6_VALID_MASK,
    "hydra27_fixed32": HYDRA27_VALID_MASK,
}

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


def active_choices(mode: Mode) -> tuple[Path, ...]:
    return tuple(
        path
        for path, enabled in zip(FIXED32_CHOICES, valid_for_mode(mode), strict=True)
        if enabled
    )


def active_child_lists(mode: Mode) -> dict[int, tuple[int, ...]]:
    """Return draft-local sampler children after validity filtering."""
    children: dict[int, list[int]] = {}
    valid = valid_for_mode(mode)
    for node, (parent, enabled) in enumerate(zip(DRAFT_PARENT, valid, strict=True)):
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


if __name__ == "__main__":
    validate_contract()
    validate_gate_contract()
    print(
        "PASS fixed32 topology: physical=31/32 active=23/27 "
        f"masks={TAIL6_VALID_MASK:#x}/{HYDRA27_VALID_MASK:#x} "
        "schedule=[1,11]/[5,7] walk_cap=12"
    )

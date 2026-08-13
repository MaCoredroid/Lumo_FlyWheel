"""FR13 GDN fused-scan rung: schedule-invariance tests (CPU only, no triton).

These pin the load-bearing structural claim of
``results/fr13_gdn_scan_20260811/design.md``: the DEPLOYED two-launch tree-GDN
scan is already critical-path-optimal for the fixed32 tree, so a fused
single-launch scan that preserves the 11-way path parallelism via per-node
ready flags cannot reduce the makespan in node-steps.

If any of these fail, the design note's verdict must be re-derived before the
lever is re-priced.

Constants are lifted out of the kernel module by AST (the same technique
``test_fr13_fixed32_gdn_schedule.py`` uses) so the suite stays importable on a
host without torch/triton.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

KERNEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lumo_flywheel_serving"
    / "fr10_gdn_tree_kernel.py"
)
CONSTANTS = {
    "_FR13_FIXED32_PARENT",
    "_FR13_FIXED32_SUBTREE_LEVELS",
    "_FR13_FIXED32_EXPORT_NODES",
    "_FR13_FIXED32_GDN_DEPTH_FIRST_GROUPS",
    "_FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID",
}
FUNCTIONS = {
    "_fr13_canonical_sha256",
    "_fr13_fixed32_gdn_single_launch_contract",
}


def _ns():
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in CONSTANTS
            for t in node.targets
        )
    ]
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    ]
    found = {
        t.id
        for node in assignments
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert found == CONSTANTS, found
    assert {n.name for n in definitions} == FUNCTIONS
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *assignments,
            *definitions,
        ],
        type_ignores=[],
    )
    ns = {"hashlib": hashlib, "json": json}
    exec(
        compile(ast.fix_missing_locations(module), str(KERNEL_PATH), "exec"),
        ns,
    )
    return ns


def _NS():
    return _ns()


# --------------------------------------------------------------------------
# tree geometry, derived from the parent vector alone
# --------------------------------------------------------------------------
def _tree_depths(parent):
    """depth[i] = number of nodes on the root->i chain, inclusive."""
    depth = [0] * len(parent)
    for i, par in enumerate(parent):
        depth[i] = 1 if par < 0 else depth[par] + 1
    return depth


def test_fixed32_tree_dependency_depth_is_12():
    """The recurrence DAG's critical path. No schedule can beat this."""
    G = _NS()
    parent = [int(p) for p in G["_FR13_FIXED32_PARENT"]]
    assert len(parent) == 32
    depth = _tree_depths(parent)
    assert max(depth) == 12
    # the deepest chain, for the record
    deepest = max(range(len(parent)), key=lambda i: depth[i])
    chain = []
    cur = deepest
    while cur >= 0:
        chain.append(cur)
        cur = parent[cur]
    assert list(reversed(chain)) == [
        0, 1, 4, 9, 14, 19, 24, 26, 28, 29, 30, 31
    ]


def test_two_launch_schedule_is_critical_path_optimal():
    """Deployed makespan (level0 depth + level1 depth) == the DAG depth.

    This is the whole verdict: the level barrier delays nothing, because the
    deepest level-1 path descends from the LAST level-0 export node.
    """
    G = _NS()
    levels = G["_FR13_FIXED32_SUBTREE_LEVELS"]
    assert len(levels) == 2
    level0_depth = max(len(path) for path, _par in levels[0])
    level1_depth = max(len(path) for path, _par in levels[1])
    assert level0_depth == 5
    assert level1_depth == 7
    deployed_waves = level0_depth + level1_depth
    parent = [int(p) for p in G["_FR13_FIXED32_PARENT"]]
    assert deployed_waves == max(_tree_depths(parent)) == 12


def test_fused_ready_flag_schedule_saves_zero_waves():
    """List-schedule the DAG with per-node ready flags; makespan is unchanged.

    Models the ladder's proposed fused kernel: a node runs in the wave after
    its parent retires, unbounded machine width. That is the most optimistic
    possible fused schedule.
    """
    G = _NS()
    parent = [int(p) for p in G["_FR13_FIXED32_PARENT"]]
    finish = _tree_depths(parent)  # earliest-possible finish wave per node
    fused_makespan = max(finish)

    levels = G["_FR13_FIXED32_SUBTREE_LEVELS"]
    deployed_makespan = max(len(p) for p, _ in levels[0]) + max(
        len(p) for p, _ in levels[1]
    )
    assert fused_makespan == deployed_makespan == 12
    assert deployed_makespan - fused_makespan == 0


def test_fused_and_deployed_execute_the_same_node_count():
    """Work term is invariant too: 32 real node-steps either way."""
    G = _NS()
    levels = G["_FR13_FIXED32_SUBTREE_LEVELS"]
    executed = sum(len(path) for level in levels for path, _par in level)
    assert executed == 32 == len(G["_FR13_FIXED32_PARENT"])


def test_padded_slots_are_descriptor_only_not_executed():
    """The path kernel's trip count is per-path ``path_lengths``, not
    MAX_PATH_LEN. The 82 'padded slots' in the ladder's model are therefore
    never executed, which is where its 3.45 ms/step came from."""
    tree = ast.parse(KERNEL_PATH.read_text())
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_tree_gdn_path_kernel"
    )
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)]
    assert len(loops) == 1, "path kernel should have exactly one node loop"
    src = ast.unparse(loops[0].iter)
    assert "path_len" in src, f"loop bound is not path_len: {src}"
    assert "MAX_PATH_LEN" not in src, f"loop bound pads: {src}"

    # and path_len is loaded from the per-path lengths tensor
    body_src = ast.unparse(fn)
    assert "path_len = tl.load(path_lengths + pid_path)" in body_src

    # descriptor padding really is 82 slots, so the two numbers differ
    G = _NS()
    levels = G["_FR13_FIXED32_SUBTREE_LEVELS"]
    padded = sum(
        max(len(p) for p, _ in level) * len(level) for level in levels
    )
    executed = sum(len(p) for level in levels for p, _ in level)
    assert padded == 82
    assert executed == 32
    assert padded != executed


def test_level1_deepest_path_roots_at_last_level0_node():
    """Why the barrier is free: nothing on the critical path is delayed."""
    G = _NS()
    levels = G["_FR13_FIXED32_SUBTREE_LEVELS"]
    root_path = list(levels[0][0][0])
    last_root_node = root_path[-1]
    deepest = max(levels[1], key=lambda item: len(item[0]))
    assert len(deepest[0]) == 7
    assert int(deepest[1]) == last_root_node == 14

    # every OTHER level-1 path finishes no later than the critical path
    parent = [int(p) for p in G["_FR13_FIXED32_PARENT"]]
    depth = _tree_depths(parent)
    for path, _par in levels[1]:
        assert max(depth[n] for n in path) <= 12


def test_single_launch_candidate_trades_depth_for_traffic():
    """The parked one-launch kernel is the OTHER horn: 32-deep, zero handoff."""
    G = _NS()
    levels = [
        [(list(p), int(par)) for p, par in level]
        for level in G["_FR13_FIXED32_SUBTREE_LEVELS"]
    ]
    contract = G["_fr13_fixed32_gdn_single_launch_contract"](levels)
    assert contract["launches"] == 1
    assert contract["critical_node_steps"] == 32
    assert contract["state_export_writes"] == 0
    assert contract["state_parent_reads"] == 0
    # strictly deeper than the deployed schedule -- that is the tradeoff the
    # cost probe measures on silicon.
    assert contract["critical_node_steps"] > 12


def test_handoff_traffic_model_is_pinned():
    """Bytes moved by the fp32 state handoff, per GDN layer per step."""
    G = _NS()
    levels = G["_FR13_FIXED32_SUBTREE_LEVELS"]
    num_vh, dim_v, dim_k = 48, 128, 128
    state_tile = num_vh * dim_v * dim_k * 4
    assert state_tile == 3_145_728  # 3.0 MiB per node state

    export_nodes = set()
    for level in levels[1:]:
        for _path, par in level:
            export_nodes.add(int(par))
    assert export_nodes == set(G["_FR13_FIXED32_EXPORT_NODES"])
    writes = len(export_nodes)
    reads = sum(len(level) for level in levels[1:])
    assert writes == 5
    assert reads == 11
    assert writes * state_tile == 15_728_640
    assert reads * state_tile == 34_603_008

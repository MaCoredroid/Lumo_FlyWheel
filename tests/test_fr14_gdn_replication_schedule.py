"""FR14 GDN root-path replication: schedule arithmetic (CPU only, no triton).

These pin the load-bearing numbers of
``results/fr14_nvfp4_port_20260816/gdn_replication_design.md``. The design's
verdict turns on exactly two counts — the replicated schedule's makespan and its
node-step total — and on the claim that a replayed prefix is the *same* node
sequence the deployed route computes, which is what makes Tier-A hold by
construction.

If any of these fail, the design note's verdict must be re-derived before the
lever is re-priced or built.

Both the topology and the schedule builder are AST-lifted from their real
sources (the kernel module and the probe), so this suite stays importable on a
host without torch/triton and cannot drift from a re-typed copy.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPO / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py"
PROBE_PATH = REPO / "scripts" / "fr13_gdn_scan_fusion_cost_probe.py"
DESIGN_PATH = (
    REPO / "results" / "fr14_nvfp4_port_20260816" / "gdn_replication_design.md"
)


def _kernel_constants() -> dict[str, object]:
    wanted = {"_FR13_FIXED32_PARENT", "_FR13_FIXED32_SUBTREE_LEVELS"}
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in wanted for t in node.targets
        )
    ]
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), KERNEL_PATH, "exec"),
         namespace)
    assert wanted <= set(namespace), sorted(wanted - set(namespace))
    return namespace


def _replicated_chains(levels):
    """The probe's REAL builder, lifted — not a re-typed copy of it."""
    tree = ast.parse(PROBE_PATH.read_text(encoding="utf-8"))
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "replicated_chains"
    )

    class _G:  # the probe reads the topology off the kernel module
        _FR13_FIXED32_SUBTREE_LEVELS = levels

    namespace: dict[str, object] = {"G": _G, "SystemExit": SystemExit}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), PROBE_PATH, "exec"),
         namespace)
    return namespace["replicated_chains"](levels)


KERNEL = _kernel_constants()
PARENT = KERNEL["_FR13_FIXED32_PARENT"]
LEVELS = KERNEL["_FR13_FIXED32_SUBTREE_LEVELS"]
CHAINS = _replicated_chains(LEVELS)


def _depth(node: int) -> int:
    depth = 1
    while PARENT[node] != -1:
        node = PARENT[node]
        depth += 1
    return depth


# --------------------------------------------------------------------------
# the two counts the verdict turns on
# --------------------------------------------------------------------------


def test_replication_reaches_the_dag_depth_and_no_further() -> None:
    """Makespan 12 = the tree's dependency depth. This is the design's ONLY
    genuine improvement over the deployed schedule — and the deployed schedule
    already achieves it, which is why the improvement is worth nothing."""
    dag_depth = max(_depth(node) for node in range(len(PARENT)))
    assert dag_depth == 12
    assert max(len(pre) + len(own) for pre, own in CHAINS) == dag_depth
    deployed_makespan = (
        len(LEVELS[0][0][0]) + max(len(p) for p, _ in LEVELS[1])
    )
    assert deployed_makespan == dag_depth, (
        "the deployed route is already critical-path-optimal; replication "
        "cannot beat it on waves"
    )


def test_replication_costs_67_node_steps_against_the_deployed_32() -> None:
    """The number that kills the design. 2.09x the work for zero waves saved."""
    deployed = len(LEVELS[0][0][0]) + sum(len(p) for p, _ in LEVELS[1])
    replicated = sum(len(pre) + len(own) for pre, own in CHAINS)
    assert deployed == 32
    assert replicated == 67
    assert replicated - deployed == 35


def test_every_node_is_still_computed_and_owned_exactly_once() -> None:
    """Replication duplicates WORK, never OUTPUT: each node has exactly one
    owning program. That is what makes REPLAY_STEPS store-suppression a
    complete fix for the multi-writer race rather than a partial one."""
    owned: list[int] = []
    for _prefix, own in CHAINS:
        owned.extend(own)
    assert sorted(owned) == list(range(len(PARENT)))


def test_a_replayed_prefix_is_exactly_the_deployed_node_sequence() -> None:
    """Tier-A by construction: the prefix a branch replays is the same node
    sequence, in the same order, that the deployed root path computes before
    exporting the state that branch reads. Same operands, same partitioning,
    same accumulation order — so the state entering the branch is bit-identical
    and no order change is forced."""
    root = tuple(int(n) for n in LEVELS[0][0][0])
    by_head = {own[0]: (prefix, parent)
               for (prefix, own), (_nodes, parent)
               in zip(CHAINS[1:], LEVELS[1])}
    for head, (prefix, parent) in by_head.items():
        assert prefix == root[: root.index(parent) + 1], head
        # the replayed prefix ends on the node whose exported state the
        # deployed route hands this branch
        assert prefix[-1] == parent, head
        assert PARENT[head] == parent, head


def test_the_root_program_replays_nothing() -> None:
    prefix, own = CHAINS[0]
    assert prefix == ()
    assert tuple(own) == tuple(int(n) for n in LEVELS[0][0][0])


def test_replication_is_dominated_by_the_kernel_already_in_the_tree() -> None:
    """single_launch reaches zero handoff at 32 node-steps; replication needs
    67 for the same zero. Under any cost model monotone in node-steps at fixed
    handoff, replication cannot win — which is the design note's §4 argument,
    and it holds without measuring anything."""
    replicated = sum(len(pre) + len(own) for pre, own in CHAINS)
    single_launch_node_steps = 32   # one program, every node once
    assert single_launch_node_steps < replicated
    # and both reach the same destination
    assert single_launch_node_steps == len(PARENT)


# --------------------------------------------------------------------------
# the probe arm that settles it must stay honest about what it measures
# --------------------------------------------------------------------------


def test_the_stage0_arm_needs_no_new_kernel() -> None:
    """The cost question is answerable with the deployed kernel's existing
    specialisations: start-from-h0 and no-export. If either disappears, the
    probe silently stops measuring the design."""
    probe = PROBE_PATH.read_text(encoding="utf-8")
    kernel = KERNEL_PATH.read_text(encoding="utf-8")
    assert "def replicated_descriptor(" in probe
    assert "def launch_descriptor(" in probe
    assert "state_source=1" in probe and "export_mode=2" in probe
    assert "if STATE_SOURCE == 1:" in kernel
    assert "if EXPORT_MODE == 1:" in kernel


def test_the_stage0_verdict_is_pre_registered_against_single_launch() -> None:
    """Beating the two-launch route is not the bar — single_launch already
    does. A verdict rule that moves after the data is not a verdict."""
    probe = PROBE_PATH.read_text(encoding="utf-8")
    assert "PRE-REGISTERED" in probe
    assert 'replication_verdict = (' in probe
    assert 'bench["single_launch"]["us_p50"] - spread' in probe
    assert '"REFUTED"' in probe and '"BUILD"' in probe


def test_the_stage0_arm_declares_its_outputs_meaningless() -> None:
    """The replicated arm races identical stores to the same out rows. An arm
    that let a reader mistake it for a correctness result would be worse than
    no arm."""
    probe = PROBE_PATH.read_text(encoding="utf-8")
    assert '"out_values_are_meaningless_by_design": True' in probe


def test_the_design_note_exists_and_states_the_counts_it_is_judged_on() -> None:
    design = DESIGN_PATH.read_text(encoding="utf-8")
    for needle in ("67", "32", "12", "REPLAY_STEPS", "Tier-A"):
        assert needle in design, needle
    # the note must carry its own refutation conditions, not just its verdict
    assert "What would change this verdict" in design

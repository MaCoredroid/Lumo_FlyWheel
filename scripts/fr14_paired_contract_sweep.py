#!/usr/bin/env python3
"""Enumerate PAIRED structures around the fixed32 drafter shape, and report stale sides.

WHY THIS EXISTS
---------------
Three campaign blockers in a row were one defect shape: **a paired structure
updated on one side only.**

  * TAW emitter vs its mirrors
  * a bash pin vs its Python twin
  * `_fr13_fixed32_drafter_proposal_end`'s census half vs its runtime-evidence
    half (the 12th site: the census half was made pass-aware when the split
    landed, the runtime half twenty lines later still said `graph_replays: 1`)

Each was found by a boot. A boot is an expensive way to learn that two literals
disagree, so this enumerates the pairs and checks both sides mechanically.

WHAT A "PAIR" IS HERE
---------------------
Two artifacts that must encode the same fact, where nothing forces them to
change together:

  emitter/validator   the runtime writes a census record; the reducer validates it
  half/half           two blocks inside one function that describe one step
  contract/consumer   a shared constant vs a hand-written literal of that constant
  twin/twin           the two launcher families, which must stay identical
  bash/python         a launcher literal vs the Python that parses or mirrors it

The split-graph change moved five quantities. Any site still carrying the
pre-split literal for one of them is a stale side:

  mtp_forward_calls   4      -> 4 (ungated) | 2 (gated)
  graph_replays       1      -> 1 | 2 (armed ungated replays lo then hi)
  graph_captures      0|1    -> 0|1|2 (a split capture opens two scopes)
  main_tail_length    6      -> 6 | 8 (gated hands off at draft position 3)
  arctic_requested    12     -> 12 | 14

Run:  python3 scripts/fr14_paired_contract_sweep.py [--json out.json]
Exit 1 if any enumerated pair has a stale side.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
CENSUS = REPO / "scripts" / "fr13_fixed32_work_census.py"
TOPOLOGY = REPO / "scripts" / "fr13_fixed32_topology.py"
DRAFTER = REPO / "scripts" / "fr13_merged_drafter.py"
LAUNCHERS = (
    REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh",
    REPO / "scripts" / "fr14_armb_leg3_launch_nomiddleware.sh",
)


def injected_blob() -> str:
    """The runtime source the patcher execs into gdn_linear_attn.

    It is string content, so the patcher's own `ast.parse` does not cover it and
    neither does any import. Everything in it has to be checked textually.
    """
    tree = ast.parse(PATCHER.read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", None)
            == "_FR13_FIXED32_OBSERVED_RUNTIME_SOURCE"
        ):
            return ast.literal_eval(node.value)
    raise SystemExit("injected runtime blob not found")


def _topology():
    sys.path.insert(0, str(REPO / "scripts"))
    import fr13_fixed32_topology as topo

    return topo


# ---------------------------------------------------------------------------
# The pairs.  Each check returns (ok, detail).
# ---------------------------------------------------------------------------

def pair_contract_vs_blob(blob, topo):
    """The shared shape contract vs the literal the blob must carry.

    The blob cannot import the contract, so this is the only thing keeping them
    equal.  This is the pair the 12th site lived in.
    """
    m = re.search(
        r'int\(proposal\["graph_replays"\]\),\s*\)\s*not in (.+)$',
        blob,
        re.M,
    )
    if m is None:
        return False, "blob no longer carries the (calls, replays) shape tuple"
    literal = ast.literal_eval(m.group(1).strip())
    if tuple(topo.LEGAL_STEP_SHAPES) != literal:
        return False, (
            f"contract {topo.LEGAL_STEP_SHAPES} != blob literal {literal}"
        )
    return True, f"agree on {literal}"


def pair_contract_vs_blob_captures(blob, topo):
    m = re.search(r'proposal\["graph_captures"\]\) not in \(([0-9, ]+)\)', blob)
    if m is None:
        return False, "blob no longer bounds graph_captures"
    literal = tuple(int(x) for x in m.group(1).replace(" ", "").split(","))
    if tuple(topo.LEGAL_GRAPH_CAPTURES) != literal:
        return False, (
            f"contract {topo.LEGAL_GRAPH_CAPTURES} != blob literal {literal}"
        )
    return True, f"agree on {literal}"


def pair_contract_vs_census(topo):
    """The census must consume the contract, not re-type it."""
    text = CENSUS.read_text()
    missing = [
        name
        for name in ("LEGAL_STEP_SHAPES", "LEGAL_GRAPH_CAPTURES",
                     "LEGAL_HANDOFF_SHAPES")
        if name not in text
    ]
    if missing:
        return False, f"census does not consume {missing}"
    return True, "census imports all three contract tuples"


def pair_census_halves():
    """`drafter` and `drafter_runtime` describe ONE step and must agree.

    The census already cross-checks them at runtime via `runtime_projection`;
    what this asserts is that neither half reintroduced a bare literal for a
    split-sensitive quantity.
    """
    text = CENSUS.read_text()
    stale = []
    if re.search(r'_expect\(\s*runtime_mtp_calls,\s*MTP_FORWARD_CALLS\b', text):
        stale.append("drafter_runtime.mtp_forward_calls pinned to 4")
    if re.search(r'"graph_replays"\)\s*,\s*\n?\s*1,', text):
        stale.append("drafter_runtime.graph_replays pinned to 1")
    if "graph_captures not in (0, 1)" in text:
        stale.append("drafter_runtime.graph_captures bounded at 0|1")
    # `reference_event` legitimately builds the UNGATED fixture; the pair that
    # matters is the validator's runtime_projection, which must derive.
    proj = text[text.index("runtime_projection = {"):]
    proj = proj[: proj.index("}")]
    if "ARCTIC_MAIN_TAIL_LENGTH" in proj:
        stale.append("runtime_projection.main_tail_length pinned to 6")
    return (not stale), (stale or "both halves derive from the observed shape")


def pair_emitter_halves(blob):
    """proposal_end writes a census half and a runtime-evidence half."""
    stale = []
    if re.search(r'"graph_replays":\s*1,', blob):
        stale.append('drafter_runtime emitter writes graph_replays: 1')
    if re.search(r'"mtp_forward_calls":\s*4,', blob):
        stale.append('drafter_runtime emitter writes mtp_forward_calls: 4')
    if re.search(r'"mtp_forward_rows":\s*4 \* batch,', blob):
        stale.append('drafter_runtime emitter writes mtp_forward_rows: 4*batch')
    if re.search(r'"main_tail_length":\s*6,', blob):
        stale.append('census emitter writes main_tail_length: 6')
    # Scope to the DRAFTER evidence chain by VARIABLE name: the target
    # forward-graph chain in the same function uses a bare `evidence` and
    # legitimately replays once per step, so a function-level scope would
    # produce a false positive on it.
    if re.search(
        r'int\(drafter_evidence\.get\("matching_replays", -1\)\)\s*\n?\s*!= 1\b',
        blob,
    ):
        stale.append("drafter_evidence demands matching_replays == 1")
    # and proposal_end's own check must be TIED to the proposal, not a literal
    if not re.search(
        r'int\(evidence\.get\("matching_replays", -1\)\)\s*\n\s*'
        r'!= int\(proposal\["graph_replays"\]\)',
        blob,
    ):
        stale.append("proposal_end evidence check is not tied to graph_replays")
    return (not stale), (stale or "both emitter halves derive from the proposal")


def pair_observer_vs_context(blob):
    """The per-forward observer vs the capture context it validates against."""
    stale = []
    if 'proposal.get("graph_captures") != 1' in blob:
        stale.append("tree-attention observer pins graph_captures to 1")
    if "tree_calls not in (0, 1, 2, 3)" in blob:
        stale.append("tree-attention observer bounds forwards at 4")
    return (not stale), (stale or "observer is segment- and pass-aware")


def pair_launcher_twins():
    """The two launcher families must carry identical FR14 blocks."""
    texts = [p.read_text() for p in LAUNCHERS]
    markers = (
        "FR14_SUFFIX_PASS_GATE",
        "FR14_GATE_SPLIT_GRAPH",
        "FR14_FUSED_DRAFT_TOPK",
        "gqa_pair_splitk",
        "_fr14_gate_incompat",
        "fr14_suffix_pass_gate.cfg",
    )
    mismatched = [
        m for m in markers
        if (m in texts[0]) != (m in texts[1])
    ]
    if mismatched:
        return False, f"present in only one launcher: {mismatched}"
    counts = {
        m: (texts[0].count(m), texts[1].count(m)) for m in markers
    }
    skewed = {m: c for m, c in counts.items() if c[0] != c[1]}
    if skewed:
        return False, f"occurrence counts differ: {skewed}"
    return True, f"{len(markers)} FR14 blocks identical across both families"


def pair_bash_cfg_vs_python_parser():
    """The launcher writes the gate sidecar; the gate module parses it."""
    launcher = LAUNCHERS[0].read_text()
    m = re.search(
        r'echo "\$\{FR14_SUFFIX_PASS_GATE_NGRAM:-\d+\} '
        r'\$\{FR14_SUFFIX_PASS_GATE_MIN_AGREE:-[\d.]+\} '
        r'\$\{FR14_SUFFIX_PASS_GATE_MIN_HISTORY:-\d+\}"',
        launcher,
    )
    if m is None:
        return False, "launcher no longer writes the 3-field sidecar"
    gate = (REPO / "scripts" / "fr14_suffix_pass_gate.py").read_text()
    if "len(raw) != 3" not in gate:
        return False, "gate parser no longer requires exactly 3 fields"
    return True, "3 fields written, 3 fields required"


def pair_topology_vs_drafter():
    """decide_fixed32's gated widths vs the topology contract."""
    topo = _topology()
    text = DRAFTER.read_text()
    for name in (
        "GATED_ARCTIC_MAIN_TAIL_LENGTH",
        "GATED_ARCTIC_LOOKUP_TOKENS_PER_REQUEST",
        "GATED_MTP_K",
    ):
        if name not in text:
            return False, f"decide_fixed32 does not consume {name}"
    if topo.GATED_MTP_K + topo.GATED_ARCTIC_MAIN_TAIL_LENGTH != 11:
        return False, "gated shape does not reach draft position 10"
    return True, "drafter consumes the topology constants"


PAIRS = (
    ("contract <-> injected blob (step shapes)", "contract/consumer",
     lambda blob, topo: pair_contract_vs_blob(blob, topo)),
    ("contract <-> injected blob (graph_captures)", "contract/consumer",
     lambda blob, topo: pair_contract_vs_blob_captures(blob, topo)),
    ("contract <-> census validator", "contract/consumer",
     lambda blob, topo: pair_contract_vs_census(topo)),
    ("census drafter <-> census drafter_runtime", "half/half",
     lambda blob, topo: pair_census_halves()),
    ("proposal_end census half <-> runtime half", "half/half",
     lambda blob, topo: pair_emitter_halves(blob)),
    ("tree-attn observer <-> capture context", "emitter/validator",
     lambda blob, topo: pair_observer_vs_context(blob)),
    ("launcher family A <-> launcher family B", "twin/twin",
     lambda blob, topo: pair_launcher_twins()),
    ("launcher sidecar <-> gate parser", "bash/python",
     lambda blob, topo: pair_bash_cfg_vs_python_parser()),
    ("topology contract <-> decide_fixed32", "contract/consumer",
     lambda blob, topo: pair_topology_vs_drafter()),
)


def sweep():
    blob = injected_blob()
    topo = _topology()
    rows = []
    for name, kind, check in PAIRS:
        ok, detail = check(blob, topo)
        rows.append(
            {"pair": name, "kind": kind, "ok": bool(ok), "detail": detail}
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = sweep()
    stale = [r for r in rows if not r["ok"]]
    for r in rows:
        print(f"[{'OK  ' if r['ok'] else 'STALE'}] {r['pair']:48s} {r['detail']}")
    print(f"\n{len(rows)} pairs enumerated, {len(stale)} stale")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "schema": "fr14.paired_contract_sweep.v1",
                    "pairs": rows,
                    "n_pairs": len(rows),
                    "n_stale": len(stale),
                },
                indent=1,
            )
        )
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
